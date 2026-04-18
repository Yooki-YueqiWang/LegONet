#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the transport (u u_x) H-block on the 1D Dirichlet (Shen–Legendre) baseplate.

PDE mechanism being learned:
    a_t = F^θ_uux(a) ≈ F^ref_uux(a) = P_b[ u u_x ]

Block parameterization (H-block, conservative):
    F^θ_uux(a) = J ∇_a H^{a,θ}(a)

where J = -M^{-1} S is the fixed skew-symmetric structure operator representing ∂_x
on the Shen space (S_ij = ⟨φ'_i, φ_j⟩), and H^{a,θ}: R^K → R is a scalar Hamiltonian
generator of the form:

    H^{a,θ}(a) = ∫_{-1}^{1} h^θ(u(x)) dx,   u(x) = Σ_k a_k φ_k(x)

The density network h^θ: R → R is trained so that dh^θ/du ≈ 0.5 u, which makes
the Hamiltonian ≈ ∫ 0.5 u^2 dx and the resulting vector field reproduce u u_x:

    J ∇_a H = J P_b[ dh^θ/du(u) ] ≈ J P_b[ 0.5 u^2 ] = P_b[ u u_x ]

This construction guarantees dH/dt = 0 along any trajectory of the isolated
sub-dynamics (Property 4 of the paper).

Training objective (instantaneous operator matching, Eq. 11):
    min_{h^θ}  E_{a ~ µ_b} [ ∫ (dh^θ/du(u(x)) - g(x))^2 dx ]
where g = P_V[ 0.5 u^2 ] is the L^2 projection of the quadratic density.

Outputs:
    models/uux_block/checkpoint.pt  — model weights
    models/uux_block/config.json    — architecture and hyperparameter record
    plots/uux_block/                — training curve and operator comparison plot
"""

import os
import json
import math
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 1234) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ============================================================
# Shen–Legendre spectral utilities
# ============================================================

def legendre_quadrature_1d(n_points: int = 256, device: str = "cpu",
                            dtype: torch.dtype = torch.float64):
    """Gauss–Legendre quadrature nodes and weights on [-1, 1]."""
    x_np, w_np = np.polynomial.legendre.leggauss(n_points)
    x = torch.tensor(x_np, dtype=dtype, device=device)
    w = torch.tensor(w_np, dtype=dtype, device=device)
    return x, w


def legendre_polynomials_and_deriv(x: torch.Tensor, max_n: int):
    """
    Evaluate Legendre polynomials P_0, …, P_{max_n} and their derivatives
    at nodes x using the three-term recurrence.
    Returns P, Pp each of shape (max_n+1, N).
    """
    x = x.reshape(-1)
    N = x.numel()
    device, dtype = x.device, x.dtype

    P = torch.zeros((max_n + 1, N), device=device, dtype=dtype)
    P[0] = 1.0
    if max_n >= 1:
        P[1] = x
    for n in range(1, max_n):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)

    Pp = torch.zeros_like(P)
    one_minus_x2 = 1.0 - x ** 2
    for n in range(1, max_n + 1):
        Pp[n] = n * (P[n - 1] - x * P[n]) / one_minus_x2
    return P, Pp


def shen_basis_and_deriv(x: torch.Tensor, order: int):
    """
    Shen–Legendre Dirichlet basis: φ_k = L_k - L_{k+2}, φ_k(±1) = 0.
    Returns phi, phi_x each of shape (order, N).
    """
    x = x.reshape(-1)
    max_n = order + 1
    P, Pp = legendre_polynomials_and_deriv(x, max_n)
    phi_list, phix_list = [], []
    for k in range(order):
        phi_list.append(P[k] - P[k + 2])
        phix_list.append(Pp[k] - Pp[k + 2])
    return torch.stack(phi_list, 0), torch.stack(phix_list, 0)


def build_mass_matrix(phi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mass matrix M_ij = ⟨φ_i, φ_j⟩_{L^2}. Shape: (K, K)."""
    wv = w.view(1, -1)
    return (phi * wv) @ phi.T


def u_to_a(u_grid: torch.Tensor, phi: torch.Tensor, w: torch.Tensor,
           M: torch.Tensor = None) -> torch.Tensor:
    """Discrete L^2 projection: a = M^{-1} Φ^T (u ⊙ w)."""
    if M is None:
        M = build_mass_matrix(phi, w)
    b = phi @ (u_grid * w)
    return torch.linalg.solve(M, b)


def a_to_u(a: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Reconstruct physical field: u = Φ a. Accepts (K,) or (B,K)."""
    return a @ phi


def build_J_shen(x_grid: torch.Tensor, w_grid: torch.Tensor, order: int):
    """
    Build the skew-symmetric structure operator J = -M^{-1} S representing ∂_x
    on the Shen trial space:
        J_ij = -[M^{-1} S]_ij,   S_ij = ⟨φ'_i, φ_j⟩_{L^2}

    The operator J is skew-adjoint with respect to the M inner product:
        (M J)^T = -(M J),   i.e.  J^T M = -M J

    This ensures H-block trajectories conserve the Hamiltonian H^{a,θ}.

    Returns J, M, S, phi, phi_x (all as tensors).
    """
    phi, phi_x = shen_basis_and_deriv(x_grid, order)
    w = w_grid.view(1, -1)
    M = (phi * w) @ phi.T
    S = (phi_x * w) @ phi.T
    J = -torch.linalg.solve(M, S)
    return J, M, S, phi, phi_x


@torch.no_grad()
def verify_J_hamiltonian(J: torch.Tensor, M: torch.Tensor) -> None:
    """
    Verify that J is skew-adjoint with respect to M:
        ||M J + (M J)^T|| / ||M J|| ≈ 0

    This is a necessary condition for the H-block to conserve its Hamiltonian.
    """
    MJ = M @ J
    skew = MJ + MJ.T
    rel = torch.norm(skew) / (torch.norm(MJ) + 1e-16)
    print(f"[J check] rel skew-adjoint error = {rel.item():.3e}")


# ============================================================
# Reference operator and training data
# ============================================================

def sample_coefficients(batch_size: int, order: int, alpha: float,
                         device: torch.device, dtype: torch.dtype,
                         gen: torch.Generator) -> torch.Tensor:
    """
    Sample from spectral-decay Gaussian prior µ_b:
        a_k ~ N(0, σ_k^2),  σ_k = 1 / (1 + k)^alpha
    """
    k = torch.arange(order, device=device, dtype=dtype)
    sigma = 1.0 / (1.0 + k) ** alpha
    z = torch.randn(batch_size, order, generator=gen, device=device, dtype=dtype)
    return z * sigma


def build_uux_true_weak_loop(a_u: torch.Tensor, phi: torch.Tensor,
                              w_grid: torch.Tensor, M: torch.Tensor,
                              J: torch.Tensor) -> torch.Tensor:
    """
    Compute the reference transport operator F^ref_uux in coefficient space:
        g^(n) = P_V[ 0.5 (u^(n))^2 ]   (L^2 projection of the density)
        F^ref_uux(a^(n)) = g^(n) @ J^T

    The density g = 0.5 u^2 comes from the quadratic Hamiltonian H = ∫ 0.5 u^2 dx,
    so that J ∇_a H = J P_V[u] = J P_V[dH/du].

    Returns tensor of shape (N, K).
    """
    N, K = a_u.shape
    out = torch.empty((N, K), device=a_u.device, dtype=a_u.dtype)
    for n in range(N):
        u = a_u[n] @ phi
        g = 0.5 * u * u
        g_a = u_to_a(g, phi, w_grid, M=M)
        out[n] = g_a @ J.T
    return out


# ============================================================
# H-block: Hamiltonian density network
# ============================================================

class HamiltonianNet(nn.Module):
    """
    Pointwise density network h^θ: R → R defining the Hamiltonian:

        H^{a,θ}(a) = ∫_{-1}^{1} h^θ(u(x)) dx

    The vector field is recovered as:
        F^θ_uux(a) = J ∇_a H^{a,θ}(a) = J P_b[ dh^θ/du(u) ]

    Training drives dh^θ/du → 0.5 u pointwise, making H ≈ ∫ 0.5 u^2 dx
    and F^θ_uux ≈ P_b[ u u_x ] (since J P_b[ u ] = P_b[ ∂_x(0.5 u^2) / u ] = P_b[u u_x]).
    """

    def __init__(self, width: int = 64, depth: int = 4, act: str = "gelu"):
        super().__init__()
        acts = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        if act.lower() not in acts:
            raise ValueError(f"Unknown activation: {act}")
        Act = acts[act.lower()]
        layers = [nn.Linear(1, width), Act()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), Act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate h^θ(u) pointwise.
        Input u: (B, Q) grid values or (Q,) single sample.
        Output: same shape as input.
        """
        if u.dim() == 1:
            return self.net(u.unsqueeze(-1)).squeeze(-1)
        return self.net(u.unsqueeze(-1)).squeeze(-1)


def uux_loss(H_net: nn.Module, a_u: torch.Tensor, a_g_true: torch.Tensor,
             phi: torch.Tensor, w_grid: torch.Tensor) -> torch.Tensor:
    """
    Operator-matching loss for the transport H-block (Eq. 12):

        L(h^θ) = ∫ (dh^θ/du(u(x)) - g_true(x))^2 dx
               ≈ Σ_q w_q (dh^θ/du(u(x_q)) - g_true(x_q))^2

    where g_true(x) = Φ a_g_true (the reference density in physical space).
    The gradient dh^θ/du is computed via autograd through H_net.
    """
    u_grid = a_to_u(a_u, phi)          # (B, Q)
    with torch.enable_grad():
        u_req = u_grid.detach().requires_grad_(True)
        f = H_net(u_req)
        g_pred = torch.autograd.grad(f.sum(), u_req, create_graph=True)[0]
    g_true = a_to_u(a_g_true.detach(), phi)
    w = w_grid.view(1, -1)
    return ((g_pred - g_true) ** 2 * w).mean()


def evaluate_uux(H_net: nn.Module, data_loader: DataLoader, phi: torch.Tensor,
                  w_grid: torch.Tensor, device: torch.device):
    """
    Evaluate the weighted relative L^2 error of the learned density:
        rel = ||dh^θ/du(u) - g_true||_{w,2} / ||g_true||_{w,2}

    Autograd is force-enabled so that dh^θ/du can be computed inside
    the evaluation loop even under torch.no_grad().

    Returns (avg_loss, rel_err).
    """
    H_net.eval()
    w = w_grid.view(1, -1).to(device)

    total_loss, total_num = 0.0, 0
    num_res_w2, num_true_w2 = 0.0, 0.0

    for a_u, a_target in data_loader:
        a_u = a_u.to(device)
        a_target = a_target.to(device)

        u_grid = a_to_u(a_u, phi)

        with torch.enable_grad():
            u_req = u_grid.detach().requires_grad_(True)
            f = H_net(u_req)
            g_pred = torch.autograd.grad(
                f.sum(), u_req, create_graph=False
            )[0]

        g_true = a_to_u(a_target, phi)
        residual = g_pred - g_true
        loss = (residual ** 2 * w).mean()

        B = a_u.size(0)
        total_loss += loss.item() * B
        total_num += B
        num_res_w2 += torch.sum(residual ** 2 * w).item()
        num_true_w2 += torch.sum(g_true ** 2 * w).item()

    avg_loss = total_loss / max(total_num, 1)
    rel_err = (num_res_w2 ** 0.5) / (num_true_w2 ** 0.5 + 1e-16)
    return avg_loss, rel_err


# ============================================================
# Plotting utilities
# ============================================================

def plot_train_curve(hist: dict, outpath: str) -> None:
    """Plot training and test loss curves on a log scale."""
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["test_loss"], label="test_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("u u_x block training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_uux_operator_compare(trained_H: nn.Module, a_u_test: torch.Tensor,
                               a_g_test: torch.Tensor, J: torch.Tensor,
                               phi: torch.Tensor, x_grid: torch.Tensor,
                               w_grid: torch.Tensor, idx: int, outpath: str,
                               cut: int = 10) -> float:
    """
    Compare the reference F^ref_uux = a_g @ J^T (solid) against the learned
    F^θ_uux = a_g_pred @ J^T (dashed) in physical space for a held-out sample.

    The learned density g_pred is obtained by evaluating dh^θ/du(u(x)) and
    projecting back to coefficient space.  Boundary points are trimmed by `cut`
    on each side to reduce endpoint interpolation artefacts.

    Returns the weighted relative L^2 error of F^θ_uux vs F^ref_uux.
    """
    # Reference transport in coefficient space
    a0 = a_u_test[idx]
    a_g_true = a_g_test[idx]
    a_uux_true = a_g_true @ J.T

    # Learned: dh^θ/du -> project -> apply J
    with torch.enable_grad():
        u = (a0 @ phi).detach()
        u_req = u.requires_grad_(True)
        f = trained_H(u_req)
        g_pred_grid = torch.autograd.grad(f.sum(), u_req, create_graph=False)[0]

    M_loc = build_mass_matrix(phi, w_grid)
    a_g_pred = u_to_a(g_pred_grid, phi, w_grid, M=M_loc)
    a_uux_pred = a_g_pred @ J.T

    # Back to physical space
    uux_true = (a_uux_true @ phi).detach().cpu()
    uux_pred = (a_uux_pred @ phi).detach().cpu()
    x = x_grid.detach().cpu()
    w = w_grid.detach().cpu()

    if cut > 0:
        x = x[cut:-cut]
        w = w[cut:-cut]
        uux_true = uux_true[cut:-cut]
        uux_pred = uux_pred[cut:-cut]

    num = torch.sqrt(torch.sum((uux_pred - uux_true) ** 2 * w))
    den = torch.sqrt(torch.sum(uux_true ** 2 * w)) + 1e-16
    rel = (num / den).item()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x.numpy(), uux_true.numpy(), "k-", lw=2.6, label="true")
    ax.plot(x.numpy(), uux_pred.numpy(), "r--", lw=2.6, label="predicted")
    ax.set_title(
        rf"$u\,u_x$ comparison" + "\n" + rf"rel err={rel:.3e}",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$(u\,u_x)(x)$", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.2)
    for lab in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
        lab.set_fontweight("bold")
    leg = ax.legend(frameon=True, fontsize=16, loc="best")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return rel


# ============================================================
# Main training routine
# ============================================================

def main():
    set_seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # --- Shen–Legendre baseplate hyperparameters ---
    order = 96
    n_points = 256
    alpha = 0.5

    # --- Dataset sizes ---
    n_train = 20000
    n_test = 2000

    # --- H-block architecture: pointwise density network h^θ: R → R ---
    width = 64
    depth = 4
    act = "gelu"

    # --- Optimizer ---
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = 128
    num_epochs = 100

    model_dir = os.path.join("models", "uux_block")
    plot_dir = os.path.join("plots", "uux_block")
    ensure_dir(model_dir)
    ensure_dir(plot_dir)

    # Build Shen–Legendre baseplate objects
    x_grid, w_grid = legendre_quadrature_1d(n_points, device=device, dtype=dtype)
    J, M, S, phi, phi_x = build_J_shen(x_grid, w_grid, order)
    verify_J_hamiltonian(J, M)

    # Pseudo-inverse of J for converting F^ref_uux to density coefficients a_g.
    # The training targets are a_g = J^{-1} a_uux, so the loss is applied
    # in the density (g) space rather than the vector-field (uux) space.
    J_pinv = torch.linalg.pinv(J.detach())

    # Build training and test coefficient datasets
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    a_u_train = sample_coefficients(n_train, order, alpha, device, dtype, gen)
    a_u_test = sample_coefficients(n_test, order, alpha, device, dtype, gen)

    # Reference transport operator evaluated on the grid
    a_uux_train = build_uux_true_weak_loop(a_u_train, phi, w_grid, M, J)
    a_uux_test = build_uux_true_weak_loop(a_u_test, phi, w_grid, M, J)

    # Convert to density (g) coefficients for training the density network
    a_g_train = a_uux_train @ J_pinv.T
    a_g_test = a_uux_test @ J_pinv.T

    train_loader = DataLoader(
        TensorDataset(a_u_train, a_g_train),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(a_u_test, a_g_test),
        batch_size=batch_size, shuffle=False,
    )

    # Initialize H-block density network
    H_net = HamiltonianNet(width=width, depth=depth, act=act).to(device).double()
    opt = torch.optim.AdamW(H_net.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train_loss": [], "test_loss": [], "test_rel": [], "lr": []}

    for ep in range(1, num_epochs + 1):
        H_net.train()
        running, total = 0.0, 0

        for a_u_b, a_g_b in train_loader:
            a_u_b = a_u_b.to(device)
            a_g_b = a_g_b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = uux_loss(H_net, a_u_b, a_g_b, phi, w_grid)
            loss.backward()
            opt.step()
            B = a_u_b.size(0)
            running += loss.item() * B
            total += B

        train_loss = running / max(total, 1)
        test_loss, test_rel = evaluate_uux(H_net, test_loader, phi, w_grid, device)
        cur_lr = opt.param_groups[0]["lr"]

        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["test_rel"].append(test_rel)
        hist["lr"].append(cur_lr)

        print(
            f"Epoch {ep:3d} | lr={cur_lr:.3e} | "
            f"train={train_loss:.3e} | test={test_loss:.3e} | rel={test_rel:.3e}"
        )

    trained = copy.deepcopy(H_net).eval()
    for p in trained.parameters():
        p.requires_grad_(False)

    # Diagnostic plots
    plot_train_curve(hist, os.path.join(plot_dir, "train_curve.png"))
    rel = plot_uux_operator_compare(
        trained, a_u_test, a_g_test, J, phi, x_grid, w_grid,
        idx=11, outpath=os.path.join(plot_dir, "uux_operator_compare.png"), cut=15,
    )
    print("[uux] operator compare rel err:", rel)

    # Save checkpoint
    ckpt = {
        "type": "uux_hamiltonian_net",
        "order": order,
        "n_points": n_points,
        "alpha": alpha,
        "arch": {"width": width, "depth": depth, "act": act},
        "state_dict": trained.state_dict(),
        "hist": hist,
    }
    torch.save(ckpt, os.path.join(model_dir, "checkpoint.pt"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(
            {k: ckpt[k] for k in ["type", "order", "n_points", "alpha", "arch"]},
            f, indent=2,
        )

    print(f"[SAVE] model  -> {model_dir}")
    print(f"[SAVE] plots  -> {plot_dir}")


if __name__ == "__main__":
    main()
