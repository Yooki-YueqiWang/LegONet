#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the diffusion (u_xx) E-block on the 1D Dirichlet (Shen–Legendre) baseplate.

PDE mechanism being learned:
    a_t = F^θ_uxx(a) ≈ F^ref_uxx(a) = P_b[ u_xx ],   u = Σ a_k φ_k

Block parameterization (E-block, dissipative):
    F^θ_uxx(a) = -G ∇_a E^{a,θ}(a),   G = M^{-1}

where E^{a,θ}: R^K → R is a scalar energy generator implemented as a small MLP,
and M is the Shen–Legendre mass matrix. The negative gradient of E gives the
vector field, which guarantees dE/dt ≤ 0 along any trajectory of the isolated
sub-dynamics (Property 3 of the paper).

In the Shen–Legendre basis the reference operator reduces to:
    F^ref_uxx(a) = -M^{-1} K a,   (K_ij = ⟨φ'_i, φ'_j⟩)
so the energy-gradient form automatically targets a quadratic energy with
Hessian ≈ K (the spectral stiffness matrix).

Training objective (instantaneous operator matching, Eq. 11):
    min_θ  E_{a ~ µ_b} [ ||∇_a E^{a,θ}(a) + F^ref_uxx(a)||^2 ]

where µ_b is a spectral-decay Gaussian prior on the Shen coefficient space.

Outputs:
    models/uxx_block/checkpoint.pt  — model weights and precomputed K matrices
    models/uxx_block/config.json    — architecture and hyperparameter record
    plots/uxx_block/                — training curve and operator comparison plot
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


# ============================================================
# Shen–Legendre spectral utilities
# ============================================================

def legendre_quadrature_1d(n_points: int = 256, device: str = "cpu",
                            dtype: torch.dtype = torch.float64):
    """
    Return Gauss–Legendre quadrature nodes and weights on [-1, 1].

    Returns
    -------
    x : Tensor of shape (n_points,) — quadrature nodes
    w : Tensor of shape (n_points,) — quadrature weights
    """
    x_np, w_np = np.polynomial.legendre.leggauss(n_points)
    x = torch.tensor(x_np, dtype=dtype, device=device)
    w = torch.tensor(w_np, dtype=dtype, device=device)
    return x, w


def legendre_polynomials_and_deriv(x: torch.Tensor, max_n: int):
    """
    Evaluate Legendre polynomials P_0, …, P_{max_n} and their first
    derivatives at the quadrature nodes x using the three-term recurrence.

    Returns
    -------
    P  : Tensor of shape (max_n+1, N)
    Pp : Tensor of shape (max_n+1, N)  — dP_n/dx
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

    # Derivative recurrence: P'_n = n (P_{n-1} - x P_n) / (1 - x^2)
    Pp = torch.zeros_like(P)
    one_minus_x2 = 1.0 - x ** 2
    for n in range(1, max_n + 1):
        Pp[n] = n * (P[n - 1] - x * P[n]) / one_minus_x2

    return P, Pp


def shen_basis_and_deriv(x: torch.Tensor, order: int):
    """
    Evaluate the Shen–Legendre Dirichlet basis functions and their derivatives.

    φ_k(x) = L_k(x) - L_{k+2}(x),  k = 0, …, order-1

    This basis satisfies φ_k(±1) = 0 by construction, making it suitable
    for homogeneous Dirichlet boundary conditions.

    Returns
    -------
    phi   : Tensor of shape (order, N)   — basis values at nodes
    phi_x : Tensor of shape (order, N)   — derivative values at nodes
    """
    x = x.reshape(-1)
    max_n = order + 1
    P, Pp = legendre_polynomials_and_deriv(x, max_n)

    phi_list, phix_list = [], []
    for k in range(order):
        phi_list.append(P[k] - P[k + 2])
        phix_list.append(Pp[k] - Pp[k + 2])

    phi = torch.stack(phi_list, dim=0)    # (K, N)
    phi_x = torch.stack(phix_list, dim=0) # (K, N)
    return phi, phi_x


def build_mass_matrix(phi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram (mass) matrix M_ij = ⟨φ_i, φ_j⟩_{L^2} via quadrature.
    Shape: (K, K).
    """
    wv = w.view(1, -1)
    return (phi * wv) @ phi.T


def build_K(phi_x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute the spectral stiffness matrix K_ij = ⟨φ'_i, φ'_j⟩_{L^2} via quadrature.
    Shape: (K, K).  The reference diffusion operator is F^ref(a) = -M^{-1} K a.
    """
    wv = w.view(1, -1)
    return (phi_x * wv) @ phi_x.T


def u_to_a(u_grid: torch.Tensor, phi: torch.Tensor, w: torch.Tensor,
           M: torch.Tensor = None) -> torch.Tensor:
    """
    Project a grid-valued function u onto the Shen basis via discrete L^2 projection:
        a = M^{-1} Φ^T (u ⊙ w)
    Returns coefficient vector of shape (K,).
    """
    if M is None:
        M = build_mass_matrix(phi, w)
    b = phi @ (u_grid * w)
    a = torch.linalg.solve(M, b)
    return a


def a_to_u(a: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a physical-space field from Shen coefficients: u = Φ a.
    Works for a of shape (K,) or (B, K).
    """
    return a @ phi


# ============================================================
# Reference operator and training data
# ============================================================

def sample_coefficients(batch_size: int, order: int, alpha: float,
                         device: torch.device, dtype: torch.dtype,
                         gen: torch.Generator) -> torch.Tensor:
    """
    Sample coefficient vectors from the spectral-decay Gaussian prior µ_b:
        a_k ~ N(0, σ_k^2),  σ_k = 1 / (1 + k)^alpha

    This prior produces smooth random fields whose amplitude decays with mode index,
    matching the coefficient distribution of typical PDE solutions on the baseplate.
    """
    k = torch.arange(order, device=device, dtype=dtype)
    sigma = 1.0 / (1.0 + k) ** alpha
    z = torch.randn(batch_size, order, generator=gen, device=device, dtype=dtype)
    return z * sigma


def apply_uxx_operator(a_u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Apply the reference diffusion operator in coefficient space:
        F^ref_uxx(a) = -K a   (with the M^{-1} factor absorbed into K here)

    K here is the full stiffness matrix K_ij = ⟨φ'_i, φ'_j⟩, which in the
    Shen basis satisfies M^{-1} K a = -Pi_V[u_xx] for u = Φ a.
    """
    return -(a_u @ K.T)


def build_uxx_datasets(n_train: int, n_test: int, order: int, alpha: float,
                        K: torch.Tensor, device: torch.device,
                        dtype: torch.dtype, seed: int = 1234):
    """
    Build paired (a_u, a_uxx) datasets for operator-matching training.
    Returns four tensors: a_u_train, a_uxx_train, a_u_test, a_uxx_test.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    a_u_train = sample_coefficients(n_train, order, alpha, device, dtype, gen)
    a_uxx_train = apply_uxx_operator(a_u_train, K)

    a_u_test = sample_coefficients(n_test, order, alpha, device, dtype, gen)
    a_uxx_test = apply_uxx_operator(a_u_test, K)

    return a_u_train, a_uxx_train, a_u_test, a_uxx_test


# ============================================================
# E-block: scalar energy generator
# ============================================================

class EnergyNet(nn.Module):
    """
    MLP parameterization of the scalar energy generator E^{a,θ}: R^K → R.

    The learned E-block vector field is:
        F^θ_uxx(a) = -G ∇_a E^{a,θ}(a) = -M^{-1} ∇_a E^{a,θ}(a)

    Gradients are computed via automatic differentiation in `energy_and_grad`.
    The architecture uses GELU activations, which provide smooth gradients
    suitable for the operator-matching loss.
    """

    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 256):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a).squeeze(-1)


def energy_and_grad(model: nn.Module, a: torch.Tensor, create_graph: bool = True):
    """
    Evaluate E^{a,θ}(a) and its gradient ∇_a E^{a,θ}(a) via autograd.

    Autograd is force-enabled so that this function works correctly even
    inside a torch.no_grad() context (e.g., during evaluation).

    Returns
    -------
    E     : Tensor of shape (B,)
    gradE : Tensor of shape (B, K)  — ∇_a E at each sample
    """
    a = a.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        E = model(a)
        gradE = torch.autograd.grad(
            outputs=E.sum(),
            inputs=a,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]
    return E, gradE


def uxx_energy_loss(model: nn.Module, a_u: torch.Tensor,
                     a_uxx: torch.Tensor) -> torch.Tensor:
    """
    Operator-matching loss for the diffusion E-block (Eq. 12):

        L(θ) = (1/M) Σ_m || ∇_a E^{a,θ}(a^(m)) - F^ref_uxx(a^(m)) ||^2
             = (1/M) Σ_m || ∇_a E^{a,θ}(a^(m)) + a_uxx^(m) ||^2

    The sign convention follows from F^θ_uxx = -∇_a E, so the residual is
    (∇E - (-a_uxx)) = (∇E + a_uxx).
    """
    _, gradE = energy_and_grad(model, a_u, create_graph=True)
    return ((gradE + a_uxx) ** 2).mean()


@torch.no_grad()
def evaluate_uxx_energy(model: nn.Module, loader: DataLoader,
                         device: torch.device):
    """
    Evaluate average operator-matching loss and weighted relative L^2 error
    on a held-out dataset.

    Returns
    -------
    avg_loss : float
    rel_err  : float — ||F^θ - F^ref|| / ||F^ref||
    """
    model.eval()
    total_loss, total_num = 0.0, 0
    num_res2, num_true2 = 0.0, 0.0

    for a_u, a_uxx in loader:
        a_u = a_u.to(device)
        a_uxx = a_uxx.to(device)
        _, gradE = energy_and_grad(model, a_u, create_graph=False)
        res = gradE + a_uxx
        loss = (res ** 2).mean()
        B = a_u.size(0)
        total_loss += loss.item() * B
        total_num += B
        num_res2 += torch.sum(res ** 2).item()
        num_true2 += torch.sum(a_uxx ** 2).item()

    avg_loss = total_loss / max(total_num, 1)
    rel_err = (num_res2 ** 0.5) / ((num_true2 ** 0.5) + 1e-16)
    return avg_loss, rel_err


@torch.no_grad()
def estimate_K_from_energy(model: nn.Module, order: int, device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor:
    """
    Estimate the effective stiffness matrix learned by the energy net by
    evaluating ∇_a E at each standard basis vector e_j.

    Since F^θ(a) ≈ -∇_a E ≈ -K a for a linear operator, the Jacobian
    of ∇_a E at a = 0 approximates K.  This provides a sanity check.
    """
    model.eval()
    K_learned = torch.zeros(order, order, device=device, dtype=dtype)
    for j in range(order):
        a = torch.zeros(1, order, device=device, dtype=dtype)
        a[0, j] = 1.0
        _, gradE = energy_and_grad(model, a, create_graph=False)
        K_learned[:, j] = gradE[0]
    return K_learned


@torch.no_grad()
def denoise_K_from_energy(model: nn.Module, order: int, n_samples: int,
                           batch_size: int, alpha: float, lam: float,
                           device: torch.device, dtype: torch.dtype,
                           seed: int = 1234) -> torch.Tensor:
    """
    Recover a symmetrized, denoised stiffness matrix from the trained energy net
    via ridge regression on random coefficient samples:

        K_ls = argmin_K  ||A K^T - G||^2_F + λ ||K||^2_F

    where A[m] = a^(m) and G[m] = ∇_a E^{a,θ}(a^(m)).

    The result is symmetrized as (K_ls + K_ls^T) / 2 to enforce the expected
    symmetry of the stiffness matrix.  This denoised K is used as the learned
    diffusion operator during rollout.
    """
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 999)

    A_list, G_list = [], []
    n_batches = math.ceil(n_samples / batch_size)
    for b in range(n_batches):
        cur = min(batch_size, n_samples - b * batch_size)
        if cur <= 0:
            break
        a = sample_coefficients(cur, order, alpha, device, dtype, gen)
        a_req = a.clone().detach().requires_grad_(True)
        _, gradE = energy_and_grad(model, a_req, create_graph=False)
        A_list.append(a.detach())
        G_list.append(gradE.detach())

    A = torch.cat(A_list, dim=0)[:n_samples]  # (M, K)
    G = torch.cat(G_list, dim=0)[:n_samples]  # (M, K)

    AtA = A.T @ A
    AtG = A.T @ G
    I = torch.eye(order, device=device, dtype=dtype)
    Kt = torch.linalg.solve(AtA + lam * I, AtG)  # (K, K)
    K_ls = Kt.T
    return 0.5 * (K_ls + K_ls.T)  # symmetrize


def learned_uxx(model: nn.Module, a: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the learned diffusion vector field F^θ_uxx(a) = -∇_a E^{a,θ}(a)
    for a single coefficient vector a of shape (K,).
    """
    model.eval()
    a_b = a.unsqueeze(0)
    _, gradE = energy_and_grad(model, a_b, create_graph=False)
    return -gradE[0]


# ============================================================
# Plotting utilities
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def plot_train_curve(hist: dict, outpath: str) -> None:
    """Plot training and test loss curves on a log scale."""
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["test_loss"], label="test_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("u_xx block training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_uxx_operator_compare(model: nn.Module, a_u_test: torch.Tensor,
                               K_exact: torch.Tensor, phi: torch.Tensor,
                               x_grid: torch.Tensor, w_grid: torch.Tensor,
                               idx: int, outpath: str, cut: int = 10) -> float:
    """
    Compare the reference operator F^ref_uxx (solid) against the learned
    F^θ_uxx (dashed) in physical space for a single held-out sample.

    Boundary quadrature points are trimmed (cut on each side) to avoid
    Runge-like oscillations at the endpoints.

    Returns the weighted relative L^2 error on the trimmed domain.
    """
    a0 = a_u_test[idx]
    a_uxx_true = -(a0 @ K_exact.T)
    a_uxx_pred = learned_uxx(model, a0)

    uxx_true = a_to_u(a_uxx_true, phi).detach().cpu()
    uxx_pred = a_to_u(a_uxx_pred, phi).detach().cpu()
    x = x_grid.detach().cpu()
    w = w_grid.detach().cpu()

    if cut > 0:
        x = x[cut:-cut]
        w = w[cut:-cut]
        uxx_true = uxx_true[cut:-cut]
        uxx_pred = uxx_pred[cut:-cut]

    num = torch.sqrt(torch.sum((uxx_pred - uxx_true) ** 2 * w))
    den = torch.sqrt(torch.sum(uxx_true ** 2 * w)) + 1e-16
    rel = (num / den).item()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x.numpy(), uxx_true.numpy(), "k-", lw=2.6, label="true")
    ax.plot(x.numpy(), uxx_pred.numpy(), "r--", lw=2.6, label="predicted")
    ax.set_title(
        rf"$u_{{xx}}$ comparison" + "\n" + rf"rel err={rel:.3e}",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("x", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$u_{xx}(x)$", fontsize=16, fontweight="bold")
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
    order = 96       # number of retained Shen modes (coefficient dimension K)
    n_points = 256   # Gauss–Legendre quadrature nodes (Q > K for accuracy)
    alpha = 0.5      # spectral decay exponent for training prior µ_b

    # --- Dataset sizes ---
    n_train = 20000
    n_test = 2000

    # --- E-block architecture ---
    num_layers = 4
    hidden_dim = 256   # wider than paper default (128) for better coverage

    # --- Optimizer and scheduler ---
    batch_size = 128
    num_epochs = 200
    lr = 1e-3

    model_dir = os.path.join("models", "uxx_block")
    plot_dir = os.path.join("plots", "uxx_block")
    ensure_dir(model_dir)
    ensure_dir(plot_dir)

    # Build Shen–Legendre baseplate objects
    x_grid, w_grid = legendre_quadrature_1d(n_points, device=device, dtype=dtype)
    phi, phi_x = shen_basis_and_deriv(x_grid, order=order)
    K_exact = build_K(phi_x, w_grid)  # reference stiffness matrix

    # Build training and test datasets
    a_u_train, a_uxx_train, a_u_test, a_uxx_test = build_uxx_datasets(
        n_train, n_test, order, alpha, K_exact, device, dtype, seed=1234
    )
    train_loader = DataLoader(
        TensorDataset(a_u_train, a_uxx_train),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(a_u_test, a_uxx_test),
        batch_size=batch_size, shuffle=False,
    )

    # Initialize E-block model
    model = EnergyNet(dim=order, num_layers=num_layers, hidden_dim=hidden_dim)
    model = model.to(device).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.3)

    hist = {"train_loss": [], "test_loss": [], "test_rel": [], "lr": []}

    for ep in range(1, num_epochs + 1):
        model.train()
        running, total = 0.0, 0
        for a_u_b, a_uxx_b in train_loader:
            a_u_b = a_u_b.to(device)
            a_uxx_b = a_uxx_b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = uxx_energy_loss(model, a_u_b, a_uxx_b)
            loss.backward()
            opt.step()
            B = a_u_b.size(0)
            running += loss.item() * B
            total += B

        train_loss = running / max(total, 1)
        test_loss, test_rel = evaluate_uxx_energy(model, test_loader, device)
        sch.step()
        cur_lr = opt.param_groups[0]["lr"]

        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["test_rel"].append(test_rel)
        hist["lr"].append(cur_lr)

        print(
            f"Epoch {ep:3d} | lr={cur_lr:.3e} | "
            f"train={train_loss:.3e} | test={test_loss:.3e} | rel={test_rel:.3e}"
        )

    # Freeze the trained model for inference
    trained = copy.deepcopy(model).eval()
    for p in trained.parameters():
        p.requires_grad_(False)

    # Precompute the learned and denoised stiffness matrices for use during rollout.
    # K_denoised is a symmetrized least-squares estimate of the effective K implied
    # by the trained energy; it is more numerically stable than differentiating
    # through the MLP at each rollout step.
    K_learned = estimate_K_from_energy(trained, order, device=device, dtype=dtype)
    K_denoised = denoise_K_from_energy(
        trained, order, n_samples=10000, batch_size=256, alpha=alpha,
        lam=1e-6, device=device, dtype=dtype,
    )

    # Diagnostic plots
    plot_train_curve(hist, os.path.join(plot_dir, "train_curve.png"))
    rel = plot_uxx_operator_compare(
        trained, a_u_test, K_exact, phi, x_grid, w_grid,
        idx=11, outpath=os.path.join(plot_dir, "uxx_operator_compare.png"), cut=10,
    )
    print("[uxx] operator compare rel err:", rel)

    # Save checkpoint
    ckpt = {
        "type": "uxx_energy_net",
        "order": order,
        "n_points": n_points,
        "alpha": alpha,
        "arch": {"num_layers": num_layers, "hidden_dim": hidden_dim},
        "state_dict": trained.state_dict(),
        "K_learned": K_learned.detach().cpu(),
        "K_denoised": K_denoised.detach().cpu(),
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
