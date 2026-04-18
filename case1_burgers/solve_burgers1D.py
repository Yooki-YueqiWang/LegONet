#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollout script for 1D viscous Burgers with Dirichlet boundaries (Case Study I).

Equation:
    u_t + u u_x = ν u_xx,   x ∈ (-1, 1),   u(±1, t) = 0

Both a reference solver (using exact spectral operators) and a learned solver
(using the pretrained E-block and H-block) are run from the same initial
condition.  The difference isolates the contribution of block learning error.

Solver design:
  - Baseplate: Shen–Legendre Dirichlet, K=96 modes, Q=256 Gauss–Legendre nodes
  - Strang splitting (symmetric, 2nd-order):

        a^{n+1} = S_N(Δt/2) ∘ S_D(Δt) ∘ S_N(Δt/2) (a^n)

    where:
      S_N — transport half-step via Heun (explicit trapezoid, 2nd-order)
      S_D — diffusion full-step via Crank–Nicolson (2nd-order)

  - An optional 2/3 dealiasing filter is applied once per full Strang step.

Block loading (from pretrained checkpoints):
  - uxx block: provides K_denoised (the symmetrized learned stiffness matrix)
  - uux block: provides HamiltonianNet h^θ; the transport update is
                u → dh^θ/du(u) → project → apply J

Outputs (saved to plots/burgers/):
  - burgers_overlay_true_vs_learned.png   — reference (solid) vs learned (markers)
  - burgers_residual_uL_minus_uT.png      — raw pointwise residual u_L - u_ref
  - burgers_relative_error_uL_minus_uT.png — normalized pointwise error e(x)
  - burgers_u_compare_final.png           — final-time comparison
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FIGSIZE_SQ = (6, 6)   # default figure size for square canvases


# ============================================================
# Utilities
# ============================================================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int = 1234) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def last_finite_step(a_hist: torch.Tensor) -> int:
    """
    Return the index of the last time step at which all coefficients are finite.
    Used to gracefully handle NaN divergence during long rollouts.
    """
    finite_mask = torch.isfinite(a_hist).all(dim=1)
    idx = torch.where(finite_mask)[0]
    if idx.numel() == 0:
        return 0
    return int(idx[-1].item())


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
    Evaluate P_0, …, P_{max_n} and their derivatives at nodes x.
    Returns (P, Pp) each of shape (max_n+1, N).
    """
    x = x.reshape(-1)
    device, dtype = x.device, x.dtype
    P = torch.zeros((max_n + 1, x.numel()), device=device, dtype=dtype)
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


def build_stiffness_matrix(phi_x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Stiffness matrix K_ij = ⟨φ'_i, φ'_j⟩_{L^2}. Shape: (K, K)."""
    wv = w.view(1, -1)
    return (phi_x * wv) @ phi_x.T


def build_J_shen(x_grid: torch.Tensor, w_grid: torch.Tensor, order: int):
    """
    Build J = -M^{-1} S representing ∂_x on the Shen trial space.
    S_ij = ⟨φ'_i, φ_j⟩_{L^2}.
    Returns J, M, S, phi, phi_x.
    """
    phi, phi_x = shen_basis_and_deriv(x_grid, order)
    w = w_grid.view(1, -1)
    M = (phi * w) @ phi.T
    S = (phi_x * w) @ phi.T
    J = -torch.linalg.solve(M, S)
    return J, M, S, phi, phi_x


def u_to_a(u_grid: torch.Tensor, phi: torch.Tensor, w: torch.Tensor,
           M: torch.Tensor = None) -> torch.Tensor:
    """Discrete L^2 projection: a = M^{-1} Φ^T (u ⊙ w)."""
    if M is None:
        M = build_mass_matrix(phi, w)
    b = phi @ (u_grid * w)
    return torch.linalg.solve(M, b)


def a_to_u(a: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Reconstruct physical field u = Φ a. Accepts (K,) or (B, K)."""
    return a @ phi


# ============================================================
# H-block model (must match the architecture in train_uux_block.py)
# ============================================================

class HamiltonianNet(nn.Module):
    """
    Pointwise Hamiltonian density network h^θ: R → R.
    The H-block vector field is F^θ_uux = J P_b[ dh^θ/du(u) ].
    Architecture must match the checkpoint saved by train_uux_block.py.
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
        if u.dim() == 1:
            return self.net(u.unsqueeze(-1)).squeeze(-1)
        return self.net(u.unsqueeze(-1)).squeeze(-1)


# ============================================================
# H-block evaluation at rollout time
# ============================================================

def uux_block_from_f(H_net: nn.Module, a_u: torch.Tensor, J: torch.Tensor,
                      phi: torch.Tensor, w_grid: torch.Tensor,
                      M: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the learned transport vector field F^θ_uux(a):

        u   = Φ a                         (reconstruct physical field)
        g   = dh^θ/du(u)                  (pointwise density derivative)
        g_a = P_b[g]                      (project back to coefficient space)
        F^θ_uux = g_a @ J^T               (apply skew-symmetric structure op)

    Autograd is force-enabled to compute dh^θ/du.
    """
    u_grid = a_to_u(a_u, phi)
    with torch.enable_grad():
        u_req = u_grid.detach().requires_grad_(True)
        f = H_net(u_req)
        g = torch.autograd.grad(f.sum(), u_req, create_graph=False)[0]
    g_a = u_to_a(g.detach(), phi, w_grid, M=M)
    return g_a @ J.T


# ============================================================
# Dealiasing (2/3 rule)
# ============================================================

def make_mode_filter(order: int, frac: float = 2 / 3, device: str = "cpu",
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Build a modal filter mask that zeros coefficients above index floor(frac * K).
    The 2/3 rule removes the top third of retained modes to prevent aliasing
    errors from pseudo-spectral nonlinear products.
    """
    cut = int(np.floor(frac * order))
    m = torch.zeros(order, device=device, dtype=dtype)
    m[:cut] = 1.0
    return m


def apply_mode_filter(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the modal filter: a <- a * mask."""
    return a * mask


# ============================================================
# Strang splitting integrator
# ============================================================

def nonlinear_step_heun(a: torch.Tensor, dt: float, nl_term_fn) -> torch.Tensor:
    """
    Heun (explicit trapezoid) step for the isolated nonlinear sub-dynamics:
        a_t = F(a)   =>   a^* = a + Δt F(a),   a^{n+1} = a + (Δt/2)(F(a) + F(a^*))

    Second-order accurate in time, consistent with Assumption 2(3) of the paper.
    """
    k1 = nl_term_fn(a)
    a_pred = a + dt * k1
    k2 = nl_term_fn(a_pred)
    return a + 0.5 * dt * (k1 + k2)


def diffusion_step_cn(a: torch.Tensor, eps: float, dt: float,
                       K_op: torch.Tensor) -> torch.Tensor:
    """
    Crank–Nicolson step for the isolated diffusion sub-dynamics:
        a_t = -ε K a

    Solves the linear system:
        (I + (Δt/2) ε K) a^{n+1} = (I - (Δt/2) ε K) a^n

    Second-order accurate in time, unconditionally stable for all Δt.
    """
    dim = a.numel()
    I = torch.eye(dim, dtype=a.dtype, device=a.device)
    A_lhs = I + 0.5 * dt * eps * K_op
    A_rhs = I - 0.5 * dt * eps * K_op
    rhs = A_rhs @ a
    return torch.linalg.solve(A_lhs, rhs)


def burgers_strang(a0: torch.Tensor, eps: float, dt: float, n_steps: int,
                   nl_term_fn, K_op: torch.Tensor, mode_mask=None,
                   stop_on_nan: bool = True,
                   filter_each_substep: bool = False) -> torch.Tensor:
    """
    Advance the Burgers equation in coefficient space using the symmetric
    Strang macro-step (Eq. 14):

        a^{n+1} = S_N(Δt/2) ∘ S_D(Δt) ∘ S_N(Δt/2) (a^n)

    where S_N uses Heun and S_D uses Crank–Nicolson.

    Parameters
    ----------
    a0              : initial coefficient vector, shape (K,)
    eps             : viscosity ν
    dt              : time step Δt
    n_steps         : number of macro steps
    nl_term_fn      : callable a -> F(a), the nonlinear (transport) vector field
    K_op            : diffusion operator matrix (K or M^{-1}K depending on form)
    mode_mask       : optional 2/3 dealiasing mask, shape (K,)
    stop_on_nan     : halt and fill remaining history with NaN on divergence
    filter_each_substep : if True, apply dealiasing after each substep (more
                          aggressive); if False (default), apply once per full step

    Returns
    -------
    a_hist : Tensor of shape (n_steps+1, K) — coefficient trajectory
    """
    a = a0.clone()
    dim = a.numel()
    a_hist = torch.zeros(n_steps + 1, dim, dtype=a.dtype, device=a.device)
    a_hist[0] = a

    half = 0.5 * dt
    for n in range(1, n_steps + 1):
        # Transport half-step (Heun)
        a = nonlinear_step_heun(a, half, nl_term_fn)
        if mode_mask is not None and filter_each_substep:
            a = apply_mode_filter(a, mode_mask)

        # Diffusion full step (Crank–Nicolson)
        a = diffusion_step_cn(a, eps, dt, K_op)
        if mode_mask is not None and filter_each_substep:
            a = apply_mode_filter(a, mode_mask)

        # Transport half-step (Heun)
        a = nonlinear_step_heun(a, half, nl_term_fn)

        # Dealiasing filter applied once per full Strang step (default)
        if mode_mask is not None and (not filter_each_substep):
            a = apply_mode_filter(a, mode_mask)

        if stop_on_nan and (not torch.isfinite(a).all()):
            print(f"[WARN] Non-finite coefficients detected at step {n}. Halting integration.")
            a_hist[n:] = float("nan")
            break

        a_hist[n] = a

    return a_hist


# ============================================================
# Plotting routines
# ============================================================

def plot_overlay_true_learned_marker(
    a_true_hist: torch.Tensor, a_learned_hist: torch.Tensor,
    phi: torch.Tensor, x_grid: torch.Tensor, steps: list,
    title: str, outpath: str,
    markevery: int = 4, marker_size: float = 4.0, cut: int = 15,
) -> None:
    """
    Overlay plot comparing reference (solid lines) and learned (hollow markers)
    solutions at selected time steps.

    Legend 1 (top): explains line style vs marker style.
    Legend 2 (bottom-right): lists the step labels by color.
    """
    x = x_grid.detach().cpu()
    fig, ax = plt.subplots(figsize=(7, 4.6))

    step_handles = []
    for s in steps:
        uT = a_to_u(a_true_hist[s], phi).detach().cpu()
        uL = a_to_u(a_learned_hist[s], phi).detach().cpu()

        xx = x[cut:-cut] if cut > 0 else x
        uT_ = uT[cut:-cut] if cut > 0 else uT
        uL_ = uL[cut:-cut] if cut > 0 else uL

        ln_T, = ax.plot(xx, uT_, lw=2.6, ls="-", zorder=2)
        c = ln_T.get_color()
        ax.plot(
            xx, uL_, ls="None", marker="s", markersize=marker_size,
            markerfacecolor="none", markeredgewidth=1.2, markeredgecolor=c,
            markevery=markevery, zorder=6,
        )
        step_handles.append(ln_T)

    style_handles = [
        Line2D([0], [0], color="k", lw=2.6, ls="-", label=r"$u_{\mathrm{ref}}$"),
        Line2D([0], [0], color="k", lw=0, marker="s", markersize=7,
               markerfacecolor="none", markeredgewidth=1.2, label=r"$u_L$"),
    ]
    leg1 = ax.legend(handles=style_handles, loc="upper center", ncol=2, frameon=True)

    step_labels = [fr"step {s}" for s in steps]
    leg2 = ax.legend(
        handles=step_handles, labels=step_labels,
        loc="lower right", bbox_to_anchor=(0.98, 0.02),
        bbox_transform=ax.transAxes, borderaxespad=0.0,
        frameon=True, fontsize=9, ncol=1,
    )
    ax.add_artist(leg1)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_residual_only(
    a_true_hist: torch.Tensor, a_learned_hist: torch.Tensor,
    phi: torch.Tensor, x_grid: torch.Tensor, steps: list,
    title: str, outpath: str, pad: float = 1.15, cut: int = 15,
) -> None:
    """
    Plot the raw pointwise residual u_L(x) - u_ref(x) at selected steps.
    The y-axis is auto-scaled symmetrically to the maximum absolute residual.
    """
    x = x_grid.detach().cpu()
    xx = x[cut:-cut] if cut > 0 else x

    max_abs_r = 0.0
    R_list = []
    for s in steps:
        uT = a_to_u(a_true_hist[s], phi).detach().cpu()
        uL = a_to_u(a_learned_hist[s], phi).detach().cpu()
        r = uL - uT
        r = r[cut:-cut] if cut > 0 else r
        R_list.append((s, r))
        max_abs_r = max(max_abs_r, r.abs().max().item())

    ymax = pad * max_abs_r if max_abs_r > 0 else 1e-12

    plt.figure(figsize=(7, 4.6))
    for (s, r) in R_list:
        plt.plot(xx, r, lw=2.0, label=fr"step {s}")
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    plt.ylim(-ymax, ymax)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u_L(x)-u_{\mathrm{ref}}(x)$")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=1, fontsize=9, frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_relative_error_only(
    a_true_hist: torch.Tensor, a_learned_hist: torch.Tensor,
    phi: torch.Tensor, x_grid: torch.Tensor, steps: list,
    title: str, outpath: str, pad: float = 1.15, cut: int = 15,
    eps_rel: float = 1e-12,
) -> None:
    """
    Plot the normalized pointwise error profile e(x) at selected steps:

        e(x_q) = (u_L(x_q) - u_ref(x_q)) / (||u_ref||_{2} + ε)

    This is the signed profile defined in Eq. (4) of the paper.  Its L^2 norm
    equals the weighted relative L^2 error up to a constant factor.
    """
    x = x_grid.detach().cpu()
    xx = x[cut:-cut] if cut > 0 else x

    E_list = []
    e_abs_max = 0.0

    for s in steps:
        uT = a_to_u(a_true_hist[s], phi).detach().cpu()
        uL = a_to_u(a_learned_hist[s], phi).detach().cpu()

        uT = uT[cut:-cut] if cut > 0 else uT
        uL = uL[cut:-cut] if cut > 0 else uL

        denom = torch.linalg.norm(uT) + eps_rel
        e = (uL - uT) / denom
        E_list.append((s, e))
        e_abs_max = max(e_abs_max, float(e.abs().max().item()))

    ymax = pad * e_abs_max if e_abs_max > 0 else 1e-12

    plt.figure(figsize=(7, 4.6))
    for (s, e) in E_list:
        plt.plot(xx, e, lw=2.0, label=fr"step {s}")
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    plt.ylim(-ymax, ymax)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$e(x)$")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=1, fontsize=9, frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_u_compare(a_hist_learned: torch.Tensor, a_hist_true: torch.Tensor,
                   phi: torch.Tensor, x_grid: torch.Tensor, step_req: int,
                   cut: int, outpath: str) -> None:
    """Final-time comparison: reference (solid) vs learned (dashed)."""
    last_true = last_finite_step(a_hist_true)
    last_learn = last_finite_step(a_hist_learned)
    used_step = int(min(step_req, last_true, last_learn))

    uL = a_to_u(a_hist_learned[used_step], phi).detach().cpu()
    uT = a_to_u(a_hist_true[used_step], phi).detach().cpu()
    x = x_grid.detach().cpu()

    if cut > 0:
        uL = uL[cut:-cut]
        uT = uT[cut:-cut]
        x = x[cut:-cut]

    rel = (torch.norm(uL - uT) / (torch.norm(uT) + 1e-12)).item()

    plt.figure(figsize=FIGSIZE_SQ)
    plt.plot(x.numpy(), uT.numpy(), "k-", lw=2, label="reference (spectral)")
    plt.plot(x.numpy(), uL.numpy(), "r--", lw=2, label="learned")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.title(f"u(x) at step {used_step} | rel err = {rel:.3e}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# Main rollout routine
# ============================================================

def main():
    # ---- PDE and integration parameters ----
    eps = 0.03           # viscosity ν
    T_final = 1.0        # final time T
    n_steps = 100000     # number of Strang macro steps (Δt = T/n_steps = 1e-5)

    use_filter_23 = True         # apply 2/3 dealiasing filter
    filter_frac = 2 / 3          # fraction of modes to retain
    filter_each_substep = False  # False: filter once per full step (recommended)

    # How K_op is formed from the learned stiffness K_denoised:
    # "notebook"  => use K directly (i.e., the ODE is a_t = -ε K a)
    # "strong"    => use M^{-1} K (i.e., the ODE is a_t = -ε M^{-1} K a)
    diffusion_form = "notebook"

    # Scale the initial condition so max|u0| = 1/scale_factor
    scale_factor = 3.0

    # Steps at which to plot snapshots
    steps_req = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]

    # Boundary trimming for plots (removes endpoints where basis gradients are large)
    cut_plot = 15
    markevery = 4
    marker_size = 4.0

    # ---- Setup ----
    set_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    plot_dir = os.path.join("plots", "burgers")
    ensure_dir(plot_dir)

    # ---- Load uxx checkpoint (provides K_denoised) ----
    uxx_ckpt = torch.load(
        os.path.join("models", "uxx_block", "checkpoint.pt"), map_location="cpu"
    )
    order = int(uxx_ckpt["order"])
    n_points = int(uxx_ckpt["n_points"])
    K_denoised = uxx_ckpt["K_denoised"].to(device=device, dtype=dtype)

    # ---- Load uux checkpoint (provides HamiltonianNet weights) ----
    uux_ckpt = torch.load(
        os.path.join("models", "uux_block", "checkpoint.pt"), map_location="cpu"
    )
    arch_uux = uux_ckpt["arch"]
    H_net = HamiltonianNet(
        width=arch_uux["width"], depth=arch_uux["depth"], act=arch_uux["act"]
    ).double().to(device)
    H_net.load_state_dict(uux_ckpt["state_dict"])
    H_net.eval()
    for p in H_net.parameters():
        p.requires_grad_(False)

    # ---- Build Shen–Legendre baseplate ----
    x_grid, w_grid = legendre_quadrature_1d(n_points, device=device, dtype=dtype)
    J, M, S, phi, phi_x = build_J_shen(x_grid, w_grid, order)
    K_exact_stiff = build_stiffness_matrix(phi_x, w_grid)

    # Select diffusion operator form
    if diffusion_form.lower() == "notebook":
        K_true = K_exact_stiff
        K_learn = K_denoised
    elif diffusion_form.lower() == "strong":
        K_true = torch.linalg.solve(M, K_exact_stiff)
        K_learn = torch.linalg.solve(M, K_denoised)
    else:
        raise ValueError("diffusion_form must be 'notebook' or 'strong'.")

    # ---- Time step ----
    dt = T_final / float(n_steps)
    print(
        f"[INFO] order={order}, Q={n_points}, ν={eps}, T={T_final}, "
        f"n_steps={n_steps}, Δt={dt:.3e}"
    )
    print(f"[INFO] diffusion_form={diffusion_form}, dealiasing={use_filter_23}")
    print(f"[INFO] Strang: N(Δt/2) → D(Δt) → N(Δt/2), filter_each_substep={filter_each_substep}")

    # ---- 2/3 dealiasing mask ----
    mode_mask = None
    if use_filter_23:
        mode_mask = make_mode_filter(order, frac=filter_frac, device=device, dtype=dtype)

    # ---- Initial condition: spectral-decay Gaussian, rescaled ----
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    k = torch.arange(order, device=device, dtype=dtype)
    sigma = 1.0 / (1.0 + k) ** 0.5
    a0_raw = torch.randn(order, generator=gen, device=device, dtype=dtype) * sigma

    u0 = a0_raw @ phi
    scale = scale_factor * u0.abs().max()
    u0_scaled = u0 / (scale + 1e-16)
    a0 = u_to_a(u0_scaled, phi, w_grid, M=M)
    print(f"[INFO] max|u0_scaled| = {u0_scaled.abs().max().item():.4f}")

    # ---- Nonlinear (transport) vector fields ----

    def nl_true(a: torch.Tensor) -> torch.Tensor:
        """Reference transport: F^ref_uux(a) = P_b[0.5 u^2] @ J^T."""
        u = a @ phi
        g = 0.5 * u * u
        g_a = u_to_a(g, phi, w_grid, M=M)
        return g_a @ J.T

    def nl_learned(a: torch.Tensor) -> torch.Tensor:
        """Learned transport: F^θ_uux(a) using HamiltonianNet."""
        return uux_block_from_f(H_net, a, J, phi, w_grid, M)

    # ---- Run Strang rollouts ----
    print("[INFO] Running reference rollout ...")
    a_true = burgers_strang(
        a0=a0, eps=eps, dt=dt, n_steps=n_steps,
        nl_term_fn=nl_true, K_op=K_true,
        mode_mask=mode_mask, stop_on_nan=True,
        filter_each_substep=filter_each_substep,
    )

    print("[INFO] Running learned rollout ...")
    a_learned = burgers_strang(
        a0=a0, eps=eps, dt=dt, n_steps=n_steps,
        nl_term_fn=nl_learned, K_op=K_learn,
        mode_mask=mode_mask, stop_on_nan=True,
        filter_each_substep=filter_each_substep,
    )

    last_true = last_finite_step(a_true)
    last_learn = last_finite_step(a_learned)
    print(f"[INFO] Last finite step — reference: {last_true}, learned: {last_learn}")

    # Clamp snapshot steps to the finite range
    steps_plot = [int(min(s, last_true, last_learn)) for s in steps_req]

    # ---- Save plots ----
    plot_overlay_true_learned_marker(
        a_true_hist=a_true, a_learned_hist=a_learned,
        phi=phi, x_grid=x_grid, steps=steps_plot,
        title=r"1D Burgers: $u_{\mathrm{ref}}$ (solid) and $u_L$ (markers)",
        outpath=os.path.join(plot_dir, "burgers_overlay_true_vs_learned.png"),
        markevery=markevery, marker_size=marker_size, cut=cut_plot,
    )

    plot_residual_only(
        a_true_hist=a_true, a_learned_hist=a_learned,
        phi=phi, x_grid=x_grid, steps=steps_plot,
        title=r"Residual: $u_L - u_{\mathrm{ref}}$",
        outpath=os.path.join(plot_dir, "burgers_residual_uL_minus_uT.png"),
        pad=1.15, cut=cut_plot,
    )

    plot_relative_error_only(
        a_true_hist=a_true, a_learned_hist=a_learned,
        phi=phi, x_grid=x_grid, steps=steps_plot,
        title="Normalized pointwise error",
        outpath=os.path.join(plot_dir, "burgers_relative_error_uL_minus_uT.png"),
        pad=1.15, cut=cut_plot,
    )

    plot_u_compare(
        a_hist_learned=a_learned, a_hist_true=a_true,
        phi=phi, x_grid=x_grid, step_req=n_steps,
        cut=cut_plot,
        outpath=os.path.join(plot_dir, "burgers_u_compare_final.png"),
    )

    print(f"[DONE] Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
