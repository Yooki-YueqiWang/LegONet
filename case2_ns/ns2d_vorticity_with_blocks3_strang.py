#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollout script for 2D forced incompressible Navier–Stokes in vorticity form (Case Study II).

Equation on the unit torus [0,1)^2 (matching FNO paper, Appendix A.3.3):
    ω_t + u · ∇ω = ν Δω + f(x,y),   ∇ · u = 0
    u = (ψ_y, -ψ_x),   -Δψ = ω

Forcing:  f(x,y) = 0.1 (sin(2π(x+y)) + cos(2π(x+y)))
Initial condition: ω_0 ~ N(0, 7^{3/2} (-Δ + 49I)^{-2.5})  (FNO Gaussian measure)

Baseplate: 2D periodic Fourier, N=64, K_cut=21, 2/3 dealiasing.

Strang macro-step (symmetric, 2nd-order):
    ω^{n+1} = L(Δt/2) ∘ N(Δt) ∘ L(Δt/2) (ω^n)

where:
    L — linear step: exact solution of ω_t = ν Δω + f in Fourier space
        (d/dt ω̂_k = -ν kfac² |k|² ω̂_k + f̂_k => ω̂_k(t) = e^{-ν kfac² |k|² t} ω̂_k(0) + ...)
    N — nonlinear step: pure advection ω_t + u · ∇ω = 0 using RK2/Heun

Both a reference trajectory (exact operators) and a learned trajectory (learned
blocks) are advanced simultaneously from the same initial condition.

Two modes of comparison:
    (A) Reference: exact Poisson inversion + exact Laplacian eigenvalues
    (B) Learned:   Poisson block from runs_poisson2d/ + Laplacian block from runs_laplace2d/

The coefficient state is stored in the real Fourier packing (RealFourierIndex),
which represents a real-valued field by keeping one element from each complex
conjugate pair and stacking real/imaginary parts of non-self-conjugate modes.
Self-conjugate modes (k = 0, Nyquist) are stored as real scalars.

Outputs (saved to runs_ns2d_vort/<run_name>_<timestamp>/):
    config.json                      — complete run configuration
    history.npz                      — relative error array over all steps
    omega_step00000_compare.png      — 3x1: learned / reference / error at step 0
    omega_steps_compare_grid.png     — 3xK: learned / reference / error at K later steps
    error_curve_omega.png            — relative L^2 error vs time

Key command-line flags (run with --help for full list):
    --N, --Kmax          grid resolution and mode cap
    --nu, --dt, --n_steps  PDE parameters
    --laplace_model      diag | mlp | fixed
    --laplace_model_dir  path to Laplace block (auto-selects latest if empty)
    --poisson_model      diag | fixed
    --poisson_model_dir  path to Poisson block (auto-selects latest if empty)
    --use_double         use float64 (recommended for long rollouts)
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D forced incompressible Navier–Stokes (vorticity-streamfunction) on the UNIT TORUS (0,1)^2,
matching FNO (ICLR 2021) Appendix A.3.3:

    w_t + u·∇w = ν Δw + f(x),   div u = 0,
    u = (ψ_y, -ψ_x),            -Δψ = w.

Now integrated with Strang splitting:
    L:  w_t = ν Δw + f     (exact-in-Fourier linear flow)
    N:  w_t + u·∇w = 0     (pseudo-spectral, RK2)

Strang macro-step:
    w^{n+1} = L(dt/2) ∘ N(dt) ∘ L(dt/2)  (applied separately to reference and learned).

Comparison:
  (A) TRUE pseudo-spectral (exact Poisson inversion + exact linear L flow)
  (B) LEARNED using saved blocks:
        - Poisson2D_Streamfunction: (-Δ)^{-1} (in packed real Fourier coords)
        - Laplace2D: approximates k^2 (in packed real Fourier coords) => used for exact linear L flow

Dealiasing: 2/3 law + Kmax cap.

Outputs:
  runs_ns2d_vort/<run_name_YYYYmmdd-HHMMSS>/
    config.json
    history.npz
    omega_step00000_compare.png                  (step=0 single)
    omega_steps_compare_grid.png                 (4 later steps in one grid)
    error_curve_omega.png                        (relative error curve)
"""

from __future__ import annotations
import os, json, math, time, glob, argparse, random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FFT_NORM = "ortho"

# ============================================================
# Seed / device
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64

# ============================================================
# FFT helpers + k grids + masks
# ============================================================

def fft2(u: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(u, norm=FFT_NORM)

def ifft2(u_hat: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(u_hat, norm=FFT_NORM)

def kgrid_int(N: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # integer modes for unit-torus basis exp(2π i (k·x)), k in Z^2
    k = torch.fft.fftfreq(N, d=1.0 / N, device=device).to(dtype)  # integers
    KX, KY = torch.meshgrid(k, k, indexing="ij")
    return KX, KY

def mask_23_and_kmax(N: int, device: torch.device, dtype: torch.dtype, Kmax: Optional[int]) -> Tuple[torch.Tensor, int]:
    KX, KY = kgrid_int(N, device, dtype)
    k23 = N // 3
    Kcut = k23 if Kmax is None else int(min(k23, Kmax))
    mask = (KX.abs() <= Kcut) & (KY.abs() <= Kcut)
    return mask, Kcut

# ============================================================
# Real Fourier coordinate system (same convention as Laplace2D scripts)
# ============================================================

@dataclass
class RealFourierIndex:
    N: int
    Kcut: int
    ij: torch.Tensor
    ij_partner: torch.Tensor
    is_self: torch.Tensor
    w: torch.Tensor
    M: int

def build_real_fourier_index(N: int, device: torch.device, dtype: torch.dtype, Kmax: Optional[int]) -> RealFourierIndex:
    mask, Kcut = mask_23_and_kmax(N, device, dtype, Kmax)
    KX, KY = kgrid_int(N, device, dtype)
    nyq = -(N // 2) if (N % 2 == 0) else None

    half = (KY > 0) | ((KY == 0) & (KX >= 0))
    if nyq is not None:
        half = half | ((KY == nyq) & (KX >= 0))

    sel = mask & half
    ij = sel.nonzero(as_tuple=False).long()
    i = ij[:, 0]
    j = ij[:, 1]
    ip = (-i) % N
    jp = (-j) % N
    ij_partner = torch.stack([ip, jp], dim=1)

    is_self = (i == ip) & (j == jp)
    w = torch.ones(ij.shape[0], device=device, dtype=dtype)
    w[~is_self] = math.sqrt(2.0)
    return RealFourierIndex(N=N, Kcut=Kcut, ij=ij, ij_partner=ij_partner, is_self=is_self, w=w, M=ij.shape[0])

def pack_hat_to_realvec(u_hat: torch.Tensor, ridx: RealFourierIndex) -> torch.Tensor:
    ij = ridx.ij
    vals = u_hat[:, ij[:, 0], ij[:, 1]]
    is_self = ridx.is_self
    w = ridx.w.to(u_hat.real.dtype)
    self_vals = (w[is_self] * vals[:, is_self].real)
    pair_re   = (w[~is_self] * vals[:, ~is_self].real)
    pair_im   = (w[~is_self] * vals[:, ~is_self].imag)
    return torch.cat([self_vals, pair_re, pair_im], dim=1)

def unpack_realvec_to_hat(a: torch.Tensor, ridx: RealFourierIndex, complex_dtype: torch.dtype) -> torch.Tensor:
    B = a.shape[0]
    N = ridx.N
    device = a.device
    real_dtype = a.dtype
    u_hat = torch.zeros(B, N, N, device=device, dtype=complex_dtype)

    is_self = ridx.is_self
    w = ridx.w.to(real_dtype)
    Ms = int(is_self.sum().item())
    Mp = ridx.M - Ms

    a_self = a[:, :Ms]
    a_re = a[:, Ms:Ms + Mp]
    a_im = a[:, Ms + Mp:Ms + 2 * Mp]

    ij = ridx.ij
    ijp = ridx.ij_partner

    if Ms > 0:
        coeff_self = (a_self / w[is_self])
        i_s = ij[is_self, 0]
        j_s = ij[is_self, 1]
        u_hat[:, i_s, j_s] = torch.complex(coeff_self, torch.zeros_like(coeff_self)).to(complex_dtype)

    if Mp > 0:
        coeff_pair = torch.complex(a_re / w[~is_self], a_im / w[~is_self]).to(complex_dtype)
        i_p = ij[~is_self, 0]
        j_p = ij[~is_self, 1]
        u_hat[:, i_p, j_p] = coeff_pair
        ip = ijp[~is_self, 0]
        jp = ijp[~is_self, 1]
        u_hat[:, ip, jp] = torch.conj(coeff_pair)

    return u_hat

def build_k2_vec_for_realcoords(ridx: RealFourierIndex, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ij = ridx.ij
    KX, KY = kgrid_int(ridx.N, device, dtype)
    k2_mode = (KX[ij[:, 0], ij[:, 1]]**2 + KY[ij[:, 0], ij[:, 1]]**2)

    is_self = ridx.is_self
    Ms = int(is_self.sum().item())
    Mp = ridx.M - Ms

    c_self = k2_mode[is_self]
    c_pair = k2_mode[~is_self]
    return torch.cat([c_self, c_pair, c_pair], dim=0)

def build_inv_k2_vec_for_realcoords(ridx: RealFourierIndex, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    k2_vec = build_k2_vec_for_realcoords(ridx, device, dtype)
    nonzero = (k2_vec > 0).to(dtype)
    inv = torch.zeros_like(k2_vec)
    inv[nonzero.bool()] = 1.0 / k2_vec[nonzero.bool()]
    return inv, nonzero

# ============================================================
# Blocks: Laplace2D
# ============================================================

class FixedLaplaceEnergy(nn.Module):
    def __init__(self, k2_vec: torch.Tensor):
        super().__init__()
        self.register_buffer("k2", k2_vec.clone().detach())
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.k2.view(1, -1) * (a * a)).sum(dim=1)

class DiagLearnableLaplaceEnergy(nn.Module):
    def __init__(self, dim: int, init: str = "rand", init_scale: float = 1.0):
        super().__init__()
        if init == "rand":
            c0 = init_scale * torch.randn(dim)
        elif init == "ones":
            c0 = torch.ones(dim)
        else:
            raise ValueError("init must be 'rand' or 'ones'")
        self.c = nn.Parameter(c0)
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        c = self.c.view(1, -1).to(a.dtype).to(a.device)
        return 0.5 * (c * (a * a)).sum(dim=1)

class MLPEnergy(nn.Module):
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 256, act: str = "gelu"):
        super().__init__()
        if act.lower() == "gelu":
            Act = nn.GELU
        elif act.lower() == "silu":
            Act = nn.SiLU
        elif act.lower() == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {act}")
        layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), Act()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a).squeeze(-1)

def energy_and_grad(model: nn.Module, a: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    a_req = a.clone().detach().requires_grad_(True)
    E = model(a_req)
    gradE = torch.autograd.grad(E.sum(), a_req, create_graph=create_graph)[0]
    return E, gradE

def laplace_pred_from_energy(model: nn.Module, a_u: torch.Tensor, create_graph: bool) -> torch.Tensor:
    _, gradE = energy_and_grad(model, a_u, create_graph=create_graph)
    return -gradE  # (Δu_hat)/scale in your training convention

# ============================================================
# Blocks: Poisson2D_Streamfunction
# ============================================================

class FixedPoissonHamiltonian(nn.Module):
    def __init__(self, inv_k2_vec: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_k2", inv_k2_vec.clone().detach())
    def forward(self, a_w: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.inv_k2.view(1, -1) * (a_w * a_w)).sum(dim=1)

class DiagLearnablePoissonHamiltonian(nn.Module):
    """
    Learns c >= 0 and forces k=0 weight to 0 via mask.
    Trained target: (inv_k2/scale) ⊙ ω  => physical inv_k2 = c*scale (still in integer-k convention).
    """
    def __init__(self, dim: int, nonzero_mask: torch.Tensor, init_scale: float = 0.1):
        super().__init__()
        self.register_buffer("mask", nonzero_mask.clone().detach())  # float 0/1
        self.raw = nn.Parameter(init_scale * torch.randn(dim))
    def c(self) -> torch.Tensor:
        return F.softplus(self.raw) * self.mask

def find_latest_model_dir(out_dir: str) -> str:
    cands = []
    for p in glob.glob(os.path.join(out_dir, "*")):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")) and os.path.exists(os.path.join(p, "model_state.pt")):
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No trained runs found under: {out_dir}")
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]

def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_laplace_block(model_dir: str, device: torch.device, dtype: torch.dtype, model_type: str):
    cfg = load_json(os.path.join(model_dir, "config.json"))
    if cfg.get("block", "") != "Laplace2D":
        raise RuntimeError(f"[FATAL] Not a Laplace2D block. cfg['block']={cfg.get('block','<missing>')} dir={model_dir}")
    w_path = os.path.join(model_dir, "model_state.pt")
    dim = int(cfg["dim"])
    if model_type == "mlp":
        mcfg = cfg["mlp"]
        model = MLPEnergy(dim=dim,
                          num_layers=int(mcfg["num_layers"]),
                          hidden_dim=int(mcfg["hidden_dim"]),
                          act=str(mcfg["act"]))
    elif model_type == "diag":
        dcfg = cfg["diag"]
        model = DiagLearnableLaplaceEnergy(dim=dim,
                                           init=str(dcfg["init"]),
                                           init_scale=float(dcfg["init_scale"]))
    elif model_type == "fixed":
        k2 = torch.tensor(cfg["k2_vec"], dtype=dtype, device=device)
        model = FixedLaplaceEnergy(k2)
    else:
        raise ValueError("unknown laplace model_type")
    model = model.to(device=device, dtype=dtype)
    state = torch.load(w_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg

def load_poisson_block(model_dir: str, device: torch.device, dtype: torch.dtype, model_type: str, nonzero_mask: torch.Tensor):
    cfg = load_json(os.path.join(model_dir, "config.json"))
    if cfg.get("block", "") != "Poisson2D_Streamfunction":
        raise RuntimeError(f"[FATAL] Not a Poisson2D_Streamfunction block. cfg['block']={cfg.get('block','<missing>')} dir={model_dir}")
    w_path = os.path.join(model_dir, "model_state.pt")
    dim = int(cfg["dim"])
    scale = float(cfg.get("scale", 1.0))
    if model_type == "fixed":
        inv_k2 = torch.tensor(cfg["inv_k2_vec"], dtype=dtype, device=device)
        model = FixedPoissonHamiltonian(inv_k2)
    elif model_type == "diag":
        init_scale = float(cfg.get("diag", {}).get("init_scale", 0.1))
        model = DiagLearnablePoissonHamiltonian(dim=dim,
                                                nonzero_mask=nonzero_mask.to(device=device, dtype=dtype),
                                                init_scale=init_scale)
    else:
        raise ValueError("unknown poisson model_type")
    model = model.to(device=device, dtype=dtype)
    state = torch.load(w_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, scale

# ============================================================
# Forcing (paper A.3.3)
# ============================================================

@torch.no_grad()
def forcing_fnoeq9_hat(N: int, L: float, amp: float,
                       device: torch.device, real_dtype: torch.dtype,
                       complex_dtype: torch.dtype,
                       mask_c: torch.Tensor) -> torch.Tensor:
    # grid x in [0,L)
    x = (torch.arange(N, device=device, dtype=real_dtype) / N) * L
    X, Y = torch.meshgrid(x, x, indexing="ij")
    phase = 2.0 * math.pi * (X + Y) / L
    f = amp * (torch.sin(phase) + torch.cos(phase))
    f_hat = fft2(f.to(complex_dtype)) * mask_c
    return f_hat

# ============================================================
# IC sampling: paper Gaussian measure
# ============================================================

@torch.no_grad()
def sample_ic_fno_gaussian_packed(ridx: RealFourierIndex, L: float,
                                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kfac = 2.0 * math.pi / L
    ij = ridx.ij
    KX, KY = kgrid_int(ridx.N, device, dtype)
    k2 = (KX[ij[:, 0], ij[:, 1]]**2 + KY[ij[:, 0], ij[:, 1]]**2)  # integer |k|^2

    lam = (7.0 ** 1.5) / ( ( (kfac*kfac) * k2 + 49.0 ) ** 2.5 )
    std = torch.sqrt(lam).to(dtype)

    is_self = ridx.is_self
    Ms = int(is_self.sum().item())
    Mp = ridx.M - Ms

    a_self = torch.randn(1, Ms, device=device, dtype=dtype) * std[is_self].view(1, -1)
    sig_p = std[~is_self].view(1, -1)
    a_re = torch.randn(1, Mp, device=device, dtype=dtype) * sig_p
    a_im = torch.randn(1, Mp, device=device, dtype=dtype) * sig_p
    return torch.cat([a_self, a_re, a_im], dim=1).squeeze(0)

# ============================================================
# TRUE/LEARNED Poisson inversion (with correct 2π scaling)
# ============================================================

@torch.no_grad()
def psi_hat_true_from_omega_hat(omega_hat: torch.Tensor,
                                k2_grid_int: torch.Tensor,
                                kfac2: float,
                                mask_c: torch.Tensor) -> torch.Tensor:
    # -Δψ = ω => (kfac^2 |k|^2) ψ_hat = ω_hat, set k=0 => 0
    psi_hat = torch.zeros_like(omega_hat)
    nz = (k2_grid_int > 0)
    psi_hat[nz] = omega_hat[nz] / (kfac2 * k2_grid_int[nz].to(omega_hat.dtype))
    return psi_hat * mask_c

@torch.no_grad()
def invk2_eff_from_poisson_model(poisson_model: nn.Module, poisson_model_type: str, poisson_scale: float) -> torch.Tensor:
    if poisson_model_type == "fixed":
        return poisson_model.inv_k2.detach()
    else:
        return (poisson_model.c().detach() * float(poisson_scale))

@torch.no_grad()
def psi_hat_learned_from_omega_hat(omega_hat: torch.Tensor,
                                  ridx: RealFourierIndex,
                                  invk2_eff_vec_intk: torch.Tensor,
                                  kfac2: float,
                                  mask_c: torch.Tensor,
                                  complex_dtype: torch.dtype) -> torch.Tensor:
    # physical inv(-Δ) = (1/kfac^2) * (1/|k|^2)
    invk2_phys = (invk2_eff_vec_intk / kfac2).to(omega_hat.real.dtype)

    a_w = pack_hat_to_realvec(omega_hat.unsqueeze(0), ridx)[0].to(invk2_phys.dtype)
    a_psi = invk2_phys.to(a_w.device) * a_w
    psi_hat = unpack_realvec_to_hat(a_psi.view(1, -1), ridx, complex_dtype)[0] * mask_c
    return psi_hat

# ============================================================
# Advection term (pseudo-spectral, dealias) with 2π scaling
# ============================================================

@torch.no_grad()
def advect_omega_hat(omega_hat: torch.Tensor, psi_hat: torch.Tensor,
                     KX_int: torch.Tensor, KY_int: torch.Tensor,
                     kfac: float,
                     mask_c: torch.Tensor,
                     complex_dtype: torch.dtype) -> torch.Tensor:
    omega_hat = omega_hat * mask_c
    psi_hat   = psi_hat   * mask_c

    ikx = (1j * kfac * KX_int).to(complex_dtype)
    iky = (1j * kfac * KY_int).to(complex_dtype)

    u_hat = iky * psi_hat          # u = ψ_y
    v_hat = (-ikx) * psi_hat       # v = -ψ_x

    omega_x = ifft2(ikx * omega_hat).real
    omega_y = ifft2(iky * omega_hat).real
    u = ifft2(u_hat).real
    v = ifft2(v_hat).real

    adv = (u * omega_x + v * omega_y)
    adv_hat = fft2(adv.to(complex_dtype)) * mask_c
    return adv_hat

# ============================================================
# Strang blocks:
#   Linear L: exact diffusion+forcing in Fourier
#   Nonlinear N: pure advection with RK2
# ============================================================

@torch.no_grad()
def linear_L_exact_hat(omega_hat: torch.Tensor,
                       f_hat: torch.Tensor,
                       k2_grid_int: torch.Tensor,
                       kfac2: float,
                       nu: float, tau: float,
                       mask_c: torch.Tensor) -> torch.Tensor:
    """
    Exact solution of: ω_t = νΔω + f over time tau in Fourier.
    d/dt ω̂ = -nu*(kfac2*k2_int)*ω̂ + f̂
    """
    k2_phys = (kfac2 * k2_grid_int).to(omega_hat.real.dtype)  # real
    E = torch.exp((-nu * tau) * k2_phys).to(omega_hat.dtype)

    omega_hat = omega_hat * mask_c
    f_hat     = f_hat     * mask_c

    # G = (1 - E)/(nu*k2_phys), with safe handling at k=0
    G = torch.zeros_like(k2_phys)
    nz = (k2_phys > 0)
    G[nz] = (1.0 - E[nz].real) / (nu * k2_phys[nz])
    # k=0: omega_hat += tau * f_hat (but f_hat mean is 0 in this forcing; keep general)
    omega_next = E * omega_hat + (G.to(omega_hat.dtype) * f_hat) + ((~nz).to(omega_hat.dtype) * (tau * f_hat))
    return omega_next * mask_c

@torch.no_grad()
def linear_L_learned_diag_hat(omega_hat: torch.Tensor,
                             f_hat: torch.Tensor,
                             k2_vec_intk: torch.Tensor,
                             ridx: RealFourierIndex,
                             kfac2: float,
                             nu: float, tau: float,
                             mask_c: torch.Tensor,
                             complex_dtype: torch.dtype) -> torch.Tensor:
    """
    Same exact linear flow, but using learned diagonal k2 in packed real coords.
    k2_phys_vec = kfac2 * k2_intk_vec.
    """
    # pack ω and f
    a_w = pack_hat_to_realvec(omega_hat.unsqueeze(0), ridx)[0]
    a_f = pack_hat_to_realvec(f_hat.unsqueeze(0), ridx)[0]

    k2_phys_vec = (kfac2 * k2_vec_intk).to(a_w.dtype).to(a_w.device)
    # stability guard: diffusion rates must be >= 0
    k2_phys_vec = torch.clamp(k2_phys_vec, min=0.0)

    E = torch.exp((-nu * tau) * k2_phys_vec)
    G = torch.zeros_like(k2_phys_vec)
    nz = (k2_phys_vec > 0)
    G[nz] = (1.0 - E[nz]) / (nu * k2_phys_vec[nz])

    a_next = E * a_w + G * a_f + ((~nz).to(a_w.dtype) * (tau * a_f))
    omega_next = unpack_realvec_to_hat(a_next.view(1, -1), ridx, complex_dtype)[0] * mask_c
    return omega_next

@torch.no_grad()
def nonlinear_N_rk2_true(omega_hat: torch.Tensor,
                         k2_grid_int: torch.Tensor,
                         KX_int: torch.Tensor, KY_int: torch.Tensor,
                         kfac: float, kfac2: float,
                         dt: float,
                         mask_c: torch.Tensor,
                         complex_dtype: torch.dtype) -> torch.Tensor:
    """
    Pure advection: ω_t + u·∇ω = 0 => ω_t = -adv(ω)
    RK2/Heun in Fourier.
    """
    psi_hat = psi_hat_true_from_omega_hat(omega_hat, k2_grid_int, kfac2, mask_c)
    adv_hat = advect_omega_hat(omega_hat, psi_hat, KX_int, KY_int, kfac, mask_c, complex_dtype)
    k1 = -adv_hat

    omega1 = (omega_hat + dt * k1) * mask_c
    psi_hat1 = psi_hat_true_from_omega_hat(omega1, k2_grid_int, kfac2, mask_c)
    adv_hat1 = advect_omega_hat(omega1, psi_hat1, KX_int, KY_int, kfac, mask_c, complex_dtype)
    k2 = -adv_hat1

    omega_next = (omega_hat + 0.5 * dt * (k1 + k2)) * mask_c
    return omega_next

@torch.no_grad()
def nonlinear_N_rk2_learned(omega_hat: torch.Tensor,
                            ridx: RealFourierIndex,
                            invk2_eff_vec_intk: torch.Tensor,
                            KX_int: torch.Tensor, KY_int: torch.Tensor,
                            kfac: float, kfac2: float,
                            dt: float,
                            mask_c: torch.Tensor,
                            complex_dtype: torch.dtype) -> torch.Tensor:
    """
    Same RK2, but Poisson inversion uses learned invk2 (packed real coords).
    """
    psi_hat = psi_hat_learned_from_omega_hat(omega_hat, ridx, invk2_eff_vec_intk, kfac2, mask_c, complex_dtype)
    adv_hat = advect_omega_hat(omega_hat, psi_hat, KX_int, KY_int, kfac, mask_c, complex_dtype)
    k1 = -adv_hat

    omega1 = (omega_hat + dt * k1) * mask_c
    psi_hat1 = psi_hat_learned_from_omega_hat(omega1, ridx, invk2_eff_vec_intk, kfac2, mask_c, complex_dtype)
    adv_hat1 = advect_omega_hat(omega1, psi_hat1, KX_int, KY_int, kfac, mask_c, complex_dtype)
    k2 = -adv_hat1

    omega_next = (omega_hat + 0.5 * dt * (k1 + k2)) * mask_c
    return omega_next

# ============================================================
# Metrics / plots  (match paper definitions)
# ============================================================

def _weighted_l2_den_ref_np(u_true: np.ndarray, eps: float = 1e-12) -> float:
    # On uniform grid, weights w_q are constant; use mean as Σ w_q |.|^2
    return float(np.sqrt(np.mean(u_true**2) + eps))

def rel_l2_grid(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """
    Scalar-weighted L2 relative error:
      rel = sqrt(Σ w |pred-true|^2) / sqrt(Σ w |true|^2).
    On uniform grid, use mean for Σ w.
    """
    diff2 = torch.mean((u_pred - u_true).reshape(-1) ** 2)
    true2 = torch.mean(u_true.reshape(-1) ** 2)
    rel = torch.sqrt(diff2 / (true2 + 1e-12))
    return float(rel.item())

def pointwise_error_profile(u_learn: np.ndarray, u_true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Pointwise error profile (paper):
      e(x_q) = (p_q - t_q) / sqrt(Σ_j w_j |t_j|^2).
    Note: THIS IS GLOBAL normalization by the reference L2 norm.
    """
    den = _weighted_l2_den_ref_np(u_true, eps=eps)
    return (u_learn - u_true) / den

def plot_step_column_learn_ref_err(u_learn: np.ndarray, u_true: np.ndarray,
                                  step: int, dt: float, out_path: str,
                                  suptitle_prefix: str = r"$\omega$"):
    """
    3x1 layout:
      row1: learned
      row2: reference
      row3: pointwise error profile
    """
    err = pointwise_error_profile(u_learn, u_true)
    # symmetric color scale for signed error
    vmax_e = float(np.max(np.abs(err)) + 1e-12)

    plt.figure(figsize=(4.6, 10.2))

    for r, (Z, ttl) in enumerate([
        (u_learn, "learned"),
        (u_true, "reference"),
        (err, "pointwise error profile"),
    ], start=1):
        ax = plt.subplot(3, 1, r)
        if r == 3:
            im = ax.imshow(Z, origin="lower", vmin=-vmax_e, vmax=vmax_e)
        else:
            im = ax.imshow(Z, origin="lower")
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"{suptitle_prefix} at step {step} (t={step*dt:.3e})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_steps_horizontal_3rows(steps: List[int],
                               true_snaps: List[np.ndarray],
                               learn_snaps: List[np.ndarray],
                               dt: float,
                               out_path: str,
                               field_name: str = r"$\omega$",
                               q_clip: float = 0.995,
                               eps: float = 1e-12):
    """
    3xK grid (Burgers-style):
      row1: learned
      row2: reference
      row3: normalized pointwise error  e = (learn - true)/||true||_2
    Robust symmetric color scale for error via quantile clip.
    """
    K = len(steps)
    assert len(true_snaps) == K and len(learn_snaps) == K

    # error profiles (paper definition)
    E = [pointwise_error_profile(learn_snaps[j], true_snaps[j], eps=eps) for j in range(K)]
    all_e = np.concatenate([ei.reshape(-1) for ei in E], axis=0)
    vmax = float(np.quantile(np.abs(all_e), q_clip)) if all_e.size > 0 else 1e-12
    vmax = max(vmax, 1e-12)

    plt.figure(figsize=(3.6 * K, 3.2 * 3))

    for j in range(K):
        step = steps[j]

        # learned
        ax = plt.subplot(3, K, 1 + j)
        im = ax.imshow(learn_snaps[j], origin="lower")
        ax.set_title(f"learned | step {step}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # reference
        ax = plt.subplot(3, K, 1 + K + j)
        im = ax.imshow(true_snaps[j], origin="lower")
        ax.set_title(f"reference | step {step}")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # normalized pointwise error (signed)
        ax = plt.subplot(3, K, 1 + 2 * K + j)
        vmax_j = np.quantile(np.abs(E[j]).ravel(), q_clip)
        im = ax.imshow(E[j], origin="lower", vmin=-vmax_j, vmax=vmax_j)
        # im = ax.imshow(E[j], origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title("normalized pointwise err")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Match your Burgers-style title/comment behavior: keep minimal, no suptitle by default
    # plt.suptitle(f"2D Navier--Stokes: {field_name} | learned vs reference vs normalized pointwise error")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_curve(err: np.ndarray, dt: float, out_path: str, name: str = r"$\omega$"):
    # Burgers-style naming and labels, but for NS
    t = dt * np.arange(err.shape[0])
    plt.figure(figsize=(6, 4))
    plt.plot(t, err)
    plt.xlabel("time")
    plt.ylabel(r"rel err")
    plt.title(f"2D Navier--Stokes: relative error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--use_double", action="store_true")

    # domain/resolution
    p.add_argument("--L", type=float, default=1.0)  # paper: 1.0
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--Kmax", type=int, default=21)

    # IC
    p.add_argument("--ic", type=str, default="fno_gaussian",
                   choices=["fno_gaussian", "white"])
    p.add_argument("--target_max_omega", type=float, default=-1.0,
                   help="if >0, rescale initial omega so max|omega| hits this target (NOT used in paper).")

    # PDE params
    p.add_argument("--nu", type=float, default=1e-4)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--print_every", type=int, default=2000)

    # forcing
    p.add_argument("--forcing", type=str, default="fnoeq9", choices=["none", "fnoeq9"])
    p.add_argument("--forcing_amp", type=float, default=0.1)

    # Laplace block
    p.add_argument("--laplace_runs_dir", type=str, default="runs_laplace2d")
    p.add_argument("--laplace_model_dir", type=str, default="")
    p.add_argument("--laplace_model", type=str, default="diag", choices=["diag", "mlp", "fixed"])

    # Poisson block
    p.add_argument("--poisson_runs_dir", type=str, default="runs_poisson2d")
    p.add_argument("--poisson_model_dir", type=str, default="")
    p.add_argument("--poisson_model", type=str, default="diag", choices=["diag", "fixed"])

    # output
    p.add_argument("--out_dir", type=str, default="runs_ns2d_vort")
    p.add_argument("--run_name", type=str, default="ns2d_forced_fnoeq9_strang")

    args = p.parse_args()

    set_seed(args.seed)
    device = default_device()
    real_dtype = torch.float64 if args.use_double else torch.float32
    complex_dtype = get_complex_dtype(real_dtype)
    print("[NS2D-FORCED-STRANG] device:", device, "| dtype:", real_dtype, "| FFT norm:", FFT_NORM)

    kfac = 2.0 * math.pi / float(args.L)
    kfac2 = kfac * kfac
    print(f"[DOMAIN] L={args.L} => kfac=2π/L={kfac:.6g}, kfac^2={kfac2:.6g}")

    # build ridx + mask
    ridx = build_real_fourier_index(args.N, device, real_dtype, Kmax=args.Kmax)
    mask_bool, Kcut = mask_23_and_kmax(args.N, device, real_dtype, Kmax=args.Kmax)
    mask_c = mask_bool.to(device=device).to(complex_dtype)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp
    print(f"[INFO] N={args.N}, Kmax={args.Kmax}, Kcut={Kcut}, dim={dim}")

    # k grids (integer)
    KX_int, KY_int = kgrid_int(args.N, device, real_dtype)
    k2_grid_int = (KX_int**2 + KY_int**2).to(real_dtype)

    # packed invk2 mask
    inv_k2_vec_true, nonzero_mask = build_inv_k2_vec_for_realcoords(ridx, device, real_dtype)

    # ---------- forcing ----------
    if args.forcing == "none":
        f_hat = torch.zeros(args.N, args.N, device=device, dtype=complex_dtype)
        print("[FORCING] none")
    else:
        f_hat = forcing_fnoeq9_hat(args.N, args.L, args.forcing_amp, device, real_dtype, complex_dtype, mask_c)
        f_grid = ifft2(f_hat).real
        print(f"[FORCING] type=fnoeq9 amp={args.forcing_amp} | max|f|={float(f_grid.abs().max().item()):.3e}")

    # ---------- load Laplace block ----------
    lap_dir = args.laplace_model_dir.strip()
    if lap_dir == "":
        lap_dir = find_latest_model_dir(args.laplace_runs_dir)
        print("[INFO] auto-selected latest Laplace2D model_dir:", lap_dir)
    else:
        print("[INFO] using Laplace2D model_dir:", lap_dir)

    lap_model, lap_cfg = load_laplace_block(lap_dir, device, real_dtype, args.laplace_model)
    lap_scale = float(lap_cfg.get("scale", 1.0))
    print("[INFO] Laplace model:", args.laplace_model, "| scale =", lap_scale)

    lap_diag_ok = hasattr(lap_model, "c") or hasattr(lap_model, "k2")
    if not lap_diag_ok:
        # Strang linear exact step needs diagonal k2. If you really want MLP, we'd need a different L integrator.
        raise RuntimeError("[FATAL] Strang(L exact) requires Laplace model to be diag or fixed (providing k2 vector).")

    if hasattr(lap_model, "c"):
        # diag: c ≈ k2/scale (integer-k convention) => physical integer-k k2 = c*scale
        k2_learn_vec_intk = (lap_model.c.detach().to(real_dtype).to(device) * lap_scale)
    else:
        k2_learn_vec_intk = lap_model.k2.detach().to(real_dtype).to(device)
    # stability guard (k2 must be nonnegative)
    k2_learn_vec_intk = torch.clamp(k2_learn_vec_intk, min=0.0)

    # ---------- load Poisson block ----------
    poi_dir = args.poisson_model_dir.strip()
    if poi_dir == "":
        poi_dir = find_latest_model_dir(args.poisson_runs_dir)
        print("[INFO] auto-selected latest Poisson2D model_dir:", poi_dir)
    else:
        print("[INFO] using Poisson2D model_dir:", poi_dir)

    poi_model, poi_cfg, poi_scale = load_poisson_block(poi_dir, device, real_dtype, args.poisson_model, nonzero_mask=nonzero_mask)
    print("[INFO] Poisson model:", args.poisson_model, "| scale =", poi_scale)

    invk2_eff_vec_intk = invk2_eff_from_poisson_model(poi_model, args.poisson_model, poi_scale).to(device=device, dtype=real_dtype)
    invk2_eff_vec_intk = invk2_eff_vec_intk * nonzero_mask.to(device=device, dtype=real_dtype)  # force k=0 to 0

    # ---------- sample IC ω ----------
    if args.ic == "fno_gaussian":
        a_w0 = sample_ic_fno_gaussian_packed(ridx, args.L, device, real_dtype)
    else:
        a_w0 = torch.randn(dim, device=device, dtype=real_dtype)

    omega_hat0 = unpack_realvec_to_hat(a_w0.view(1, -1), ridx, complex_dtype)[0] * mask_c
    w0 = ifft2(omega_hat0).real
    print(f"[IC] max|ω|={float(w0.abs().max().item()):.3e} (before optional rescale)")

    if args.target_max_omega > 0:
        s = max(float(w0.abs().max().item()) / args.target_max_omega, 1e-12)
        omega_hat0 = (omega_hat0 / s) * mask_c
        w0 = ifft2(omega_hat0).real
        print(f"[IC] rescaled => max|ω|={float(w0.abs().max().item()):.3e} (target {args.target_max_omega})")

    # histories
    err_w = np.zeros(args.n_steps + 1, dtype=np.float64)

    # snapshots
    snap_steps_all = [0, args.n_steps//4, args.n_steps//2, 3*args.n_steps//4, args.n_steps]
    snap_steps_all = sorted(list(set([int(s) for s in snap_steps_all])))
    true_snaps_all, learn_snaps_all = [], []

    # run
    omega_hat_T = omega_hat0.clone()
    omega_hat_L = omega_hat0.clone()

    t0 = time.time()
    half = 0.5 * args.dt

    for n in range(args.n_steps + 1):
        wT = ifft2(omega_hat_T).real
        wL = ifft2(omega_hat_L).real
        err_w[n] = rel_l2_grid(wL, wT)

        if (n % args.print_every == 0) or (n == args.n_steps):
            print(f"[step {n:6d}/{args.n_steps}] t={n*args.dt:.3e} | rel_omega={err_w[n]:.3e} | max|ω_ref|={float(wT.abs().max()):.3e}")

        if n in snap_steps_all:
            true_snaps_all.append(wT.detach().cpu().numpy())
            learn_snaps_all.append(wL.detach().cpu().numpy())

        if n == args.n_steps:
            break

        # ---------------------------
        # Strang: L(dt/2) -> N(dt) -> L(dt/2)
        # Reference
        omega_hat_T = linear_L_exact_hat(omega_hat_T, f_hat, k2_grid_int, kfac2, args.nu, half, mask_c)
        omega_hat_T = nonlinear_N_rk2_true(omega_hat_T, k2_grid_int, KX_int, KY_int, kfac, kfac2,
                                           args.dt, mask_c, complex_dtype)
        omega_hat_T = linear_L_exact_hat(omega_hat_T, f_hat, k2_grid_int, kfac2, args.nu, half, mask_c)

        # Learned
        omega_hat_L = linear_L_learned_diag_hat(omega_hat_L, f_hat, k2_learn_vec_intk, ridx, kfac2,
                                               args.nu, half, mask_c, complex_dtype)
        omega_hat_L = nonlinear_N_rk2_learned(omega_hat_L, ridx, invk2_eff_vec_intk, KX_int, KY_int,
                                              kfac, kfac2, args.dt, mask_c, complex_dtype)
        omega_hat_L = linear_L_learned_diag_hat(omega_hat_L, f_hat, k2_learn_vec_intk, ridx, kfac2,
                                               args.nu, half, mask_c, complex_dtype)

    print(f"[DONE] runtime: {time.time()-t0:.2f}s")

    # save
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"{args.run_name}_{run_stamp}")
    os.makedirs(run_dir, exist_ok=True)

    cfg_out: Dict[str, Any] = {
        "pde": "2D forced incompressible NS (vorticity-streamfunction) on (0,1)^2",
        "integrator": {
            "macro": "Strang splitting",
            "L_block": "exact Fourier flow for diffusion+forcing",
            "N_block": "RK2/Heun for pure advection (pseudo-spectral)"
        },
        "equation": [
            "w_t + u·∇w = nu Δw + f(x)",
            "u=(psi_y,-psi_x),  -Δpsi=w",
            "f(x)=0.1(sin(2π(x1+x2))+cos(2π(x1+x2))) [FNO App.A.3.3]",
            "w0 ~ N(0, 7^{3/2}(-Δ+49I)^(-2.5)) [FNO App.A.3.3]",
        ],
        "L": float(args.L),
        "N": int(args.N),
        "Kmax": int(args.Kmax),
        "Kcut": int(Kcut),
        "nu": float(args.nu),
        "dt": float(args.dt),
        "n_steps": int(args.n_steps),
        "T": float(args.dt * args.n_steps),
        "forcing": args.forcing,
        "forcing_amp": float(args.forcing_amp),
        "ic": args.ic,
        "target_max_omega": float(args.target_max_omega),
        "fft_norm": FFT_NORM,
        "dealias": "2/3 law + Kmax cap",
        "laplace_model_dir": lap_dir,
        "laplace_model_type": args.laplace_model,
        "laplace_cfg": lap_cfg,
        "laplace_scale": float(lap_scale),
        "poisson_model_dir": poi_dir,
        "poisson_model_type": args.poisson_model,
        "poisson_cfg": poi_cfg,
        "poisson_scale": float(poi_scale),
        "snap_steps": snap_steps_all,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    np.savez(os.path.join(run_dir, "history.npz"), err_omega=err_w)

    # ---- plots you requested ----
    # step=0 alone: 3x1 (learned / reference / error profile)
    plot_step_column_learn_ref_err(
        u_learn=learn_snaps_all[0],
        u_true=true_snaps_all[0],
        step=snap_steps_all[0],
        dt=args.dt,
        out_path=os.path.join(run_dir, "omega_step00000_compare.png"),
        suptitle_prefix=r"$\omega$",
    )

    # later 4 steps in one horizontal figure: 3 rows x 4 cols
    plot_steps_horizontal_3rows(
        steps=snap_steps_all[1:],
        true_snaps=true_snaps_all[1:],
        learn_snaps=learn_snaps_all[1:],
        dt=args.dt,
        out_path=os.path.join(run_dir, "omega_steps_compare_grid.png"),
    )

    # relative error curve (already stored in err_w)
    plot_error_curve(err_w, args.dt, os.path.join(run_dir, "error_curve_omega.png"))

    print("[SAVE] run_dir:", run_dir)

if __name__ == "__main__":
    main()
