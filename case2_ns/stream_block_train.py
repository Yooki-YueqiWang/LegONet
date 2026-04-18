#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poisson inverse / streamfunction Hamiltonian block on 2D periodic Fourier grid.

Goal:
    -Δ ψ = ω   on T^2
Fourier:
    ψ̂(k) = ω̂(k) / |k|^2   for k != 0,   ψ̂(0)=0

We train a diagonal Hamiltonian/EnergyNet:
    Hθ(a) = 0.5 * sum_i c_i * a_i^2
so that
    ∇_a Hθ(a_ω) ≈ a_ψ
i.e. learn c_i ≈ 1/|k|^2 in real-packed Fourier coordinates.

This file includes:
    - train / eval / save_fixed
    - verify: coefficient error, grid error, velocity error, weights error, plots

PowerShell examples:
  python .\\stream_block_train.py --mode train --model diag --N 64 --Kmax 21 --epochs 200 --use_double
  python .\\stream_block_train.py --mode eval  --model diag --model_dir runs_poisson2d\\poisson2d_diag --N 64 --Kmax 21 --use_double --verify --n_verify 512
"""

from __future__ import annotations
import argparse, copy, json, math, os, random, time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
# Real Fourier coordinate system (same as your Laplace2D script)
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
    pair_re  = (w[~is_self] * vals[:, ~is_self].real)
    pair_im  = (w[~is_self] * vals[:, ~is_self].imag)
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


# ============================================================
# Sampling ω coefficients
# ============================================================

@torch.no_grad()
def sample_coefficients(batch: int,
                        ridx: RealFourierIndex,
                        alpha: float,
                        amp: float,
                        device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
    """
    Sample omega coefficients in packed real coords.
    sigma_mode ~ amp/(1+|k|)^alpha
    """
    ij = ridx.ij
    KX, KY = kgrid_int(ridx.N, device, dtype)
    kx = KX[ij[:, 0], ij[:, 1]]
    ky = KY[ij[:, 0], ij[:, 1]]
    kmag = torch.sqrt(kx ** 2 + ky ** 2)
    sigma_mode = amp / (1.0 + kmag).pow(alpha)

    is_self = ridx.is_self
    Ms = int(is_self.sum().item())
    Mp = ridx.M - Ms

    a_self = torch.randn(batch, Ms, device=device, dtype=dtype) * sigma_mode[is_self].view(1, -1)
    sig_p = sigma_mode[~is_self].view(1, -1)
    a_re = torch.randn(batch, Mp, device=device, dtype=dtype) * sig_p
    a_im = torch.randn(batch, Mp, device=device, dtype=dtype) * sig_p
    return torch.cat([a_self, a_re, a_im], dim=1)


# ============================================================
# Build inv_k2 vector in real-packed coords: a_psi = inv_k2 ⊙ a_omega
# ============================================================

def build_k2_vec_for_realcoords(ridx: RealFourierIndex,
                                device: torch.device,
                                dtype: torch.dtype) -> torch.Tensor:
    ij = ridx.ij
    KX, KY = kgrid_int(ridx.N, device, dtype)
    k2_mode = (KX[ij[:, 0], ij[:, 1]]**2 + KY[ij[:, 0], ij[:, 1]]**2)

    is_self = ridx.is_self
    Ms = int(is_self.sum().item())
    Mp = ridx.M - Ms

    c_self = k2_mode[is_self]
    c_pair = k2_mode[~is_self]
    return torch.cat([c_self, c_pair, c_pair], dim=0)

def build_inv_k2_vec_for_realcoords(ridx: RealFourierIndex,
                                   device: torch.device,
                                   dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      inv_k2_vec: (dim,)
      nonzero_mask: (dim,) float mask (1 for k!=0, 0 for k=0)
    """
    k2_vec = build_k2_vec_for_realcoords(ridx, device, dtype)
    nonzero = (k2_vec > 0).to(dtype)
    inv = torch.zeros_like(k2_vec)
    inv[nonzero.bool()] = 1.0 / k2_vec[nonzero.bool()]
    return inv, nonzero

@torch.no_grad()
def build_dataset(n_samples: int,
                  ridx: RealFourierIndex,
                  alpha: float,
                  amp: float,
                  device: torch.device,
                  dtype: torch.dtype,
                  scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    a_w: sampled ω coeffs (real-packed)
    a_psi_true_scaled: (inv_k2 ⊙ a_w) / scale
    loss_mask: float mask to ignore k=0 mode in loss
    """
    a_w = sample_coefficients(n_samples, ridx, alpha, amp, device, dtype)
    inv_k2_vec, nonzero_mask = build_inv_k2_vec_for_realcoords(ridx, device, dtype)
    a_psi_true = inv_k2_vec.view(1, -1) * a_w
    a_psi_true_scaled = a_psi_true / float(scale)
    return a_w, a_psi_true_scaled, nonzero_mask  # (dim,)


# ============================================================
# Energy models (Hamiltonian) -> streamfunction via grad
# ============================================================

class FixedPoissonHamiltonian(nn.Module):
    def __init__(self, inv_k2_vec: torch.Tensor):
        super().__init__()
        self.register_buffer("inv_k2", inv_k2_vec.clone().detach())

    def forward(self, a_w: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.inv_k2.view(1, -1) * (a_w * a_w)).sum(dim=1)

class DiagLearnablePoissonHamiltonian(nn.Module):
    """
    Learn inv_k2 diagonals with positivity + force k=0 weight to 0.
    """
    def __init__(self, dim: int, nonzero_mask: torch.Tensor, init_scale: float = 0.1):
        super().__init__()
        self.register_buffer("mask", nonzero_mask.clone().detach())  # float 0/1
        self.raw = nn.Parameter(init_scale * torch.randn(dim))

    def c(self) -> torch.Tensor:
        return F.softplus(self.raw) * self.mask  # positive on nonzero, exactly 0 at k=0

    def forward(self, a_w: torch.Tensor) -> torch.Tensor:
        c = self.c().view(1, -1).to(a_w.dtype).to(a_w.device)
        return 0.5 * (c * (a_w * a_w)).sum(dim=1)

def energy_and_grad(model: nn.Module, a: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    a_req = a.clone().detach().requires_grad_(True)
    H = model(a_req)
    gradH = torch.autograd.grad(H.sum(), a_req, create_graph=create_graph)[0]
    return H, gradH

def stream_pred_from_energy(model: nn.Module, a_w: torch.Tensor, create_graph: bool) -> torch.Tensor:
    # ψ = δH/δω  -> use +grad (no minus)
    _, gradH = energy_and_grad(model, a_w, create_graph=create_graph)
    return gradH


# ============================================================
# Metrics + verification helpers
# ============================================================

def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-12)).item()

@torch.no_grad()
def coeff_to_grid_scalar(a: torch.Tensor, ridx: RealFourierIndex, mask_bool: torch.Tensor) -> torch.Tensor:
    """
    a: (B, dim) real-packed coeffs
    returns u: (B, N, N) real grid, with low-mode mask applied.
    """
    device = a.device
    dtype = a.dtype
    ctype = get_complex_dtype(dtype)
    mask_c = mask_bool.to(ctype)
    u_hat = unpack_realvec_to_hat(a, ridx, ctype) * mask_c
    u = ifft2(u_hat).real
    return u

@torch.no_grad()
def psi_to_velocity_grid(a_psi: torch.Tensor, ridx: RealFourierIndex, mask_bool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    u = (ψ_y, -ψ_x)
    """
    device = a_psi.device
    dtype = a_psi.dtype
    ctype = get_complex_dtype(dtype)

    KX, KY = kgrid_int(ridx.N, device, dtype)
    ikx = (1j * KX).to(ctype)
    iky = (1j * KY).to(ctype)

    mask_c = mask_bool.to(ctype)
    psi_hat = unpack_realvec_to_hat(a_psi, ridx, ctype) * mask_c

    u_hat = iky * psi_hat
    v_hat = -ikx * psi_hat
    u = ifft2(u_hat).real
    v = ifft2(v_hat).real
    return u, v

def plot_weights(inv_k2_true: np.ndarray, c_learn: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(inv_k2_true, label="true inv_k2")
    plt.plot(c_learn, label="learned c")
    plt.yscale("log")
    plt.xlabel("mode index in packed vector")
    plt.ylabel("value (log scale)")
    plt.title("Diagonal weights: true vs learned")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_psi_compare(w: np.ndarray, psi_true: np.ndarray, psi_pred: np.ndarray, out_path: str, suptitle: str) -> None:
    diff = psi_pred - psi_true
    plt.figure(figsize=(12, 3.6))
    for i, (Z, ttl) in enumerate([(w, r"$\omega$"),
                                 (psi_true, r"$\psi$ true"),
                                 (psi_pred, r"$\psi$ pred"),
                                 (diff, r"$\psi$ pred - true")], start=1):
        ax = plt.subplot(1, 4, i)
        im = ax.imshow(Z, origin="lower")
        ax.set_title(ttl)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# IO
# ============================================================

def save_run(run_dir: str, model: nn.Module, cfg: Dict[str, Any], history: Dict[str, List[float]]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "model_state.pt"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    np.savez(os.path.join(run_dir, "history.npz"), **{k: np.array(v) for k, v in history.items()})
    print("Saved to:", run_dir)

def load_config(model_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_block(model_dir: str, device: torch.device, dtype: torch.dtype,
               model_type: str, nonzero_mask: torch.Tensor) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg = load_config(model_dir)

    # ---- SAFETY CHECK: avoid accidentally loading Laplace block ----
    if cfg.get("block", "") != "Poisson2D_Streamfunction":
        raise RuntimeError(
            f"[FATAL] model_dir seems NOT a Poisson block. cfg['block']={cfg.get('block','<missing>')}. "
            f"Did you point to a Laplace2D run by mistake?"
        )

    dim = int(cfg["dim"])
    if model_type == "fixed":
        inv_k2 = torch.tensor(cfg["inv_k2_vec"], dtype=dtype, device=device)
        model = FixedPoissonHamiltonian(inv_k2)
    elif model_type == "diag":
        init_scale = float(cfg.get("diag", {}).get("init_scale", 0.1))
        model = DiagLearnablePoissonHamiltonian(dim=dim, nonzero_mask=nonzero_mask.to(device=device, dtype=dtype),
                                                init_scale=init_scale)
    else:
        raise ValueError("unknown model_type")

    w_path = os.path.join(model_dir, "model_state.pt")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing model_state.pt in {model_dir}")
    state = torch.load(w_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg


# ============================================================
# Train / Eval / Verify
# ============================================================

def _auto_scale_if_needed(args, inv_k2_vec: torch.Tensor) -> float:
    if args.scale > 0:
        return float(args.scale)
    # max(inv_k2)=1 for k_min=1, so usually 1
    s = float(inv_k2_vec.max().item())
    return max(s, 1.0)

def eval_rel(model: nn.Module, a_w: torch.Tensor, psi_true_scaled: torch.Tensor,
             scale: float, loss_mask: torch.Tensor) -> float:
    model.eval()
    # 关键：这里必须 enable_grad，因为要用 autograd.grad 求 ∇H
    with torch.enable_grad():
        psi_pred_scaled = stream_pred_from_energy(model, a_w, create_graph=False)

    m = loss_mask.view(1, -1).to(a_w.device).to(a_w.dtype)
    psi_pred = (psi_pred_scaled * scale) * m
    psi_true = (psi_true_scaled * scale) * m
    return rel_l2(psi_pred, psi_true)


@torch.no_grad()
def verify_report(model: nn.Module,
                  ridx: RealFourierIndex,
                  inv_k2_true: torch.Tensor,
                  scale: float,
                  alpha: float,
                  amp: float,
                  n_verify: int,
                  out_dir: str,
                  tag: str) -> Dict[str, float]:
    device = inv_k2_true.device
    dtype = inv_k2_true.dtype
    mask_bool, _ = mask_23_and_kmax(ridx.N, device, dtype, Kmax=ridx.Kcut)

    # sample omega
    a_w, psi_true_scaled, loss_mask = build_dataset(n_verify, ridx, alpha, amp, device, dtype, scale=scale)
    with torch.enable_grad():
        psi_pred_scaled = stream_pred_from_energy(model, a_w, create_graph=False)

    m = loss_mask.view(1, -1).to(dtype)

    # coeff errors (rescaled back)
    psi_true = psi_true_scaled * scale * m
    psi_pred = psi_pred_scaled * scale * m
    rel_coeff_psi = rel_l2(psi_pred, psi_true)

    # grid errors (psi)
    psi_true_grid = coeff_to_grid_scalar(psi_true, ridx, mask_bool)
    psi_pred_grid = coeff_to_grid_scalar(psi_pred, ridx, mask_bool)
    rel_grid_psi = rel_l2(psi_pred_grid, psi_true_grid)

    # velocity errors
    u_true, v_true = psi_to_velocity_grid(psi_true, ridx, mask_bool)
    u_pred, v_pred = psi_to_velocity_grid(psi_pred, ridx, mask_bool)
    # stack two components
    rel_grid_u = rel_l2(torch.stack([u_pred, v_pred], dim=1), torch.stack([u_true, v_true], dim=1))

    # weights error (if diag)
    rel_weights_c = float("nan")
    c_learn = None
    if hasattr(model, "c"):
        c_learn = model.c().detach()
        rel_weights_c = rel_l2(c_learn, inv_k2_true)

    os.makedirs(out_dir, exist_ok=True)

    # plots: weights + one-sample compare
    if c_learn is not None:
        plot_weights(inv_k2_true.detach().cpu().numpy(),
                     c_learn.detach().cpu().numpy(),
                     os.path.join(out_dir, f"weights_compare_{tag}.png"))

    # one sample plot
    w0 = coeff_to_grid_scalar(a_w[:1], ridx, mask_bool)[0].detach().cpu().numpy()
    pt0 = psi_true_grid[0].detach().cpu().numpy()
    pp0 = psi_pred_grid[0].detach().cpu().numpy()
    plot_psi_compare(w0, pt0, pp0, os.path.join(out_dir, f"psi_compare_{tag}.png"),
                     suptitle=f"verify: rel_coeff_psi={rel_coeff_psi:.3e}, rel_grid_psi={rel_grid_psi:.3e}, rel_grid_u={rel_grid_u:.3e}")

    report = {
        "rel_coeff_psi": float(rel_coeff_psi),
        "rel_grid_psi": float(rel_grid_psi),
        "rel_grid_u": float(rel_grid_u),
        "rel_weights_c": float(rel_weights_c) if not math.isnan(rel_weights_c) else float("nan"),
    }
    with open(os.path.join(out_dir, f"report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

def train(args) -> str:
    set_seed(args.seed)
    device = default_device()
    dtype = torch.float64 if args.use_double else torch.float32
    print("[TRAIN] device:", device, "| dtype:", dtype, "| FFT norm:", FFT_NORM)

    ridx = build_real_fourier_index(args.N, device, dtype, Kmax=args.Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(args.N, device, dtype, Kmax=args.Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp
    print(f"[TRAIN] N={args.N}, Kmax={args.Kmax}, Kcut={Kcut}, dim={dim}")

    inv_k2_vec, nonzero_mask = build_inv_k2_vec_for_realcoords(ridx, device, dtype)
    scale = _auto_scale_if_needed(args, inv_k2_vec)
    if args.scale <= 0:
        print(f"[TRAIN] auto scale enabled: scale = {scale:.6g} (≈ max inv_k2)")
    else:
        print(f"[TRAIN] scale = {scale:.6g}")

    # data
    a_tr, psi_tr, loss_mask = build_dataset(args.n_train, ridx, args.alpha, args.amp, device, dtype, scale=scale)
    a_te, psi_te, _ = build_dataset(args.n_test,  ridx, args.alpha, args.amp, device, dtype, scale=scale)

    a_tr, psi_tr = a_tr.cpu(), psi_tr.cpu()
    a_te, psi_te = a_te.to(device), psi_te.to(device)
    loss_mask = loss_mask.to(device=device, dtype=dtype)
    inv_k2_vec = inv_k2_vec.to(device=device, dtype=dtype)

    train_loader = DataLoader(TensorDataset(a_tr, psi_tr),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    # model
    if args.model == "fixed":
        model = FixedPoissonHamiltonian(inv_k2_vec)
    elif args.model == "diag":
        model = DiagLearnablePoissonHamiltonian(dim=dim, nonzero_mask=nonzero_mask, init_scale=args.diag_init_scale)
    else:
        raise ValueError("Unknown model")
    model = model.to(device=device, dtype=dtype)

    # fixed: just save
    if args.model == "fixed":
        rel = eval_rel(model, a_te, psi_te, scale, loss_mask)
        print(f"[FIXED] test_rel_L2 = {rel:.3e}")
        run_dir = os.path.join(args.out_dir, args.run_name)
        cfg = {
            "block": "Poisson2D_Streamfunction",
            "model": "fixed",
            "fft_norm": FFT_NORM,
            "domain": "[0,2π)^2 periodic",
            "dealiasing": "2/3 law + Kmax cap",
            "N": int(args.N),
            "Kmax": int(args.Kmax),
            "Kcut": int(Kcut),
            "dim": int(dim),
            "alpha": float(args.alpha),
            "amp": float(args.amp),
            "scale": float(scale),
            "dtype": "float64" if args.use_double else "float32",
            "inv_k2_vec": inv_k2_vec.detach().cpu().tolist(),
            "note": "H(ω)=0.5*<ω,(-Δ)^{-1}ω>; grad gives ψ.",
        }
        history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}
        save_run(run_dir, model, cfg, history)
        return run_dir

    # ---- diag hyperparam fix (match your Laplace2D style) ----
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    if lr < 1e-2:
        lr = 5e-2
    weight_decay = 0.0
    print(f"[TRAIN][diag] using lr={lr:.3e}, weight_decay={weight_decay:.1e}, init_scale={args.diag_init_scale}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}
    best_rel, best_ep, best_state = float("inf"), -1, None

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, total = 0.0, 0

        for a_b, psi_true_scaled_b in train_loader:
            a_b = a_b.to(device)
            psi_true_scaled_b = psi_true_scaled_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            psi_pred_scaled = stream_pred_from_energy(model, a_b, create_graph=True)
            m = loss_mask.view(1, -1).to(a_b.dtype)
            loss = F.mse_loss(psi_pred_scaled * m, psi_true_scaled_b * m)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            B = a_b.size(0)
            run_loss += loss.item() * B
            total += B

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        train_mse = run_loss / max(total, 1)
        rel = eval_rel(model, a_te, psi_te, scale, loss_mask)

        if ep <= 3 or ep % args.diag_print_every == 0:
            with torch.no_grad():
                c = model.c().detach()
                print(f"[diag stats] c(min/mean/max) = {float(c.min()):.3e} / {float(c.mean()):.3e} / {float(c.max()):.3e}")

        print(f"Epoch {ep:4d} | lr={lr_now:.3e} | train_mse={train_mse:.3e} | test_rel_L2={rel:.3e}")

        history["epoch"].append(ep)
        history["lr"].append(lr_now)
        history["train_mse"].append(train_mse)
        history["test_rel_l2"].append(rel)

        if rel < best_rel - 1e-15:
            best_rel, best_ep = rel, ep
            best_state = copy.deepcopy(model.state_dict())

    print(f"[DONE] training time: {time.time()-t0:.2f}s")

    if best_state is None:
        best_state = model.state_dict()

    best_model = copy.deepcopy(model).eval()
    best_model.load_state_dict(best_state)
    for p in best_model.parameters():
        p.requires_grad_(False)

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg: Dict[str, Any] = {
        "block": "Poisson2D_Streamfunction",
        "model": "diag",
        "fft_norm": FFT_NORM,
        "domain": "[0,2π)^2 periodic",
        "dealiasing": "2/3 law + Kmax cap",
        "N": int(args.N),
        "Kmax": int(args.Kmax),
        "Kcut": int(Kcut),
        "dim": int(dim),
        "alpha": float(args.alpha),
        "amp": float(args.amp),
        "scale": float(scale),
        "dtype": "float64" if args.use_double else "float32",
        "best_epoch": int(best_ep),
        "best_test_rel_l2": float(best_rel),
        "diag": {"init_scale": float(args.diag_init_scale), "positivity": "softplus(raw)*mask_nonzero", "k0_mode": "forced to 0"},
        "inv_k2_vec": inv_k2_vec.detach().cpu().tolist(),  # store true for verification / reproducibility
    }

    torch.save(best_model.state_dict(), os.path.join(run_dir, "model_state.pt"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    np.savez(os.path.join(run_dir, "history.npz"), **{k: np.array(v) for k, v in history.items()})
    print(f"Saved BEST to: {run_dir} (best_ep={best_ep}, best_rel={best_rel:.3e})")
    return run_dir

def evaluate(args) -> None:
    set_seed(args.seed)
    device = default_device()
    dtype = torch.float64 if args.use_double else torch.float32
    print("[EVAL] device:", device, "| dtype:", dtype)

    if not args.model_dir:
        raise ValueError("--model_dir is required for eval")

    # Use config to avoid mismatches
    cfg = load_config(args.model_dir)
    N = int(cfg["N"])
    Kmax = int(cfg["Kmax"])
    scale = float(cfg.get("scale", 1.0))
    print(f"[EVAL] using cfg: N={N}, Kmax={Kmax}, scale={scale}")

    ridx = build_real_fourier_index(N, device, dtype, Kmax=Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(N, device, dtype, Kmax=Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp
    print(f"[EVAL] N={N}, Kmax={Kmax}, Kcut={Kcut}, dim={dim}")

    inv_k2_true, nonzero_mask = build_inv_k2_vec_for_realcoords(ridx, device, dtype)

    # load model (with strict block check)
    model, cfg_loaded = load_block(args.model_dir, device, dtype, args.model, nonzero_mask=nonzero_mask)

    # eval set
    a_te, psi_te, loss_mask = build_dataset(args.n_test, ridx, args.alpha, args.amp, device, dtype, scale=scale)
    rel = eval_rel(model, a_te, psi_te, scale, loss_mask.to(device=device, dtype=dtype))
    print(f"[EVAL] rel_L2(coeff psi) = {rel:.6e}")

    if args.verify:
        out_dir = os.path.join(args.eval_out_dir, args.eval_name)
        tag = f"N{N}_K{Kmax}_B{args.n_verify}"
        rep = verify_report(model, ridx, inv_k2_true, scale, args.alpha, args.amp, args.n_verify, out_dir, tag)
        print("[VERIFY]", " | ".join([f"{k}={v:.6g}" for k, v in rep.items()]))
        print("[VERIFY] saved report to:", out_dir)

def save_fixed_only(args) -> None:
    set_seed(args.seed)
    device = default_device()
    dtype = torch.float64 if args.use_double else torch.float32

    ridx = build_real_fourier_index(args.N, device, dtype, Kmax=args.Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(args.N, device, dtype, Kmax=args.Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp

    inv_k2_vec, _ = build_inv_k2_vec_for_realcoords(ridx, device, dtype)
    scale = _auto_scale_if_needed(args, inv_k2_vec)

    model = FixedPoissonHamiltonian(inv_k2_vec).to(device=device, dtype=dtype)

    run_dir = os.path.join(args.out_dir, args.run_name)
    history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}
    cfg = {
        "block": "Poisson2D_Streamfunction",
        "model": "fixed",
        "fft_norm": FFT_NORM,
        "domain": "[0,2π)^2 periodic",
        "dealiasing": "2/3 law + Kmax cap",
        "N": int(args.N),
        "Kmax": int(args.Kmax),
        "Kcut": int(Kcut),
        "dim": int(dim),
        "alpha": float(args.alpha),
        "amp": float(args.amp),
        "scale": float(scale),
        "dtype": "float64" if args.use_double else "float32",
        "inv_k2_vec": inv_k2_vec.detach().cpu().tolist(),
    }
    save_run(run_dir, model, cfg, history)


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "save_fixed"])

    p.add_argument("--N", type=int, default=64)  #64 128
    p.add_argument("--Kmax", type=int, default=21)  #21 42
    p.add_argument("--alpha", type=float, default=0.0, help="建议 Poisson 用 0.0（各模同方差）更好学")
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=-1.0)

    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_test", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--model", type=str, default="diag", choices=["fixed", "diag"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_double", action="store_true")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--step_size", type=int, default=80)
    p.add_argument("--gamma", type=float, default=0.3)

    p.add_argument("--diag_init_scale", type=float, default=0.1)
    p.add_argument("--diag_print_every", type=int, default=10)

    p.add_argument("--out_dir", type=str, default="runs_poisson2d")
    p.add_argument("--run_name", type=str, default="poisson2d_diag")
    p.add_argument("--model_dir", type=str, default="")

    p.add_argument("--verify", action="store_true")
    p.add_argument("--n_verify", type=int, default=512)
    p.add_argument("--eval_out_dir", type=str, default="runs_poisson2d_eval")
    p.add_argument("--eval_name", type=str, default="poisson2d_verify")

    return p

def main():
    args = build_parser().parse_args()

    if args.mode == "save_fixed":
        args.model = "fixed"
        save_fixed_only(args)
        return

    if args.mode == "train":
        if args.run_name == "poisson2d_diag":
            args.run_name = f"poisson2d_{args.model}"
        train(args)
        return

    if args.mode == "eval":
        evaluate(args)
        return

if __name__ == "__main__":
    main()
