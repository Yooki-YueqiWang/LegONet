#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the 2D Laplacian E-block on the 2D periodic Fourier baseplate (Case Study II).

PDE mechanism being learned:
    F^θ_Δ(a) ≈ F^ref_Δ(a) = P_b[ Δu ]

Block parameterization (E-block, dissipative):
    F^θ_Δ(a) = -G ∇_a E^{a,θ}(a),   G = I  (Fourier coefficient metric)

Because the Laplacian is mode-decoupled in the Fourier basis, the energy generator
is restricted to a diagonal quadratic form:

    E^{a,θ}(a) = (1/2) Σ_k c^θ_k a_k^2   (default: model=diag)

The parameter c^θ ∈ R^K is learned directly from data.  The resulting vector field
F^θ_Δ(a) = -diag(c^θ) a approximates the exact Laplacian eigenvalues (kfac^2 |k|^2)
up to a global scale factor.

This script also supports:
  - model=mlp   : full MLP energy generator (higher capacity, less interpretable)
  - model=fixed : saves exact analytical Laplacian block without training

Training objective (instantaneous operator matching, Eq. 11):
    min_{c^θ} E_{a ~ µ_b} [ || -diag(c^θ) a - Δu_coeff(a) ||^2 / scale^2 ]

where the target Δu_coeff(a) = -(k^2 / scale) ⊙ a is normalized by scale ≈ max(k^2)
to avoid gradient scaling issues.

Stability features:
  - Best checkpoint tracking: saves the model with the lowest test relative error
  - Optional early stopping (--early_stop) based on tolerance or no-improvement count
  - For diag model: learning rate is automatically raised to 5e-2 and weight decay
    is forced to 0 to improve convergence of the single-vector parameter c^θ

Usage:
    # Train diagonal E-block (recommended for Case II)
    python laplace2d_block_2.py --mode train --model diag --N 64 --Kmax 21 \
        --scale -1 --epochs 80 --use_double

    # Save exact fixed block (no training)
    python laplace2d_block_2.py --mode save_fixed --N 64 --Kmax 21 --use_double

    # Evaluate a saved block
    python laplace2d_block_2.py --mode eval --model_dir runs_laplace2d/laplace2d_diag \
        --model diag --N 64 --Kmax 21 --use_double --vis

Outputs:
    runs_laplace2d/<run_name>/model_state.pt  — best model weights
    runs_laplace2d/<run_name>/config.json     — full configuration record
    runs_laplace2d/<run_name>/history.npz     — training history
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D periodic Laplacian (u_xx + u_yy) block trained like your 1D EnergyNet u_xx block.

Key fixes for diag training:
  - allow auto scaling: if --scale <= 0 => scale = k2_max (≈ 2*Kcut^2)
  - for diag: default lr boosted (if too small) and weight_decay forced to 0
  - diag init_scale default = 1.0 (not 1e-3), still random init (not truth)
  - add diagnostics: print c stats & grad stats

NEW (stability):
  - always track and save BEST checkpoint (by smallest test_rel_L2)
  - optional early-stop:
      * stop if test_rel_L2 <= tol for N consecutive epochs
      * or stop if no improvement for M epochs
  - final saved model_state.pt = BEST (not the last drifting epoch)
  - optional also save the LAST model_state_last.pt

Usage examples:
  # diag with auto scale (recommended)
  python laplace2d_block_train_like_1d_fixed_beststop.py --mode train --model diag --N 64 --Kmax 21 \
      --scale -1 --epochs 80 --use_double

  # mlp
  python laplace2d_block_train_like_1d_fixed_beststop.py --mode train --model mlp --N 64 --Kmax 21 \
      --scale -1 --epochs 80 --lr 1e-3 --use_double

  # fixed (exact)
  python laplace2d_block_train_like_1d_fixed_beststop.py --mode save_fixed --N 64 --Kmax 21 --use_double
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# avoid CPU oversubscription
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


# ============================================================
# FFT helpers + k grids + masks
# ============================================================

def fft2(u: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(u, norm=FFT_NORM)

def ifft2(u_hat: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(u_hat, norm=FFT_NORM)

def kgrid_int(N: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    k = torch.fft.fftfreq(N, d=1.0 / N, device=device).to(dtype)
    KX, KY = torch.meshgrid(k, k, indexing="ij")
    return KX, KY

def mask_23_and_kmax(N: int, device: torch.device, dtype: torch.dtype, Kmax: Optional[int]) -> Tuple[torch.Tensor, int]:
    KX, KY = kgrid_int(N, device, dtype)
    k23 = N // 3
    Kcut = k23 if Kmax is None else int(min(k23, Kmax))
    mask = (KX.abs() <= Kcut) & (KY.abs() <= Kcut)
    return mask, Kcut


# ============================================================
# Real Fourier coordinate system
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
# Sampling
# ============================================================

@torch.no_grad()
def sample_coefficients(batch: int,
                        ridx: RealFourierIndex,
                        alpha: float,
                        amp: float,
                        device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
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
# Laplace target
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
    k2_vec = torch.cat([c_self, c_pair, c_pair], dim=0)
    return k2_vec

@torch.no_grad()
def build_dataset(n_samples: int,
                  ridx: RealFourierIndex,
                  alpha: float,
                  amp: float,
                  device: torch.device,
                  dtype: torch.dtype,
                  scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    a_u = sample_coefficients(n_samples, ridx, alpha, amp, device, dtype)
    k2_vec = build_k2_vec_for_realcoords(ridx, device, dtype).view(1, -1)
    lap_true = -(k2_vec * a_u)
    lap_true_scaled = lap_true / float(scale)
    return a_u, lap_true_scaled


# ============================================================
# Energy models
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


# ============================================================
# Energy -> Laplace via -grad
# ============================================================

def energy_and_grad(model: nn.Module, a: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    a_req = a.clone().detach().requires_grad_(True)
    E = model(a_req)
    gradE = torch.autograd.grad(E.sum(), a_req, create_graph=create_graph)[0]
    return E, gradE

def laplace_pred_from_energy(model: nn.Module, a_u: torch.Tensor, create_graph: bool) -> torch.Tensor:
    _, gradE = energy_and_grad(model, a_u, create_graph=create_graph)
    return -gradE


# ============================================================
# Metrics / IO
# ============================================================

def compute_rel_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (torch.linalg.norm(pred - true) / (torch.linalg.norm(true) + 1e-12)).item()

def save_run(run_dir: str, model: nn.Module, cfg: Dict[str, Any], history: Dict[str, List[float]]) -> None:
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "model_state.pt"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    np.savez(os.path.join(run_dir, "history.npz"), **{k: np.array(v) for k, v in history.items()})
    print("Saved to:", run_dir)

def load_block(model_dir: str,
               device: torch.device,
               dtype: torch.dtype,
               model_type: str) -> Tuple[nn.Module, Dict[str, Any]]:
    cfg_path = os.path.join(model_dir, "config.json")
    w_path = os.path.join(model_dir, "model_state.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing config.json/model_state.pt in {model_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dim = int(cfg["dim"])
    if model_type == "mlp":
        mcfg = cfg["mlp"]
        model = MLPEnergy(dim=dim, num_layers=int(mcfg["num_layers"]),
                          hidden_dim=int(mcfg["hidden_dim"]), act=str(mcfg["act"]))
    elif model_type == "diag":
        dcfg = cfg["diag"]
        model = DiagLearnableLaplaceEnergy(dim=dim, init=str(dcfg["init"]), init_scale=float(dcfg["init_scale"]))
    elif model_type == "fixed":
        k2 = torch.tensor(cfg["k2_vec"], dtype=dtype, device=device)
        model = FixedLaplaceEnergy(k2)
    else:
        raise ValueError("unknown model_type")

    model = model.to(device=device, dtype=dtype)
    state = torch.load(w_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


# ============================================================
# Visualization
# ============================================================

def visualize_one_sample(model: nn.Module,
                         ridx: RealFourierIndex,
                         a: torch.Tensor,
                         scale: float,
                         out_path: str) -> None:
    device = a.device
    real_dtype = a.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    u_hat = unpack_realvec_to_hat(a.unsqueeze(0), ridx, complex_dtype)[0]
    u = ifft2(u_hat).real

    KX, KY = kgrid_int(ridx.N, device, real_dtype)
    k2 = (KX**2 + KY**2)
    lap_hat_true = -(k2.to(complex_dtype) * u_hat)
    lap_true = ifft2(lap_hat_true).real

    with torch.enable_grad():
        lap_pred_scaled = laplace_pred_from_energy(model, a.unsqueeze(0), create_graph=False)[0]
    lap_pred = (lap_pred_scaled * scale)
    lap_hat_pred = unpack_realvec_to_hat(lap_pred.unsqueeze(0), ridx, complex_dtype)[0]
    lap_pred_grid = ifft2(lap_hat_pred).real

    u_np = u.detach().cpu().numpy()
    lt_np = lap_true.detach().cpu().numpy()
    lp_np = lap_pred_grid.detach().cpu().numpy()
    diff_np = lp_np - lt_np

    plt.figure(figsize=(12, 3.6))
    for i, (Z, ttl) in enumerate([(u_np, "u(x,y)"),
                                 (lt_np, "true Δu(x,y)"),
                                 (lp_np, "pred Δu(x,y) = -∇E")], start=1):
        ax = plt.subplot(1, 3, i)
        im = ax.imshow(Z, origin="lower")
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    plt.figure(figsize=(4, 3.6))
    ax = plt.gca()
    im = ax.imshow(diff_np, origin="lower")
    ax.set_title("pred - true (Δu)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_diff.png"), dpi=200)
    plt.close()


# ============================================================
# Train / Eval
# ============================================================

def eval_rel(model: nn.Module, a: torch.Tensor, lap_true_scaled: torch.Tensor, scale: float) -> float:
    model.eval()
    with torch.enable_grad():
        lap_pred = laplace_pred_from_energy(model, a, create_graph=False)
    lap_pred_u = lap_pred * scale
    lap_true_u = lap_true_scaled * scale
    return compute_rel_l2(lap_pred_u, lap_true_u)

def _auto_scale_if_needed(args, ridx: RealFourierIndex, device, dtype) -> float:
    if args.scale > 0:
        return float(args.scale)
    k2_vec = build_k2_vec_for_realcoords(ridx, device, dtype)
    k2_max = float(k2_vec.max().item())
    return max(k2_max, 1.0)

def train(args) -> str:
    set_seed(args.seed)
    device = default_device()
    print("[TRAIN] device:", device, "| FFT norm:", FFT_NORM)

    dtype = torch.float64 if args.use_double else torch.float32
    ridx = build_real_fourier_index(args.N, device, dtype, Kmax=args.Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(args.N, device, dtype, Kmax=args.Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp
    print(f"[TRAIN] N={args.N}, Kmax={args.Kmax}, Kcut={Kcut}, dim={dim}")

    # ---- AUTO SCALE ----
    scale = _auto_scale_if_needed(args, ridx, device, dtype)
    if args.scale <= 0:
        print(f"[TRAIN] auto scale enabled: scale = {scale:.6g} (≈ max k^2)")
    else:
        print(f"[TRAIN] scale = {scale:.6g}")

    # data
    a_tr, lap_tr = build_dataset(args.n_train, ridx, args.alpha, args.amp, device, dtype, scale=scale)
    a_te, lap_te = build_dataset(args.n_test,  ridx, args.alpha, args.amp, device, dtype, scale=scale)
    a_tr, lap_tr = a_tr.cpu(), lap_tr.cpu()
    a_te, lap_te = a_te.cpu(), lap_te.cpu()

    train_loader = DataLoader(TensorDataset(a_tr, lap_tr),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    # model
    if args.model == "mlp":
        model = MLPEnergy(dim=dim, num_layers=args.num_layers, hidden_dim=args.hidden_dim, act=args.act)
    elif args.model == "diag":
        model = DiagLearnableLaplaceEnergy(dim=dim, init=args.diag_init, init_scale=args.diag_init_scale)
    elif args.model == "fixed":
        k2 = build_k2_vec_for_realcoords(ridx, device, dtype)
        model = FixedLaplaceEnergy(k2)
    else:
        raise ValueError("Unknown model")

    model = model.to(device=device, dtype=dtype)

    # ---- fixed: just save ----
    if args.model == "fixed":
        rel = eval_rel(model, a_te.to(device), lap_te.to(device), scale)
        print(f"[FIXED] test_rel_L2 = {rel:.3e}")
        run_dir = os.path.join(args.out_dir, args.run_name)
        cfg = {
            "block": "Laplace2D",
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
            "k2_vec": build_k2_vec_for_realcoords(ridx, device, dtype).detach().cpu().tolist(),
        }
        history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}
        save_run(run_dir, model, cfg, history)
        return run_dir

    # ---- hyperparam adjustments for diag ----
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    if args.model == "diag":
        if lr < 1e-2:
            lr = 5e-2
        if weight_decay != 0.0:
            weight_decay = 0.0
        print(f"[TRAIN][diag] using lr={lr:.3e}, weight_decay={weight_decay:.1e}, init={args.diag_init}, init_scale={args.diag_init_scale}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}

    # ---- BEST CHECKPOINT + EARLY STOP ----
    best_rel = float("inf")
    best_ep = -1
    best_state = None

    good_count = 0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, total = 0.0, 0

        for a_b, lap_true_scaled_b in train_loader:
            a_b = a_b.to(device)
            lap_true_scaled_b = lap_true_scaled_b.to(device)

            optimizer.zero_grad(set_to_none=True)

            lap_pred_scaled = laplace_pred_from_energy(model, a_b, create_graph=True)
            loss = F.mse_loss(lap_pred_scaled, lap_true_scaled_b)

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

        rel = eval_rel(model, a_te.to(device), lap_te.to(device), scale)

        # diagnostics for diag
        if args.model == "diag" and (ep <= 3 or ep % args.diag_print_every == 0):
            with torch.no_grad():
                c = model.c.detach()
                c_mean = c.abs().mean().item()
                c_max = c.abs().max().item()
                g_mean = float("nan")
                g_max = float("nan")
                if model.c.grad is not None:
                    g = model.c.grad.detach()
                    g_mean = g.abs().mean().item()
                    g_max = g.abs().max().item()
            print(f"[diag stats] |c|mean={c_mean:.3e} |c|max={c_max:.3e} |grad|mean={g_mean:.3e} |grad|max={g_max:.3e}")

        print(f"Epoch {ep:4d} | lr={lr_now:.3e} | train_mse={train_mse:.3e} | test_rel_L2={rel:.3e}")

        history["epoch"].append(ep)
        history["lr"].append(lr_now)
        history["train_mse"].append(train_mse)
        history["test_rel_l2"].append(rel)

        # ---- update best ----
        if rel + args.best_eps < best_rel:
            best_rel = rel
            best_ep = ep
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        # ---- early stop (tol for consecutive epochs) ----
        if args.early_stop and (rel <= args.early_stop_tol):
            good_count += 1
        else:
            good_count = 0

        if args.early_stop and good_count >= args.early_stop_patience:
            print(f"[EARLY STOP] rel_L2 <= {args.early_stop_tol:.3e} for {args.early_stop_patience} consecutive epochs. Stop at ep={ep}.")
            break

        # ---- early stop (no improvement) ----
        if args.early_stop_patience_no_improve > 0 and no_improve >= args.early_stop_patience_no_improve:
            print(f"[EARLY STOP] no improvement for {args.early_stop_patience_no_improve} epochs. Best ep={best_ep}, best_rel={best_rel:.3e}. Stop at ep={ep}.")
            break

    # choose BEST to save (prevents late drift)
    if best_state is None:
        best_state = model.state_dict()
        best_ep = history["epoch"][-1] if len(history["epoch"]) > 0 else -1
        best_rel = history["test_rel_l2"][-1] if len(history["test_rel_l2"]) > 0 else float("inf")

    best_model = copy.deepcopy(model).eval()
    best_model.load_state_dict(best_state)
    for p in best_model.parameters():
        p.requires_grad_(False)

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # (optional) also save LAST
    if args.save_last_too:
        last_state_path = os.path.join(run_dir, "model_state_last.pt")
        torch.save(model.state_dict(), last_state_path)

    cfg: Dict[str, Any] = {
        "block": "Laplace2D",
        "model": args.model,
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
        "early_stop": bool(args.early_stop),
        "early_stop_tol": float(args.early_stop_tol),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_patience_no_improve": int(args.early_stop_patience_no_improve),
    }
    if args.model == "mlp":
        cfg["mlp"] = {"num_layers": int(args.num_layers), "hidden_dim": int(args.hidden_dim), "act": str(args.act)}
    if args.model == "diag":
        cfg["diag"] = {"init": str(args.diag_init), "init_scale": float(args.diag_init_scale)}

    # save best as model_state.pt
    torch.save(best_model.state_dict(), os.path.join(run_dir, "model_state.pt"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    np.savez(os.path.join(run_dir, "history.npz"), **{k: np.array(v) for k, v in history.items()})
    print(f"Saved BEST to: {run_dir}  (best_ep={best_ep}, best_rel={best_rel:.3e})")
    return run_dir


def evaluate(args) -> None:
    set_seed(args.seed)
    device = default_device()
    print("[EVAL] device:", device, "| FFT norm:", FFT_NORM)

    dtype = torch.float64 if args.use_double else torch.float32
    ridx = build_real_fourier_index(args.N, device, dtype, Kmax=args.Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(args.N, device, dtype, Kmax=args.Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp
    print(f"[EVAL] N={args.N}, Kmax={args.Kmax}, Kcut={Kcut}, dim={dim}")

    model, cfg = load_block(args.model_dir, device, dtype, args.model)

    scale = float(cfg.get("scale", args.scale if args.scale > 0 else 1.0))
    if scale <= 0:
        scale = _auto_scale_if_needed(args, ridx, device, dtype)

    a_te, lap_te = build_dataset(args.n_test, ridx, args.alpha, args.amp, device, dtype, scale=scale)
    with torch.enable_grad():
        lap_pred = laplace_pred_from_energy(model, a_te, create_graph=False) * scale
    lap_true = lap_te * scale
    rel = compute_rel_l2(lap_pred, lap_true)
    print(f"[EVAL] scale={scale:.6g} | rel_L2 = {rel:.3e}")

    if args.vis:
        idx = int(args.vis_index)
        idx = max(0, min(idx, a_te.shape[0] - 1))
        out_dir = os.path.join(args.eval_out_dir, args.eval_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"vis_sample_{idx:04d}.png")
        visualize_one_sample(model, ridx, a_te[idx].to(device), scale, out_path)
        print("[EVAL] saved visualization to:", out_path)


def save_fixed_only(args) -> None:
    set_seed(args.seed)
    device = default_device()
    dtype = torch.float64 if args.use_double else torch.float32
    ridx = build_real_fourier_index(args.N, device, dtype, Kmax=args.Kmax)
    _mask_bool, Kcut = mask_23_and_kmax(args.N, device, dtype, Kmax=args.Kmax)

    Ms = int(ridx.is_self.sum().item())
    Mp = ridx.M - Ms
    dim = Ms + 2 * Mp

    k2 = build_k2_vec_for_realcoords(ridx, device, dtype)
    model = FixedLaplaceEnergy(k2).to(device=device, dtype=dtype)

    scale = _auto_scale_if_needed(args, ridx, device, dtype)

    run_dir = os.path.join(args.out_dir, args.run_name)
    history = {"epoch": [], "lr": [], "train_mse": [], "test_rel_l2": []}
    cfg = {
        "block": "Laplace2D",
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
        "k2_vec": k2.detach().cpu().tolist(),
    }
    save_run(run_dir, model, cfg, history)


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "save_fixed"])

    # shared
    p.add_argument("--N", type=int, default=64)  #64 128
    p.add_argument("--Kmax", type=int, default=21)  #21 42
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--amp", type=float, default=1.0)

    # scale<=0 triggers auto scale
    p.add_argument("--scale", type=float, default=-1.0)

    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_test", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=128)

    p.add_argument("--model", type=str, default="diag", choices=["fixed", "diag", "mlp"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_double", action="store_true")

    # train hyperparams
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--step_size", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.3)

    # diag
    p.add_argument("--diag_init", type=str, default="rand", choices=["rand", "ones"])
    p.add_argument("--diag_init_scale", type=float, default=1.0)
    p.add_argument("--diag_print_every", type=int, default=10)

    # NEW: best/early-stop controls
    p.add_argument("--early_stop", action="store_true",
                   help="enable early stopping based on test_rel_L2")
    p.add_argument("--early_stop_tol", type=float, default=1e-6,
                   help="if test_rel_L2 <= tol, count as good epoch")
    p.add_argument("--early_stop_patience", type=int, default=5,
                   help="stop after this many consecutive good epochs")
    p.add_argument("--early_stop_patience_no_improve", type=int, default=30,
                   help="stop if no improvement for this many epochs (0 disables)")
    p.add_argument("--best_eps", type=float, default=0.0,
                   help="treat improvement only if rel decreases by at least eps (avoid tiny numeric flips)")
    p.add_argument("--save_last_too", action="store_true",
                   help="also save last model to model_state_last.pt")

    # mlp params
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--act", type=str, default="gelu")

    # IO
    p.add_argument("--out_dir", type=str, default="runs_laplace2d")
    p.add_argument("--run_name", type=str, default="laplace2d_block")
    p.add_argument("--model_dir", type=str, default="")

    # eval viz
    p.add_argument("--vis", action="store_true")
    p.add_argument("--vis_index", type=int, default=3)
    p.add_argument("--eval_out_dir", type=str, default="runs_laplace2d_eval")
    p.add_argument("--eval_name", type=str, default="laplace2d_vis")

    return p

def main():
    args = build_parser().parse_args()

    if args.mode == "save_fixed":
        args.model = "fixed"
        save_fixed_only(args)
        return

    if args.mode == "train":
        if args.run_name == "laplace2d_block":
            args.run_name = f"laplace2d_{args.model}"
        train(args)
        return

    if args.mode == "eval":
        if not args.model_dir:
            raise ValueError("--model_dir is required for eval")
        evaluate(args)
        return

if __name__ == "__main__":
    main()
