# LegONet: Plug-and-Play Structure-Preserving Neural Operator Blocks for Compositional PDE Learning

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](YOUR_ARXIV_LINK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official implementation of **LegONet**, a compositional framework for time-dependent PDEs that builds solvers from plug-and-play, structure-preserving operator blocks.

> **Jiahao Zhang, Yueqi Wang, Guang Lin**  
> Department of Mathematics, Purdue University  
> Corresponding: guanglin@purdue.edu

This repository contains training and evaluation code for:
- **Case Study I** — 1D Viscous Burgers equation with Dirichlet boundary conditions
- **Case Study II** — 2D Incompressible Navier–Stokes in vorticity form (turbulent regime)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Case Study I: 1D Burgers (Dirichlet)](#case-study-i-1d-burgers-dirichlet)
- [Case Study II: 2D Navier–Stokes (Turbulent)](#case-study-ii-2d-navier-stokes-turbulent)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Reproducing Baselines](#reproducing-baselines)
- [Citation](#citation)

---

## Overview

LegONet replaces monolithic neural PDE solvers with a library of reusable, structure-preserving operator blocks. Its design rests on two separations:

1. **Boundary handling** is separated from mechanism learning. All blocks operate on a shared coefficient state `a(t)` that lives in a boundary-adapted spectral trial space, so boundary conditions are satisfied by construction throughout rollout.

2. **Mechanism learning** is separated from time integration. Blocks are pretrained offline by instantaneous operator matching and then assembled via symmetric Strang splitting — no trajectory fitting, no retraining when the PDE changes.

Each evolution block follows the structured template:

```
F^θ_i(a) = -G_i ∇_a E^{a,θ}_i(a)   [E-block, dissipative]
           + J_i ∇_a H^{a,θ}_i(a)   [H-block, conservative]
           + R^a_i(a)                [R-block, residual]
```

where `G_i ⪰ 0` (symmetric positive semidefinite) and `J_i` is skew-symmetric, so discrete structure is guaranteed at the block level and preserved under composition.

---

## Repository Structure

```
LegONet/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
│
├── legonet/                         # Core library
│   ├── __init__.py
│   ├── baseplates/
│   │   ├── __init__.py
│   │   ├── shen_legendre.py         # 1D Dirichlet baseplate (Shen–Legendre basis)
│   │   └── fourier_2d.py            # 2D periodic baseplate (Fourier/FFT)
│   │
│   ├── blocks/
│   │   ├── __init__.py
│   │   ├── block_base.py            # Abstract base class for all blocks
│   │   ├── e_block.py               # Generic dissipative E-block (-G ∇E)
│   │   ├── h_block.py               # Generic conservative H-block (J ∇H)
│   │   ├── generators.py            # MLP scalar generators, density networks, quadratic forms
│   │   └── auxiliary.py             # Non-evolution maps (e.g. Poisson inversion)
│   │
│   ├── splitting/
│   │   ├── __init__.py
│   │   └── strang.py                # Symmetric Strang macro-step, within-block integrators
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── operator_matching.py     # Instantaneous operator-matching objective (Eq. 11/12)
│   │   ├── samplers.py              # Spectral-decay Gaussian prior µ_b (Eq. S3/S4)
│   │   └── trainer.py               # Training loop, AdamW + StepLR scheduling
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py               # Weighted relative L2 error, energy diagnostics
│       └── visualization.py         # Snapshot plots, pointwise error maps, diagnostic curves
│
├── case1_burgers/                   # Case Study I: 1D Viscous Burgers
│   ├── configs/
│   │   └── burgers.yaml             # All hyperparameters for this experiment
│   ├── data/
│   │   └── generate_reference.py    # Generate Galerkin reference trajectories
│   ├── train_blocks.py              # Train diffusion (uxx) and transport (u ux) blocks
│   ├── run_rollout.py               # Assemble blocks and run closed-loop rollout
│   ├── run_baselines.py             # Train and evaluate FNO, DeepONet, PINN
│   └── plot_results.py              # Reproduce Figures 4a–4f from the paper
│
├── case2_ns/                        # Case Study II: 2D Navier–Stokes
│   ├── configs/
│   │   └── ns2d.yaml                # All hyperparameters for this experiment
│   ├── data/
│   │   └── generate_reference.py    # Generate vorticity reference trajectories
│   ├── train_blocks.py              # Train Laplacian, transport, Poisson-inversion blocks
│   ├── run_rollout.py               # Assemble blocks and run closed-loop rollout
│   ├── run_ablation.py              # LegONet-unconstrained ablation
│   ├── run_baselines.py             # Train and evaluate FNO, DeepONet
│   └── plot_results.py              # Reproduce Figures 5a–5b from the paper
│
├── checkpoints/                     # Pretrained block weights
│   ├── case1/
│   │   ├── diffusion_block.pt
│   │   └── transport_block.pt
│   └── case2/
│       ├── laplacian_block.pt
│       ├── transport_x_block.pt
│       ├── transport_y_block.pt
│       └── poisson_block.pt
│
└── notebooks/
    ├── case1_demo.ipynb             # Interactive walkthrough of Case I
    └── case2_demo.ipynb             # Interactive walkthrough of Case II
```

---

## Installation

We recommend creating a dedicated conda environment.

```bash
git clone https://github.com/Yooki-YueqiWang/LegONet.git
cd LegONet
conda env create -f environment.yml
conda activate legonet
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

**Core dependencies:** `torch >= 2.0`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`

GPU is strongly recommended for Case II (2D Navier–Stokes). All experiments were run on a single NVIDIA A100 80 GB GPU.

---

## Quick Start

To train all blocks and run a full rollout for Case I in one command:

```bash
cd case1_burgers
python train_blocks.py --config configs/burgers.yaml
python run_rollout.py  --config configs/burgers.yaml --checkpoint ../checkpoints/case1/
python plot_results.py --config configs/burgers.yaml
```

For Case II:

```bash
cd case2_ns
python generate_reference.py --config configs/ns2d.yaml   # ~30 min on A100
python train_blocks.py       --config configs/ns2d.yaml
python run_rollout.py        --config configs/ns2d.yaml --checkpoint ../checkpoints/case2/
python plot_results.py       --config configs/ns2d.yaml
```

To skip training and use pretrained checkpoints directly, add `--skip-training` to `run_rollout.py`.

---

## Case Study I: 1D Burgers (Dirichlet)

**Equation:**

```
u_t + u u_x = ν u_xx,   x ∈ (-1, 1),   u(±1, t) = 0
```

with `ν = 0.03`, `Δt = 1e-5`, final time `T = 1`.

### Baseplate: 1D Dirichlet (Shen–Legendre)

The homogeneous component of the solution is expanded in Shen's Legendre basis:

```
φ_k(x) = L_{k-1}(x) - L_{k+1}(x),   k = 1, …, K
```

which satisfies `φ_k(±1) = 0` by construction. Grid evaluation uses `Q = 256` Gauss–Legendre quadrature nodes. The retained coefficient dimension is `K = 96`. The projection `P_b` is the discrete L² projection via the mass matrix `M_{ij} = ⟨φ_i, φ_j⟩`. The structure operators are `G = M⁻¹` and `J = M⁻¹S` where `S_{ij} = ⟨∂_x φ_i, φ_j⟩`.

### Block 1: Diffusion block (`u → u_xx`, E-block)

The scalar energy generator `E^{a,θ}_{uxx}: ℝ^K → ℝ` is parameterized by an MLP with 4 hidden layers, width 128, and GELU activations. The learned vector field is:

```
F^θ_{uxx}(a) = -G ∇_a E^{a,θ}_{uxx}(a)
```

Gradients are computed by automatic differentiation through the MLP. This construction guarantees monotone energy dissipation along any trajectory of the isolated sub-dynamics.

**Training:** AdamW (`lr=1e-3`), StepLR (`step_size=50`, `gamma=0.3`), 200 epochs, batch size 128, 20 000 coefficient samples from the spectral-decay Gaussian prior (`amp=1`, `α=0.5`).

### Block 2: Transport block (`u → u u_x`, H-block)

A pointwise density network `h^θ: ℝ → ℝ` (depth 4, width 128, GELU) is used to construct the Hamiltonian generator via the integral form:

```
H^{a,θ}_{uux}(a) = ∫_Ω h^θ(u_0(x)) dx,   u_0(x) = Σ_k a_k φ_k(x)
```

The learned vector field is:

```
F^θ_{uux}(a) = J ∇_a H^{a,θ}_{uux}(a)
```

This construction preserves the Hamiltonian `H^{a,θ}_{uux}` exactly along any trajectory of the isolated sub-dynamics.

**Training:** AdamW (`lr=1e-4`, `weight_decay=1e-4`), 100 epochs, batch size 128.

### Strang Composition

The assembled Burgers solver advances one macro step `Δt` via:

```
a_{n+1} = S^θ_{uux, Δt/2} ∘ S^θ_{ν·uxx, Δt} ∘ S^θ_{uux, Δt/2} (a_n)
```

The diffusion substep uses a Crank–Nicolson update in coefficient space. The transport substep uses second-order Heun integration on the coefficient state.

### Expected Results

| Method | Mean Rel. L² Error | Max Rel. L² Error |
|---|---|---|
| LegONet | ~1e-14 | ~1e-13 |
| FNO | ~1e-2 | ~1e-1 |
| DeepONet | ~1e-2 | ~1e-1 |
| PINN | ~1e-2 | ~1e-1 |

The diffusion E-block produces monotone per-step energy decay (`-ΔE ≥ 0`). The transport H-block preserves its Hamiltonian to numerical precision across both half-steps. See paper Figure 4 for full diagnostics.

---

## Case Study II: 2D Navier–Stokes (Turbulent)

**Equation (vorticity form):**

```
ω_t + u · ∇ω = ν Δω + f(x,y),   ∇ · u = 0,   (x,y) ∈ [0, 2π)²
```

with Kolmogorov forcing `f(x,y) = 0.1(sin(x+y) + cos(x+y))`, viscosity `ν = 1e-4`, `Δt = 1e-3`, final time `T = 50`. The divergence-free velocity is recovered via a streamfunction: `u = (ψ_y, -ψ_x)` with `-Δψ = ω`.

### Baseplate: 2D Periodic Fourier

Fields are represented by a band-limited Fourier expansion on a `64 × 64` uniform grid. Retained modes satisfy `|j|, |ℓ| ≤ K_cut = 21`. Coefficients are stored in a real Hermitian-packed vector `a ∈ ℝ^K`. Forward and inverse FFT provide `Φ_b` and `P_b`. Nonlinear products are computed pseudospectrally with 2/3 dealiasing. The structure operators are `G = I` and `J_x, J_y` are the fixed spectral representations of `∂_x, ∂_y` on retained modes.

### Block 1: Laplacian block (`u → Δu`, E-block)

Because the Laplacian is mode-decoupled in the Fourier basis, the energy generator is restricted to a structured diagonal quadratic form:

```
E^{a,θ}_Δ(a) = ½ aᵀ diag(c^θ) a,   c^θ ∈ ℝ^K
```

The learned vector field is:

```
F^θ_Δ(a) = -G ∇_a E^{a,θ}_Δ(a) = -diag(c^θ) a
```

**Training:** AdamW (`lr=1e-3`), StepLR (`step_size=40`, `gamma=0.3`), 80 epochs, batch size 128.

### Blocks 2 & 3: Transport blocks (`u → u u_x` and `u → u u_y`, H-blocks)

Two independent density networks `ρ^θ_x` and `ρ^θ_y` (depth 4, width 128, GELU) are trained for the two directional advection terms. The vector fields are:

```
F^θ_{uux}(a) = J_x ∇_a H^{a,θx}_{uux}(a)
F^θ_{uuy}(a) = J_y ∇_a H^{a,θy}_{uuy}(a)
```

**Training:** AdamW (`lr=5e-4`, `weight_decay=1e-6`), StepLR (`step_size=150`, `gamma=0.5`), 500 epochs, batch size 16.

### Block 4: Poisson inversion block (`Δψ = -ω`, auxiliary)

Given vorticity coefficients, this block predicts the streamfunction coefficients on the same retained Fourier modes. A diagonal quadratic generator with softplus-constrained weights is fitted to exact Fourier inversion targets.

**Training:** AdamW, StepLR (`step_size=80`, `gamma=0.3`), 200 epochs, batch size 128.

### Strang Composition

The assembled Navier–Stokes solver follows the recipe in Figure 2c of the paper:

```
a_{n+1} = S^θ_{-u·∇ω, Δt/2} ∘ S^θ_{νΔω, Δt/2} ∘ S_{f(x,t), Δt}
          ∘ S^θ_{νΔω, Δt/2} ∘ S^θ_{-u·∇ω, Δt/2} (a_n)
```

At each transport half-step, the velocity `u` is recovered from the current vorticity coefficients via the Poisson inversion block, then the advection term `u · ∇ω` is evaluated pseudospectrally.

### Ablation: LegONet-unconstrained

This ablation replaces the structured Laplacian E-block with a parameter-matched but unconstrained MLP vector field (no generator form, same parameter count). It is controlled by the flag `--unconstrained` in `run_ablation.py`. Results show that removing structure causes immediate rollout degradation and large energy drift even at identical model capacity, isolating the contribution of mechanism-level structure.

### Expected Results

| Method | Mean Rel. L² Error (T=50) | Max Rel. L² Error (T=50) |
|---|---|---|
| LegONet | < 4% | < 4% |
| LegONet-unconstrained | >> 4% | >> 4% |
| FNO | accumulates drift | — |
| DeepONet | accumulates drift | — |

See paper Figure 5 for vorticity snapshots, pointwise error maps, and long-horizon diagnostic curves.

---

## Pretrained Checkpoints

Pretrained block weights are provided in `checkpoints/` and can be loaded directly for rollout without retraining:

```python
import torch
from legonet.blocks import EBlock
from legonet.baseplates import ShenLegendre

baseplate = ShenLegendre(Q=256, K=96)
diffusion_block = EBlock(K=96, hidden_dim=128, n_layers=4)
diffusion_block.load_state_dict(torch.load("checkpoints/case1/diffusion_block.pt"))
```

A helper function `load_assembly()` in `legonet/splitting/strang.py` loads and assembles a complete solver from a checkpoint directory:

```python
from legonet.splitting.strang import load_assembly

solver = load_assembly("checkpoints/case1/", config="case1_burgers/configs/burgers.yaml")
a_trajectory = solver.rollout(a0, T=1.0)
```

---

## Reproducing Baselines

All baseline models (FNO, DeepONet, PINN) are implemented in `case1_burgers/run_baselines.py` and `case2_ns/run_baselines.py`. Parameter counts are matched to LegONet as described in the paper.

**Case I baseline configurations:**

| Model | Modes/Rank | Width | Depth | Rollout window |
|---|---|---|---|---|
| FNO | m=16 | 40 | 2 | K_roll=25 |
| DeepONet | r=145 | 128 | 3 | K_roll=25 |
| PINN | — | 149 | 6 (tanh) | space–time surrogate |

**Case II baseline configurations:**

| Model | Modes/Rank | Width | Depth | Rollout window |
|---|---|---|---|---|
| FNO | m=12 | 64 | 4 | K_roll=10 |
| DeepONet | r=128 | 256 | 3 | K_roll=10 |

All supervised baselines use the rollout-aware training objective over short unrolled windows (Eq. S18 in the paper) to mitigate teacher-forcing mismatch. The training-sample budget is matched to LegONet across experiments.

To run all baselines for Case I:

```bash
cd case1_burgers
python run_baselines.py --config configs/burgers.yaml --models fno deeponet pinn
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025legonet,
  title={LegONet: Plug-and-Play Structure-Preserving Neural Operator Blocks for Compositional PDE Learning},
  author={Zhang, Jiahao and Wang, Yueqi and Lin, Guang},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Acknowledgments

This work was supported by the National Science Foundation (DMS-2533878, DMS-2053746, DMS-2134209, ECCS-2328241, CBET-2347401, OAC-2311848), the U.S. Department of Energy Office of Science Advanced Scientific Computing Research program (DE-SC0023161), the SciDAC LEADS Institute, and DOE Fusion Energy Science (DE-SC0024583).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
