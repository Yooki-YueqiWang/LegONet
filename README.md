# LegONet: Plug-and-Play Structure-Preserving Neural Operator Blocks for Compositional PDE Learning

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2603.07882)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official implementation for Case Studies I and II of the paper:

> **LegONet: Plug-and-Play Structure-Preserving Neural Operator Blocks for Compositional PDE Learning**  
> Jiahao Zhang, Yueqi Wang, Guang Lin  
> Department of Mathematics, Purdue University  
> Correspondence: guanglin@purdue.edu

---

## Overview

LegONet builds PDE solvers from a library of reusable, structure-preserving operator blocks. Each block is pretrained independently on a shared coefficient-space representation (the *baseplate*) via instantaneous operator matching — no trajectory fitting required. Blocks are then assembled into a solver through symmetric Strang splitting.

Two block types are supported:

- **E-block** (dissipative): `F^θ(a) = -G ∇_a E^{a,θ}(a)`, with `G ⪰ 0`. Energy is non-increasing along any trajectory of the isolated sub-dynamics.
- **H-block** (conservative): `F^θ(a) = J ∇_a H^{a,θ}(a)`, with `J` skew-symmetric. The Hamiltonian `H` is exactly conserved.

Structure is preserved under composition: the Strang macro-step inherits the per-block dissipation/conservation guarantees (Theorem 1 of the paper).

---

## Repository Structure

```
LegONet/
├── README.md
├── requirements.txt
├── environment.yml           
│
├── case1_burgers/               # Case Study I: 1D Viscous Burgers
│   ├── train_uxx_block.py       # Train diffusion E-block (u_xx)
│   ├── train_uux_block.py       # Train transport H-block (u u_x)
│   ├── solve_burgers1D.py  # Strang rollout and plots
│   └── checkpoints/
│       ├── uxx_block/
│       │   └── checkpoint.pt    # Pretrained E-block weights + K_denoised
│       └── uux_block/
│           └── checkpoint.pt    # Pretrained H-block weights
│
└── case2_ns/                    # Case Study II: 2D Navier–Stokes
    ├── laplace2d_block.py     # Train Laplacian E-block (Δu, diagonal)
    ├── stream_block_train.py    # Train Poisson inversion block
    ├── ns2d_vorticity_with_blocks.py  # Strang rollout and plots
    └── checkpoints/
        ├── laplace2d_diag/
        │       ├── model_state.pt
        │       └── config.json
        └── poisson2d_diag/
                ├── model_state.pt
                └── config.json
```

---

## Installation

```bash
git clone https://github.com/Yooki-YueqiWang/LegONet.git
cd LegONet
```

**With conda (recommended):**
```bash
conda env create -f environment.yml
conda activate legonet
```

**With pip:**
```bash
pip install -r requirements.txt
```

Core dependencies: `torch >= 2.0`, `numpy >= 1.24`, `matplotlib >= 3.7`.  
GPU is recommended for Case II. All experiments were run on a single NVIDIA A100.

---

## Case Study I — 1D Viscous Burgers (Dirichlet)

**Equation:**
```
u_t + u u_x = ν u_xx,   x ∈ (-1, 1),   u(±1, t) = 0
```
ν = 0.03, Δt = 1e-5, T = 1.

**Baseplate:** 1D Dirichlet (Shen–Legendre), K = 96 modes, Q = 256 Gauss–Legendre nodes.  
**Structure operators:** G = M⁻¹, J = −M⁻¹S where M_ij = ⟨φ_i, φ_j⟩ and S_ij = ⟨φ'_i, φ_j⟩.

### Step 1 — Train the diffusion E-block (u_xx)

```bash
cd case1_burgers
python train_uxx_block.py
```

The energy generator `E^{a,θ}: R^K → R` is an MLP (4 layers, width 256, GELU).  
Training minimizes `||∇_a E + F^ref_uxx(a)||^2` over 20 000 coefficient samples drawn from the spectral-decay Gaussian prior (α = 0.5).  
Optimizer: AdamW (lr=1e-3), StepLR (step=50, γ=0.3), 200 epochs.

After training, a symmetrized least-squares estimate `K_denoised` of the effective stiffness matrix is extracted and saved in the checkpoint. This is used as the learned diffusion operator during rollout.

### Step 2 — Train the transport H-block (u u_x)

```bash
python train_uux_block.py
```

The Hamiltonian density `h^θ: R → R` (4 layers, width 64, GELU) is trained so that `dh^θ/du ≈ 0.5 u`, making `H ≈ ∫ 0.5 u^2 dx` and reproducing the transport mechanism.  
Optimizer: AdamW (lr=1e-4, wd=1e-4), 100 epochs.

### Step 3 — Run Strang rollout

```bash
python solve_burgers1D.py
```

Assembles the trained blocks via symmetric Strang splitting:
```
a^{n+1} = S_N(Δt/2) ∘ S_D(Δt) ∘ S_N(Δt/2) (a^n)
```
S_N uses Heun (explicit trapezoid); S_D uses Crank–Nicolson.  
Both a reference trajectory (exact spectral operators) and a learned trajectory are produced from the same initial condition.

**Outputs** saved to `plots/burgers/`:
- Overlay of reference vs learned at selected steps
- Pointwise residual `u_L - u_ref`
- Normalized pointwise error profile `e(x) = (u_L - u_ref) / ||u_ref||`

### Using pretrained checkpoints

Place the provided checkpoint files at:
```
case1_burgers/models/uxx_block/checkpoint.pt
case1_burgers/models/uux_block/checkpoint.pt
```
Then run `solve_burgers1D.py` directly — training is not required.

---

## Case Study II — 2D Forced Navier–Stokes (Turbulent)

**Equation (vorticity form):**
```
ω_t + u · ∇ω = ν Δω + f(x,y),   ∇ · u = 0,   (x,y) ∈ [0,1)^2
```
Kolmogorov forcing: `f = 0.1(sin(2π(x+y)) + cos(2π(x+y)))`,  
ν = 1e-4, Δt = 1e-3, T = 50.

**Baseplate:** 2D periodic Fourier, N = 64, K_cut = 21 (retained modes per direction), 2/3 dealiasing.  
**Structure operators:** G = I, J_x/J_y are spectral representations of ∂_x, ∂_y.

### Step 1 — Train the Laplacian E-block

```bash
cd case2_ns
python laplace2d_block.py --mode train --model diag --N 64 --Kmax 21 \
    --scale -1 --epochs 80 --use_double
```

The energy generator is restricted to a diagonal quadratic form:
```
E^{a,θ}(a) = (1/2) aᵀ diag(c^θ) a
```
because the Laplacian is mode-decoupled in the Fourier basis. The parameter `c^θ ∈ R^K` is learned directly. Setting `--scale -1` enables auto-scaling by `max(k^2)`.

The script saves the best checkpoint (lowest test relative error) automatically and supports early stopping. Trained output: `runs_laplace2d/laplace2d_diag/`.

**Alternative modes:**
- `--model fixed` — save the exact analytical Laplacian block (no training, for reference)
- `--model mlp` — use a full MLP energy generator (larger capacity, less interpretable)

### Step 2 — Train the Poisson inversion H-block

The Poisson block solves `−Δψ = ω` to recover the velocity field at each advection substep.
It is parameterized as a diagonal H-block with learnable weights `c^θ_k → 1/|k|²`.

```bash
python stream_block_train.py --mode train --model diag --N 64 --Kmax 21 \
    --epochs 200 --use_double
```
Saves the best checkpoint (lowest test relative L² error) to `runs_poisson2d/poisson2d_diag/`.

### Step 3 — Run Strang rollout

```bash
python ns2d_vorticity_with_blocks3_strang.py \
    --N 64 --Kmax 21 --nu 1e-4 --dt 1e-3 --n_steps 50000 \
    --laplace_model diag \
    --poisson_model diag \
    --use_double
```

The Strang macro-step is:
```
ω^{n+1} = L(Δt/2) ∘ N(Δt) ∘ L(Δt/2) (ω^n)
```
where L is the exact-in-Fourier linear diffusion+forcing step and N is the pure advection step (RK2/Heun with pseudo-spectral evaluation).

The script runs both a reference trajectory (exact operators) and a learned trajectory (learned Laplacian + learned Poisson), logs relative errors every 2000 steps, and saves comparison plots.

**Outputs** saved to `runs_ns2d_vort/<run_name>/`:
- `omega_step00000_compare.png` — initial vorticity field
- `omega_steps_compare_grid.png` — 3×4 grid: learned / reference / normalized error at 4 later steps
- `error_curve_omega.png` — relative L^2 error over time
- `history.npz` — full error array
- `config.json` — complete run configuration

**Command-line options** (key flags for `ns2d_vorticity_with_blocks3_strang.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | 64 | Grid size |
| `--Kmax` | 21 | Maximum retained Fourier mode |
| `--nu` | 1e-4 | Viscosity |
| `--dt` | 1e-3 | Time step |
| `--n_steps` | 50000 | Total steps (T = n_steps * dt) |
| `--laplace_model` | diag | `diag`, `mlp`, or `fixed` |
| `--laplace_model_dir` | (auto) | Path to Laplace block dir; auto-selects latest if empty |
| `--poisson_model` | diag | `diag` or `fixed` |
| `--poisson_model_dir` | (auto) | Path to Poisson block dir; auto-selects latest if empty |
| `--use_double` | False | Use float64 (recommended for long rollouts) |

### Using pretrained checkpoints

Place the provided checkpoint directories at:
```
case2_ns/runs_laplace2d/laplace2d_diag/model_state.pt
case2_ns/runs_laplace2d/laplace2d_diag/config.json
case2_ns/runs_poisson2d/poisson2d_diag/model_state.pt
case2_ns/runs_poisson2d/poisson2d_diag/config.json
```
Then run the NS rollout directly — both blocks will be auto-selected by `find_latest_model_dir()`.

---

## Checkpoint Contents

**Case I:**

| File | Contents |
|------|----------|
| `uxx_block/checkpoint.pt` | `state_dict`, `K_learned`, `K_denoised`, architecture config, training history |
| `uux_block/checkpoint.pt` | `state_dict`, architecture config, training history |

**Case II:**

| File | Contents |
|------|----------|
| `laplace2d_diag/model_state.pt` | `c` parameter (diagonal k² approximation) |
| `laplace2d_diag/config.json` | `block`, `dim`, `scale`, `N`, `Kmax`, `best_epoch`, etc. |
| `poisson2d_diag/model_state.pt` | `raw` parameter (softplus-constrained diagonal weights c^θ) |
| `poisson2d_diag/config.json` | `block`, `dim`, `scale`, `N`, `Kmax`, `inv_k2_vec` (true values stored for verification), etc. |

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

Supported by NSF (DMS-2533878, DMS-2053746, DMS-2134209, ECCS-2328241, CBET-2347401, OAC-2311848), DOE Office of Science ASCR (DE-SC0023161), the SciDAC LEADS Institute, and DOE Fusion Energy Science (DE-SC0024583).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
