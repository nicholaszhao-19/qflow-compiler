> These notes cover the original MPS-V experiments. For the smooth flow and inference benchmarks, see [Reproduction](reproduction.md).

# Learned potentials and data provenance

The repository contains **three actual trained MPS-V potentials**. They were
trained with Sygaldry's released PyTorch implementation, not fitted to invented
phase tables or substituted with an analytic potential. The two-dimensional
rank-four checkpoint is the primary generative model; the rank-two checkpoint
provides the smaller oracle case. The four-dimensional rank-two checkpoint
adds a learned higher-dimensional test.

## Upstream source and reproduction scope

The public source is [sygaldry-tech/tnwf-paper](https://github.com/sygaldry-tech/tnwf-paper),
commit `11358d3dca52193f93f037046229797296a2063e`, retrieved on 29 October 2026.
It accompanies [Kodama et al., arXiv:2608.21700](https://arxiv.org/abs/2608.21700).
The source is preserved under `vendor/tnwf-paper`, with its MIT license. See
`vendor/UPSTREAM.md` and `THIRD_PARTY_NOTICES.md` for attribution.

`scripts/train_potential.py` calls `tnwf.mps_v.train.train`. A wrapper around
`jam_conservative_loss` records every scalar loss without modifying the loss,
optimizer, data sampler, or architecture. The model is upstream's
`MPSScalarPotentialTimeSite`: one spatial core per dimension and one time core,
with periodic multilinear spatial interpolation and linear time interpolation.

The public trainer explicitly omits the unpublished definition of the
`unitarity_weight=0.1` regularizer used for the paper's archived checkpoints.
These are therefore **fresh training runs with the public code**, not exact
reproductions of the paper's reported training or numerical tables. No
archived Sygaldry result is presented as our execution result.

## Dataset and objective

For spatial dimension `d`, the source is an isotropic Gaussian with standard
deviation `4/3`. The target is an equally weighted mixture of `2d` isotropic
Gaussians with standard deviation `0.5`, centered at `+3 e_j` and `-3 e_j` for
each coordinate direction. Thus the 2D target has four components and the 4D
target has eight. The public sampler returns balanced component counts, then
shuffles the pool.

The target pool contains 12,000 samples. The trainer holds out 20% (2,400)
and draws training minibatches from the remaining 9,600. Its validation batch
draws with replacement from the held-out pool and fixes source samples and
times to make checkpoint selection comparable across iterations. The saved
weights are restored from the best held-out loss, not necessarily the last
optimization step.

The released objective, called JAM in the repository, is conditional flow
matching restricted to a gradient velocity:

\[
\mathcal L=\mathbb E\left[\tfrac12\|\nabla V_t(x_t)\|^2
 -\nabla V_t(x_t)\cdot(x_1-x_0)\right],
\qquad x_t=(1-t)x_0+t x_1.
\]

The implementation distinguishes this objective from its separate variational
action-matching loss. The objective can be negative; a negative loss is not a
probability or a distribution-error estimate.

## Exact training configurations

All runs used seed 42, CPU PyTorch `2.8.0+cpu`, one intra-op and one inter-op
thread, deterministic algorithms, Adam, batch size 256, constant learning rate
0.003, global gradient clipping at 5, and validation every 100 steps. Early
stopping patience was 100 evaluations; no run stopped early. The domain has
side length `L=8`, the native spatial grid has 16 points per dimension, and the
time core has eight sites. The source width was set explicitly to `4/3`.

| Checkpoint | Dimension | TT bond cap | Iterations | Best held-out loss | Final 100-step training mean | Measured training time |
|---|---:|---:|---:|---:|---:|---:|
| `gmm_2d_rank2.pt` | 2 | 2 | 2,000 | -2.442702 | -2.432895 | 3.332 s |
| `gmm_2d_rank4.pt` | 2 | 4 | 3,000 | -2.554511 | -2.620592 | 4.956 s |
| `gmm_4d_rank2.pt` | 4 | 2 | 3,000 | -2.424381 | -2.545696 | 7.378 s |

The timings cover the trainer call, including checkpoint saving, on the
execution CPU. They exclude package installation, extraction validation, and
subsequent diagnostics. They are measurements of these small runs, not
hardware-independent performance claims. The final training mean is recorded
before the trainer restores its best validation weights, so it is not a
fresh loss evaluation of the saved checkpoint.

The corresponding configuration files are `configs/train_gmm_2d.json`,
`configs/train_gmm_2d_rank4.json`, and `configs/train_gmm_4d.json`.
For every checkpoint, the adjacent `.losses.csv` contains all training and
validation loss calls; `.metadata.json` contains the exact arguments,
validation trajectory, duration, versions, checksum, and extraction checks.

## Checkpoint identity and deterministic replay

The original checkpoint byte hashes are:

| Checkpoint | SHA-256 |
|---|---|
| `gmm_2d_rank2.pt` | `b2ed3f8d071dda76da392b1727a780dd60e36d4ddfb23ed0a3b80d277cc366f9` |
| `gmm_2d_rank4.pt` | `1790e269f42b2c07c947864e38b38963d5761bd70fddb455cfcee78750c03880` |
| `gmm_4d_rank2.pt` | `14febd05eb261d4518f9039fe0443a92ede2ed859b06b62677ef4e1558f4fe24` |

An independent replay of the rank-two 2D training configuration was executed
with the same CPU environment. Every stored model tensor, the final loss, and
the complete validation history were bitwise identical. The replay record is
`data/checkpoints/gmm_2d_rank2.replay.json`. This is tensor/value reproducibility;
PyTorch's serialized container bytes need not be identical across a new save.
No multi-seed robustness claim is made.

## Grid, tensor extraction, and floating-point reference

The circuit grid is centered and periodic:

\[
x_j=-L/2+jL/N,\qquad j=0,\ldots,N-1.
\]

The flattened tensor uses NumPy C order, so the last coordinate occupies the
least significant address register. The model is evaluated on this same grid
for all methods. A core-wise periodic interpolation resamples the native
16-point cores onto the circuit grid. On the 4-point and 8-point grids this is
simply selection of native grid nodes. It does not learn a new TT or discard
singular values.

The benchmark's reference function interprets the stored float32 model weights
as real numbers and performs spatial and time interpolation/contraction in
float64. This uses the same algebra as upstream's `get_mps_cores`, but avoids
its float32 time-site rounding jitter, which otherwise obstructed convergence
of the adaptive numerical evolution reference. The checkpoint bytes were not
changed. The implemented time dependence is piecewise linear, verified to
floating-point precision by a dedicated test.

All grid inputs at `N=4,8,16` and times `0,0.125,0.5,0.875,1` were independently
compared with the original PyTorch model's float32 forward function. Maximum
potential-value discrepancies were `2.3914e-6` (2D rank two), `2.5407e-6` (2D
rank four), and `1.2054e-5` (4D rank two). These are evaluation-precision
differences, not training errors. Oracle errors in the benchmark are measured
against the explicitly defined float64 function, not against a rounded
float32 forward call. Tests also cover a non-native grid size and the
upstream provider at time knots.

## Independent classical quality diagnostics

`scripts/evaluate_training.py` evaluates the learned gradient on fresh
Gaussian-bridge pairs and runs a separate classical particle ODE with RK4.
It uses seed 1729, 200 fixed random projection directions, and independent
target samples. The 2D runs use 2,048 particles and the 4D run uses 1,024.
The scalar metric below is the average projected **1-Wasserstein** distance,
matching upstream's `sliced_wasserstein`, not its square or a 2-Wasserstein
metric. These diagnostics do not execute a quantum circuit.

| Diagnostic | 2D rank two | 2D rank four | 4D rank two |
|---|---:|---:|---:|
| Source-to-target sliced W1 | 0.891360 | 0.891360 | 0.323915 |
| Learned ODE-to-target sliced W1, 256 steps | 0.199868 | 0.074102 | 0.181954 |
| Learned ODE-to-target sliced W1, 128 steps | 0.196951 | 0.074060 | 0.181949 |
| Independent target-to-target sample floor | 0.037432 | 0.037432 | 0.043545 |
| Velocity MSE against analytic conditional bridge | 0.560991 | 0.232622 | 0.735582 |
| Zero-velocity MSE against the same bridge | 5.197147 | 5.197147 | 5.723526 |
| Particle RMS difference, 128 versus 256 RK4 steps | 0.223220 | 0.013902 | 0.004192 |

The analytic diagnostic velocity follows ordinary Gaussian conditioning for
independent Gaussian endpoints. It is not used as a training teacher. Raw
source, target, ODE endpoint, and bridge-velocity arrays are preserved in the
adjacent `.quality.npz` files; exact metrics and seeds are in `.quality.json`.

The rank-two 2D model has a substantial particle-level RK4 refinement
discrepancy despite a fairly stable projected distribution metric. Its
continuum generation is **not established as converged**. Rank four improves
both the distribution metric and the refinement diagnostic, which motivates
using it for the primary flow experiment. The two separately trained ranks
also have different functions and training durations; their comparison does
not isolate rank alone. Controlled synthetic TT experiments serve that role.

## Scientific limits of these learned models

The target is an unbounded Euclidean mixture while the learned potential is
periodic on a finite box. In the quality runs, about 2.4% of 2D target samples
and 2.6% of 4D target samples lie outside the box. This domain mismatch is
reported, not silently removed. The native interpolation is piecewise
multilinear, so its spatial gradient has jumps at grid boundaries. RK4's usual
smooth-field order and the continuum wavefunction-flow paper's smoothness
conditions do not automatically apply to this checkpoint. No rigorous
continuum discretization certificate is claimed.

A correctly compiled finite-grid Hamiltonian can therefore have very high
fidelity to its numerical reference and still reproduce the target mixture
poorly, especially on `4x4` and `8x8` grids. The benchmark reports circuit
error separately from these training/domain/discretization limits. High
circuit fidelity is not evidence of accurate continuum generative modeling or
of computational quantum advantage.

## Reproduction commands

Run from the repository root with the pinned environment installed:

```bash
python scripts/train_potential.py --config configs/train_gmm_2d.json
python scripts/train_potential.py --config configs/train_gmm_2d_rank4.json
python scripts/train_potential.py --config configs/train_gmm_4d.json
python scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_2d_rank2.pt
python scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_2d_rank4.pt
python scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_4d_rank2.pt --samples 1024
python -m pytest tests/test_potentials.py
```

Training overwrites the corresponding checkpoint and its logs. The shipped
checkpoints already allow the circuit experiments to run without retraining.
