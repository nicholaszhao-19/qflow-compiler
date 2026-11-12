# Original MPS-V experiment provenance

The original MPS-V benchmarks ran on 29 October 2026 in a Linux x86-64 CPU
environment. See the [reproduction guide](reproduction.md) for the later smooth
flow and inference experiments.

## Inputs and training

The Sygaldry source snapshot is pinned to
`11358d3dca52193f93f037046229797296a2063e`. The snapshot and its license are
included so a checkout does not depend on the upstream default branch.
`vendor/UPSTREAM.md` records the source and attribution. The original study
used the upstream public trainer and saved weights, losses and diagnostics in
`data/checkpoints/`.

The public trainer omits the unpublished regularizer used for the paper's
archived checkpoints. The three provided models therefore reproduce the
public architecture and objective, not the paper's exact archived training.
Each checkpoint has a SHA-256 recorded in its metadata. The rank-two 2D
training replay was bitwise identical; its replay record is included. Other
platforms may differ in floating-point reduction or training behavior even
with the same seed.

## Executed scientific workflow

The code is installed and tested in a fresh, isolated Python 3.12.14 virtual
environment from `requirements.lock.txt`, without system site packages.
The recorded correctness run contains 103 passing tests, no failures, errors,
or skipped tests. `results/pytest.xml` preserves the original test record.

The delivered experiments contain:

- 42 TT arithmetic-precision sweep rows.
- 21 learned matched comparisons, including 174 compiled baseline candidates.
- 13 synthetic matched comparisons, including 114 baseline candidates.
- Six full learned-flow cases: two grids times three step counts, each with
  Walsh, SelectSwap, TT, and an explicitly uncosted exact-phase reference.
- Training-quality and numerical-reference diagnostics, including spatial
  grids with 4, 8, 16, and 32 points per coordinate.

Every compilation row records its actual gate counts, allocated logical
wires, preprocessing/compilation timing, approximation error, and validation
coverage. The higher-dimensional cases are actual compiled circuits, not
gate-count extrapolations. Their gate-level checking is sampled; classical
finite-grid quantization error is exhaustive. The primary 2D 4x4 and 8x8
oracles and all complete-flow potential blocks have exhaustive address checks.

All whole-flow state results execute exact native rotations. The
Clifford+T synthesis models are prospective estimates and are not included
as stochastic noise or approximate gate sequences in those state results.
The reference `exact_formula` entries have null resource fields and
`costed=false`; they are not zero-cost quantum circuits.

## Timing and numerical reproducibility

Training, compilation, gate-path simulation, and dense reference times are
single-run wall-clock observations. They are not controlled multi-run
performance benchmarks and have no confidence intervals. The final scaling
rerun used `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`
and completed in 191.797 seconds. Dense numerical routines use the thread
limits declared in their code. The virtualized final verification environment
reports an AMD EPYC 9V74 CPU; host load and allocation are not controlled.

`results/environment.json` records the scientific package versions and the
upstream commit. The package lock pins direct and transitive Python packages.
The final CSV and JSON tables were checked for identical case identities and
key error and resource values. `figures/manifest.json` records the source
table hashes for the figures from this study.

The expected reproducible outputs are the circuit structure, qubit/gate
counts, precision selections, and numerical results within floating-point
tolerance. Timings, file timestamps, plot metadata, and stochastic training
on different numerical platforms need not be byte-identical. No error bars
or statistical generalization are inferred from the single-seed model and
synthetic families.
