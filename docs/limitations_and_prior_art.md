# Scope and limits of the baseline study

## Demonstrated scope

The baseline experiment is a two-dimensional, nonseparable Gaussian mixture
benchmark. It trains one seeded smooth scalar potential by sampled conditional
flow matching, validates its ODE and Fourier wavefunction flows, compiles
signed reversible TT and lookup phase circuits, and executes a complete small
learned preparation and its inverse in amplitude-estimation circuits.
The baseline outputs are `results/smooth`, `results/oracle-study` and
`results/inference`. See [benchmark provenance](archive/README.md) for the other studies.

## Prior art

The flow-to-Hamiltonian mapping, Fourier discretization and commutator formula
are from Layden et al. Kodama et al. already propose reversible TT arithmetic
and demonstrate classical tensor-network wavefunction simulation. Che and
af Klinteberg already demonstrate QROM phase circuits and QNN alternatives.
CFM, TT-SVD, ripple-carry arithmetic, Walsh diagonal synthesis, SelectSwap,
amplitude amplification and maximum-likelihood amplitude estimation are
established techniques. See `literature_verified.md` for checked primary
locations and `THIRD_PARTY_NOTICES.md` for attribution.

This release is an implementation, validation and resource audit, with no
claim of a new asymptotic algorithm, first phase oracle, globally optimal
compiler, or computational quantum advantage.

## Training and spatial discretization

The smooth architecture deliberately uses finite periodic Fourier modes and a
time MLP. With maximum mode 5, the sampled 2D potential matrix has rank at most
11 (or the grid size, if smaller), because each coordinate lies in an
11-dimensional Fourier span. Thus compression observations concern an
intentionally structured architecture. They do not establish that generic
neural potentials are compressible. Retained SVD rank is explicitly distinguished
from minimal numerical rank.

Training uses Euclidean Gaussian interpolation; continuum validation uses the
learned periodic velocity and a wrapped Gaussian source. Boundary tails are
quantified rather than assumed absent. The small box-tail probability does
not prove a uniform bound on the error of periodizing an arbitrary learned
flow. Direct same-field characteristics and convergence studies provide the
numerical validation used here.

The independent sample sliced-W1 metric and normalized grid TV answer different
questions. For the recorded seed, a small sample sliced-W1 does **not** imply
negligible model error: the fine-grid learned/target discrepancy remains about
0.116 TV. The fine-grid wave/learned discrepancy is about 0.000279 TV. These
measure model error and simulation error separately. Normalized node densities
are collocation diagnostics, not exact cell probabilities or certified
continuum-TV estimates; the coarse 8x8 density quadrature is particularly poor.

The previous MPS-V experiment used a piecewise-multilinear potential and did not
establish continuum convergence. Its good discrete-state fidelity remains valid
within that finite-grid scope; it is not used to certify the new smooth model.

## Oracle compilation and resource comparison

Small dense potential tables are explicitly constructed, decomposed and
quantized classically. That preprocessing is timed. Two-dimensional TT-SVD is
matrix SVD. No polynomial-in-dimension data-loading claim follows from this
workflow. The independent coherent evaluator itself performs coordinate-wise
coefficient lookup and signed multiplication, retaining all required history.

The TT precision search takes the first passing fractional precision for each
examined retained rank, then the least expensive candidate under the declared
T-cost model. It is not a proof of globally minimal gate cost; gauge optimization,
other arithmetic, relative-phase constructions and measurement-assisted lookup
could change constants. Every TT address in the new sweep is gate-validated.
Larger baseline portfolios disclose sampled gate-validation scope separately
from exhaustive table-error calculations.

Native arbitrary rotations are simulated exactly to floating-point precision.
The matched operator tolerance assigns half its budget to measured
approximation and half to prospective rotation synthesis. Clifford+T rotation
sequences are not generated or executed. Reported T counts use a stated
Ross-Selinger-inspired planning model with a sensitivity slope, not a rigorous
finite-instance synthesis upper bound. The selected QROM/Walsh/SelectSwap
portfolio is not every possible phase-synthesis algorithm. Logical connectivity
is unrestricted; routing, code distance, physical qubits, factories and hardware
wall time are outside scope.

## Inference

MLAE uses an actual preparation A, its actual inverse A-dagger, an event phase
oracle and a zero-state reflection. It requires no controlled A in this
construction, so no controlled-call cost is omitted. It estimates the event
probability of the **compiled prepared state**. Product-formula, spatial and
model biases are retained separately and do not vanish with more shots.

Native gates are executed once along the required statevector trajectories;
independent repeated measurements are then simulated from those resulting Born
probabilities. That is noiseless classical circuit simulation, not hardware
execution. Empirical RMSE and quantiles across seeded repetitions are not
certified finite-shot confidence intervals for the MLAE estimator. Likelihood
alias failures and preparation bias are recorded.

Deterministic dense and two-site MPS contractions are inexpensive for the tested
small problem. Dense-to-MPS SVD setup is included; the contraction result does
not imply scalable MPS state construction in higher dimensions. Event structure,
potential rank and state rank are distinct. A lower query count than Monte Carlo
alone cannot establish lower physical cost after complete state preparation.
Break-even formulas are conditional on unknown fault-tolerant operation times
and do not establish a measured crossover.

## Reproducibility limits

Training and results are seeded, but a single training seed is not a
multi-seed model-quality study. A same-seed parameter replay checks reproducibility,
not generalization over optimization randomness. Tests, measured commands and
checkpoint hashes are included. Runtime timings are CPU measurements from this
environment; library versions, BLAS and concurrent work can affect them.

No GQSP/QSVT or reversible-chain oracle is inserted without a concrete cost
comparison. A learned potential does not automatically supply the access
assumptions of a Szegedy-walk QSAMPLE algorithm.
