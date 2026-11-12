# Resource model and interpretation

## What is measured

Every reported oracle is an actual Qiskit circuit containing elementary native
gates. Measured quantities include allocated logical wires, workspace wires,
X/CX/CCX/P/Rz counts, circuit depth in that gate basis, classical compilation
time, and gate-level numerical checks. Preparation and Fourier-kinetic circuits
also include H gates after their established decompositions.

These native counts are **not physical fault-tolerant resource measurements**.
The code does not choose a code distance, compile a two-dimensional layout,
place magic-state factories, calculate a physical error rate, or estimate
wall-clock quantum runtime. The architecture used for logical estimates has
unrestricted logical connectivity and assumes sufficient magic-state supply.

The oracle contract requires clean ancillas. Register counts include
coefficient access, arithmetic scratch, retained contraction registers,
phase registers, and full uncomputation. A qubit is counted even if a particular
constant-data instance never excites it. No free QRAM, coefficient oracle,
state-preparation oracle, or opaque dense quantum unitary is assumed.

## Clifford+T decomposition assumptions

`resources.py` uses one consistent ancilla-free exact Toffoli template:

| Native gate | T count | CNOT count after decomposition | T-depth macro duration |
|---|---:|---:|---:|
| CX | 0 | 1 | 0 |
| CCX | 7 | 6 | 4 |
| RCCX or its inverse, if present | 4 | 3 | 4 |
| SWAP | 0 | 3 | 0 |
| T or T† | 1 | 0 | 1 |
| Clifford single-qubit gate | 0 | 0 | 0 |

The six-CNOT Toffoli construction has T-depth four after commuting gates.
The different ancilla-free T-depth-three construction of Amy et al. uses
seven CNOTs; its best entries are not mixed with the six-CNOT template here.
See Selinger, *Quantum circuits of T-depth one*, Eqs. (1)–(3), and Amy et al.,
*A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum
circuits*, Fig. 13. The actual benchmark arithmetic uses exact CCX gates;
measurement-assisted AND/uncompute substitutions are not silently assumed.

For an arbitrary single-qubit P or Rz rotation with allocated operator error
\(\delta\), the primary **planning model** is

\[
T_{\mathrm{rot}}(\delta)
=\left\lceil 3\log_2(1/\delta)+10\right\rceil.
\]

The leading logarithmic term is motivated by Ross–Selinger synthesis. The
constant ten is an explicit modelling choice. This formula is not asserted
to be a rigorous finite-precision upper bound for every angle. No actual
Clifford+T rotation sequences are synthesized or executed in this project.
The oracle and scaling tables include a sensitivity model that replaces the
factor three with four and re-evaluates the winning baseline. Complete-flow
resource totals use the primary factor-three model. These estimates should
be replaced by executed synthesis before making a hardware proposal or a
certified resource claim.

Stored binary64 angles are interpreted as exact dyadic reals. A value computed
as a multiple of stored `pi/4` is generally not an exact mathematical multiple
of \(\pi/4\). A rational enclosure of pi bounds the error of replacing
such a rotation by its Clifford+T representative. Even multiples cost zero T
gates and odd multiples cost one, but the replacement error is charged to the
synthesis allowance. If that allowance cannot cover the charge, the gate is
costed as an arbitrary rotation. No numerical closeness test makes an angle exact.
Global phases of uncontrolled single-qubit gates are tracked analytically
and do not incur a physical gate cost. The numerical oracle comparisons retain
an explicit global-phase convention and compare absolute diagonal unitaries.

## Error allocation

For a matched oracle experiment with target tolerance \(\epsilon\), at most
\(\epsilon/2\) is assigned to potential approximation and at most
\(\epsilon/2\) to prospective rotation synthesis. TT precision and phase-table
word width are selected using the actual finite-grid operator error. Sparse
Walsh synthesis uses its stated coefficient-L1 pruning bound. A recorded
finite-grid error is not a continuum approximation guarantee.

If a block contains \(m\) arbitrary rotations and its synthesis allowance is
\(\epsilon_s\), first reserve the total special-angle replacement error \(E\).
Each arbitrary rotation is then assigned an allowance no larger than
\((\epsilon_s-E)/m\), rounded downward. For repeated preparation, allocations
use the declared decimal budgets as exact rational numbers; the code verifies
that the summed replacement and arbitrary-rotation allowances fit the budget.
The standard unitary telescoping inequality makes the sum of these allowances
a valid block error allowance **if** rotations meeting those individual
operator errors are synthesized. The project executes ideal native rotations,
so measured fidelity excludes this prospective synthesis error.

For a complete flow with \(r\) time slices, each eight-factor commutator step
uses four potential and four kinetic blocks. Including one Gaussian
preparation block gives \(8r+1\) blocks. The total synthesis allowance is
divided equally among these blocks before allocating it to their rotations.
Potential approximation receives a separate allowance divided by \(4r\).
This is a simple reproducible allocation, not an optimization of total T count
over all possible nonuniform error allocations.

Measured error components are reported separately: time freezing, the
commutator product formula, compiled-potential approximation, and numerical
reference refinement. They are vector-norm differences between successive
reference levels; they need not add in quadrature. Unitary telescoping gives
an additive upper bound for accumulated block errors. Training and spatial
discretization diagnostics are separate from this finite-grid circuit error.

## Depth and complete-flow accounting

`native_depth` is the Qiskit depth of the recorded native gate list, treating
each native gate as one unit and respecting wire conflicts. It is not a
Clifford+T circuit depth, routed depth, or physical runtime.

The T-depth estimate is a conservative macro schedule for the selected
templates. A CCX reserves its three wires for four T layers. An arbitrary
single-qubit rotation reserves its wire for one layer per modelled T gate.
Clifford gates synchronize dependency times on their participating wires but
add no T layers. Disjoint macros can run in parallel. This preserves causal
dependencies but does not attempt commuting, cancellation, phase-polynomial
optimization, or cross-template scheduling.

Whole-flow resources are accumulated by `SerialResourceCounter`. It adds the
actual gate counts of the preparation, four kinetic blocks per time slice,
and four compiled-potential blocks per time slice. Inverse blocks have the
same gate costs. Workspace is reused, so the whole-flow width is the maximum
block width, not the sum of widths across time. Depths are summed across
explicit serial block boundaries. This is a valid declared schedule and may
overestimate a globally optimized circuit.

Each compiled potential block is executed gate by gate on every address of the
small end-to-end grids. Its verified clean-ancilla restriction is then applied
to the address state as a diagonal matrix. The kinetic circuit is independently
converted to its matrix and checked against the dense Fourier-kinetic
exponential. This is an exact blockwise simulation of the compiled circuit
up to floating-point arithmetic. It is not a dense statevector simulation of
all workspace wires and does not demonstrate a quantum computational advantage.
The `exact_formula` rows are classical numerical references. Their null
resource fields are placeholders meaning that no implementation was costed;
they do not describe a zero-cost quantum circuit.

## Baseline portfolio and crossover claims

The baseline portfolio contains sparse Walsh diagonal synthesis, coherent
phase-table QROM, and coherent clean-bank SelectSwap circuits at each compiled
power-of-two block size. Lookup, word selection, phase application and
uncomputation are charged. Their actual code implements exact CCX circuits;
it does not claim the lower constants of measurement-assisted or dirty-ancilla
QROM variants. Alternative lookup schemes, arithmetic implementations,
rotation synthesizers or global circuit optimizers could change the crossover.

The experiment chooses the smallest modelled T count among baseline candidates
that meet the same error target and requested qubit cap. The comparison with
the complete compiled portfolio means no **additional** cap beyond the
configuration's portfolio limit; it does not mean an unrestricted global
optimization over all quantum circuits.

`scripts/analyze_crossover.py` recomputes comparisons at 128, 256 and 512 logical
qubits, and at each TT circuit's exact width. A TT circuit exceeding a cap is
excluded from the win/loss counts for that cap. The script also tests Pareto
dominance in the two objectives of logical qubits and modelled T count and
records the interval of integer qubit caps over which a TT point strictly wins.
Using the TT width is a legitimate equal-budget comparison, but a narrow
interval must be reported as a limited space-versus-T tradeoff, not as an
unqualified advantage. SelectSwap's tested block sizes are powers of two;
other implementations might fill gaps between these resource points.

Higher-dimensional synthetic tests isolate tensor structure. They do not
establish performance on a learned high-dimensional generative model. Large
tables have exhaustive classical approximation-error checks and actual built
circuits, but gate-level circuit checking is sampled or omitted as recorded in
the CSV. The two-dimensional 4×4 and 8×8 oracle grids, and every end-to-end
grid, receive exhaustive address validation. A sampling flag must not be
upgraded to exhaustive validation in any report.

## Primary references

1. P. Selinger, *Quantum circuits of T-depth one* (2013),
   [arXiv:1210.0974](https://arxiv.org/abs/1210.0974), Eqs. (1)–(3).
2. M. Amy, D. Maslov, M. Mosca and M. Roetteler,
   *A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum
   circuits* (2013), [arXiv:1206.0758](https://arxiv.org/abs/1206.0758), Fig. 13.
3. N. J. Ross and P. Selinger, *Optimal ancilla-free Clifford+T approximation
   of z-rotations* (2014), [arXiv:1403.2975](https://arxiv.org/abs/1403.2975).
4. G. H. Low, V. Kliuchnikov and L. Schaeffer, *Trading T-gates for dirty
   qubits in state preparation and unitary synthesis* (2018),
   [arXiv:1812.00954](https://arxiv.org/abs/1812.00954).

## Sensitivity to the rotation cost model

`results/v03/audit_sensitivity.json` re-scores the fixed qualified candidate set with rotation slopes 3 and 4 and offsets 0, 10 and 20. Walsh or SelectSwap has the lowest projected T count in all 16 matched groups for each of the six models. This does not cover unseen circuits or arbitrary synthesis models. The archived baseline records use the resource conventions from their original studies.
