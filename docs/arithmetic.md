# Reversible arithmetic and access model

## What the circuit computes

For sampled open-boundary tensor cores `cores[j]` of shape
`(r_left, grid_size, r_right)`, the scalar potential is

\[
V(x)=A_1(x_1)A_2(x_2)\cdots A_d(x_d).
\]

The first and last bond dimensions are one. The compiler rounds each core
entry independently to the nearest integer after multiplication by `2**f`
(ties to even). Write these integer cores as \(C_j\). The implemented potential
is exactly

\[
\widetilde V(x)=2^{-df}C_1(x_1)\cdots C_d(x_d).
\]

The multiplication and addition circuits perform **exact integer contraction**.
There is no intermediate fixed-point rounding, division oracle, floating-point
gate, or ignored overflow. The approach deliberately trades growing register
width for a simple, auditable precision model. It is not claimed to be the
optimal reversible fixed-point contraction implementation.

The oracle is `compute -> signed phase -> compute.inverse()`. Its action on
clean ancillas is

\[
|x\rangle|0\rangle\longmapsto
e^{i\beta\widetilde V(x)}|x\rangle|0\rangle.
\]

The circuit is a well-defined unitary on the whole Hilbert space, but the
stated phase-oracle contract assumes clean ancillas. Tests verify all allocated
ancillas, including coefficient, multiplier, carry and QROM-selection wires.

## Address conventions

Address wires are the first `n_address` wires. The final coordinate occupies
the least significant address bits; thus integer address `x` corresponds to
the C-order flattened grid. Within every arithmetic register, the first wire
is the least significant bit. Signed integers use two's complement.

Grids may have different power-of-two sizes. A one-point coordinate is allowed
and has no address bit. Time is a classical compilation parameter; this code
does not provide coherent access to a time register or claim to charge for it.

## Coefficient access

Coefficient tables are compiled into an explicit coherent QROM circuit. They
are not assumed to be free quantum data. A table with `N=2**k` rows and `w`
output bits uses a shared prefix tree with up to `k-1` clean scratch bits.
At a branching node with active prefix `p` and address bit `a`, the circuit
forms `p AND NOT a`, visits the left subtree, XORs `p` into the flag to obtain
`p AND a`, visits the right subtree, then clears the flag. Only X, CX and CCX
gates occur. With no pruning, a whole lookup uses at most `2*(N-2)` CCX gates
for `k>=1`, independent of output width; each selected set output bit costs
a CNOT. Constant columns and identical subtrees are simplified at compilation.

This follows the established coherent unary-selection/QROM access model used
in Babbush et al. The particular implementation retains exact CCX gates and
is **not** the measurement-assisted temporary-AND construction. In particular,
the measurement-assisted 4T lookup/unlookup numbers must not be assigned to
this circuit without actually changing the implementation and its fault-
tolerant assumptions.

For the first core, the lookup directly initializes the first contraction
vector. For each subsequent core, all variable matrix coefficients are loaded
through one shared QROM tree into a reusable coefficient bank. The contraction
is computed, and the bank is unloaded. Reversing the full computation after
the phase incurs the corresponding access costs again. Thus later-core
coefficient data is loaded/unloaded twice over the complete oracle. Constant
coefficients use classical constant arithmetic and do not occupy the bank.

Classical preprocessing scales with the number of sampled core entries.
`materialize_values=False` avoids constructing an exponentially large dense
potential merely to compile the circuit. A dense validation grid, when
requested, is separately timed in `dense_validation_preprocess_seconds`.

## Adders and multipliers

`modular_add` implements the MAJ/UMA ripple-carry construction of Cuccaro et al.,
using the omission of the high carry for addition modulo a power of two.
The included elementary variant uses one clean carry bit and, for width
`w>=2`, `2*(w-1)` CCX and `4*w-2` CX gates; for `w=1` it is a CNOT.
It is a direct, auditable variant of the known construction, not a new adder.
Subtraction reverses its elementary gate sequence. More aggressive published
adder schedules can change constants; they are not silently assumed here.

`multiply_accumulate` uses the standard reversible shift-and-add construction.
For each non-sign bit of the coefficient, it forms a controlled shifted copy
of the multiplicand in clean workspace, adds it, and uncomputes the copy. For
the sign bit it subtracts. Multiplicand sign extension uses its sign bit as
the control for higher positions and does not require an uncharged register.
Leading zero shifts reduce the adder width. Known classical coefficients use
their signed magnitude, omit zero bits, and replace partial-product CCXs with
CNOTs. The quantum coefficient register and multiplicand are restored.

If the coefficient width is \(b\) and output width is \(w\), one multiplication
uses \(O(bw)\) elementary reversible gates with ripple-carry depth of the same
order. A TT contraction includes each actual bond-matrix multiplication,
\(\sum_j r_{j-1}r_j\) terms, as well as coefficient access, phase application,
and full reversal. **The number of stored TT parameters is not the quantum
gate complexity.**

### Exact gate formulas for this implementation

For one variable-coefficient multiply-accumulate, let \(b\) be the signed
coefficient width, \(w\) the output width, and \(m=\min(b,w)\). Shift \(j\)
acts on \(w-j\) output bits. Its partial-product compute and uncompute cost
\(2(w-j)\) CCX gates; its ripple adder costs \(2(w-j-1)\) CCX gates.
Consequently the exact multiply-accumulate counts are

\[
N_{\mathrm{CCX}}=2m(2w-m),\qquad
N_{\mathrm{CX}}=2m(2w-m)-\mathbf 1_{m=w}.
\]

These counts include partial-product uncomputation. They do not include
coefficient loading or reversal of the whole TT computation after the phase.
For a classical constant coefficient, sum only over its nonzero signed-
magnitude bit positions \(j<w\). Each such position has
\(2(w-j-1)\) CCX and \(6(w-j)-2-\mathbf 1_{w-j=1}\) CX gates.

Let \(Q_j\) denote the **measured** gate count of the simplified QROM load for
core \(j\), for either gate type separately. Let \(M_j\) be the sum of
multiply-accumulate counts over that core's matrix entries. The full oracle
has

\[
N=2Q_1+4\sum_{j=2}^d Q_j+2\sum_{j=2}^d M_j
\]

for CCX and CX. The first-core vector is loaded once in the forward
computation; later cores use a load/unload pair. The outer factor two pays
for reversal after phase application. There are at most \(w_d\) additional
single-qubit phase rotations. The tables' set bits, constant columns and
repeated subtrees affect actual CNOT counts, so substituting just a parameter
count for \(Q_j\) would be incorrect.

With \(q_j\) variable coefficient-bank bits at site \(j\), the exact allocated
qubit count for \(d>1\) is

\[
\sum_j\log_2 n_j
+\sum_j r_jw_j
+\max_{j\ge2}q_j
+\max_{j\ge2}w_j
+1+\max_j(\log_2n_j-1,0).
\]

The terms are address, retained contraction vectors, reusable coefficient
bank, partial-product workspace, carry, and shared QROM prefix workspace.
For \(d=1\), omit coefficient bank, partial-product workspace, and carry.

For uniformly bounded core entries, bond rank \(r\), grid size \(n\), and
core precision \(f\), exact contraction widths grow as
\(w_j=O(j(f+\log r))\). Thus variable-coefficient arithmetic can scale as
\(O(d^2r^2(f+\log r)^2)\) gates in this deliberately nontruncating design,
in addition to coordinate-local QROM. The full dense table instead has
\(n^d\) entries. This difference creates a possible asymptotic crossover,
but does not establish a finite-instance saving or quantum computational
advantage. The benchmark compares actual gate lists and explicit synthesis
models to find finite-instance crossovers.

## Overflow and finite precision

The compiler starts with the scalar bound one. At each contraction it bounds
the next component by summing previous component bounds times maximum absolute
integer core entries over that coordinate grid. Each component register is
wide enough for both positive and negative values up to the largest such
bound. Bounds and register widths are recorded in oracle metadata.

Arithmetic is modular inside these registers. A temporary partial sum can
wrap, but the final integer residue equals the exact contraction; the absolute
bound guarantees that its signed interpretation is unique and correct. It is
therefore unnecessary to claim that every temporary partial sum avoids
wraparound.

For two's-complement bits \(z_0,\ldots,z_{w-1}\), signed phase application uses
\(P(\beta 2^{j-df})\) on each non-sign bit and
\(P(-\beta 2^{w-1-df})\) on the sign bit. This directly produces the desired
phase, including its sign and absolute global-phase convention. Angles are
reduced modulo \(2\pi\) before storage. The measured circuits retain arbitrary
P rotations; fault-tolerant rotation synthesis must be estimated or implemented
separately and assigned an error budget.

## Approximation bound

Let \(\widetilde A_j=2^{-f}C_j\). Apply the existing telescoping product identity
and induced matrix infinity norm to obtain

\[
\|V-\widetilde V\|_\infty
\leq
\sum_{j=1}^d
\left(\prod_{i<j}\|\widetilde A_i\|_\infty\right)
\|A_j-\widetilde A_j\|_\infty
\left(\prod_{i>j}\|A_i\|_\infty\right).
\]

Here each core norm is maximized over its supplied coordinate grid and is
the induced row-sum norm of the bond matrix. `quantization_bound` evaluates
this analytic upper bound in float64. It is not an interval-arithmetic
certificate. Dense finite-grid comparisons provide a usually much tighter
measured error. The standard phase Lipschitz bound is

\[
\|U_V-U_{\widetilde V}\|
\leq |\beta|\,\|V-\widetilde V\|_\infty.
\]

For a finite grid, the actual diagonal-unitary error is exactly the maximum
of \(|e^{i\beta V(x)}-e^{i\beta\widetilde V(x)}|\) over all addresses. The
comparator must include arithmetic approximation **and** any separate
rotation-synthesis budget.

## Validation strategy and scope

The tests exercise all input pairs for small adders, signed multiply-
accumulates including nonzero output registers, all QROM addresses and data
words on small tables, inverses, constant and repeated table branches,
four-dimensional contraction, and integer values beyond 64-bit precision.
Small adders and one-dimensional phase circuits are cross-checked against
Qiskit's dense statevector/operator routines.

Large arithmetic circuits contain only basis permutations and diagonal phases.
An independent integer/complex interpreter follows every elementary gate on
every address with clean ancillas, verifying the complete output bit string
and phase. Because no such gate branches a basis state, this also determines
the exact linear action on all superpositions supported on the clean-ancilla
subspace. It is a sparse exact gate simulation, not a dense simulation of the
entire ancilla Hilbert space and not evidence of a computational quantum
advantage.

## Primary references

1. S. A. Cuccaro, T. G. Draper, S. A. Kutin and D. P. Moulton,
   *A new quantum ripple-carry addition circuit*, 2004,
   [arXiv:quant-ph/0410184](https://arxiv.org/abs/quant-ph/0410184), Sections 2
   and 4.1.
2. V. Vedral, A. Barenco and A. Ekert, *Quantum Networks for Elementary
   Arithmetic Operations*, 1995,
   [arXiv:quant-ph/9511018](https://arxiv.org/abs/quant-ph/9511018).
3. R. Babbush et al., *Encoding Electronic Spectra in Quantum Circuits with
   Linear T Complexity*, Physical Review X 8, 041015 (2018),
   [arXiv:1805.03662](https://arxiv.org/abs/1805.03662),
   [DOI:10.1103/PhysRevX.8.041015](https://doi.org/10.1103/PhysRevX.8.041015).
