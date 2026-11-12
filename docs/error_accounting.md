# Error accounting and circuit semantics

This document distinguishes established tools and direct finite-dimensional
consequences from measurements. It gives no new asymptotic algorithm or
uniform guarantee over untested model distributions.

## The same learned velocity on both paths

The trained real scalar potential is $V_\theta(t,x)$ and the velocity is
$v_\theta=\nabla_xV_\theta$. On a periodic domain, with a smooth potential,

$$
H_t=i[K,V_t],\qquad K=-\tfrac12\Delta,
\qquad \partial_t\psi_t=[K,V_t]\psi_t
=-v_t\cdot\nabla\psi_t-\tfrac12(\nabla\cdot v_t)\psi_t.
$$

Thus $p_t=|\psi_t|^2$ satisfies the continuity equation. This is the established
wavefunction-flow construction of Layden et al., not a contribution of this
benchmark. Fourier collocation gives a finite Hermitian $K_N$. Since $V_N$ is
real diagonal, $i[K_N,V_N]$ is Hermitian exactly in real arithmetic. Floating
point Hermiticity and norm errors are measured, not silently renormalized away.

The smooth experiment uses a periodic potential and a large finite box.
Euclidean Gaussian tails and periodic boundary treatment are explicitly
recorded. Spatial convergence is measured against reverse characteristics of
the **same learned velocity**, not against the intended target alone. Density
samples at grid nodes normalized to a discrete distribution are not cell
integrals. Agreement of those grid distributions is a collocation diagnostic.

## What each error means

1. **Model error:** learned classical pushforward versus the known target GMM.
   Held-out CFM loss includes irreducible conditional noise. Its oracle
   diagnostic uses the analytically conditioned marginal velocity. Distribution
   error is evaluated independently with ODE samples and density diagnostics.
2. **Domain and spatial error:** Fourier-grid evolution versus the learned
   continuous transport; periodic tail mass, density quadrature normalization,
   and spatial refinement are reported separately.
3. **Time error:** midpoint time freezing plus the existing eight-factor
   commutator product formula. The generic local error is \(O(h^2)\) and global error
   is \(O(h)\); midpoint evaluation does not change that global order.
4. **Oracle error:** rank truncation, core quantization and finite signed
   arithmetic. Exact native phase rotations are used during circuit simulation.
5. **Prospective synthesis error:** a separately allocated budget for replacing
   arbitrary rotations by Clifford+T sequences. Those sequences are not executed.
6. **Inference error:** preparation/observable bias plus finite-shot estimation
   error. An accurately estimated discrete probability need not equal the
   continuum target probability.

High fidelity to a discrete reference only constrains errors against that
reference. Sample Wasserstein distance and state norm error measure different
quantities.

## Tensor truncation and uniform fixed-point contraction

This section covers the uniform precision baseline. The [mixed precision
analysis](v03_precision.md) includes the intermediate rounding terms.

At each fixed classical time, a small dense potential table is decomposed by
TT-SVD. In two dimensions this is matrix SVD, with cores of shape $(1,N,r)$
and $(r,N,1)$. The experiment measures both parameter count and sup-norm error;
low rank is not assumed. Dense table construction and SVD are charged classical
preprocessing. They are not a high-dimensional oracle construction.

Each core entry is rounded to an integer multiple of $2^{-b}$, with ties to
nearest even. The circuit contracts these integer cores exactly, retaining
intermediate contraction vectors and applying full compute--phase--uncompute.
For $d$ cores the output has $db$ fractional bits. Register bounds are computed
from sums of absolute products. Internal modular wraparound is harmless when
the final signed interpretation is within those bounds; no destructive
intermediate rounding is used. Consequently there is no omitted per-product
rounding error in this circuit. The wider accumulators and retained history
are included in qubit and gate counts.

For a grid address $x$, write exact and quantized core slices as $A_j(x_j)$
and $B_j(x_j)$. Define $a_j=\max_{x_j}\|A_j(x_j)\|_\infty$,
$b_j=\max_{x_j}\|B_j(x_j)\|_\infty$, and
$e_j=\max_{x_j}\|A_j(x_j)-B_j(x_j)\|_\infty$, using the induced infinity norm.
The finite product identity gives the direct bound

$$
\left|\prod_{j=1}^{d} A_j(x_j)-\prod_{j=1}^{d}B_j(x_j)\right|
\le \sum_{j=1}^{d}\left(\prod_{k<j} b_k\right)e_j
                         \left(\prod_{k>j} a_k\right).
$$

This is exactly the telescoping perturbation bound evaluated by the code.
Adding measured TT truncation error gives a bound for the original grid table.
Norms are evaluated in float64, so these are analytic inequalities checked
numerically, not interval-arithmetic certificates.

## Phase and accumulated simulation error

For real tables $V$ and $\widetilde V$, the exact diagonal operator error on
clean work registers is

$$
\delta_V=\max_x|e^{i\beta V(x)}-e^{i\beta\widetilde V(x)}|
\le \min\{2,|\beta|\|V-\widetilde V\|_\infty\}.
$$

Global phases are retained in standalone oracle checks. For circuits with
workspace, the comparison is between isometries on $|x\rangle|0\rangle$,
not between their extensions to arbitrary dirty workspaces. Every clean
address is gate-simulated for the new TT instances, so linearity establishes
the claimed action on arbitrary address superpositions. A separate Qiskit
full-statevector test checks the gate-path interpreter on a bounded instance.

For unitary blocks, repeated insertion of $U_j-\widetilde U_j$ and the triangle
inequality give $\|U_m\cdots U_1-\widetilde U_m\cdots\widetilde U_1\|
\le\sum_j\|U_j-\widetilde U_j\|$. Each commutator step contains four potential
calls, so a uniform per-call error $\delta_V$ contributes at most $4r\delta_V$
across $r$ steps, in addition to the kinetic, preparation and formula errors.
In the standard balanced formula $\beta=O(r^{-1/2})$, a fixed potential
sup-error can therefore have a conservative accumulated bound growing like
$\sqrt r$. Increasing step count alone is not a complete error strategy.

For continuous discrete-Hamiltonian evolution, the corresponding direct
Duhamel bound is

$$
\|U(T)-\widetilde U(T)\|
\le\int_0^T\|H_t-\widetilde H_t\|\,dt
\le 2\|K_N\|\int_0^T\|V_t-\widetilde V_t\|_\infty\,dt.
$$

Thus finer grids can amplify a fixed potential error through $\|K_N\|$.
An empirically small potential L2 error alone is insufficient to certify
transport or derivative accuracy.

For normalized pure states and any event projector $\Pi$, the probability
bias obeys $|\langle\psi|\Pi|\psi\rangle-\langle\phi|\Pi|\phi\rangle|
\le\sqrt{1-|\langle\psi|\phi\rangle|^2}\le\|\psi-\phi\|_2$.
These bounds constrain preparation bias; they do not justify a finite-shot
confidence claim for an arbitrary estimator.

## Gate counts versus fault-tolerant projections

The saved circuits contain elementary X, H, P/Rz, CX and CCX operations, not
opaque diagonal matrices. The native gate counts, allocated wires and depth
are measured. The Clifford+T model assigns seven T gates and six CNOTs to CCX
and a stated precision-dependent planning cost to arbitrary rotations. That
rotation formula is not an executed synthesis or a rigorous finite-instance
upper bound. Physical qubits, code distance, routing, distillation throughput
and hardware time are not estimated. Full inference uses an uncontrolled
MLAE construction; no free controlled preparation is assumed.

## Sources

- Layden et al., [Wavefunction Flows](https://arxiv.org/html/2510.08462v1),
  Theorems 1--2 and Eq. (36).
- Kodama et al., [tensor-network wavefunction flows](https://arxiv.org/html/2608.21700v1),
  Supplement S1.8, Proposition 2 (proposed TT arithmetic route).
- Oseledets, [Tensor-Train Decomposition](https://doi.org/10.1137/090752286), 2011.
- Cuccaro et al., [A new quantum ripple-carry addition circuit](https://arxiv.org/abs/quant-ph/0410184).
- Suzuki et al., [Amplitude estimation without phase estimation](https://arxiv.org/pdf/1904.10246),
  Sections 2--3. Empirical finite-shot performance is kept separate from
  asymptotic Fisher-information statements.
