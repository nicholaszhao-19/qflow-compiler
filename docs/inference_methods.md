# Inference from the compiled learned flow

This extension estimates one bounded observable of a **fully compiled learned
wavefunction-flow preparation**. It implements amplitude amplification and
maximum-likelihood amplitude estimation (MLAE), executes their native gates in
a noiseless Qiskit statevector, and compares their statistical and complete
preparation costs with classical methods. No quantum hardware or synthesized
approximate Clifford+T rotation sequences are used.

Run from the repository root:

```bash
python -m pytest -q tests/test_inference.py
python scripts/run_inference.py
```

The trained checkpoint is `data/checkpoints/smooth_gmm.pt`. Its SHA-256, exact
configuration, versions, command, raw measurement counts, independent trial
estimates, and numerical results are saved under `results/inference/`. The
bundled `prepared_flow.qpy.gz` contains the original Qiskit QPY circuit. A fresh
run writes `prepared_flow.qpy`. The event and zero reflection are in OpenQASM 3.

## Recorded default-run findings

The following measured probabilities refer to the same cell observable, with
the quantum values interpreted on their specified grid:

| Quantity | Probability or error |
|---|---:|
| Target GMM cell probability | 0.1391909763 |
| Continuous **learned** flow cell probability | 0.1537007642 |
| Continuous-time 16-by-16 Hamiltonian reference | 0.1457935716 |
| Fully compiled 64-step prepared state | 0.1452353756 |
| Full-circuit versus block-propagation state error | 5.59e-13 |
| Full-circuit fidelity to same-grid reference | 0.999927959 |
| Final MLAE RMSE to the prepared probability | 0.00221771 |
| Equal-query MC RMSE to the prepared probability | 0.00691582 |
| Final MLAE RMSE to the continuous learned-flow cell probability | 0.00874537 |

The continuous cell integral changes by `1.62e-7` from 24 to 32 quadrature
nodes per axis and by `8.07e-11` when doubling reverse-RK4 steps. The circuit's
total bias to that integral is `-0.00846539`, mostly spatial quadrature/
discretization bias at this resolution; its product-formula contribution
relative to the same-grid reference is `-0.000558196`. More measurement shots
cannot remove either bias. The target-GMM mismatch is separately a model
error and should not be disguised by cancellation with grid bias.

The preparation contains **186,312 native gates**, including **90,646 CNOTs**,
on eight data qubits. Paying for the complete default likelihood experiment
gives **23,034,088,640 estimated T gates** under the documented synthesis model.
This is not a measured Clifford+T sequence. The classical finite-grid reference
was constructed in about 0.094 seconds in the recorded CPU run; exact and
truncated tensor-event contraction are also available. There is no
end-to-end quantum speedup demonstrated here.

At absolute prepared-probability tolerances 0.001, 0.002, 0.005, and 0.01, the
final likelihood estimator succeeded in 83, 167, 249, and 256 of 256 trials.
For example, the observed success frequency at tolerance 0.005 is 0.9727, with
an individual 95% exact binomial interval `[0.9445, 0.9889]`. These are success-
frequency estimates for this noiseless shot experiment, not guarantees about
the original target distribution. The shorter `[0,1]` amplification schedule
had occasional likelihood aliases and worse empirical RMSE than equal-query
MC, so a favorable deepest-schedule result is not a universal finite-shot
advantage claim.

## 1. Precisely defined observable

The default circuit uses a 16-by-16 periodic Fourier grid of side length 12,
with coordinate nodes

\[
x_i=-6+\frac{12i}{16},\qquad i=0,\ldots,15.
\]

The marked event is

\[
E=\{(i,j): i\geq 10,\ j\geq10\},\qquad
f(i,j)=\mathbf 1_E(i,j)\in[0,1].
\]

Thus the selected grid nodes have both coordinates at least 1.5. This is a
bounded-observable workload; it is not advertised as a rare-event experiment.
The event is selected before inspecting trained-state probabilities.

For the separate classical continuous ODE control, samples are assigned to
their nearest periodic grid node. The matching continuum cell event is
`[1.125, 5.625)` in each coordinate, interpreted periodically. Simply testing
continuous samples against 1.5 would define a different observable. The
collocation Born masses approximate the masses of these cells only as the grid
resolves the density. The small-grid circuit still has spatial bias, which is
not part of amplitude-estimation shot noise.

The event phase oracle is either a compiled union of disjoint coordinate-prefix
boxes or a fully materialized Walsh mask. Both are checked against the entire
expected diagonal matrix, and the implementation with the smaller estimated T
count at a fixed comparison budget is selected. A one-coordinate event and an
irregular mask with the same number of marked addresses provide observable-
structure cost ablations. They do not change the primary inference target.

### A synthesis assumption caught by operator validation

In Qiskit 2.2.2, generic transpilation defaults to the assumption that all input
qubits are initially zero. That assumption is appropriate for a preparation
from zero but is not appropriate for a phase oracle acting on arbitrary data.
During this extension, the full-operator check caught a structured event
compiled with unused **address** bits borrowed as clean auxiliaries. The event
and zero reflection now explicitly use `qubits_initially_zero=False`. A
16-by-16 regression checks all 256 addresses, including the spectator bits.
This finding concerns an explicitly controlled transpiler assumption, not an
incorrect quantum synthesis algorithm.

## 2. The preparation is included in full

Let \(A|0\rangle=|\widetilde\psi\rangle\). The circuit contains:

1. The specified periodized source, compiled coordinate by coordinate with
   Qiskit's `StatePreparation` decomposition. It is checked against the exact
   supplied grid amplitudes. A truncated Gaussian is not silently substituted.
2. Every kinetic Fourier/QFT operation in the commutator formula.
3. Every native Walsh potential-phase operation, evaluated at the learned
   model's midpoint time values.
4. All inverse potential and kinetic operations.

The eight chronological factors per step are
\(K_+,V_+,K_-,V_-,K_-,V_-,K_+,V_+\), with
\(K_+=e^{i\alpha K}\), \(V_+=e^{i\beta V}\), and
\(\alpha\beta=\Delta t/2\). This is the established commutator construction in
Layden et al., Eq. (36). Its fixed-Hamiltonian local error is of order
\(\Delta t^2\), with first-order global convergence. It is not an ordinary
second-order Strang formula.

This driver chooses a time-global, sampled potential half-range and the kinetic
half-range to balance the coefficients. The sampled maximum is a heuristic for
choosing coefficients, not a certified supremum bound or error certificate.
The numerical refinement therefore uses **this driver's own preparation** at
32, 64, and 128 steps. It does not reuse temporal error figures obtained with a
different balance rule.

An independent FFT/DOP853 integration of the same learned finite-grid
Hamiltonian, with the same source, gives the reference state. A tighter
tolerance run checks reference stability. The primary circuit is executed
gate by gate and checked against independent block propagation. The wider
reversible tensor-train phase oracles are assessed separately by the main
oracle benchmark; this full-statevector inference experiment uses the compact
native Walsh route and does not claim to simulate hundreds of arithmetic
workspace qubits densely.

## 3. Amplification and likelihood estimation

Define

\[
\widetilde p=\langle\widetilde\psi|\Pi_E|\widetilde\psi\rangle,
\qquad \theta=\arcsin\sqrt{\widetilde p},
\]

\[
S_E=I-2\Pi_E,\qquad S_0=I-2|0\rangle\langle0|,
\qquad Q=-A S_0 A^\dagger S_E.
\]

The measured success probability after \(Q^k A\) is the established amplitude-
amplification identity

\[
q_k=\sin^2((2k+1)\theta).
\]

The implementation executes the actual native gates of \(A\), \(A^\dagger\),
the event oracle and zero reflection. It checks this identity numerically at
\(k=0,1,2,4,8\). The statevector trajectory is reused at intermediate depths
to avoid simulating the same classical prefix twice; the resource report
still pays for every independent physical shot.

For \(h_k\) marked outcomes in \(N_k\) shots, the log-likelihood is

\[
\ell(\theta)=\sum_k\left[
h_k\log q_k+(N_k-h_k)\log(1-q_k)\right].
\]

The parameter search covers \([0,\pi/2]\), brackets every local maximum detected
on a frequency-resolving grid, refines every bracket, and compares the candidates
including the endpoints. This follows the likelihood construction of Suzuki
et al., Section 3.1, Eqs. (4)--(6). It avoids choosing one favorable local mode.
A deterministic test doubles the search-grid density. This is numerical
validation, not a proof that finite-shot likelihood ambiguity is absent.

The default study uses 64 shots per selected depth and 256 independent
simulated shot experiments. Binomial counts are sampled from the probabilities
obtained by native-gate execution. Reported RMSE, error quantiles, outlier
fractions, and RMSE standard errors are **empirical statistics**. No finite-shot
confidence theorem for MLAE is asserted. In particular, the QPE-based amplitude-
estimation guarantee in Brassard et al. is not applied to this different
estimator. Controlled \(A\) operations are unnecessary for this variant and
their count is explicitly zero.

Success counts at absolute tolerances 0.001, 0.002, 0.005, and 0.01 have
Clopper--Pearson intervals across the 256 independent experiment repetitions.
Each is an individual interval for the success frequency of this particular
estimator, state and noiseless shot model. They are not simultaneous intervals,
a universal AE guarantee, or a guarantee on target/continuum accuracy.

## 4. Error accounting

There are three distinct comparison targets:

| Target | What the reported difference measures |
|---|---|
| The same compiled probability \(\widetilde p\) | Shot noise and estimator error |
| The continuous-time **finite-grid** Hamiltonian probability | Persistent product-formula/preparation bias plus shot noise |
| The classical continuous learned-flow cell probability or target GMM cell probability | Also includes spatial and learned-model differences |

Because every repetition uses a fixed compiled unitary and its exact inverse,
the amplification identity remains exact for \(\widetilde p\). If the prepared
state differs from a desired state, its target bias persists as shots increase.
It is not multiplied by query count merely because the same unitary is reused:

\[
|\widehat p-p|\leq |\widehat p-\widetilde p|+|\widetilde p-p|,
\qquad
|\widetilde p-p|\leq
\operatorname{TV}(|\widetilde\psi|^2,|\psi|^2).
\]

Prospective errors in **implementing** the prescribed rotations are different.
For that unexecuted stage, a conservative unitary telescoping budget is divided
over all block occurrences in the complete repeated experiment. The resulting
T counts use the repository's declared planning model; they are not measured
Clifford+T sequences or a hardware error guarantee.

## 5. Classical comparisons and complete resource counts

The experiment includes all of the following.

- **Same-distribution MC:** actual categorical samples from the computed
  prepared Born distribution, paying for CDF construction and separately
  reporting the cost of constructing that distribution. Independent binomial
  repetitions additionally compare sampling error at equal paid A/A† query
  budgets. Their cheap random-number generation time is not treated as the
  cost of generative-model sampling.
- **Actual learned-flow MC:** 4,096 independent Gaussian source samples pushed
  through the learned velocity by RK4, timed in a real batch. The exact
  Clopper--Pearson interval describes Bernoulli sampling uncertainty for this
  numerical pushforward. A paired 64/128-step run checks trajectory refinement
  and event classifications. This interval is not a model-error or
  discretization guarantee.
- **Deterministic dense evaluation:** construct the state and sum the marked
  Born probabilities. Both construction and the subsequent event sum are
  reported.
- **Continuous learned-flow cell integral:** integrate the learned density over
  the same fixed cell rectangle by backward characteristics and tensor-product
  Gauss--Legendre quadrature. Orders 16, 24, and 32, followed by a 128/256-step
  reverse-RK4 check, isolate continuum-model probability from quantum-grid
  bias. These are observed convergence diagnostics, not certified enclosures.
- **Two-site amplitude MPS:** a dense SVD converts the small \(16\times16\)
  amplitude array into \(L R\). With product event factors \(E_x,E_y\),
  evaluate
  \(\operatorname{Tr}[(L^\dagger E_xL)(R E_yR^\dagger)]\).
  The SVD setup is counted. Ranks 1, 2, 4, 8, and 16 reveal truncation effects;
  the full-rank contraction is exact up to roundoff. For normalized SVD
  truncation retaining squared singular-value weight \(w\), the event error is
  bounded by \(\sqrt{1-w}\), the pure-state trace distance. This is not a claim
  that a high-dimensional state can be obtained by a cheap dense SVD.

One shot at depth \(k\) pays for \(k+1\) forward preparations, \(k\) inverse
preparations, \(k\) event oracles and \(k\) zero reflections. For the full
default schedule there are 320 shots, **1,280 forward A calls, 960 inverse A
calls, 960 event calls and 960 zero reflections**. The total is 2,240 paid
preparation/inverse queries, with 17 in the deepest coherent circuit. Native
gate counts, logical width, CNOT equivalents, and scheduled depths include
every call. The prospective synthesis allocation also reflects these
repetitions.

The crossover report first matches the **observed** MLAE RMSE with the
Bernoulli-variance formula \(N_{\rm MC}\simeq
\widetilde p(1-\widetilde p)/\mathrm{RMSE}^2\). This is an accuracy-sensitivity
calculation, not a guaranteed-success comparison. A necessary time condition
is then displayed symbolically:

\[
N_A t_A+N_E t_E+N_0 t_0+t_{\rm readout}+t_{\rm MLE}
<N_{\rm MC}t_{\rm classical\ draw}.
\]

A conversion of the measured classical ODE throughput into seconds per
effective serial T is recorded only as a conditional sensitivity proxy. There
is no physical qubit count, routing, factory, parallelization or hardware clock
model. Equal observables and **equal total approximation error** would be
required for a real end-to-end speed comparison. The available deterministic
dense and tensor contractions already remove an inference advantage on this
small materialized-grid workload.

## References

- D. Layden et al., *Wavefunction Flows*, arXiv:2510.08462,
  Eq. (36) and Theorem 2: <https://arxiv.org/html/2510.08462v1>.
- Y. Suzuki et al., *Amplitude estimation without phase estimation*, Quantum
  Information Processing **19**, 75 (2020), Sections 2 and 3.1--3.2,
  Eqs. (1), (3)--(6), (10)--(11): <https://arxiv.org/abs/1904.10246>,
  <https://doi.org/10.1007/s11128-019-2565-2>.
- G. Brassard, P. Høyer, M. Mosca and A. Tapp, *Quantum Amplitude Amplification
  and Estimation*, arXiv:quant-ph/0005055:
  <https://arxiv.org/abs/quant-ph/0005055>.
- Kodama et al., *Scalable quantum simulation of continuous-time generative
  models via tensor networks*, arXiv:2608.21700, Supplement S1.1 explicitly
  distinguishes its idealized inference protocols from a compiled AE circuit:
  <https://arxiv.org/html/2608.21700v1>.
