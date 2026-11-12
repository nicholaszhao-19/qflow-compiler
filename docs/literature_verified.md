# FlowOracleBench literature verification

Verified on 2026-11-05 by reading the primary arXiv HTML/PDF versions and the public repository. This is a bounded related-work check, not a priority claim or exhaustive systematic review.

## Source and implementation map

| Source | Verified location | Established contribution and relevant scope |
|---|---|---|
| Layden et al., *Wavefunction Flows: Efficient Quantum Simulation of Continuous Flow Models*, arXiv:2510.08462v1 | Sections III–V; Theorems 1–2; Eqs. (24)–(29), (36)–(41) | Continuity-to-Schrodinger mapping; Fourier collocation and eight-factor commutator formula; approximation analysis conditional on regularity and potential-oracle access. Potential work-register precision is explicitly left outside the reported system-qubit count after Eq. (34). |
| Kodama et al., *Scalable quantum simulation of continuous-time generative models via tensor networks*, arXiv:2608.21700v1 | Sections 2.1–2.5, 4; Supplement S1.8 Proposition 2; S1.9 | Dense and MPS classical simulations; learned scalar MPS potentials; an arithmetic-oracle construction-level cost estimate that explicitly assumes negligible rounding; idealized rare-event amplification. The work states that it does not compile amplitude estimation or complete preparation/amplification to gates. |
| Che and af Klinteberg, *An Efficient Quantum Circuit for Flow Model Execution Using Quantum Neural Networks*, arXiv:2610.11537v1 | Sections 3.1–3.2, 4.2–4.6 | QROM compute–phase–uncompute and learned QNN encoders; PennyLane circuit simulation; resource and precision experiments. Therefore a gate-level QROM wavefunction-flow demonstration alone is existing work. Its published resource comparison is with its particular multi-controlled-X lookup, not a universal lower bound on lookup. |
| Sygaldry `tnwf-paper` | https://github.com/sygaldry-tech/tnwf-paper | MIT-licensed reproduction code; pin `11358d3dca52193f93f037046229797296a2063e`. Dense Fourier evolution, analytic GMM potentials, JAM training, TT/MPS methods, sample metrics, and idealized amplification costs are implemented. |

Primary URLs:

- https://arxiv.org/html/2510.08462v1
- https://arxiv.org/html/2608.21700v1
- https://arxiv.org/html/2610.11537v1
- https://github.com/sygaldry-tech/tnwf-paper/tree/11358d3dca52193f93f037046229797296a2063e

## Exact established tools to apply

**Conditional flow matching.** Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747v2, Theorem 2: with positive marginal path densities, the CFM and marginal FM objectives differ by a parameter-independent constant and have identical parameter gradients. The positive Gaussian-mixture path used here meets the positivity hypothesis. Restricting the learned vector field to a scalar gradient remains a modeling restriction; for the independent Gaussian-to-isotropic-mixture path below the exact marginal velocity is itself conservative.

https://arxiv.org/html/2210.02747v2

**Amplitude amplification.** Brassard et al., arXiv:quant-ph/0005055, gives the reflection construction and the amplified good probability. Theorem 12 supplies a finite-confidence guarantee for the original phase-estimation-based amplitude estimator; it must not be transferred to a different maximum-likelihood estimator without an argument.

https://arxiv.org/pdf/quant-ph/0005055

**Maximum-likelihood amplitude estimation.** Suzuki et al., *Amplitude estimation without phase estimation*, Quantum Information Processing 19, 75 (2020), arXiv:1904.10246v2: Sec. 2 Eqs. (1)–(3) define the reflection iterate, and Sec. 3.1 Eqs. (4)–(6) define the product likelihood and MLE. Sec. 3.2 Eq. (11) counts preparation and inverse calls as `sum_k N_k*(2*m_k+1)`. This variant needs no controlled preparation or phase-estimation register. Fisher-information comparisons are asymptotic; periodic likelihood aliases, finite-shot failure, and noisy-circuit bias require separate analysis. Repeated-shot empirical RMSE/quantiles are appropriate evidence, not a certified confidence interval.

https://arxiv.org/pdf/1904.10246

## Independent implementation checks and direct consequences

1. The eight-factor commutator formula is generically first order globally, even when the time-dependent potential is sampled at the midpoint. It contains four potential and four kinetic factors, applied right-to-left. For the literature's choice, alpha=(L/(pi*N))*sqrt(h/d) and beta=(pi*N/(2*L))*sqrt(d*h), hence alpha*beta=h/2. It is not the ordinary second-order Strang splitting for K+V.
2. The Fourier domain is a torus. Smoothness at periodic boundaries is part of the spatial-error theorem, not an optional interpretation. A neural potential clipped to a box, or multilinearly interpolated sampled cores, does not inherit this theorem's regularity automatically. Using a smooth periodic architecture, the same velocity for both ODE and quantum paths, boundary diagnostics, and measured convergence addresses the concrete issue.
3. A potential can be shifted by an x-independent constant at every time without changing i[K,V]. Standalone phase-oracle validation must nevertheless record whether it compares absolute phases or equivalence up to global phase. Controlled circuits require tracking such phases.
4. Core compression rank, wavefunction rank, and observable rank are distinct quantities. A low-rank potential does not guarantee an efficiently contractible evolved state. Conversely, when both the evolved state and event/observable are compact tensors, deterministic contraction is a mandatory classical comparator.
5. Validating compute–phase–uncompute needs both phase accuracy and complete workspace restoration. For an approximate learned encoder that produces a superposition of integer values, applying its inverse after phases does not automatically clean the workspace. Exact arithmetic avoids that additional approximation, and the implementation should test every clean address for small instances.
6. To compare two diagonal phase operators, use the exact finite-grid maximum of |exp(i beta V(x))-exp(i beta Vtilde(x))|; its standard upper bound is |beta| times the sup-norm potential error. Core-wise rounding requires a separate multilinear/telescoping bound or direct exhaustive validation. Count lookup, arithmetic, retained intermediate vectors, and full reversal; do not equate stored TT parameter count with gates or infer workspace reversibility from an irreversible classical contraction.
7. Native arbitrary phase gates are not synthesized Clifford+T gates. Rotation-cost planning formulas, routing assumptions, and physical error-correction costs must remain separate from executed circuit counts.

## Analytic diagnostic for the genuine trained flow

Let x0 be N(0,s0^2 I), and x1 a mixture of N(mu_k,s1^2 I), independently. The linear interpolant x_t=(1-t)x0+t*x1 has s_t^2=(1-t)^2*s0^2+t^2*s1^2 and posterior component weights w_k(x,t) proportional to pi_k*exp(-||x-t*mu_k||^2/(2*s_t^2)). Define c_t=t*s1^2-(1-t)*s0^2. Gaussian conditioning gives

v_t(x) = (c_t/s_t^2)*x + ((1-t)*s0^2/s_t^2)*sum_k w_k(x,t)*mu_k.

For t>0 it is the gradient of

V_t(x) = c_t*||x||^2/(2*s_t^2)
       + ((1-t)*s0^2/t)*log(sum_k pi_k*exp((t*mu_k.x-t^2*||mu_k||^2/2)/s_t^2)),

up to an x-independent gauge. At t=0 the exact velocity limit is E[x1]-x for unit-variance source. This is a trusted diagnostic and provides the known target marginal; it should not replace CFM training in the demonstrated learned-model experiment. A periodic finite-box implementation needs its own boundary/reference treatment rather than asserting this Euclidean formula is exactly periodic.

## Defensible project positioning

Implement and measure the previously proposed TT arithmetic oracle, characterize core quantization and exact reversible arithmetic, compare to strong implemented lookup/diagonal baselines, and include a genuine coherent inference workload. The scientific value is a reproducible end-to-end diagnosis of where the method succeeds and where total cost or approximation dominates. No claim of a new flow-to-Hamiltonian mapping, new TT-oracle idea, first QROM circuit, or demonstrated fault-tolerant quantum advantage is justified merely by this implementation.
