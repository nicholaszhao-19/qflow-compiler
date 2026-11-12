# Smooth learned Gaussian-mixture flow: methods and validation

This extension closes the continuum-validation gap in the earlier public MPS-V
benchmark without replacing its reversible arithmetic circuits. The new model
has smooth periodic spatial features, a genuine sampled conditional
flow-matching objective, an independently evaluated classical pushforward,
and explicit spatial and temporal refinement. It is a deliberately small
research benchmark, with one trained seed; no population generalization,
high-dimensional compression, or quantum speedup is inferred.

## Reproduce

From the project root in the recorded Python environment:

```bash
python scripts/run_smooth_flow.py --retrain
python scripts/run_smooth_flow.py
python scripts/run_smooth_flow.py --verify-training-replay
python -m pytest -q tests/test_smooth_flow.py
```

The first command trains and evaluates; the second reuses the saved model;
the third retrains in memory and compares every parameter/buffer and recorded
loss against the checkpoint without overwriting it. Configuration is in
`configs/smooth_flow.json`. Raw measurements, environment, losses, states,
samples, training metadata, and the replay outcome are in `results/smooth/`.
The checkpoint is `data/checkpoints/smooth_gmm.pt`.

## 1. Generative model and genuine training

The source is a standard normal in two dimensions. The Euclidean target is

\[
p_1(x)=\sum_{k=1}^{4}\pi_k\,\mathcal N(x;\mu_k,0.55^2 I),
\]

with means, in order,
\((-1.25,-1.25),(-1.25,1.25),(1.25,-1.25),(1.25,1.25)\),
and weights \((0.4,0.1,0.1,0.4)\). This is not a product distribution:
the two coordinates are correlated. Its symmetry under joint inversion
\(x\mapsto-x\), also shared by the source, is imposed in the potential model.

The scalar potential is

\[
V_\theta(t,x)=\sum_{k\in\mathcal K}
 a_{\theta,k}(t)\,\frac{\cos(k\cdot x)}{\lVert k\rVert_2}.
\]

Here \(k=(2\pi/12)(m_1,m_2)\), with one representative of each nonzero
\(\pm m\) pair in \(\{-5,\ldots,5\}^2\). There are 60 real modes, including
mixed spatial modes. A two-hidden-layer, width-48 tanh network maps
\((t,\sin(\pi t),\cos(\pi t))\) to the coefficients. It has 5,484 trainable
parameters. Both \(V_\theta\) and its velocity are periodic on
\([-6,6)^2\), and the mathematical model is smooth in space and time.
Spatial periodicity is built into the function, not imposed by clipping its
values at a box boundary.

Independent samples \(x_0\sim p_0\), \(x_1\sim p_1\), and
\(t\sim\mathrm{Uniform}[0,1]\) define

\[
x_t=(1-t)x_0+t x_1,\qquad u=x_1-x_0.
\]

The training objective is sampled conditional flow matching,

\[
\mathbb E\bigl[\lVert\nabla V_\theta(t,x_t)-u\rVert_2^2\bigr]
 +10^{-5}\mathbb E_t\left[\frac1{60}\sum_k a_{\theta,k}(t)^2\right].
\]

The coefficient penalty is disclosed as part of the objective. The implementation
uses the exact analytic Fourier derivatives and PyTorch automatic differentiation
with respect to the network parameters. Independent tests compare the analytic
velocity and divergence against first and second spatial derivatives computed
by PyTorch autograd. This avoids repeatedly constructing spatial Hessians in
the characteristic-density solver without approximating the gradient field.

Seed 42, 2,500 updates, batch 1,024, float64, and Adam with a cosine learning-rate
schedule are recorded. Every batch contains fresh stochastic source/target
pairs. An analytic conditional field is used for diagnostics only; it is never
a training label. Conditional flow matching itself is established work; see
[Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
and [Tong et al., Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482).

### Analytic held-out reference

For independent isotropic Gaussian source and common-covariance GMM target,
ordinary Gaussian conditioning gives

\[
s_t^2=(1-t)^2\sigma_0^2+t^2\sigma_1^2,\qquad
c_t=t\sigma_1^2-(1-t)\sigma_0^2,
\]

\[
w_k(t,x)=
\frac{\pi_k\exp[-\lVert x-t\mu_k\rVert^2/(2s_t^2)]}
{\sum_j\pi_j\exp[-\lVert x-t\mu_j\rVert^2/(2s_t^2)]},
\]

\[
\mathbb E[u\mid x_t=x]
=\frac{c_t}{s_t^2}x+
\frac{(1-t)\sigma_0^2}{s_t^2}\sum_k w_k(t,x)\mu_k.
\]

This is a direct Gaussian-conditioning calculation, not a novelty claim.
The implementation is tested independently using the per-component posterior
means of \(x_0\) and \(x_1\), and at \(t=0,1\). Squared-loss regression is
minimized by the conditional mean; therefore its held-out loss provides the
appropriate noisy-label reference. Neither a nonzero training loss nor its
raw decrease alone establishes distributional accuracy.

The analytic field belongs to the Euclidean interpolation problem. A periodic
finite Fourier model approximates it on the region carrying the probability
mass; no claim is made that it exactly realizes the Euclidean conditional
velocity on all of \(\mathbb R^2\).

## 2. Boundary conditions and source preparation

Continuum validation uses the learned periodic velocity and the **periodized**
Gaussian source on \([-6,6)^2\). The source density is the product of one-dimensional
Gaussian image sums. The code evaluates images \(j=-2,\ldots,2\) after wrapping
the evaluation point. The exact infinite image sum is a smooth periodic density.
The omitted total source mass is at most

\[
4\Phi(-30)<2\times10^{-197},
\]

where \(\Phi\) is the standard normal CDF. This is a numerical truncation far
below the reported errors. The Euclidean Gaussian probability outside the
box is approximately \(3.94635\times10^{-9}\). The target outside-box union
bound is about \(5.80\times10^{-18}\); the code retains this cancellation-free
bound rather than interpreting floating-point subtraction that rounds to zero
as exact zero probability.

The normalized finite-grid source is

\[
\psi_0[j]=\sqrt{p_0(x_j)\big/\sum_\ell p_0(x_\ell)}.
\]

The new reference and inference circuits use this same state. It is slightly
different from the repository's earlier normalized *unperiodized* Gaussian
grid restriction. The difference in vector norm is explicitly recorded:
\(5.59\times10^{-5}\) on the 8-by-8 grid and \(2.07\times10^{-5}\) on the
64-by-64 grid. The inference preparation includes this source's actual gates;
no free amplitude-loading assumption is used.

## 3. Independent classical and quantum-style evolution

For a smooth real potential with \(v=\nabla V\), the established continuity
Hamiltonian is

\[
H_t=i[K,V_t],\qquad K=-\tfrac12\Delta.
\]

Expanding its action gives
\([K,V]\psi=-\nabla V\cdot\nabla\psi-\tfrac12(\Delta V)\psi\).
Consequently \(\partial_t\psi=[K,V_t]\psi\) transports
\(|\psi|^2\) by the continuity equation
\(\partial_t p=-\nabla\cdot(vp)\), with the stated smooth periodic assumptions.
This construction and the Fourier/product-formula approach are from
[Layden et al., Wavefunction Flows, Theorems 1–2 and Eq. (36)](https://arxiv.org/abs/2510.08462).
The periodic model and smooth periodized source remove the gradient jumps and
uncontrolled periodic-edge mismatch in the earlier learned MPS-V path.

### Classical density reference

Particle generation integrates \(\dot x=v_\theta(t,x)\) using RK4.
A separate deterministic density calculation uses backward characteristics.
For each final evaluation point \(x\), set \(y(1)=x\), \(J(1)=0\), and integrate

\[
\dot y=v_\theta(t,y),\qquad \dot J=\nabla\cdot v_\theta(t,y)
\]

**backwards** to zero. Then

\[
p_1^{\rm learned}(x)=p_0(y(0))\exp(J(0)).
\]

The positive sign in this final expression follows because
\(J(0)=-\int_0^1\nabla\cdot v_\theta(t,y(t))\,dt\).
A unit test checks the entire computation against the exact periodic flow
\(\dot x=a\sin x\) and its analytic density. This reference does not use target
samples, evolve a wavefunction, or reuse a discretized Hamiltonian.

### Fourier-collocation wavefunction reference

The grid is \(x_j=-6+12j/N\) in each coordinate. The kinetic operator is
implemented by FFT, with diagonal Fourier eigenvalues
\(\frac12(k_1^2+k_2^2)\). Adaptive DOP853 integrates the same learned potential's
finite-dimensional equation

\[
\dot\psi=K(V_t\psi)-V_t(K\psi).
\]

The solver does not renormalize its output. Hermiticity and norm drift are
reported, and the finest-grid solution is rerun with tighter tolerances.
Tests compare the FFT action against an independently constructed dense
Fourier-collocation matrix, and a time-independent reference against the exact
matrix exponential. Tolerance settings are not represented as rigorous global
error certificates.

### Product formula

The implemented eight-factor formula is the established Layden construction,
not ordinary second-order Strang splitting. Chronological order is

\[
K_+,V_+,K_-,V_-,K_-,V_-,K_+,V_+,
\]

where \(K_\pm=e^{\pm i\alpha K}\),
\(V_\pm=e^{\pm i\beta V(t_{j+1/2})}\), and
\(\alpha\beta=\Delta t/2\). The local frozen-generator error is
\(O(\Delta t^2)\), giving **first-order global** convergence. Midpoint time
sampling does not turn this into a second-order global method. For a fixed
grid, the code balances the finite-grid kinetic norm and half the potential
range to choose \(\alpha,\beta\); its numerical values are saved.
There are four potential and four kinetic blocks per step.

The FFT implementation uses exact floating-point phases, and is explicitly
identified as a numerical simulation. The separate oracle and inference stages
compile native circuits and count their gates. Those native phases are not
misrepresented as synthesized finite Clifford+T sequences.

## 4. Measured outcomes and their limits

The source-to-target empirical sliced W1 is **0.41268**, while the learned ODE
samples give **0.03380**. Independent target samples give **0.02918** under the
same metric. Each uses 4,096 samples and 128 seeded projection directions.
These are finite-sample sliced-W1 diagnostics from one training seed. They do
**not** establish a small total-variation error or that the learned distribution
is indistinguishable from the target.

Indeed, the independent characteristic density yields approximately **0.1162
normalized grid TV** from the target on the fine grid. This is an essential
qualification: the learned model captures the broad multimodal geometry well
while retaining visible density errors. Wavefunction evolution faithfully
transports that learned distribution; it cannot repair the model's error.

| Points per axis | Wave vs same learned classical density: normalized grid TV | Classical density quadrature mass |
|---:|---:|---:|
| 8 | 0.254974 | 1.529296 |
| 16 | 0.099238 | 1.006545 |
| 32 | 0.009636 | 1.0000298 |
| 64 | 0.000278853 | 1.00000137 |

The small grids are severely under-resolved. On the 8-by-8 grid, some learned
potential modes are already above Nyquist; 16-by-16 resolves the potential
frequencies but not necessarily the evolving wavefunction. In particular, 8-by-8 circuit
accuracy is accuracy relative to a coarse discrete problem; it is not evidence
of continuum sampling accuracy. The classical density mass is retained before
normalization so that this failure is visible. Comparing normalized sampled
densities is a grid diagnostic, not a theorem bounding the TV distance between
continuum measures or an exact cell-probability calculation.

At 64-by-64, doubling characteristic RK4 steps from 128 to 256 changes the
normalized-grid density by approximately \(6.63\times10^{-10}\) in TV.
Tightening DOP853 from \((10^{-10},10^{-12})\) to
\((10^{-11},10^{-13})\) changes the state by approximately
\(5.00\times10^{-10}\) in vector norm. These controls make the measured spatial
error interpretable; neither is a machine-certified interval bound.

The fixed-grid product-formula state error decreases from **0.016236** at
32 steps to **0.00107076** at 512 steps on the 16-by-16 grid, consistent with
first-order global behavior. Increasing the number of steps improves the
finite-grid simulation while leaving the spatial/model errors unchanged.

## 5. Compression and research scope

`SmoothPotential.cores(t,n,max_rank=...)` materializes a small spatial table,
performs an ordinary two-site TT-SVD (matrix SVD), and balances its two factors
with the square roots of the singular values. This uses the established
[TT-SVD method of Oseledets](https://doi.org/10.1137/090752286). The independent
oracle stage measures singular values, truncation error, coefficient precision,
and actual reversible circuit costs.

The chosen finite Fourier model is intentionally structured. At full spatial
resolution its two-dimensional table has rank at most \(2M+1=11\), because
its rows and columns lie in the corresponding univariate Fourier spans. This
is an architecture-imposed upper bound, not evidence that arbitrary trained
neural potentials compress efficiently. The actual rank/error tradeoff still
depends on the trained coefficients and is measured. All TT extraction here
uses materialized small dense grids and therefore includes classical
preprocessing; no high-dimensional scalable extraction theorem is claimed.

The substantive result of this extension is a working, reproducible bridge
between genuine scalar-potential learning, an independent classical density
reference, convergent wavefunction evolution, and the existing explicit
reversible oracle compiler. It supports resource/error analysis with clearly
separated model, spatial, temporal, arithmetic, and synthesis assumptions.
