# Sygaldry source provenance

`tnwf-paper` contains the files used by QFlow Compiler from
<https://github.com/sygaldry-tech/tnwf-paper> at commit
`11358d3dca52193f93f037046229797296a2063e`, retrieved 2026-10-29.
The retained source, data tables and example checkpoint are byte-identical to that revision. The full upstream checkout, including demos and upstream test suites, is available at the pinned revision. Its MIT license and copyright notice are preserved in `tnwf-paper/LICENSE`.
The upstream package and data are associated with Kodama et al.,
*Scalable quantum simulation of continuous-time generative models via tensor
networks*, arXiv:2608.21700.

FlowOracleBench calls the public `tnwf.mps_v.train.train` function and its
`jam_conservative_loss` unchanged. The log wrapper in our training script only
records scalar loss values. Our loader calls `tnwf.mps_v.load_mps_v`; core
extraction uses the algebra of native `get_mps_cores(t)` with float64 arithmetic
and then performs the same periodic multilinear interpolation at our chosen
centered grid points.

The public training code explicitly says the definition of the paper's
`unitarity_weight=0.1` regularizer is unavailable and omits it. Therefore our
fresh public-code training runs are **not exact reproductions of the paper's
trained checkpoints**. They do provide a real learned MPS-V potential from the
released architecture and objective. No upstream bulk-data result is reported
as having been recomputed by FlowOracleBench.

The native extraction uses float32-trained coefficients and a float32 time-site
contraction, then casts the spatial cores to float64. We instead interpret the
stored float32 weights as real numbers and evaluate spatial and time
interpolation/contraction in float64. This avoids float32 time interpolation
jitter in adaptive ODE references without retraining or changing stored
parameters. Independent model-forward and native-provider comparisons record
the small float32 evaluation discrepancy explicitly.
