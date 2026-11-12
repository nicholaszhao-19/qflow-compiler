# Benchmark provenance

QFlow Compiler v1 includes the data from the studies below. The labels and paths identify the original experiments.

| Original study | Scope and retained evidence |
| --- | --- |
| v0.1 | Original MPS-V extraction and arithmetic experiments. Historical numerical tables at the root of `results/` document this stage. |
| v0.2.0 | Smooth conditional-flow-matching model, grid and temporal refinements, eight uniform-TT comparisons, and the small inference example. See `results/smooth/`, `results/oracle-study/` and `results/inference/`. |
| v0.3.0 | Mixed-precision compiler, controlled public-checkpoint slices, and matched candidate and preparation benchmarks. |
| v0.3.1 | Exact special-angle and aggregate synthesis budget audit. The main comparison results are retained in `results/v03/`. |

The [release audit](../release_audit.md) records the verified results. The
[reproduction guide](../reproduction.md) describes the current benchmark and the
input needed to replay the frozen v0.2.0 release.
