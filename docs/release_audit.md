# Benchmark validation

12 November 2026 · [Execution and comparison records](../results/release_validation_v032.json)

## Checks

The v1 checkout passed **160** tests. A fresh compiler run reproduced the non-timing fields in all **252** oracle, **63** Pareto, **466** width cap and **320** preparation midpoint rows. The smooth flow pipeline reproduced the eight baseline ratios and the macOS inference results below.

The earlier macOS ARM64 audit used Python 3.12.15 and the pinned package distributions.

- **Original v0.2.0:** Verified all **319** archived manifest hashes, passed **134** tests from the extraction and completed the smooth-flow, oracle and full native inference replay. The 68 TT sweep circuits covered **26,816** clean address cases. The patched one-command driver completed the same numerical stages.
- **Training:** Ran all **2,500** updates at seed 42, drawing **2.56 million** pairs. Two of eight tensors were bitwise equal to the archived Linux checkpoint. The largest absolute weight difference was **7.77×10⁻¹⁶**. The main reruns used the bundled checkpoint to hold the model fixed.
- **Earlier compiler replay:** Passed **155** tests. Reproduced **15,600** clean address checks and the rankings under six rotation cost models. The complete 8×8 preparation agreed with the saved native error bounds and resource counts.
- **Baseline convergence and inference:** At 64×64, normalized grid TV to the same learned ODE was **2.79×10⁻⁴**. The learned ODE to target TV was **0.1162**. The fixed-grid product-formula convergence orders were **0.96–0.99** under step doubling. The 8-qubit inference reproduced prepared event probability **0.1452353756**, MLAE RMSE **0.00222** and continuum-cell bias **−0.00847** at the stated precision.

### Platform differences

The saved Linux QPY contains **186,312** native gates. The fresh macOS QPY contains **186,260**, with the entire difference in the number of Rz angles below 10⁻¹⁶ radians. The maximum absolute discrepancy in the gate-executed amplification identity was **6.1×10⁻¹²** in the macOS replay. The complete 2,240-call schedule projects **23.034B** T gates in the archived Linux run and **23.020B** in the macOS replay. The 68 archived TT rows reproduced their logical widths, CCX counts, arbitrary rotation counts and projected T counts. Native gate count and depth differed in 35 and 56 rows respectively. The eight matched T ratios reproduced exactly.

The nested environment field in the older `results/v03/baseline_replay.json` is a historical copy of the Linux record. The fresh environment and measurements above come from the new execution.

## Interpretation

For each error target, the compiler benchmark compares the lowest projected T count in each tested family. The precision-aware TT implementation saves **20.9–60.6%** relative to the uniform precision implementation across **16** pairs. The comparison measures the combined effect of register sizing, precision and the candidate search. Some selected plans use uniform coefficient precision. SelectSwap or Walsh wins each pair. In the separate baseline sweep, uniform precision TT uses **12.7–53.2×** the best tested alternative across **eight** pairs. [Claim checker](../scripts/check_release_claims.py)

The exact phase-path checks apply to binary64 finite-grid inputs and clean ancillas. The measured TV, continuum discretization, model fit, preparation bias and finite-shot estimator error address different comparisons. The synthesis allowance is conditional on the gate tolerance and phase convention. Native gates and logical qubits are circuit counts. T counts use a model, and physical qubits and runtime require an error-correction and hardware model. **No end-to-end quantum advantage has been demonstrated.** [Error accounting](error_accounting.md) · [Resource model](resource_model.md)

The original release verified 131 vendored files against the pinned upstream Git blobs. The current tree retains the upstream source, data tables, checkpoint, citation and licenses used by the benchmark. The benchmark uses slices of the released checkpoint. The original audit ran on macOS and used the saved results for the earliest experiment sweep. [Upstream provenance](../vendor/UPSTREAM.md) · [Historical versions](archive/README.md)
