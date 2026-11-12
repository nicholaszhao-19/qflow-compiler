#!/usr/bin/env python3
"""Execute compression, reversible arithmetic and matched-accuracy baselines.

All circuit resources are obtained from constructed Qiskit circuits. TT-SVD is
classical preprocessing of the materialized small grid, and is explicitly timed.
No dense table access is claimed to be efficient in high dimension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from qiskit import qpy, qasm2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floworaclebench.baselines import phase_error
from floworaclebench.benchmark import audited_row, baseline_portfolio, dump_json, write_rows
from floworaclebench.potentials import contract_cores
from floworaclebench.reversible import build_tt_oracle, quantize_cores
from floworaclebench.reversible import contract_cores as integer_contract


def quantized_table(cores: list[np.ndarray], bits: int) -> np.ndarray:
    """Exact Python-integer contraction, scaled once after all products."""
    integers = integer_contract(quantize_cores(cores, bits))
    return np.asarray(integers, dtype=float).reshape(-1) * 2.0 ** (-len(cores) * bits)


def run(config: dict, out: Path) -> dict:
    from floworaclebench.smooth_flow import load_smooth_potential

    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / config["checkpoint"]
    potential = load_smooth_potential(checkpoint)
    time, beta = float(config["time"]), float(config["phase_multiplier"])
    rows, matched, portfolios, compression = [], [], [], []
    start = perf_counter()
    saved_circuit = False
    for n in config["grids"]:
        begin = perf_counter()
        values = potential.values(time, n).reshape(-1)
        table_seconds = perf_counter() - begin
        singular_values = np.linalg.svd(values.reshape(n, n), compute_uv=False)
        architecture_rank_bound = min(n, 2 * potential.config.max_mode + 1)
        ranks = sorted({min(int(rank), architecture_rank_bound) for rank in config["rank_caps"]})
        cached = {}
        for rank in ranks:
            begin = perf_counter()
            cores = potential.cores(time, n, max_rank=rank)
            core_construction_seconds = perf_counter() - begin
            reconstructed = contract_cores(cores).reshape(-1)
            truncation = float(np.max(np.abs(reconstructed - values)))
            rank_error = phase_error(beta * reconstructed, beta * values)
            context = {
                "dataset": "smooth_cfm_gmm", "grid": n, "dimension": 2,
                "rank_cap": rank, "retained_rank": int(cores[0].shape[-1]),
                "time": time, "beta": beta, "table_points": n * n,
                "tt_coefficients": sum(core.size for core in cores),
                "coefficient_to_dense_ratio": sum(core.size for core in cores) / (n * n),
                "singular_values": singular_values.tolist(),
                "architecture_rank_upper_bound": architecture_rank_bound,
                "svd_tail_frobenius": float(np.linalg.norm(singular_values[rank:])),
                "table_evaluation_seconds": table_seconds, "core_construction_seconds": core_construction_seconds,
                "truncation_sup_error": truncation, "rank_phase_error": rank_error,
                "classical_preprocessing": "materialized dense grid then 2D TT-SVD (matrix SVD)",
            }
            compression.append(context)
            cached[rank] = (cores, context)
            for bits in config["fraction_bits"]:
                oracle = build_tt_oracle(cores, beta, int(bits))
                row = audited_row(
                    oracle, beta * oracle.quantized_values,
                    {**context, "experiment": "rank_precision"},
                    config["rotation_synthesis_budget"], max_inputs=n * n,
                )
                row["quantization_sup_error"] = float(np.max(np.abs(oracle.quantized_values - reconstructed)))
                row["total_potential_sup_error"] = float(np.max(np.abs(oracle.quantized_values - values)))
                row["total_operator_error"] = phase_error(beta * oracle.quantized_values, beta * values)
                row["operator_error_bound"] = min(2.0, abs(beta) * (truncation + oracle.metadata["core_quantization_bound"]))
                if row["total_operator_error"] > row["operator_error_bound"] + 2e-11:
                    raise AssertionError("Phase error exceeds analytic bound")
                rows.append(row)
                if n == 4 and rank == 2 and bits == 5:
                    target = ROOT / "data/oracle_circuits/smooth_gmm_n4_rank2_f5.qpy"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as stream:
                        qpy.dump(oracle.circuit, stream)
                    target.with_suffix(".qasm").write_text(qasm2.dumps(oracle.circuit) + "\n")
                    dump_json(target.with_suffix(".json"), row)
                    np.savez_compressed(target.with_suffix(".npz"), potential_values=values, quantized_values=oracle.quantized_values, beta=beta, n_address=oracle.n_address)
                    saved_circuit = True
            print(f"compiled/exhaustively validated n={n} rank={rank}", flush=True)
        if config.get("matched_search_all_ranks_through_architecture_bound", False):
            for rank in range(1, architecture_rank_bound + 1):
                if rank in cached:
                    continue
                begin = perf_counter()
                cores = potential.cores(time, n, max_rank=rank)
                duration = perf_counter() - begin
                approx = contract_cores(cores).reshape(-1)
                context = {**next(iter(cached.values()))[1],
                    "rank_cap": rank, "retained_rank": int(cores[0].shape[-1]),
                    "tt_coefficients": sum(core.size for core in cores),
                    "coefficient_to_dense_ratio": sum(core.size for core in cores)/(n*n),
                    "core_construction_seconds": duration,
                    "truncation_sup_error": float(np.max(np.abs(approx-values))),
                    "rank_phase_error": phase_error(beta*approx,beta*values),
                    "svd_tail_frobenius": float(np.linalg.norm(singular_values[rank:]))}
                cached[rank] = (cores, context)
        for tolerance in config["matched_tolerances"]:
            candidates = []
            for rank, (cores, context) in cached.items():
                for bits in range(config["max_fraction_bits"] + 1):
                    table = quantized_table(cores, bits)
                    err = phase_error(beta * table, beta * values)
                    if err <= tolerance / 2:
                        oracle = build_tt_oracle(cores, beta, bits)
                        row = audited_row(oracle, beta * table,
                            {**context, "experiment": "matched", "tolerance": tolerance,
                             "approximation_budget": tolerance / 2},
                            tolerance / 2, max_inputs=n * n)
                        row["total_operator_error"] = err
                        row["operator_error"] = err
                        candidates.append(row)
                        break
            if not candidates:
                raise RuntimeError(f"No TT meets n={n}, tolerance={tolerance}")
            best_tt = min(candidates, key=lambda row: (row["t_count_estimate"], row["logical_qubits"]))
            _, best_baseline, all_baselines = baseline_portfolio(
                values, beta, tolerance,
                {"dataset": "smooth_cfm_gmm", "grid": n, "time": time,
                 "beta": beta, "tolerance": tolerance, "experiment": "matched"},
                config["baseline_max_qubits"], validate=True,
            )
            # Portfolio defaults to at most 64 branch paths; disclose that field
            # explicitly (all finite-grid phase errors are still exhaustive).
            best_tt["ratio_to_best_baseline"] = best_tt["t_count_estimate"] / max(1, best_baseline["t_count_estimate"])
            best_tt["baseline_method"] = best_baseline["method"]
            equal_width = min((row for row in all_baselines if row["logical_qubits"] <= best_tt["logical_qubits"]), key=lambda row: row["t_count_estimate"])
            best_tt["same_width_baseline_method"] = equal_width["method"]
            best_tt["same_width_baseline_t_estimate"] = equal_width["t_count_estimate"]
            best_tt["same_width_tt_t_ratio"] = best_tt["t_count_estimate"] / max(1, equal_width["t_count_estimate"])
            matched.extend([best_tt, best_baseline])
            portfolios.extend(all_baselines)
            print(f"matched n={n} tol={tolerance}: rank={best_tt['retained_rank']} f={best_tt['core_fraction_bits']}, TT/baseline={best_tt['ratio_to_best_baseline']:.3f}", flush=True)
    write_rows(out / "rank_precision.csv", rows)
    write_rows(out / "compression.csv", compression)
    write_rows(out / "matched.csv", matched)
    write_rows(out / "baseline_portfolio.csv", portfolios)
    summary = {
        "config": config, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "rank_precision_instances": len(rows), "matched_pairs": len(matched) // 2,
        "all_tt_branches_exhaustive": all(row["exhaustive"] for row in rows),
        "max_gate_vs_integer_phase_error": max(row["basis_phase_error"] for row in rows),
        "all_ancillas_clean": all(row["ancillas_clean"] and row["inverse_ancillas_clean"] for row in rows),
        "max_observed_error_minus_bound": max(row["total_operator_error"]-row["operator_error_bound"] for row in rows),
        "minimum_tt_to_baseline_t_ratio": min(row["ratio_to_best_baseline"] for row in matched if "ratio_to_best_baseline" in row),
        "maximum_tt_to_baseline_t_ratio": max(row["ratio_to_best_baseline"] for row in matched if "ratio_to_best_baseline" in row),
        "saved_qiskit_qpy_and_qasm": saved_circuit,
        "rank_semantics": "retained SVD bond size, not certified minimal numerical rank",
        "precision_selection": "first fitting fraction bits per rank; min modelled T among retained candidates; not global optimization",
        "time_scope": "one fixed snapshot; no uniform-in-time compressibility claim",
        "architecture_scope": "finite Fourier max_mode=5 implies matrix rank<=11; measured spectra do not establish generic neural compressibility",
        "elapsed_seconds": perf_counter() - start,
        "fault_tolerant_status": "native rotations exact; Clifford+T rotation costs modelled, not synthesized",
        "compression": compression, "matched": matched,
    }
    dump_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oracle_study.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/oracle-study")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
