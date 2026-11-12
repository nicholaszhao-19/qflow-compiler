#!/usr/bin/env python3
"""Recompute headline comparisons and inference statistics from saved records."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path

import numpy as np
from qiskit import qpy


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def resource_key(row):
    return int(row["t_count_estimate"]), int(row["logical_qubits"])


def check_v02(root):
    results = root / "results"
    study = results / "oracle-study"
    matched = defaultdict(list)
    portfolio = defaultdict(list)
    for table, groups in [("matched.csv", matched), ("baseline_portfolio.csv", portfolio)]:
        for row in read_rows(study / table):
            if row["experiment"] == "matched":
                groups[int(row["grid"]), Fraction(row["tolerance"])].append(row)
    expected = {(grid, tolerance) for grid in (4, 8, 16, 32)
                for tolerance in (Fraction(1, 100), Fraction(1, 1000))}
    assert set(matched) == set(portfolio) == expected
    comparisons = []
    for (grid, tolerance), pair in sorted(matched.items()):
        tt, = [row for row in pair if row["method"] == "reversible_tt"]
        reported, = [row for row in pair if row["method"] != "reversible_tt"]
        qualified = [row for row in portfolio[grid, tolerance]
                     if Fraction(row["operator_error"]) + Fraction(row["rotation_error_budget"]) <= tolerance]
        best = min(qualified, key=resource_key)
        assert best["method"] == reported["method"] and resource_key(best) == resource_key(reported)
        assert Fraction(tt["operator_error"]) + Fraction(tt["rotation_error_budget"]) <= tolerance
        ratio = Fraction(resource_key(tt)[0], resource_key(best)[0])
        assert ratio > 1 and math.isclose(float(ratio), float(tt["ratio_to_best_baseline"]), rel_tol=1e-12)
        comparisons.append({"grid": grid, "tolerance": float(tolerance), "alternative": best["method"],
                            "uniform_tt_t": resource_key(tt)[0], "alternative_t": resource_key(best)[0],
                            "uniform_to_alternative_ratio": float(ratio)})

    smooth = json.loads((results / "smooth/metrics.json").read_text())
    grids = smooth["grids"]
    assert [row["n"] for row in grids] == [8, 16, 32, 64]
    tv = [row["wave_vs_classical_tv"] for row in grids]
    assert all(before > after for before, after in zip(tv, tv[1:]))
    errors = [row["state_l2_error"] for row in smooth["product_formula"]]
    orders = [math.log2(before / after) for before, after in zip(errors, errors[1:])]
    assert all(0.9 < order < 1.1 for order in orders)

    inference = results / "inference"
    record = json.loads((inference / "results.json").read_text())
    prep = record["preparation"]
    resource = prep["native_gate_resources"]
    circuit_path = inference / "prepared_flow.qpy"
    if circuit_path.is_file():
        stream = circuit_path.open("rb")
    else:
        stream = gzip.open(circuit_path.with_suffix(".qpy.gz"), "rb")
    with stream:
        circuit, = qpy.load(stream)
    assert circuit.num_qubits == resource["logical_qubits"]
    assert len(circuit.data) == resource["gate_count"] and circuit.depth() == resource["native_depth"]
    assert dict(circuit.count_ops()) == resource["native_gate_counts"]
    full = record["complete_repeated_query_resources"][-1]
    depths = record["config"]["depths"]
    shots = record["config"]["shots_per_depth"]
    forward = shots * sum(k + 1 for k in depths)
    inverse = shots * sum(depths)
    assert (full["A_forward_calls"], full["A_inverse_calls"]) == (forward, inverse)
    assert full["event_calls"] == full["zero_reflection_calls"] == inverse
    assert full["A_or_inverse_calls"] == forward + inverse
    assert full["circuit_shots"] == shots * len(depths)
    blocks = [(full["A_resources_at_allocated_budget"], forward + inverse),
              (full["event_resources_at_allocated_budget"], inverse),
              (full["zero_reflection_resources_at_allocated_budget"], inverse)]
    for field in ("gate_count", "t_count_estimate"):
        assert full[field] == sum(block[field] * count for block, count in blocks)
    with np.load(inference / "states_and_samples.npz", allow_pickle=False) as samples:
        p_prepared = float(np.sum(abs(samples["prepared_state"][samples["event_mask"]]) ** 2))
        estimates = samples[f"mle_prefix{len(depths)}"]
        assert estimates.size == record["config"]["independent_repetitions"]
    assert math.isclose(p_prepared, prep["prepared_event_probability"], abs_tol=1e-12, rel_tol=0)
    rmse = float(np.sqrt(np.mean((estimates - p_prepared) ** 2)))
    assert math.isclose(rmse, full["empirical_rmse"], abs_tol=1e-12, rel_tol=0)
    bias = p_prepared - record["classical_continuum_cell_reference"]["probability"]
    assert math.isclose(bias, prep["persistent_continuum_cell_probability_bias"], abs_tol=1e-12, rel_tol=0)
    ratios = [row["uniform_to_alternative_ratio"] for row in comparisons]
    return {
        "matched_pairs": comparisons, "uniform_to_alternative_ratio_range": [min(ratios), max(ratios)],
        "grid_sizes": [row["n"] for row in grids], "wave_vs_same_learned_ode_grid_tv": tv,
        "finest_grid_learned_ode_vs_target_tv": grids[-1]["classical_model_vs_target_tv"],
        "product_formula_empirical_temporal_orders": orders,
        "saved_inference_circuit": {"qubits": circuit.num_qubits, "native_gates": len(circuit.data),
                                    "native_depth": circuit.depth()},
        "inference": {"prepared_probability": p_prepared, "learned_continuum_event_bias": bias,
                      "empirical_mlae_rmse_to_prepared_distribution": rmse,
                      "repetitions": len(estimates), "forward_or_inverse_calls": forward + inverse,
                      "native_gates_per_schedule": full["gate_count"],
                      "projected_t_per_schedule": full["t_count_estimate"]},
    }


def check_v03(root):
    results = root / "results/v03"
    config = json.loads((results / "oracles.json").read_text())["config"]
    groups = defaultdict(list)
    for row in read_rows(results / "oracles.csv"):
        if row["experiment"] == "matched" and row["meets_target"] == "True":
            groups[row["case"], float(row["tolerance"])].append(row)
    assert set(groups) == {(case["case"], tol) for case in config["cases"] for tol in config["tolerances"]}
    comparisons = []
    for (case, tolerance), group in sorted(groups.items()):
        uniform = min((r for r in group if r["method"] == "reversible_tt"), key=resource_key)
        mixed = min((r for r in group if r["method"] == "precision_tt"), key=resource_key)
        best = min(group, key=resource_key)
        assert best["method"] in {"walsh", "selectswap"}
        assert resource_key(best)[0] < resource_key(mixed)[0] < resource_key(uniform)[0]
        comparisons.append({"case": case, "tolerance": tolerance, "winner": best["method"],
                            "uniform_tt_t": resource_key(uniform)[0], "mixed_tt_t": resource_key(mixed)[0],
                            "winner_t": resource_key(best)[0],
                            "tt_t_saving_percent": 100 * (1 - resource_key(mixed)[0] / resource_key(uniform)[0])})
    prep = json.loads((results / "preparation.json").read_text())
    case, = [case for case in prep["cases"] if case["grid"] == 8 and case["steps"] == 64]
    methods = {}
    for method in case["methods"]:
        resource = method["per_A_and_inverse_resource_accounting"]
        bound = Fraction(method["native_state_and_tv_bound_exact"])
        assert bound <= Fraction(str(prep["config"]["native_potential_budget"]))
        measured = method["measured_metrics_to_native_block_reference"]["tv_distance"]
        assert measured <= float(bound) + 5e-12
        methods[method["method"]] = {"logical_qubits": resource["logical_qubits"],
                                     "projected_t": resource["t_count_estimate"],
                                     "native_gates": resource["native_gates"],
                                     "native_state_and_tv_bound": method["native_state_and_tv_bound"],
                                     "observed_tv": measured}
    savings = [row["tt_t_saving_percent"] for row in comparisons]
    return {"matched_pairs": comparisons, "tt_t_saving_percent_range": [min(savings), max(savings)],
            "preparation_8x8_64_steps": methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v02-root", type=Path, default=ROOT)
    parser.add_argument("--v03-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({"v0.2": check_v02(args.v02_root), "v0.3": check_v03(args.v03_root)}, indent=2))


if __name__ == "__main__":
    main()
