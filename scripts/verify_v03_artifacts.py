#!/usr/bin/env python3
"""Read-only consistency checks for the measured v0.3 release records."""

from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/v03"


def _rows(name):
    with (RESULTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _record(name):
    return json.loads((RESULTS / name).read_text())


def _hash(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    summary = _record("oracles.json")
    for label, path in summary["config"]["checkpoints"].items():
        assert _hash(path) == summary["checkpoints_sha256"][label]
    rows = _rows("oracles.csv")
    assert len(rows) == summary["circuits"]
    assert sum(int(row["exhaustive_exact_clean_branches"]) for row in rows) == summary["exact_clean_branch_cases"]
    qualified = [row for row in rows if row["experiment"] == "matched" and row["meets_target"] == "True"]
    assert len(qualified) == summary["qualified_matched_circuits"]
    for row in rows:
        allocated = Fraction(row["rotation_error_budget_exact"])
        special = Fraction(row["special_rotation_replacement_error_bound_exact"])
        assert special + int(row["arbitrary_rotations"]) * Fraction(float(row["per_rotation_error"])) <= allocated
        assert allocated == Fraction(str(float(row["rotation_error_budget"])))
        assert int(row["exhaustive_exact_clean_branches"]) == 2 ** int(row["address_qubits"])
        assert int(row["clean_ancillas"]) + int(row["dirty_ancillas"]) == int(row["ancilla_qubits"])
        assert int(row["address_qubits"]) + int(row["ancilla_qubits"]) == int(row["logical_qubits"])
    assert all(float(row["total_bound_conditional_on_synthesis"]) <= float(row["tolerance"]) + 1e-15 for row in qualified)

    groups = defaultdict(list)
    for row in qualified:
        resources = (float(row["total_bound_conditional_on_synthesis"]), int(row["t_count_estimate"]), int(row["logical_qubits"]))
        groups[row["case"], row["tolerance"]].append((row["candidate"], resources))
    expected_frontier = set()
    for group in groups.values():
        for candidate, resources in group:
            if not any(other != resources and all(a <= b for a, b in zip(other, resources, strict=True)) for _, other in group):
                expected_frontier.add(candidate)
    observed_frontier = _rows("oracles_pareto.csv")
    assert {row["candidate"] for row in observed_frontier} == expected_frontier

    preparation = _record("preparation.json")
    config = preparation["config"]
    assert _hash(config["checkpoint"]) == preparation["checkpoint_sha256"]
    midpoint = defaultdict(list)
    for row in _rows("preparation_midpoints.csv"):
        midpoint[int(row["grid"]), int(row["steps"]), row["method"]].append(row)
    for case in preparation["cases"]:
        grid, steps = case["grid"], case["steps"]
        assert case["kinetic_calls_per_A"] == case["phase_calls_per_A"] == 4 * steps
        for method in case["methods"]:
            records = midpoint[grid, steps, method["method"]]
            assert {int(row["midpoint_index"]) for row in records} == set(range(steps)) and len(records) == steps
            assert all(int(row["exhaustive_clean_addresses"]) == grid**2 for row in records)
            native_bound = Fraction(method["native_state_and_tv_bound_exact"])
            assert native_bound == sum(Fraction(row["centered_step_bound_exact"]) for row in records)
            assert native_bound <= Fraction(str(config["native_potential_budget"]))
            assert native_bound <= Fraction(method["native_state_and_tv_bound"])
            assert method["measured_metrics_to_native_block_reference"]["tv_distance"] <= float(native_bound) + 5e-12
            for account, budget_key in (("per_A_and_inverse_resource_accounting", "preparation_synthesis_budget"),
                                       ("repeated_amplification_accounting", "repeated_synthesis_budget")):
                resource = method[account]
                allocated = Fraction(resource["allocated_rotation_synthesis_budget_exact"])
                assert allocated == Fraction(str(config[budget_key]))
                assert Fraction(resource["used_rotation_synthesis_allowance_exact"]) <= allocated
            cost = method["per_A_and_inverse_resource_accounting"]
            assert cost["native_gates"] == sum(cost["native_gate_counts"].values())
            check = method["full_native_gate_integration"]
            if check is not None:
                assert check["full_gate_count"] == cost["native_gates"]
                assert check["full_native_depth"] <= cost["serial_native_depth_upper_bound"]
    baseline = _record("baseline_replay.json")
    assert baseline["status"] == "passed" and baseline["validation"]["status"] == "passed"
    assert baseline["tests"] == {"tests": 134, "failures": 0, "errors": 0, "skipped": 0}
    assert baseline["rank_precision_cases"] == 68
    print(f"Verified {len(rows)} oracle rows, {len(observed_frontier)} Pareto rows, "
          f"{sum(map(len, midpoint.values()))} midpoint rows, and the v0.2 replay record.")


if __name__ == "__main__":
    main()
