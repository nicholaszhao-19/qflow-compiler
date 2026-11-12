#!/usr/bin/env python3
"""Scientific consistency gate for the executed v0.2 pipeline.

These are experiment acceptance checks, not statistical theorems. Assertions
address source identity, numerical-reference quality, circuit correctness and
reported resource scope. They do not mandate a positive quantum advantage.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print results without overwriting the saved record")
    args = parser.parse_args()
    smooth = read("results/smooth/metrics.json")
    oracle = read("results/oracle-study/summary.json")
    inference = read("results/inference/results.json")
    checkpoint_hash = hashlib.sha256((ROOT / "data/checkpoints/smooth_gmm.pt").read_bytes()).hexdigest()
    assert smooth["checkpoint_sha256"] == checkpoint_hash
    assert oracle["checkpoint_sha256"] == checkpoint_hash
    assert inference["checkpoint_sha256"] == checkpoint_hash
    checks = {
        "shared_checkpoint_sha256": checkpoint_hash,
        "grid_sizes": [row["n"] for row in smooth["grids"]],
        "finest_grid_wave_classical_tv": smooth["grids"][-1]["wave_vs_classical_tv"],
        "finest_grid_model_target_tv": smooth["grids"][-1]["classical_model_vs_target_tv"],
        "hermiticity_frobenius_error": smooth["hermiticity_frobenius_error_n8"],
        "phase_oracle_count": oracle["rank_precision_instances"],
        "all_tt_ancillas_clean": oracle["all_ancillas_clean"],
        "all_tt_address_checks_exhaustive": oracle["all_tt_branches_exhaustive"],
        "maximum_phase_gate_error": oracle["max_gate_vs_integer_phase_error"],
        "complete_flow_native_gate_vs_block_error": inference["preparation"]["full_native_vs_independent_block_l2_error"],
        "maximum_amplification_identity_error": max(row["amplification_identity_abs_error"] for row in inference["amplification"]),
        "hardware_execution": inference["interpretation"]["hardware_execution"],
    }
    assert checks["all_tt_ancillas_clean"] and checks["all_tt_address_checks_exhaustive"]
    assert checks["maximum_phase_gate_error"] < 3e-10
    assert checks["hermiticity_frobenius_error"] < 1e-11
    assert checks["complete_flow_native_gate_vs_block_error"] < 1e-9
    assert checks["maximum_amplification_identity_error"] < 1e-8
    assert not checks["hardware_execution"]
    for row in smooth["grids"]:
        assert row["wavefunction"]["norm_drift"] < 1e-7
    fine = smooth["grids"][-1]
    assert fine["density_refinement"]["normalized_tv"] < 1e-6
    assert fine["wave_solver_refinement"]["metrics"]["state_l2_error"] < 1e-6
    assert fine["wave_vs_classical_tv"] < smooth["grids"][0]["wave_vs_classical_tv"]
    assert smooth["product_formula"][-1]["state_l2_error"] < smooth["product_formula"][0]["state_l2_error"]
    if not args.check_only:
        out = ROOT / "results/current_validation.json"
        out.write_text(json.dumps({"status": "passed", "scope": "recorded scientific experiments; no quantum-advantage assertion", **checks}, indent=2) + "\n")
    print(json.dumps({"status": "passed", **checks}, indent=2))


if __name__ == "__main__":
    main()
