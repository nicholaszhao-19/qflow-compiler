#!/usr/bin/env python3
"""Reload the saved real Qiskit circuit and recheck every clean address."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from floworaclebench.simulation import validate_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Print results without overwriting the saved record")
    args = parser.parse_args()
    base = ROOT / "data/oracle_circuits/smooth_gmm_n4_rank2_f5"
    with base.with_suffix(".qpy").open("rb") as stream:
        circuit = qpy.load(stream)[0]
    with np.load(base.with_suffix(".npz")) as ref:
        validation = validate_oracle(circuit, int(ref["n_address"]), float(ref["beta"]) * ref["quantized_values"])
    assert validation["exhaustive"] and validation["address_preserved"]
    assert validation["ancillas_clean"] and validation["inverse_ancillas_clean"]
    assert validation["basis_phase_error"] < 3e-10
    assert validation["superposition_error"] < 3e-10
    assert validation["inverse_error"] < 3e-10
    if not args.check_only:
        output = ROOT / "results/oracle-study/saved_circuit_reloaded.json"
        output.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
