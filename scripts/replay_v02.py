#!/usr/bin/env python3
"""Reproduce the frozen v0.2 scientific pipeline in an isolated extraction."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "f393b8fb0034e0d21bd3092eff982ff9528ae764"
ARCHIVE_SHA256 = "f909ca5c9d78929d182d05f43fd50bea7f36b12aad893b705f2d706ff4dae141"


def _extract(destination, archive):
    if archive:
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Input is not the preserved FlowOracleBench v0.2.0 release")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        return destination / "FlowOracleBench"
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--quiet", "v0.2.0^{commit}"],
            cwd=ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit("Pass --zip /path/to/FlowOracleBench-v0.2.0.zip to replay the original release.") from exc
    if actual != CHECKPOINT:
        raise ValueError("Tag v0.2.0 is not the verified release checkpoint")
    data = subprocess.check_output(["git", "archive", CHECKPOINT], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(destination, filter="data")
    return destination


def _read(checkout, path):
    return json.loads((checkout / path).read_text())


def _rows(checkout, path):
    with (checkout / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def replay(archive, log):
    with tempfile.TemporaryDirectory(prefix="floworacle-v02-") as temp:
        checkout = _extract(Path(temp), archive)
        oracle_csv = "results/oracle-study/rank_precision.csv"
        original_rows = _rows(checkout, oracle_csv)
        inference_file = "results/inference/results.json"
        original = _read(checkout, inference_file)
        env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join(str(checkout / p) for p in ("src", "vendor/tnwf-paper/src")))
        commands = [["-m", "pytest", "-q", "--junitxml=results/pytest-current.xml"]]
        commands += [[f"scripts/{name}.py"] for name in (
            "run_smooth_flow", "run_oracle_study", "verify_saved_oracle", "plot_oracle_study",
            "run_inference", "validate_results")]
        for args in commands:
            print("v0.2 replay", " ".join(args), flush=True)
            proc = subprocess.run([sys.executable, *args], env=env, cwd=checkout, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
            log.write("\n" + " ".join(args) + "\n" + proc.stdout.replace(str(checkout), "<v0.2-extraction>"))
            log.flush()
            proc.check_returncode()
        suite = ET.parse(checkout / "results/pytest-current.xml").getroot().find("testsuite")
        rows = _rows(checkout, oracle_csv)
        assert len(rows) == len(original_rows)
        assert all(a[key] == b[key] for a, b in zip(original_rows, rows, strict=True)
                   for key in ("grid", "rank_cap", "core_fraction_bits"))
        changes = {key: sum(a[key] != b[key] for a, b in zip(original_rows, rows, strict=True))
                   for key in ("logical_qubits", "gate_count", "native_depth", "ccx", "arbitrary_rotations", "t_count_estimate")}
        measured = _read(checkout, inference_file)
        return {
            "status": "passed", "v02_import_checkpoint": CHECKPOINT, "python": platform.python_version(),
            "platform": platform.platform(), "commands": commands,
            "environment": _read(checkout, "results/smooth/environment.json"),
            "tests": {key: int(suite.get(key)) for key in ("tests", "failures", "errors", "skipped")},
            "rank_precision_cases": len(rows), "rank_precision_fields_differing_from_release": changes,
            "tt_clean_addresses_checked": sum(int(row["validated_inputs"]) for row in rows),
            "validation": _read(checkout, "results/current_validation.json"),
            "inference": {"release": original["preparation"], "replayed": measured["preparation"]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Use the preserved release ZIP instead of the local tag")
    parser.add_argument("--out", type=Path, default=ROOT / "results/v03/baseline_replay.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".log").open("w") as log:
        result = replay(args.zip, log)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
