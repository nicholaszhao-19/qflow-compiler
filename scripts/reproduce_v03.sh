#!/usr/bin/env bash
# Execute v0.3 tests and both benchmark drivers. The v0.2 replay is separate.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${FLOWORACLEBENCH_PYTHON:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$root/src:$root/vendor/tnwf-paper/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p results/v03
"$python_bin" -m pytest -q --junitxml=results/v03/pytest.xml
"$python_bin" - <<'PY'
from pathlib import Path
from xml.etree import ElementTree

path = Path("results/v03/pytest.xml")
document = ElementTree.parse(path)
for suite in document.iter("testsuite"):
    suite.attrib.pop("hostname", None)
document.write(path, encoding="utf-8", xml_declaration=True)
PY
"$python_bin" scripts/run_v03_oracles.py
"$python_bin" scripts/run_v03_preparation.py
"$python_bin" scripts/verify_v03_artifacts.py
"$python_bin" scripts/audit_v031_sensitivity.py
