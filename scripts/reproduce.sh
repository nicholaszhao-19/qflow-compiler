#!/usr/bin/env bash
# Complete v0.2 scientific pipeline; the default reuses the shipped checkpoint.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOWORACLEBENCH_PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--retrain" ) ]]; then
    printf 'Usage: scripts/reproduce.sh [--retrain]\n' >&2
    exit 2
fi
"$python_bin" -m pytest -q --junitxml=results/pytest-current.xml
"$python_bin" scripts/run_smooth_flow.py "$@"
"$python_bin" scripts/run_oracle_study.py
"$python_bin" scripts/verify_saved_oracle.py
"$python_bin" scripts/plot_oracle_study.py
"$python_bin" scripts/run_inference.py
"$python_bin" scripts/validate_results.py
