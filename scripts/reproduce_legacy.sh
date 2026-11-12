#!/usr/bin/env bash
# Run from a pinned, installed Python environment. Optional --retrain replaces
# the shipped checkpoints and training logs with deterministic fresh runs.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOWORACLEBENCH_PYTHON:-python}"

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--retrain" ) ]]; then
    printf 'Usage: scripts/reproduce_legacy.sh [--retrain]\n' >&2
    exit 2
fi
if [[ "${1:-}" == "--retrain" ]]; then
    "$python_bin" scripts/fetch_upstream.py
    "$python_bin" scripts/train_potential.py --config configs/train_gmm_2d.json
    "$python_bin" scripts/train_potential.py --config configs/train_gmm_2d_rank4.json
    "$python_bin" scripts/train_potential.py --config configs/train_gmm_4d.json
    "$python_bin" scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_2d_rank2.pt
    "$python_bin" scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_2d_rank4.pt
    "$python_bin" scripts/evaluate_training.py --checkpoint data/checkpoints/gmm_4d_rank2.pt --samples 1024
fi

"$python_bin" -m pytest
"$python_bin" scripts/run_benchmarks.py --stage all
"$python_bin" scripts/run_reference_diagnostics.py
"$python_bin" scripts/analyze_crossover.py
"$python_bin" scripts/plot_results.py
