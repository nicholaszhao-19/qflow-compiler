# Reproduction

Use Python 3.12 and the [installation commands](../README.md#run-the-checks). Use a separate checkout or extraction when regenerating results so you can compare them with the saved data.

## Compiler benchmark

```bash
bash scripts/reproduce_v03.sh
python scripts/check_release_claims.py
```

The benchmark runs the tests, builds the oracle candidates and both preparation schedules, and checks how the rankings change under six rotation cost models. It writes the results to `results/v03/` and the figure to `figures/v03_oracle_frontier.png`. These paths use the original study names. The software release is v1.

## Smooth flow baseline and training

```bash
# Run the baseline pipeline with the bundled checkpoint.
bash scripts/reproduce.sh

# Train from seed 42 without overwriting the bundled weights.
python scripts/run_smooth_flow.py --retrain --train-only \
  --checkpoint /tmp/qflow-training/smooth_gmm.pt --out /tmp/qflow-training
```

Training runs for 2,500 updates. Platform math libraries can change the last bits of the weights, the SVD basis and the number of small nonzero Walsh rotations. Use the recorded platform when comparing errors and projected gate counts. [Validation record](release_audit.md)

To replay the original code, supply the archived FlowOracleBench v0.2.0 ZIP from your own copy. The script checks its SHA-256 digest before extracting it.

```bash
python scripts/replay_v02.py --zip /path/to/FlowOracleBench-v0.2.0.zip --out /tmp/baseline-replay.json
```

## Original MPS-V benchmark

Run `bash scripts/reproduce_legacy.sh` with the bundled checkpoints to regenerate the earliest results. Its optional `--retrain` flag replaces those checkpoints in the working copy. [Execution provenance](provenance.md)

## Source archive

```bash
python scripts/package_release.py
python scripts/smoke_release.py --archive dist/QFlow-Compiler-v1.zip \
  --destination /tmp/qflow-extracted --record-dir /tmp/qflow-smoke
```

Choose an extraction directory that does not exist. The smoke check uses your installed dependencies, verifies the archive hashes, imports from the extracted source, runs the tests, and checks the saved benchmark data. Use `python -m pip check` to verify the installed dependencies.
