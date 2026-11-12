#!/usr/bin/env python3
"""Train upstream JAM/MPS-V and retain raw losses, metadata and extraction QA.

This invokes the public trainer itself.  A loss wrapper records its calls but
does not modify the objective or optimizer.  The public trainer omits the
unreleased regularizer used for the paper's archived checkpoints; these runs
are therefore fresh public-code training, not an exact paper reproduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from floworaclebench.potentials import UPSTREAM_COMMIT, ensure_upstream, load_potential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_gmm_2d.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    threads = int(config.pop("torch_threads", 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    torch.use_deterministic_algorithms(True)
    out_path = ROOT / config.pop("out_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_upstream()
    training = importlib.import_module("tnwf.mps_v.train")
    loss_function = training.jam_conservative_loss
    history = []
    batch_size = int(config["batch_size"])

    def record_loss(model, x0, x1, t):
        loss = loss_function(model, x0, x1, t)
        history.append({"call": len(history) + 1, "batch_size": len(x0),
                        "kind": "train" if len(x0) == batch_size else "validation",
                        "loss": float(loss.detach().cpu())})
        return loss

    training.jam_conservative_loss = record_loss
    start = time.perf_counter()
    checkpoint = training.train(out_path=str(out_path), **config)
    elapsed = time.perf_counter() - start
    training.jam_conservative_loss = loss_function
    raw_path = out_path.with_suffix(".losses.csv")
    with raw_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["call", "batch_size", "kind", "loss"])
        writer.writeheader()
        writer.writerows(history)
    potential = load_potential(out_path)
    extraction = []
    for n in (4, 8, 16):
        for t in (0.0, 0.125, 0.5, 0.875, 1.0):
            extracted = potential.values(t, n)
            direct = potential.direct_values(t, n)
            extraction.append({"n": n, "t": t, "max_error": float(np.max(np.abs(extracted-direct))),
                               "min_value": float(extracted.min()), "max_value": float(extracted.max())})
    metadata = {
        "training_kind": "fresh training using public upstream JAM/MPS-V implementation",
        "not_exact_paper_reproduction": "public trainer explicitly omits unpublished unitarity_weight regularizer",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_url": "https://github.com/sygaldry-tech/tnwf-paper",
        "config": config | {"torch_threads": threads, "out_path": str(out_path.relative_to(ROOT))},
        "checkpoint_args": checkpoint["args"],
        "final_training_loss": checkpoint["final_loss"],
        "validation_history": checkpoint["val_history"],
        "elapsed_seconds": elapsed,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "checkpoint_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "extraction_validation": extraction,
        "grid": "x_j = -L/2 + j L/n; C-order spatial indices",
        "potential_truth": "stored float32 model weights interpreted as real numbers; spatial and time interpolation/contraction evaluated float64",
        "time_contraction": "float64 equivalent of upstream get_mps_cores algebra; avoids float32 time interpolation jitter in ODE references",
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(out_path), "elapsed_seconds": elapsed,
                      "final_loss": checkpoint["final_loss"],
                      "extraction_max_error": max(row["max_error"] for row in extraction)}, indent=2))


if __name__ == "__main__":
    main()
