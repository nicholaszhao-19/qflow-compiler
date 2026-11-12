#!/usr/bin/env python3
"""Plot measured oracle accuracy and native/modelled resources."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/oracle-study"


def rows(name: str) -> list[dict]:
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    precision, compression, matched = rows("rank_precision.csv"), rows("compression.csv"), rows("matched.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.titleweight": "bold"})
    colors = {4: "#1a6189", 8: "#d66a30", 16: "#6d5193", 32: "#419c86"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for n in (4, 8, 16, 32):
        data = [row for row in compression if int(row["grid"]) == n]
        axes[0, 0].semilogy([int(row["retained_rank"]) for row in data],
                            [max(float(row["rank_phase_error"]), 1e-15) for row in data],
                            "o-", color=colors[n], label=f"{n} x {n} grid")
    axes[0, 0].set(title="Compression measured on the learned potential", xlabel="Retained TT bond size", ylabel="Phase operator error (exact grid maximum)")
    axes[0, 0].legend(frameon=False)
    for rank in (1, 2, 4, 8):
        data = [row for row in precision if int(row["grid"]) == 8 and int(row["retained_rank"]) == rank]
        axes[0, 1].semilogy([int(row["core_fraction_bits"]) for row in data],
                            [max(float(row["total_operator_error"]), 1e-15) for row in data],
                            "o-", label=f"rank {rank}")
    axes[0, 1].set(title="Rank sets an error floor; precision adds cost", xlabel="Fractional bits per core entry", ylabel="Total phase operator error (8 x 8)")
    axes[0, 1].legend(frameon=False, ncols=2)
    for n in (4, 8, 16, 32):
        data = [row for row in matched if int(row["grid"]) == n and row["method"] == "reversible_tt"]
        axes[1, 0].plot([float(row["tolerance"]) for row in data],
                       [float(row["ratio_to_best_baseline"]) for row in data],
                       "o-", color=colors[n], label=f"{n} x {n}")
    axes[1, 0].axhline(1, color="#888888", linestyle="--", lw=1)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Matched accuracy includes both implementations", xlabel="Total allowed operator error", ylabel="TT / best baseline estimated T count")
    axes[1, 0].legend(frameon=False)
    data = [row for row in rows("baseline_portfolio.csv") if int(row["grid"]) == 8 and float(row["tolerance"]) == .001]
    for method, marker, color in [("walsh", "s", "#419c86"), ("qrom", "^", "#1a6189"), ("selectswap", "o", "#d66a30")]:
        selected = [row for row in data if row["method"] == method]
        axes[1, 1].scatter([int(row["logical_qubits"]) for row in selected],
                           [int(row["t_count_estimate"]) for row in selected],
                           label=method, marker=marker, color=color, s=45)
    tt = next(row for row in matched if int(row["grid"]) == 8 and float(row["tolerance"]) == .001 and row["method"] == "reversible_tt")
    axes[1, 1].scatter([int(tt["logical_qubits"])], [int(tt["t_count_estimate"])], label="reversible TT", marker="*", color="#6d5193", s=140)
    axes[1, 1].set(xscale="log", yscale="log", title="Explicit qubit / estimated-T tradeoff (8 x 8)", xlabel="Allocated logical qubits", ylabel="Estimated T count at tolerance 0.001")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for axis in axes.ravel():
        axis.grid(alpha=.18)
    fig.suptitle("FlowOracleBench: learned-potential accuracy and circuit resources", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "accuracy_resources.png", dpi=180)
    fig.savefig(OUT / "accuracy_resources.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
