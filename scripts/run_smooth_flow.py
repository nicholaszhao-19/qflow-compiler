#!/usr/bin/env python3
"""Execute genuine CFM training and same-potential classical/wave validation.

Default reuses the saved checkpoint. Use --retrain for a fresh seeded training
run, or --train-only to stop after producing the checkpoint and training log.
All reported diagnostics are produced by executed code in this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from threadpoolctl import threadpool_limits

from floworaclebench.evolution import state_metrics
from floworaclebench.potentials import centered_grid
from floworaclebench.smooth_flow import (
    SmoothConfig, SmoothPotential, boundary_diagnostics, heldout_diagnostics,
    hermiticity_error, load_smooth_potential, particle_flow, product_formula,
    reverse_density, sample_source, sample_target, save_model, sliced_wasserstein_1,
    train, wavefunction_reference,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_training_replay(args):
    """Retrain in memory and compare every saved tensor without overwriting it."""
    checkpoint = ROOT / args.checkpoint
    output = ROOT / args.out
    original = load_smooth_potential(checkpoint)
    model, losses, metadata = train(original.config)
    old_state, new_state = original.model.state_dict(), model.state_dict()
    same_keys = set(old_state) == set(new_state)
    differences = {name: float(np.max(np.abs(value.detach().numpy() - new_state[name].detach().numpy())))
                   for name, value in old_state.items()}
    bitwise = all(np.array_equal(value.detach().numpy(), new_state[name].detach().numpy())
                  for name, value in old_state.items())
    with (output / "losses.csv").open() as handle:
        old_losses = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    loss_history_equal = old_losses == losses
    result = {"command": " ".join(sys.argv), "checkpoint": args.checkpoint,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "config": asdict(original.config), "training_metadata": metadata,
              "same_tensor_keys": same_keys, "all_tensors_bitwise_equal": bitwise,
              "maximum_absolute_tensor_difference": max(differences.values()),
              "tensor_maximum_absolute_differences": differences,
              "loss_history_equal": loss_history_equal, "loss_history_rows": len(losses),
              "scope": "one exact CPU environment and seed; trained weights compared in memory; saved checkpoint not changed"}
    write_json(output / "training_replay.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not (same_keys and bitwise and loss_history_equal):
        raise AssertionError("Training replay differed from saved parameters or loss history")


def run(args):
    start = perf_counter()
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / args.checkpoint
    if args.retrain or not checkpoint.exists():
        config = SmoothConfig(**json.loads((ROOT / args.config).read_text()))
        print(f"Training seed {config.seed}: {config.steps} updates, batch {config.batch_size}", flush=True)
        model, losses, metadata = train(config)
        save_model(model, config, checkpoint, metadata)
        with (output / "losses.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(losses[0]))
            writer.writeheader()
            writer.writerows(losses)
        write_json(output / "training.json", metadata)
        potential = SmoothPotential(model, config)
        print(f"Saved {checkpoint.name}; training took {metadata['seconds']:.2f} s", flush=True)
    else:
        potential = load_smooth_potential(checkpoint)
        config = potential.config
    write_json(output / "config.json", asdict(config))
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "torch", "matplotlib", "qiskit")},
                   "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "command": " ".join(sys.argv),
                   "date_utc": datetime.now(timezone.utc).isoformat(),
                   "dtype": "float64 training and evaluation; complex128 evolution",
                   "threads": config.threads}
    write_json(output / "environment.json", environment)
    if args.train_only:
        return
    diagnostics = {"config": asdict(config), "checkpoint_sha256": environment["checkpoint_sha256"],
                   "heldout": heldout_diagnostics(potential),
                   "boundary": boundary_diagnostics(potential),
                   "hermiticity_frobenius_error_n8": hermiticity_error(potential)}
    print("Held-out diagnostics:", diagnostics["heldout"], flush=True)
    t0 = perf_counter()
    source = sample_source(config, config.samples, config.seed + 1000)
    target = sample_target(config, config.samples, config.seed + 2000)
    target_other = sample_target(config, config.samples, config.seed + 3000)
    coarse = particle_flow(potential, source, config.particle_steps // 2)
    fine = particle_flow(potential, source, config.particle_steps)
    refinement = np.linalg.norm(fine - coarse, axis=1)
    diagnostics["particles"] = {
        "n": config.samples, "coarse_steps": config.particle_steps // 2,
        "fine_steps": config.particle_steps,
        "rk4_refinement_rms_endpoint_change": float(np.sqrt(np.mean(refinement**2))),
        "rk4_refinement_max_endpoint_change": float(np.max(refinement)),
        "sliced_w1_source_target": sliced_wasserstein_1(source, target),
        "sliced_w1_learned_target": sliced_wasserstein_1(fine, target),
        "sliced_w1_independent_target_floor": sliced_wasserstein_1(target_other, target),
        "outside_box_sample_count": int(np.sum(np.any(np.abs(fine) >= config.length / 2, axis=1))),
        "seconds_including_metrics": perf_counter() - t0,
        "interpretation": "one seeded training run; empirical sliced W1 over 128 fixed directions, not population TV"}
    print("Particle diagnostics:", diagnostics["particles"], flush=True)
    arrays = {"source_samples": source, "target_samples": target,
              "target_samples_independent": target_other, "learned_samples": fine,
              "learned_samples_coarse_ode": coarse}
    grid_rows, wave_states, spatial_rows = [], {}, []
    previous = None
    for n in config.grids:
        points = centered_grid(n, 2, config.length)
        density, density_metadata = reverse_density(potential, points, config.density_steps)
        state, wave_metadata = wavefunction_reference(potential, n)
        wave_states[n] = state
        h = config.length / n
        mass = float(density.sum() * h**2)
        classical_probability = density / density.sum()
        born_probability = abs(state)**2
        born_probability /= born_probability.sum()
        target_density = potential.target_density(points)
        target_mass = float(target_density.sum() * h**2)
        target_probability = target_density / target_density.sum()
        row = {"n": n, "density": density_metadata, "wavefunction": wave_metadata,
               "classical_density_quadrature_mass": mass,
               "target_density_quadrature_mass": target_mass,
               "wave_vs_classical_tv": float(0.5 * np.sum(abs(born_probability - classical_probability))),
               "classical_model_vs_target_tv": float(0.5 * np.sum(abs(classical_probability - target_probability))),
               "wave_vs_target_tv": float(0.5 * np.sum(abs(born_probability - target_probability))),
               "wave_vs_classical_sqrt_state": state_metrics(state, np.sqrt(classical_probability)),
               "born_raw_vs_unnormalized_density_l1": float(np.sum(abs(abs(state)**2 - density * h**2)))}
        if n == max(config.grids):
            refined_density, refined_metadata = reverse_density(potential, points, 2 * config.density_steps)
            row["density_refinement"] = {
                "metadata": refined_metadata,
                "raw_grid_l1": float(np.sum(abs(refined_density - density)) * h**2),
                "normalized_tv": float(0.5 * np.sum(abs(refined_density / refined_density.sum() - classical_probability))),
                "mass": float(refined_density.sum() * h**2)}
            refined_state, refined_wave_meta = wavefunction_reference(potential, n, rtol=1e-11, atol=1e-13)
            row["wave_solver_refinement"] = {"metadata": refined_wave_meta,
                                             "metrics": state_metrics(state, refined_state)}
            arrays[f"reference_refined_n{n}"] = refined_state
            arrays[f"classical_density_refined_n{n}"] = refined_density
        if previous is not None:
            old_n, old_state = previous
            interpolated = old_state.reshape(old_n, old_n)
            for axis in range(2):
                interpolated = resample(interpolated, n, axis=axis)
            interpolated = interpolated.reshape(-1) * (old_n / n)
            spatial_rows.append({"old_n": old_n, "new_n": n,
                                 "interpolation": "Fourier resample; includes normalized-grid amplitude factor",
                                 "metrics": state_metrics(interpolated, state)})
        previous = n, state
        arrays[f"reference_n{n}"] = state
        arrays[f"initial_n{n}"] = potential.initial_state(n)
        arrays[f"classical_density_n{n}"] = density
        arrays[f"target_density_n{n}"] = target_density
        grid_rows.append(row)
        print(f"Grid {n}: TV(wave,classical)={row['wave_vs_classical_tv']:.6g}; "
              f"density mass={mass:.8g}; wave solve {wave_metadata['seconds']:.2f}s", flush=True)
        diagnostics["grids"] = grid_rows
        write_json(output / "metrics_partial.json", diagnostics)
    diagnostics["grids"] = grid_rows
    diagnostics["spatial_refinement"] = spatial_rows
    formula_rows = []
    n = config.formula_grid
    reference = wave_states[n]
    for steps in config.formula_steps:
        state, metadata = product_formula(potential, n, steps)
        metrics = state_metrics(state, reference)
        formula_rows.append({**metadata, **metrics})
        arrays[f"formula_n{n}_steps{steps}"] = state
        print(f"Product formula n={n}, steps={steps}: state L2={metrics['state_l2_error']:.6g}", flush=True)
    diagnostics["product_formula"] = formula_rows
    diagnostics["seconds_total"] = perf_counter() - start
    np.savez_compressed(output / "states.npz", **arrays)
    write_json(output / "metrics.json", diagnostics)
    plot(output, diagnostics, arrays)
    print(f"Saved complete numerical diagnostics to {output}; total {diagnostics['seconds_total']:.2f} s", flush=True)


def plot(output, metrics, arrays):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, key, title in zip(axes[0], ["source_samples", "learned_samples", "target_samples"],
                              ["Gaussian source", "Learned ODE samples", "Independent GMM target"]):
        x = arrays[key]
        ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.18, rasterized=True)
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal", title=title, xlabel="x", ylabel="y")
    grids = metrics["grids"]
    n = grids[-1]["n"]
    length = metrics["config"]["length"]
    h = length / n
    extent = [-length / 2 - h / 2, length / 2 - h / 2] * 2
    maps = [(abs(arrays[f"reference_n{n}"])**2 * (n / length)**2,
             "Born density: finite-grid reference"),
            (arrays[f"classical_density_n{n}"], "Same-potential reverse-ODE density"),
            (arrays[f"target_density_n{n}"], "Exact periodized GMM target density")]
    vmax = max(np.max(x) for x, _ in maps)
    for ax, (density, title) in zip(axes[1], maps):
        ax.imshow(density.reshape(n, n).T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), title=title, xlabel="x", ylabel="y")
    fig.savefig(output / "learned_flow.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    if (output / "losses.csv").exists():
        losses = np.genfromtxt(output / "losses.csv", delimiter=",", names=True)
        axes[0].plot(losses["step"], losses["cfm_loss"], lw=1, label="training batch CFM")
        axes[0].axhline(metrics["heldout"]["analytic_conditional_cfm_mse"], color="black", ls=":", label="held-out analytic conditional mean")
        axes[0].set(xlabel="Optimizer updates", ylabel="Squared velocity loss", title="Genuine sampled CFM training")
        axes[0].legend(fontsize=8)
    axes[1].loglog([x["n"] for x in grids], [x["wave_vs_classical_tv"] for x in grids], "o-", label="wave vs same learned ODE")
    axes[1].loglog([x["n"] for x in grids], [x["classical_model_vs_target_tv"] for x in grids], "s--", label="learned ODE vs target")
    axes[1].set(xlabel="Grid points per axis", ylabel="Normalized grid TV", title="Simulation error vs model error")
    axes[1].legend(fontsize=8)
    rows = metrics["product_formula"]
    steps = np.array([r["steps"] for r in rows])
    error = np.array([r["state_l2_error"] for r in rows])
    axes[2].loglog(steps, error, "o-", label="eight-factor formula")
    axes[2].loglog(steps, error[-1] * steps[-1] / steps, "k:", label="slope -1 guide")
    axes[2].set(xlabel="Time steps", ylabel="State vector L2 error", title=f"Fixed-grid temporal refinement (N={rows[0]['n']})")
    axes[2].legend(fontsize=8)
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smooth_flow.json")
    parser.add_argument("--checkpoint", default="data/checkpoints/smooth_gmm.pt")
    parser.add_argument("--out", default="results/smooth")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--verify-training-replay", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.verify_training_replay:
            verify_training_replay(args)
        else:
            run(args)
