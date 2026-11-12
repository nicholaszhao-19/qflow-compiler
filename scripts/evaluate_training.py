#!/usr/bin/env python3
"""Check learned velocity quality and a separate classical particle ODE.

These are diagnostics of training, not quantum-circuit accuracy or proof of
continuum convergence.  The conditional Gaussian-mixture bridge velocity is
available analytically; no learned teacher is used for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import softmax
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from floworaclebench.potentials import DEFAULT_CHECKPOINT, ensure_upstream, load_potential


def velocity(model, x: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
    z = x.detach().requires_grad_(True)
    if isinstance(t, float):
        t = torch.full((len(z), 1), t, dtype=z.dtype)
    value = model(z, t)
    return torch.autograd.grad(value.sum(), z)[0].detach()


def rk4(model, source: np.ndarray, steps: int) -> np.ndarray:
    x = torch.from_numpy(source.copy()).float()
    h = 1 / steps
    for k in range(steps):
        t = k * h
        k1 = velocity(model, x, float(t))
        k2 = velocity(model, x + (h/2)*k1, float(t+h/2))
        k3 = velocity(model, x + (h/2)*k2, float(t+h/2))
        k4 = velocity(model, x + h*k3, float(t+h))
        x = (x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)).detach()
    return x.numpy()


def conditional_velocity(x: np.ndarray, t: np.ndarray, centers: np.ndarray,
                         sigma0: float, sigma1: float) -> np.ndarray:
    """Gaussian conditioning for independent endpoint CFM pairs on R^d."""
    variance = (1-t)**2*sigma0**2 + t**2*sigma1**2
    covariance = t*sigma1**2 - (1-t)*sigma0**2
    residual = x[:, None, :] - t[:, :, None]*centers[None, :, :]
    weights = softmax(-np.sum(residual**2, axis=-1)/(2*variance), axis=1)
    components = centers[None, :, :] + (covariance/variance)[:, :, None]*residual
    return np.sum(weights[:, :, None]*components, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ensure_upstream()
    from tnwf.data.gaussian_mixture import gm_mode_centers, sample_gaussian_mixture
    from tnwf.metrics.sw import sliced_wasserstein

    start = time.perf_counter()
    potential = load_potential(args.checkpoint)
    model = potential.model
    d = potential.dimension
    sigma0 = potential.sigma0
    sigma1 = float(potential.config["std"])
    scale = float(potential.config["scale"])
    rng = np.random.default_rng(args.seed)
    source = rng.normal(0, sigma0, (args.samples, d)).astype(np.float32)
    target = sample_gaussian_mixture(args.samples, d=d, std=sigma1, scale=scale, seed=args.seed+1)
    target2 = sample_gaussian_mixture(args.samples, d=d, std=sigma1, scale=scale, seed=args.seed+2)
    t = rng.random((args.samples, 1)).astype(np.float32)
    x = (1-t)*source + t*target
    learned_v = velocity(model, torch.from_numpy(x), torch.from_numpy(t)).numpy()
    conditional_v = conditional_velocity(x, t, gm_mode_centers(d, scale=scale), sigma0, sigma1)
    endpoint_coarse = rk4(model, source, args.steps // 2)
    endpoint_fine = rk4(model, source, args.steps)

    def sw(a, b):
        return sliced_wasserstein(a, b, n_projections=200, rng=np.random.default_rng(args.seed+3))

    report = {
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "dimension": d, "samples": args.samples, "seed": args.seed,
        "rk4_steps": [args.steps // 2, args.steps], "sw_projections": 200,
        "fresh_pair_jam_loss": float(np.mean(0.5*np.sum(learned_v**2,axis=1)-np.sum(learned_v*(target-source),axis=1))),
        "velocity_mse_against_conditional_gaussian_bridge": float(np.mean(np.sum((learned_v-conditional_v)**2,axis=1))),
        "zero_velocity_mse_against_conditional_gaussian_bridge": float(np.mean(np.sum(conditional_v**2,axis=1))),
        "source_target_sw1": sw(source, target),
        "learned_ode_target_sw1": sw(endpoint_fine, target),
        "learned_ode_coarse_target_sw1": sw(endpoint_coarse, target),
        "target_target_sw1_sample_floor": sw(target, target2),
        "rk4_refinement_particle_rms": float(np.sqrt(np.mean(np.sum((endpoint_fine-endpoint_coarse)**2,axis=1)))),
        "source_outside_box_fraction": float(np.any(np.abs(source)>=potential.length/2,axis=1).mean()),
        "target_outside_box_fraction": float(np.any(np.abs(target)>=potential.length/2,axis=1).mean()),
        "ode_outside_box_fraction": float(np.any(np.abs(endpoint_fine)>=potential.length/2,axis=1).mean()),
        "elapsed_seconds": time.perf_counter()-start,
        "interpretation": "Classical training diagnostics only. The analytic bridge is unbounded; the learned model is periodic. Low-resolution quantum-grid dynamics need not match this particle ODE. RK4 refinement is empirical, not a rigorous continuum error bound."
    }
    destination = args.checkpoint.with_suffix(".quality.json")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.checkpoint.with_suffix(".quality.npz"),
                        source=source, target=target, target2=target2,
                        endpoint_coarse=endpoint_coarse, endpoint_fine=endpoint_fine,
                        bridge_points=x, bridge_times=t, learned_velocity=learned_v,
                        conditional_velocity=conditional_v)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
