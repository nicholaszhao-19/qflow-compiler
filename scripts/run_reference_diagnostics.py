#!/usr/bin/env python3
"""Reproduce numerical reference checks; these are not compiled-circuit results.

Compares DOP853 tolerances, two parameter choices for the established eight-
letter formula, and empirical grid/target diagnostics. The trained potential's
piecewise-linear spatial interpolation does not satisfy the smoothness
hypotheses needed to invoke a continuum convergence theorem.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.linalg import expm
from scipy.signal import resample
from threadpoolctl import threadpool_limits

from floworaclebench.evolution import (
    apply_product_formula, dense_reference, gaussian_state, kinetic_matrix, state_metrics,
)
from floworaclebench.potentials import UPSTREAM_COMMIT, centered_grid, load_potential


def grid_target_probability(points, centers, std, length, image_radius=0):
    """Normalized grid-point probabilities of an equal-weight Gaussian mixture.

    A positive image_radius sums periodic images in [-radius,radius]^d.
    This approximates the periodized density, not continuum cell integration.
    It should not be interpreted as an independently trained model density.
    """
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if points.ndim != 2 or centers.ndim != 2 or centers.shape[1] != points.shape[1]:
        raise ValueError("points and centers must be matrices with matching dimension")
    if not np.isfinite(std) or std <= 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("positive finite standard deviation and side length required")
    if not isinstance(image_radius, int) or image_radius < 0:
        raise ValueError("image_radius must be a nonnegative integer")
    values = np.zeros(len(points))
    images = range(-image_radius, image_radius + 1)
    for integer_shift in itertools.product(images, repeat=points.shape[1]):
        shift = length * np.asarray(integer_shift)
        distance2 = np.sum((points[:, None, :] - centers[None, :, :] + shift)**2, axis=-1)
        values += np.exp(-distance2 / (2 * std**2)).sum(axis=1)
    if not np.isfinite(values).all() or values.sum() == 0:
        raise ValueError("target density underflowed or contains nonfinite values")
    return values / values.sum()


def interpolate_state(state, old_n, new_n, dimension):
    """Fourier interpolate normalized collocation amplitudes to a finer grid.

    Includes the grid-spacing amplitude factor (old_n/new_n)^(d/2). SciPy's
    even-grid Nyquist splitting convention is used; this is a diagnostic
    interpolation, not a unitary embedding. Preserve the raw norm and let
    state_metrics normalize only its fidelity/TV calculations.
    """
    if new_n < old_n or old_n < 2 or dimension < 1:
        raise ValueError("a valid grid refinement is required")
    array = np.asarray(state, dtype=complex).reshape((old_n,) * dimension)
    for axis in range(dimension):
        array = resample(array, new_n, axis=axis)
    return array.reshape(-1) * (old_n / new_n)**(dimension / 2)


def kinetic_exponential(n, dimension, length, alpha):
    """Exact separable dense exp(i alpha K), without a full cubic matrix expm."""
    single = expm(1j * alpha * kinetic_matrix(n, 1, length))
    result = single
    for _ in range(dimension - 1):
        result = np.kron(result, single)
    return result


def run(config):
    start = perf_counter()
    path = Path(config["checkpoint"])
    if not path.is_absolute():
        path = ROOT / path
    potential = load_potential(path)
    from tnwf.data.gaussian_mixture import gm_mode_centers

    dimension = potential.dimension
    target_config = {name: potential.config[name] for name in ["d", "std", "scale", "arrangement"]}
    if "nmodes" in potential.config:
        target_config["nmodes"] = potential.config["nmodes"]
    centers = gm_mode_centers(
        d=dimension, scale=target_config["scale"], arrangement=target_config["arrangement"],
        nmodes=int(target_config.get("nmodes", 4)),
    )
    target_config["centers"] = centers.tolist()
    target_config["weights"] = (np.ones(len(centers)) / len(centers)).tolist()
    target_config["definition"] = "normalized sampled density at grid nodes, not integrated cell mass"
    states = {}
    refs = {}
    reference_rows, formula_rows, spatial_rows = [], [], []
    previous_n = None
    knot_times = np.linspace(0, 1, int(potential.config["N_t"]))
    final_time = float(config["final_time"])

    for n in config["grids"]:
        n = int(n)
        reference, metadata = dense_reference(potential, n, final_time=final_time,
                                              rtol=config["rtol"], atol=config["atol"])
        refined, refined_metadata = dense_reference(
            potential, n, final_time=final_time,
            rtol=config["refined_rtol"], atol=config["refined_atol"],
        )
        reference_rows.append({"n": n, "metadata": metadata, "refined_metadata": refined_metadata,
                               "refinement_metrics": state_metrics(reference, refined)})
        refs[n] = reference
        states[f"reference_n{n}"] = reference
        states[f"reference_refined_n{n}"] = refined
        initial = gaussian_state(n, dimension, potential.length, potential.sigma0)
        states[f"initial_n{n}"] = initial
        points = centered_grid(n, dimension, potential.length)
        target = grid_target_probability(points, centers, target_config["std"], potential.length)
        periodized = grid_target_probability(points, centers, target_config["std"], potential.length,
                                             image_radius=int(config["periodic_image_radius"]))
        states[f"target_probability_n{n}"] = target
        states[f"periodized_target_probability_n{n}"] = periodized
        spatial = {
            "n": n,
            "grid_target_tv": state_metrics(reference, np.sqrt(target))["tv_distance"],
            "periodized_grid_target_tv": state_metrics(reference, np.sqrt(periodized))["tv_distance"],
            "initial_grid_target_tv": state_metrics(initial, np.sqrt(target))["tv_distance"],
            "reference_norm_error": float(abs(np.linalg.norm(reference) - 1)),
        }
        if previous_n is not None:
            interpolated = interpolate_state(refs[previous_n], previous_n, n, dimension)
            states[f"interpolated_n{previous_n}_to_n{n}"] = interpolated
            spatial["coarse_n"] = previous_n
            spatial["interpolation_metrics"] = state_metrics(interpolated, reference)
            spatial["interpolation_interpretation"] = (
                "Fourier interpolation with SciPy even-grid Nyquist convention; "
                "fidelity and TV normalize inputs, raw norms retained. This is not a "
                "certified continuum error or a unitary cross-grid state embedding."
            )
        spatial_rows.append(spatial)
        previous_n = n
        potential_half_range = float(max(np.ptp(potential.values(float(t), n)) / 2 for t in knot_times))
        kinetic_half_range = float(dimension * np.pi**2 * n**2 / (4 * potential.length**2))
        if potential_half_range <= 0:
            raise ValueError("norm balancing requires a nonconstant spatial potential")

        for steps in config["steps"]:
            steps = int(steps)
            h = final_time / steps
            for rule in config["balance_rules"]:
                if rule == "paper":
                    alpha = potential.length / (np.pi * n) * np.sqrt(h / dimension)
                    beta = np.pi * n / (2 * potential.length) * np.sqrt(dimension * h)
                elif rule == "spectral_range":
                    alpha = np.sqrt(h * potential_half_range / (2 * kinetic_half_range))
                    beta = np.sqrt(h * kinetic_half_range / (2 * potential_half_range))
                else:
                    raise ValueError(f"Unknown balance rule: {rule}")
                kinetic = kinetic_exponential(n, dimension, potential.length, alpha)
                state = initial.copy()
                evolution_start = perf_counter()
                for step in range(steps):
                    t = (step + 0.5) * h
                    phases = np.exp(1j * beta * potential.values(float(t), n))
                    state = apply_product_formula(state, phases, kinetic)
                formula_rows.append({
                    "n": n, "steps": steps, "balance_rule": rule,
                    "alpha": float(alpha), "beta": float(beta),
                    "potential_half_range": potential_half_range,
                    "kinetic_half_range": kinetic_half_range,
                    "seconds": perf_counter() - evolution_start,
                    "norm_error": float(abs(np.linalg.norm(state) - 1)),
                    **state_metrics(state, reference),
                })
                states[f"formula_{rule}_n{n}_steps{steps}"] = state
        best = [row for row in formula_rows if row["n"] == n and row["balance_rule"] == "spectral_range"][-1]
        print(f"n={n}: dense norm drift {metadata['norm_drift']:.3g}; "
              f"{best['steps']}-step balanced fidelity {best['fidelity']:.9f}; "
              f"target-grid TV {spatial['grid_target_tv']:.6f}", flush=True)

    return {
        "schema_version": 1,
        "config": config,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "upstream_commit": UPSTREAM_COMMIT,
        "potential_config": potential.config,
        "source_sigma": potential.sigma0,
        "time_knots": knot_times.tolist(),
        "target": target_config,
        "method": {
            "reference": "segmented DOP853 on Fourier-collocation Hamiltonian i[K,V_t]",
            "formula": "Layden Eq. (36), midpoint V_t, exact real-angle diagonal phases",
            "kinetic_exponential": "Kronecker product of exact single-axis dense exponentials",
            "spectral_range_rule": "alpha=sqrt(h*a/(2*k)), beta=sqrt(h*k/(2*a)); "
                "a=max over time knots range(V_t)/2; k=d*pi^2*n^2/(4*L^2)",
            "spectral_range_bound": "range of an affine potential vector is convex between time knots; "
                "endpoint maximum bounds all times. Rule is independent of the initial state.",
            "compiled_circuits": False,
        },
        "interpretation": (
            "Numerical reference and exact-phase product-formula diagnostics only. "
            "These rows are not compiled fixed-point-oracle results. Training, "
            "finite-box effects and spatial discretization contribute jointly to "
            "target mismatch. The model is spatially piecewise linear, so the "
            "smoothness assumptions in Layden's continuum theorem have not been "
            "established. No continuum or target-distribution convergence is claimed."
        ),
        "environment": {"python": platform.python_version(),
                        **{package: importlib.metadata.version(package)
                           for package in ["numpy", "scipy", "torch", "threadpoolctl"]}},
        "reference": reference_rows, "formula": formula_rows, "spatial": spatial_rows,
        "elapsed_seconds": perf_counter() - start,
    }, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reference.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reference_diagnostics.json")
    parser.add_argument("--states-output", type=Path, default=ROOT / "results/reference_states.npz")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=int(config["blas_threads"])), np.errstate(invalid="raise", over="raise"):
        results, states = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.states_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.states_output, **states)
    print(f"Wrote {args.output} and {args.states_output}")


if __name__ == "__main__":
    main()
