#!/usr/bin/env python3
"""Run the compiled learned-state inference experiment and its classical controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3, qpy
from qiskit.quantum_info import Operator, Statevector
from scipy.stats import binomtest, norm
from threadpoolctl import threadpool_limits
import torch

from floworaclebench.evolution import state_metrics
from floworaclebench.inference import (
    box_event, compile_flow_preparation, execute_amplification, lookup_event,
    probability, repeated_circuit_resources, sampling_benchmark,
    two_site_contraction, zero_reflection,
)
from floworaclebench.resources import estimate_resources
from floworaclebench.smooth_flow import (
    load_smooth_potential, particle_flow, reverse_density, sample_source, wavefunction_reference,
)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _classical_ode_control(potential, n: int, lo: int, sample_count: int, seed: int) -> tuple[dict, dict]:
    """Actually integrate independent particles, with a paired RK4 refinement."""
    source = sample_source(potential.config, sample_count, seed)
    start = perf_counter()
    coarse = particle_flow(potential, source, steps=64)
    coarse_seconds = perf_counter() - start
    start = perf_counter()
    samples = particle_flow(potential, source, steps=128)
    fine_seconds = perf_counter() - start
    spacing = potential.length / n

    def event(points):
        # C-order coordinate bins: nearest periodic grid point.  Boundaries
        # have probability zero for the continuous source/model.
        indices = np.floor((points + potential.length / 2) / spacing + 0.5).astype(np.int64) % n
        return np.all(indices >= lo, axis=1)

    selected, selected_coarse = event(samples), event(coarse)
    successes = int(selected.sum())
    interval = binomtest(successes, sample_count).proportion_ci(confidence_level=0.95, method="exact")
    cell_lower = -potential.length / 2 + (lo - 0.5) * spacing
    cell_upper = potential.length / 2 - 0.5 * spacing
    # The GMM is periodized in the reference.  Sum independent 1D cell masses
    # over five image boxes.  The omitted tails are negligible for this fixed
    # benchmark; record a union bound rather than calling truncation exact.
    center = potential.config.centers
    sigma = potential.config.target_std
    one_axis = np.zeros_like(center)
    for shift in range(-2, 3):
        one_axis += norm.cdf((cell_upper + shift * potential.length - center) / sigma)
        one_axis -= norm.cdf((cell_lower + shift * potential.length - center) / sigma)
    target_probability = float(np.dot(np.asarray(potential.config.weights), np.prod(one_axis, axis=1)))
    tail_z = (2 * potential.length - np.max(abs(center))) / sigma
    log_tail_bound = float(np.log(4) + norm.logsf(tail_z))
    # Preserve a positive conservative displayed bound when float64 exp
    # underflows; zero would not be a mathematical upper bound on a tail.
    omitted_tail_bound = float(max(np.exp(log_tail_bound), 1e-300))
    info = {
        "samples": sample_count, "successes": successes, "estimate": successes / sample_count,
        "clopper_pearson_95_interval": [float(interval.low), float(interval.high)],
        "interval_covers": "Monte Carlo uncertainty for the specified RK4 pushforward; not model/discretization bias",
        "rk4_fine_steps": 128, "rk4_coarse_steps": 64,
        "fine_ode_batch_seconds": fine_seconds, "coarse_ode_batch_seconds": coarse_seconds,
        "fine_seconds_per_particle_at_measured_batch_size": fine_seconds / sample_count,
        "same_source_rk4_refinement_rms_endpoint_change": float(np.sqrt(np.mean(np.sum((samples - coarse)**2, axis=1)))),
        "same_source_event_classifications_changed": int(np.sum(selected != selected_coarse)),
        "event": "nearest periodic grid point has both coordinate indices >= lower index",
        "equivalent_cell_interval_per_coordinate": [cell_lower, cell_upper],
        "equivalent_cell_interval_is_periodic": True,
        "target_GMM_probability_of_same_periodic_cell_event": target_probability,
        "target_GMM_omitted_periodization_tail_union_bound": omitted_tail_bound,
        "target_GMM_tail_union_bound_log10": log_tail_bound / np.log(10),
        "target_GMM_displayed_tail_bound_floor": 1e-300,
        "target_probability_is_learned_model_probability": False,
    }
    return info, {"ode_source": source, "ode_final": samples, "ode_event_indicators": selected}


def continuum_cell_reference(potential, n: int, lo: int) -> dict[str, Any]:
    """Integrate the same learned-flow cell probability by backward characteristics.

    Product Gauss--Legendre quadrature is refined on the fixed physical cell
    rectangle.  A second RK4 step count separates temporal from quadrature
    error.  These are numerical convergence checks, not interval certificates.
    """
    spacing = potential.length / n
    lower = -potential.length / 2 + (lo - 0.5) * spacing
    upper = potential.length / 2 - spacing / 2
    if not 0 <= lo < n:
        raise ValueError("nonempty nonwrapping cell event required")
    start = perf_counter()
    rows = []
    for order, steps in [(16, 128), (24, 128), (32, 128), (32, 256)]:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        nodes = (lower + upper) / 2 + (upper - lower) / 2 * nodes
        weights = (upper - lower) / 2 * weights
        points = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
        density, metadata = reverse_density(potential, points, steps=steps)
        value = float(np.sum(density.reshape(order, order) * weights[:, None] * weights[None, :]))
        rows.append({"quadrature_nodes_per_axis": order, "reverse_RK4_steps": steps,
                     "probability": value, **metadata})
    return {
        "method": "backward learned-flow characteristics with product Gauss-Legendre quadrature",
        "cell_interval_per_axis": [lower, upper], "refinements": rows,
        "probability": rows[-1]["probability"],
        "last_spatial_quadrature_change": abs(rows[-2]["probability"] - rows[-3]["probability"]),
        "last_reverse_RK4_change": abs(rows[-1]["probability"] - rows[-2]["probability"]),
        "certified_error_bound": False, "total_seconds": perf_counter() - start,
    }


def plot_results(output: Path, amplification: list[dict], estimates: list[dict],
                 refinement: list[dict], tn: list[dict], bias: float,
                 continuum_bias: float | None = None) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    x = [row["k"] for row in amplification]
    ax.plot(x, [row["amplification_identity_prediction"] for row in amplification], "-", color="#666666", label="Exact amplification identity")
    ax.plot(x, [row["gate_executed_event_probability"] for row in amplification], "o", color="#155e75", label="Native-gate statevector")
    ax.set(xlabel="Amplification iterations k", ylabel="Marked-event probability", title="Whole learned circuit and its inverse")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for method, label, color in [
        ("maximum_likelihood_AE", "Likelihood AE", "#155e75"),
        ("Monte_Carlo_equal_A_query_budget", "MC, equal query budget", "#b45309"),
    ]:
        rows = [row for row in estimates if row["method"] == method]
        ax.errorbar([row["paid_preparation_queries_per_repetition"] for row in rows],
                    [row["rmse"] for row in rows], yerr=[row["rmse_standard_error_delta_method"] for row in rows],
                    fmt="o-", capsize=2, color=color, label=label)
    if bias > 0:
        ax.axhline(bias, color="#b91c1c", linestyle="--", label="Product-formula event bias")
    if continuum_bias is not None and continuum_bias > 0:
        ax.axhline(continuum_bias, color="#7e22ce", linestyle=":", label="Total bias to learned continuum event")
    ax.set(xscale="log", yscale="log", xlabel="Paid A or A† queries", ylabel="Empirical RMSE to prepared probability",
           title=f"Shot noise; {estimates[0]['independent_repetitions']} repeated experiments")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.loglog([row["steps"] for row in refinement], [row["state_l2_error"] for row in refinement], "o-", label="State error to same-grid ODE")
    ax.loglog([row["steps"] for row in refinement], [row["event_probability_abs_error"] for row in refinement], "s-", label="Event-probability bias")
    ax.set(xlabel="Commutator steps", ylabel="Absolute error", title="Own compiled-preparation refinement")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.semilogy([row["rank"] for row in tn], [max(1e-16, row["event_absolute_error"]) for row in tn], "o-", color="#15803d", label="Measured event error")
    ax.semilogy([row["rank"] for row in tn], [max(1e-16, row["pure_state_trace_distance_event_bound"]) for row in tn], "--", color="#64748b", label="Pure-state trace-distance bound")
    ax.set(xlabel="Two-site amplitude MPS rank", ylabel="Absolute event error", title="Classical tensor contraction")
    ax.legend(fontsize=8)
    fig.suptitle("FlowOracleBench: inference costs and persistent preparation bias", fontsize=14)
    fig.savefig(output / "inference_summary.png", dpi=180)
    plt.close(fig)


def run(args) -> None:
    start = perf_counter()
    config = json.loads((ROOT / args.config).read_text())
    if config["final_time"] != 1.0:
        raise ValueError("the inference driver currently requires final_time=1 for matching reference and particle controls")
    if config["depths"] != sorted(set(config["depths"])) or config["depths"][0] != 0:
        raise ValueError("inference depths must be strictly increasing and start at zero")
    output = ROOT / args.out
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / config["checkpoint"]
    potential = load_smooth_potential(checkpoint)
    n, steps, lo = int(config["grid"]), int(config["steps"]), int(config["event_lower_index"])
    intervals = [(lo, n), (lo, n)]
    mask, structured_event = box_event(n, intervals)
    generic_event = lookup_event(mask)
    event_candidates = [("coordinate_prefix", structured_event), ("Walsh_mask", generic_event)]
    event_rows = []
    for method, circuit in event_candidates:
        desired = np.diag(1 - 2 * mask.astype(float))
        error = float(np.linalg.norm(Operator(circuit).data - desired, ord=2))
        if error > 2e-11:
            raise AssertionError("compiled event does not mark exactly the specified addresses")
        event_rows.append({"observable": "two_coordinate_box", "method": method, "marked_addresses": int(mask.sum()),
                           "operator_error": error, **estimate_resources(circuit, circuit.num_qubits, 1e-6)})
    best = min(range(len(event_candidates)), key=lambda j: event_rows[j]["t_count_estimate"])
    event_method, event = event_candidates[best]
    # Observable-structure ablations retain both implementations where available.
    axis_mask, axis_event = box_event(n, [(lo, n), (0, n)])
    for method, circuit in [("coordinate_prefix", axis_event), ("Walsh_mask", lookup_event(axis_mask))]:
        event_rows.append({"observable": "one_coordinate_tail", "method": method, "marked_addresses": int(axis_mask.sum()),
                           **estimate_resources(circuit, circuit.num_qubits, 1e-6)})
    rng = np.random.default_rng(config["seed"] + 99)
    irregular = np.zeros(n * n, dtype=bool)
    irregular[rng.choice(n * n, int(mask.sum()), replace=False)] = True
    irregular_circuit = lookup_event(irregular)
    event_rows.append({"observable": "seeded_irregular_mask_same_support_size", "method": "Walsh_mask", "marked_addresses": int(irregular.sum()),
                       "selection_seed": int(config["seed"] + 99),
                       **estimate_resources(irregular_circuit, irregular_circuit.num_qubits, 1e-6)})
    print(f"Inference: grid {n}², {steps} flow steps, event implementation {event_method}", flush=True)
    reference_start = perf_counter()
    reference, reference_metadata = wavefunction_reference(potential, n, rtol=1e-11, atol=1e-13)
    reference_seconds = perf_counter() - reference_start
    p_reference = probability(reference, mask)
    refined_reference, _ = wavefunction_reference(potential, n, rtol=2e-12, atol=2e-14)
    reference_metadata["tolerance_refinement_l2_change"] = float(np.linalg.norm(reference - refined_reference))
    reference_metadata["event_probability"] = p_reference
    refinement = []
    prepared = None
    for count in config["time_refinement_steps"]:
        candidate = compile_flow_preparation(potential, n, int(count), final_time=config["final_time"])
        row = {"steps": int(count), **candidate.metadata, **state_metrics(candidate.block_state, reference),
               "event_probability": probability(candidate.block_state, mask),
               "event_probability_abs_error": abs(probability(candidate.block_state, mask) - p_reference),
               "resource_counts": estimate_resources(candidate.circuit, candidate.circuit.num_qubits, 1e-3),
               "refinement_state_source": "same formula as compiled circuit; primary row independently executed gate by gate"}
        refinement.append(row)
        if count == steps:
            prepared = candidate
        print(f"  steps {count}: event bias {row['event_probability_abs_error']:.6g}, state error {row['state_l2_error']:.6g}", flush=True)
    if prepared is None:
        raise ValueError("primary steps must appear in the refinement schedule")
    write_rows(output / "preparation_refinement.csv", refinement)
    print(f"Executing native A/A† amplification through k={max(config['depths'])}; {len(prepared.circuit.data):,} gates in A", flush=True)
    state, amplification = execute_amplification(prepared.circuit, event, mask, config["depths"])
    circuit_vs_block = float(np.linalg.norm(state - prepared.block_state))
    if circuit_vs_block > 2e-9:
        raise AssertionError(f"full native circuit disagrees with block propagation: {circuit_vs_block}")
    if max(row["amplification_identity_abs_error"] for row in amplification) > 2e-8:
        raise AssertionError("executed amplification violates the identity beyond roundoff tolerance")
    p_prepared = probability(state, mask)
    print(f"Native prepared event p={p_prepared:.10f}; discrete reference p={p_reference:.10f}", flush=True)
    # Actual categorical Monte Carlo from the computed distribution, including
    # CDF preparation, is a separate executed classical inference baseline.
    probability_table = np.abs(state)**2
    probability_table /= probability_table.sum()
    rng = np.random.default_rng(config["seed"] + 200)
    categorical_samples = int(config["shots_per_depth"] * sum(2 * np.asarray(config["depths"]) + 1))
    sample_start = perf_counter()
    cdf = np.cumsum(probability_table)
    sampled_indices = np.searchsorted(cdf, rng.random(categorical_samples), side="right")
    categorical_seconds = perf_counter() - sample_start
    sample_successes = int(mask[sampled_indices].sum())
    categorical_interval = binomtest(sample_successes, categorical_samples).proportion_ci(confidence_level=0.95, method="exact")
    dense_start = perf_counter()
    exact_dense = float(np.sum(probability_table[mask]))
    dense_sum_seconds = perf_counter() - dense_start
    estimates, shot_arrays = sampling_benchmark(
        [row["gate_executed_event_probability"] for row in amplification], config["depths"],
        shots_per_depth=config["shots_per_depth"], repetitions=config["independent_repetitions"],
        seed=config["seed"], reference_probability=p_reference,
    )
    cost_rows = []
    for row in estimates:
        if row["method"] == "maximum_likelihood_AE":
            cost_rows.append({"schedule": row["schedule"], "empirical_rmse": row["rmse"],
                              **repeated_circuit_resources(prepared.circuit, event, row["schedule"], config["shots_per_depth"],
                                                           total_rotation_budget=config["total_rotation_synthesis_budget"])})
    print("Running actual classical RK4 particle baseline and same-source refinement", flush=True)
    classical_ode, particle_arrays = _classical_ode_control(potential, n, lo, config["classical_particle_count"], config["seed"] + 301)
    continuum = continuum_cell_reference(potential, n, lo)
    continuum["finite_grid_reference_probability_bias"] = p_reference - continuum["probability"]
    continuum["compiled_probability_bias"] = p_prepared - continuum["probability"]
    continuum["learned_model_minus_target_GMM_cell_probability"] = continuum["probability"] - classical_ode["target_GMM_probability_of_same_periodic_cell_event"]
    for prefix in range(1, len(config["depths"]) + 1):
        for method, key in [("maximum_likelihood_AE", "mle"), ("Monte_Carlo_equal_A_query_budget", "mc")]:
            row = next(row for row in estimates if row["method"] == method and len(row["schedule"]) == prefix)
            samples = shot_arrays[f"{key}_prefix{prefix}"]
            row["rmse_to_continuum_cell_reference"] = float(np.sqrt(np.mean((samples - continuum["probability"])**2)))
            row["rmse_to_target_GMM_cell_probability"] = float(np.sqrt(np.mean((samples - classical_ode["target_GMM_probability_of_same_periodic_cell_event"])**2)))
    tn = two_site_contraction(state, n, intervals, config["tensor_ranks"])
    tensor_reference = two_site_contraction(reference / np.linalg.norm(reference), n, intervals, config["tensor_ranks"])
    final_stat = [row for row in estimates if row["method"] == "maximum_likelihood_AE"][-1]
    final_cost = cost_rows[-1]
    n_mc = int(final_stat["MC_draws_matching_empirical_MLAE_rmse_variance_formula"])
    # A necessary wall-clock crossover condition, explicitly conditional on
    # rates/accuracy/workload.  It is not a hardware runtime prediction.
    extrapolated_classical = n_mc * classical_ode["fine_seconds_per_particle_at_measured_batch_size"]
    crossover = {
        "empirical_MLAE_RMSE_to_prepared_probability": final_stat["rmse"],
        "Bernoulli_MC_samples_matching_this_variance_at_same_p": n_mc,
        "comparison_is_failure_probability_guaranteed": False,
        "current_compiled_probability_bias_to_continuum": continuum["compiled_probability_bias"],
        "current_continuum_bias_exceeds_empirical_shot_RMSE": abs(continuum["compiled_probability_bias"]) > final_stat["rmse"],
        "current_quantum_vs_classical_total_error_is_matched": False,
        "classical_ODE_batch_throughput_seconds_per_sample": classical_ode["fine_seconds_per_particle_at_measured_batch_size"],
        "extrapolated_ODE_seconds_for_variance_matched_samples": extrapolated_classical,
        "quantum_total_T_count_estimate": final_cost["t_count_estimate"],
        "necessary_seconds_per_effective_serial_T_if_all_other_quantum_costs_zero": extrapolated_classical / final_cost["t_count_estimate"],
        "symbolic_condition": "N_A * time(A) + N_E * time(event) + N_0 * time(S0) + measurement/reset + classical MLE < N_MC * time(classical ODE draw)",
        "conditions": [
            "equal target observable and total approximation error are required before this comparison is meaningful",
            "variance matching uses the measured prepared p and empirical MLAE RMSE, not a finite-shot success theorem",
            "batch ODE throughput is extrapolated linearly only as a sensitivity reference",
            "T costs use the documented prospective synthesis model, not synthesized rotations",
            "serial-T threshold is a conditional conversion factor, not a fault-tolerant architecture or clock model",
            "classical dense/TT construction and contraction are available and faster on this small grid",
        ],
    }
    spacing = potential.length / n
    metadata = {
        "config": config, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "command": " ".join(sys.argv), "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "qiskit", "torch", "matplotlib"]},
                        "numerical_threads": 1},
        "event": {"definition": "indicator that both discrete coordinate indices lie in [lower_index,n)",
                  "grid": n, "lower_index": lo, "grid_node_interval": [-potential.length / 2 + lo * spacing, potential.length / 2],
                  "marked_addresses": np.flatnonzero(mask).tolist(), "bounded_observable_range": [0, 1],
                  "rare_event_claimed": False, "selected_compiler": event_method,
                  "selection_rule": "minimum estimated T among the two executed native event implementations at a fixed 1e-6 rotation budget"},
        "preparation": {**prepared.metadata, "native_gate_resources": estimate_resources(prepared.circuit, prepared.circuit.num_qubits, 1e-3),
                        "full_native_vs_independent_block_l2_error": circuit_vs_block,
                        "native_state_vs_reference": state_metrics(state, reference),
                        "prepared_event_probability": p_prepared,
                        "finite_grid_reference_probability": p_reference,
                        "persistent_preparation_probability_bias": p_prepared - p_reference,
                        "persistent_continuum_cell_probability_bias": continuum["compiled_probability_bias"],
                        "controlled_preparation_used": False, "all_workspace_qubits": 0},
        "reference": {**reference_metadata, "measured_reference_construction_seconds": reference_seconds},
        "amplification": amplification, "likelihood_statistics": estimates,
        "complete_repeated_query_resources": cost_rows,
        "classical_same_prepared_distribution": {
            "dense_probability_sum": exact_dense, "dense_probability_sum_seconds": dense_sum_seconds,
            "dense_statevector_setup_seconds": amplification[0]["cumulative_native_statevector_seconds"],
            "categorical_samples": categorical_samples, "categorical_event_successes": sample_successes,
            "categorical_estimate": sample_successes / categorical_samples,
            "categorical_95_exact_binomial_interval": [float(categorical_interval.low), float(categorical_interval.high)],
            "cdf_setup_and_actual_categorical_sampling_seconds": categorical_seconds,
            "distribution_setup_is_paid_separately": True,
        },
        "classical_ODE_Monte_Carlo": classical_ode,
        "classical_continuum_cell_reference": continuum,
        "prepared_state_tensor_contraction": tn,
        "reference_state_tensor_contraction": tensor_reference,
        "event_structure_ablations": event_rows,
        "preparation_refinement": refinement,
        "crossover_conditions": crossover,
        "interpretation": {
            "experiment_type": "native-gate noiseless classical statevector execution plus simulated independent measurement counts",
            "hardware_execution": False, "controlled_A_count": 0,
            "estimated_T_counts_are_measured_Clifford_T_sequences": False,
            "inference_bias": "MLAE estimates the same compiled prepared-state probability; additional shots do not remove model, spatial, or product-formula bias",
            "synthesis_error": "prospective rotation implementation error is separately budgeted over repeated gate occurrences",
            "classical_dominance_scope": "this small materialized finite-grid workload; not an asymptotic high-dimensional theorem",
            "continuum_vs_grid": "ODE samples are quantized to nearest periodic grid nodes; coarse-grid Born masses are only approximations to those cell probabilities",
        },
        "total_experiment_seconds": perf_counter() - start,
    }
    write_json(output / "results.json", metadata)
    write_json(output / "config.json", config)
    write_rows(output / "amplification.csv", amplification)
    write_rows(output / "likelihood_statistics.csv", estimates)
    write_rows(output / "complete_resources.csv", cost_rows)
    write_rows(output / "tensor_contraction.csv", tn)
    write_rows(output / "event_resources.csv", event_rows)
    write_json(output / "continuum_cell_reference.json", continuum)
    np.savez_compressed(output / "states_and_samples.npz", prepared_state=state, reference_state=reference,
                        event_mask=mask, actual_categorical_indices=sampled_indices, **shot_arrays, **particle_arrays)
    with (output / "prepared_flow.qpy").open("wb") as handle:
        qpy.dump(prepared.circuit, handle)
    (output / "event_oracle.qasm").write_text(qasm3.dumps(event))
    (output / "zero_reflection.qasm").write_text(qasm3.dumps(zero_reflection(prepared.circuit.num_qubits)))
    plot_results(output, amplification, estimates, refinement, tn, abs(p_prepared - p_reference),
                 abs(continuum["compiled_probability_bias"]))
    print(f"Saved inference results; final MLAE RMSE {final_stat['rmse']:.6g}, full experiment T estimate {final_cost['t_count_estimate']:,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.json")
    parser.add_argument("--out", default="results/inference")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(arguments)
