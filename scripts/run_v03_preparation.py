#!/usr/bin/env python3
"""Compile all midpoint oracles, compose their certified clean actions, and cost A/A†."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from run_v03_oracles import ROOT, _upward_float, certified_phase_error, exact_monomial_phases
from run_inference import _classical_ode_control, continuum_cell_reference

from floworaclebench.baselines import build_walsh_oracle
from floworaclebench.benchmark import dump_json, write_rows
from floworaclebench.evolution import apply_product_formula, build_kinetic_circuit, state_metrics
from floworaclebench.inference import box_event, compile_initial_state, two_site_contraction, zero_reflection
from floworaclebench.precision_tt import build_precision_tt_oracle, plan_precision_tt
from floworaclebench.resources import estimate_resources
from floworaclebench.reversible import build_tt_oracle, contract_cores, quantize_cores
from floworaclebench.smooth_flow import load_smooth_potential, wavefunction_reference
from floworaclebench.spectral import build_fourier_oracle


def _exact_grid_error(integers, fraction_bits, reference, beta):
    denominator = 1 << fraction_bits
    return max(abs(beta * Fraction(int(integer), denominator) - truth)
               for integer, truth in zip(integers.flat, reference, strict=True))


def _oracle_candidates(potential, time, n, beta, values, budget):
    """Choose the first feasible uniform bits and screen a small mixed-width menu."""
    beta_exact = Fraction(beta)
    reference = tuple(beta_exact * Fraction(float(v)) for v in values)
    start = perf_counter()
    cores = potential.cores(time, n)  # Full available matrix rank in this experiment.
    for bits in range(25):
        integers = contract_cores(quantize_cores(cores, bits))
        if _exact_grid_error(integers, 2 * bits, reference, beta_exact) <= budget * Fraction(999, 1000):
            break
    else:
        raise ValueError("Uniform precision search did not meet its midpoint budget")
    baseline = build_tt_oracle(cores, beta, bits, materialize_values=False)

    plans = []
    for first, second in sorted({(bits, bits), (max(0, bits - 2), bits + 2),
                                (bits + 2, max(0, bits - 2)), (bits + 1, bits + 1)}):
        for final in sorted({first + second, min(first + second, bits + 4)}):
            plan = plan_precision_tt(cores, (first, second), (first, final))
            error = _exact_grid_error(plan.evaluate_exact(), final, reference, beta_exact)
            error += plan.phase_parameter_error_bound(beta_exact)
            if error <= budget:
                # Deterministic, inexpensive selection proxy for the common
                # two-site shape. No global circuit optimality is asserted.
                raw = plan.raw_widths[-1][0]
                key = (raw * second, sum(plan.retained_widths[0]) + raw, final, first, second)
                plans.append((key, plan))
    if not plans:
        raise ValueError("No precision-aware plan meets its midpoint budget")
    plan = min(plans, key=lambda item: item[0])[1]
    precision = build_precision_tt_oracle(plan, beta, materialize_values=False)
    preparation = perf_counter() - start
    return reference, {
        "uniform_tt": (baseline, {"core_bits": [bits, bits], "stage_bits": [bits, 2 * bits]}),
        "precision_tt": (precision, {"core_bits": plan.core_fraction_bits,
                                     "stage_bits": plan.stage_fraction_bits}),
        "walsh": (build_walsh_oracle(values, beta), {}),
        "direct_fourier": (build_fourier_oracle(potential, time, n, beta), {}),
    }, preparation


def _resource_total(blocks, global_budget):
    """Same rotation policy as the oracle benchmark, uniform across block occurrences."""
    counts = Counter()
    for resource, multiplicity in blocks:
        for gate, count in resource["native_gate_counts"].items():
            counts[gate] += multiplicity * count
    exact_budget = Fraction(str(global_budget))  # Same decimal convention as the error certificate.
    budget_per_block = exact_budget / sum(multiplicity for _, multiplicity in blocks)
    def special_error(resource):
        return Fraction(resource.get("special_rotation_replacement_error_bound_exact",
                                     resource.get("special_rotation_replacement_error_bound", 0.0)))
    if any(special_error(resource) >= budget_per_block for resource, _ in blocks):
        raise ValueError("Block synthesis allocation does not cover special-angle replacement")
    assigned = Fraction(0)
    t_count = 0
    for resource, multiplicity in blocks:
        arbitrary = resource["arbitrary_rotations"]
        cost = resource["t_count_estimate"]
        if arbitrary:
            slope, offset = resource["rotation_model_slope"], resource["rotation_model_offset"]
            old = math.ceil(slope * math.log2(1 / resource["per_rotation_error"]) + offset)
            special = special_error(resource)
            remaining = budget_per_block - special
            if remaining <= 0:
                raise ValueError("Block synthesis allocation does not cover special-angle replacement")
            per_rotation = float(remaining / arbitrary)
            if Fraction(per_rotation) > remaining / arbitrary:
                per_rotation = math.nextafter(per_rotation, 0.0)
            new = math.ceil(slope * math.log2(1 / per_rotation) + offset)
            assigned += multiplicity * arbitrary * Fraction(per_rotation)
            cost += arbitrary * (new - old)
        assigned += multiplicity * special_error(resource)
        t_count += multiplicity * cost
    if assigned > exact_budget:
        raise AssertionError("Aggregate synthesis allocation exceeded exact declared budget")
    return {
        "logical_qubits": max(resource["logical_qubits"] for resource, _ in blocks),
        "clean_ancillas": max(resource["ancilla_qubits"] for resource, _ in blocks),
        "dirty_ancillas": 0,
        "native_gates": sum(counts.values()),
        "ccx": counts["ccx"],
        "native_gate_counts": dict(counts),
        "serial_native_depth_upper_bound": sum(resource["native_depth"] * multiple for resource, multiple in blocks),
        "arbitrary_rotations": sum(resource["arbitrary_rotations"] * multiple for resource, multiple in blocks),
        "t_count_estimate": t_count,
        "allocated_rotation_synthesis_budget": global_budget,
        "allocated_rotation_synthesis_budget_exact": str(exact_budget),
        "used_rotation_synthesis_allowance_exact": str(assigned),
        "special_rotation_replacement_error_bound": _upward_float(sum(
            special_error(resource) * multiple for resource, multiple in blocks)),
        "rotation_synthesis_executed": False,
    }


def _centered_step_bound(angles, reference, uncentered):
    """The two P/P† pairs cancel a common phase shift exactly."""
    differences = [actual - target for actual, target in zip(angles, reference, strict=True)]
    center = (max(differences) + min(differences)) / 2
    centered = certified_phase_error(angles, (target + center for target in reference))
    return min(Fraction(2), 4 * min(uncentered, centered))


def run_case(potential, case, config):
    n, steps = case["grid"], case["steps"]
    dt = 1 / steps
    n_address = 2 * (n.bit_length() - 1)
    budget = Fraction(str(config["native_potential_budget"]))
    if not 0 < budget < 1:
        raise ValueError("Native preparation error budget must be in (0, 1)")
    per_oracle = budget / (4 * steps)
    initial_circuit, ideal_initial, source_meta = compile_initial_state(potential, n)
    source = Statevector.from_instruction(initial_circuit).data
    sampled = [potential.values(float(time), n) for time in np.linspace(0, 1, 17)]
    # Same balancing rule as v0.2. The balance samples are a schedule choice.
    potential_half_range = max(float(np.ptp(values)) / 2 for values in sampled)
    kinetic_half_range = 2 * np.pi**2 * n**2 / (4 * potential.length**2)
    if potential_half_range <= 0:
        raise ValueError("The benchmark expects a nonconstant learned potential")
    alpha = math.sqrt(dt * potential_half_range / (2 * kinetic_half_range))
    beta = math.sqrt(dt * kinetic_half_range / (2 * potential_half_range))
    kinetic_circuit = build_kinetic_circuit(n, 2, potential.length, alpha)
    kinetic = Operator(kinetic_circuit).data
    source_resource = estimate_resources(initial_circuit, n_address)
    kinetic_resource = estimate_resources(kinetic_circuit, n_address)
    blocks = {method: [(source_resource, 1), (kinetic_resource, 4 * steps)]
              for method in ("uniform_tt", "precision_tt", "walsh", "direct_fourier")}
    states = {method: source.copy() for method in blocks}
    full_circuits = {"walsh": QuantumCircuit(n_address), "direct_fourier": QuantumCircuit(n_address + 1)}
    for circuit in full_circuits.values():
        circuit.compose(initial_circuit, range(n_address), inplace=True)
    kinetic_inverse = kinetic_circuit.inverse()
    reference_state = source.copy()
    per_step = []
    bounds = Counter()
    start = perf_counter()
    for step in range(steps):
        time = (step + 0.5) * dt
        values = np.asarray(potential.values(time, n), dtype=float).ravel()
        reference, oracles, selection_seconds = _oracle_candidates(potential, time, n, beta, values, per_oracle)
        reference_state = apply_product_formula(reference_state, np.exp(1j * np.asarray(reference, dtype=float)), kinetic)
        for method, (oracle, planning) in oracles.items():
            angles = exact_monomial_phases(oracle.circuit, n_address)
            error = certified_phase_error(angles, reference)
            if error > per_oracle:
                raise AssertionError(f"{method} failed its exact midpoint certificate at t={time}: {float(error)}")
            step_bound = _centered_step_bound(angles, reference, error)
            bounds[method] += step_bound
            states[method] = apply_product_formula(states[method], np.exp(1j * np.asarray(angles, dtype=float)), kinetic)
            if method in full_circuits:
                inverse = oracle.circuit.inverse()
                full = full_circuits[method]
                for kinetic_block, phase_block in ((kinetic_circuit, oracle.circuit), (kinetic_inverse, inverse),
                                                   (kinetic_inverse, inverse), (kinetic_circuit, oracle.circuit)):
                    full.compose(kinetic_block, range(n_address), inplace=True)
                    full.compose(phase_block, range(phase_block.num_qubits), inplace=True)
            resources = estimate_resources(oracle.circuit, n_address)
            blocks[method].append((resources, 4))
            per_step.append({"grid": n, "steps": steps, "midpoint_index": step, "time": time,
                             "beta": beta, "method": method, "exhaustive_clean_addresses": len(angles),
                             "native_phase_bound": _upward_float(error), "native_phase_bound_exact": str(error),
                             "centered_step_bound": _upward_float(step_bound), "centered_step_bound_exact": str(step_bound),
                             "tt_candidate_selection_seconds_shared": selection_seconds,
                             **planning, **resources})
        if (step + 1) % 8 == 0 or step + 1 == steps:
            print(f"preparation n={n}, r={steps}: {step + 1}/{steps} midpoints", flush=True)

    lo = 5 * n // 8
    event_mask, event = box_event(n, [(lo, n), (lo, n)])
    same_grid, solver = wavefunction_reference(potential, n, rtol=1e-11, atol=1e-13)
    refined_grid, _ = wavefunction_reference(potential, n, rtol=2e-13, atol=1e-15)
    event_probability = lambda state: float(np.sum(np.abs(state[event_mask])**2))
    reference_p = event_probability(reference_state)
    grid_p = event_probability(refined_grid)
    classical, _ = _classical_ode_control(potential, n, lo, config["classical_particles"], config["seed"])
    continuum = continuum_cell_reference(potential, n, lo)
    tensor = two_site_contraction(reference_state, n, [(lo, n), (lo, n)], [1, 2, n])
    depths, shots = config["amplification_depths"], config["shots_per_depth"]
    inverse_calls = shots * sum(depths)
    forward_calls = shots * sum(depth + 1 for depth in depths)
    calls = forward_calls + inverse_calls
    event_resource = estimate_resources(event, n_address)
    # On the invariant clean subspace this address-only reflection is the
    # required reflection of |0_address,0_workspace>; extra wires stay clean.
    reflection_resource = estimate_resources(zero_reflection(n_address), n_address)
    summaries = []
    for method, state in states.items():
        resources = _resource_total(blocks[method], config["preparation_synthesis_budget"])
        repeated_blocks = [(resource, multiple * calls) for resource, multiple in blocks[method]]
        repeated_blocks.extend([(event_resource, inverse_calls), (reflection_resource, inverse_calls)])
        metrics = state_metrics(state, reference_state)
        if metrics["tv_distance"] > float(bounds[method]) + 5e-12:
            raise AssertionError("Numerical TV check exceeded certified native preparation bound")
        native_bound = bounds[method]
        synthesis_bound = min(Fraction(1), native_bound + Fraction(str(config["preparation_synthesis_budget"])))
        event_p = event_probability(state)
        event_theta = math.asin(math.sqrt(min(1.0, event_p)))
        reference_theta = math.asin(math.sqrt(min(1.0, reference_p)))
        amplification = []
        for depth in depths:
            query_count = 2 * depth + 1
            observed = math.sin(query_count * event_theta)**2
            reference_observed = math.sin(query_count * reference_theta)**2
            bound = min(Fraction(1), query_count * native_bound)
            if abs(observed - reference_observed) > float(bound) + 5e-12:
                raise AssertionError("Amplified event difference exceeded the native bound")
            amplification.append({"depth": depth, "preparation_calls": query_count,
                                  "event_probability_from_amplification_identity": observed,
                                  "reference_probability_from_amplification_identity": reference_observed,
                                  "event_bias_bound_to_reference_amplified_circuit": _upward_float(bound)})
        integration = None
        if method in full_circuits:
            full = full_circuits[method]
            prepared = Statevector.from_instruction(full)
            clean, leaked = prepared.data[: len(state)], prepared.data[len(state):]
            recovered = prepared.evolve(full.inverse()).data
            global_aligned_error = state_metrics(clean, state)["phase_aligned_l2_error"]
            clean_workspace_error = float(np.linalg.norm(leaked))
            inverse_error = float(np.linalg.norm(recovered - np.eye(1, len(recovered))[0]))
            if max(global_aligned_error, clean_workspace_error, inverse_error) > 2e-10:
                raise AssertionError(f"Full native {method} preparation or inverse failed validation")
            integration = {"full_simulated_qubits": full.num_qubits, "full_gate_count": len(full.data),
                           "full_native_depth": full.depth(), "clean_workspace_l2_error": clean_workspace_error,
                           "phase_aligned_l2_error_to_block_composition": global_aligned_error,
                           "full_inverse_l2_error_to_zero_state": inverse_error}
        summaries.append({"method": method, "event_probability": event_p,
                          "event_error_to_product_formula": abs(event_p - reference_p),
                          "native_state_and_tv_bound": _upward_float(native_bound),
                          "native_state_and_tv_bound_exact": str(native_bound),
                          "tv_bound_conditional_on_rotation_synthesis": _upward_float(synthesis_bound),
                          "measured_metrics_to_native_block_reference": metrics,
                          "full_native_gate_integration": integration,
                          "amplified_event_comparison": amplification,
                          "per_A_and_inverse_resource_accounting": resources,
                          "repeated_amplification_accounting": _resource_total(repeated_blocks, config["repeated_synthesis_budget"])})
    return {
        "grid": n, "steps": steps, "alpha": alpha, "beta": beta, "n_address": n_address,
        "phase_calls_per_A": 4 * steps, "kinetic_calls_per_A": 4 * steps,
        "source": source_meta, "source_norm_deviation": abs(float(np.linalg.norm(source)) - 1),
        "source_state_l2_difference_empirical": float(np.linalg.norm(source - ideal_initial)),
        "certificate_reference": "Exact binary64 grid phases and the same native source and kinetic blocks",
        "simulation": "Exact exhaustive monomial phase paths and dense logical source/kinetic composition; wide workspace statevector not allocated",
        "kinetic_source_and_formula_cost_scope": "Compiled native blocks, serial depth bound; A-dagger reverses the same blocks",
        "reference_event_probability": reference_p, "same_grid_ode_probability": grid_p,
        "product_formula_and_native_block_event_difference_empirical": reference_p - grid_p,
        "same_grid_ode_reference_refinement": state_metrics(same_grid, refined_grid),
        "same_grid_ode_solver": solver,
        "learned_continuum_cell_probability": continuum["probability"],
        "spatial_and_event_binning_difference_empirical": grid_p - continuum["probability"],
        "learned_vs_target_event_difference_empirical": continuum["probability"] - classical["target_GMM_probability_of_same_periodic_cell_event"],
        "classical_ode_and_sampling_control": classical, "continuum_refinement": continuum,
        "classical_two_site_amplitude_contraction": tensor,
        "amplification_schedule": {"depths": depths, "shots_per_depth": shots,
                                   "forward_A_calls": forward_calls, "inverse_A_calls": inverse_calls,
                                   "event_and_zero_reflection_calls_each": inverse_calls,
                                   "controlled_preparation_calls": 0},
        "methods": summaries, "total_seconds": perf_counter() - start,
    }, per_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v03_preparation.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/v03")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    checkpoint = ROOT / config["checkpoint"]
    potential = load_smooth_potential(checkpoint)
    cases, rows = [], []
    for case in config["cases"]:
        summary, midpoint_rows = run_case(potential, case, config)
        cases.append(summary)
        rows.extend(midpoint_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    dump_json(args.out / "preparation.json", {"config": config, "cases": cases,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    write_rows(args.out / "preparation_midpoints.csv", rows)


if __name__ == "__main__":
    main()
