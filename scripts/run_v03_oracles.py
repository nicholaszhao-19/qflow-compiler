#!/usr/bin/env python3
"""Certify and compare native reversible phase-oracle circuits on finite grids."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floworaclebench.baselines import build_qrom_oracle, build_walsh_oracle, select_phase_bits
from floworaclebench.benchmark import dump_json, write_rows
from floworaclebench.potentials import LearnedPotential, centered_grid, contract_cores, ensure_upstream
from floworaclebench.resources import estimate_resources
from floworaclebench.reversible import build_tt_oracle, quantize_cores
from floworaclebench.reversible import contract_cores as integer_contract
from floworaclebench.selectswap import build_selectswap_oracle
from floworaclebench.simulation import validate_oracle


@lru_cache(maxsize=1)
def _two_pi_interval() -> tuple[Fraction, Fraction]:
    """Enclose 2π via Machin's identity and alternating-series remainders."""
    def arctan_inverse(q: int) -> tuple[Fraction, Fraction]:
        terms = 32  # Even partial sum is the lower bound.
        lower = sum(Fraction((-1)**k, (2*k+1)*q**(2*k+1)) for k in range(terms))
        upper = lower + Fraction(1, (2*terms+1)*q**(2*terms+1))
        return lower, upper

    a, b = arctan_inverse(5), arctan_inverse(239)
    return 32*a[0] - 8*b[1], 32*a[1] - 8*b[0]


def certified_phase_error(phases, reference) -> Fraction:
    """Rational upper bound on the operator norm, including global phase."""
    lower, upper = _two_pi_interval()
    midpoint = (lower + upper) / 2
    maximum = Fraction(0)
    for phase, target in zip(phases, reference, strict=True):
        delta = Fraction(phase) - Fraction(target)
        winding = round(delta / midpoint)
        bound = min(abs(delta), max(abs(delta - winding*lower), abs(delta - winding*upper)))
        maximum = max(maximum, min(Fraction(2), bound))
    return maximum


def _upward_float(value: Fraction) -> float:
    estimate = float(value)
    return math.nextafter(estimate, math.inf) if Fraction(estimate) < value else estimate


def exact_monomial_phases(circuit, n_address: int) -> tuple[Fraction, ...]:
    """Evaluate exact dyadic angles on all clean inputs and verify cleanup.

    Gates are interpreted using their stored binary64 angles as exact reals.
    In particular, Rz(a) contributes a(b−1/2), and P(a) contributes ab.
    The path proof is exact over Boolean states, without dense statevectors.
    """
    addresses = np.arange(1 << n_address, dtype=np.int64)
    input_bits = ((addresses[None, :] >> np.arange(n_address)[:, None]) & 1).astype(bool)
    bits = np.zeros((circuit.num_qubits, len(addresses)), dtype=bool)
    bits[:n_address] = input_bits
    angle_steps = []
    global_angle = Fraction(float(circuit.global_phase))
    qindex = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for item in circuit.data:
        gate = item.operation.name
        qubits = [qindex[qubit] for qubit in item.qubits]
        if gate == "x":
            bits[qubits[0]] ^= True
        elif gate == "cx":
            bits[qubits[1]] ^= bits[qubits[0]]
        elif gate == "ccx":
            bits[qubits[2]] ^= bits[qubits[0]] & bits[qubits[1]]
        elif gate in {"p", "rz", "u1"}:
            angle = Fraction(float(item.operation.params[0]))
            if gate == "rz":
                global_angle -= angle / 2
            angle_steps.append((angle, bits[qubits[0]].copy()))
        elif gate not in {"barrier", "id"}:
            raise ValueError(f"Unsupported gate for exact monomial analysis: {gate}")
    if not np.array_equal(bits[:n_address], input_bits) or bits[n_address:].any():
        raise AssertionError("Circuit does not preserve addresses with clean ancillas")
    # All angle denominators are powers of two, so use a single exact denominator.
    denominator = max([global_angle.denominator, *(angle.denominator for angle, _ in angle_steps)])
    numerators = np.full(len(addresses), int(global_angle * denominator), dtype=object)
    for angle, selected in angle_steps:
        numerators[selected] += int(angle * denominator)
    return tuple(Fraction(value, denominator) for value in numerators)


def exact_tt_values(cores) -> tuple[Fraction, ...]:
    """Contract supplied binary64 core coefficients as exact dyadic numbers."""
    denominators = []
    integer_cores = []
    for core in cores:
        entries = [float(value).as_integer_ratio() for value in core.flat]
        denominator = max(d for _, d in entries)
        denominators.append(denominator)
        coefficients = [n * (denominator // d) for n, d in entries]
        integer_cores.append(np.asarray(coefficients, dtype=object).reshape(core.shape))
    denominator = math.prod(denominators)
    return tuple(Fraction(int(value), denominator) for value in integer_contract(integer_cores).flat)


def _max_difference(left, right) -> Fraction:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _dense_tt(values: np.ndarray, rank_cap: int) -> list[np.ndarray]:
    """TT-SVD with balanced factors at each cut, as in the v0.2 2D fit."""
    cores = []
    carry, left_rank = values, 1
    for size in values.shape[:-1]:
        u, s, vh = np.linalg.svd(carry.reshape(left_rank * size, -1), full_matrices=False)
        rank = min(rank_cap, len(s))
        root = np.sqrt(s[:rank])
        cores.append((u[:, :rank] * root).reshape(left_rank, size, rank))
        carry = root[:, None] * vh[:rank]
        left_rank = rank
    return [*cores, carry.reshape(left_rank, values.shape[-1], 1)]


def _balance_bonds(cores: list[np.ndarray]) -> list[np.ndarray]:
    """Bounded diagonal gauge that changes no non-underflowed binary64 TT value."""
    result = [core.copy() for core in cores]
    for left, right in zip(result[:-1], result[1:]):
        a, b = np.max(np.abs(left), axis=(0, 1)), np.max(np.abs(right), axis=(1, 2))
        exponents = np.zeros(a.shape, dtype=int)
        nonzero = (a > 0) & (b > 0)
        exponents[nonzero] = np.clip(np.rint((np.log2(b[nonzero]) - np.log2(a[nonzero])) / 2), -8, 8).astype(int)
        left[:] = np.ldexp(left, exponents[None, None, :])
        right[:] = np.ldexp(right, -exponents[:, None, None])
    return result


def _target(models, case, t: float):
    model = models[case["dataset"]]
    n = case["grid"]
    if case["dataset"] == "smooth_cfm_gmm":
        return model.values(t, n).reshape((n,) * 2), None
    dimension = case["dimension"]
    native = model.cores(t, n)
    tail = np.ones(1)
    fixed = case["fixed_indices"]
    if len(fixed) != model.dimension - dimension or any(x not in range(n) for x in fixed):
        raise ValueError("Public slice must fix each omitted coordinate on its resampled grid")
    for core, x in zip(reversed(native[dimension:]), reversed(fixed), strict=True):
        tail = core[:, x, :] @ tail
    cores = [*native[:dimension-1], (native[dimension-1] @ tail)[..., None]]
    return contract_cores(cores), cores


def _case_inputs(models, case, config):
    dt = 1 / config["steps"]
    t = (case["step_index"] + 0.5) * dt
    begin = perf_counter()
    values, native = _target(models, case, t)
    table_seconds = perf_counter() - begin
    begin = perf_counter()
    sampled_half_range = max(
        float(np.ptp(_target(models, case, float(time))[0])) / 2
        for time in np.linspace(0, 1, config["balance_samples"])
    )
    kinetic_half_range = values.ndim * np.pi**2 * case["grid"]**2 / (4 * models[case["dataset"]].length**2)
    beta = math.sqrt(dt * kinetic_half_range / (2 * sampled_half_range)) if sampled_half_range else math.sqrt(dt / 2)
    return values, native, t, beta, table_seconds, perf_counter() - begin


def _model_paths(config):
    return {name: ROOT / path for name, path in config["checkpoints"].items()}


def _load_models(config):
    from floworaclebench.smooth_flow import load_smooth_potential

    paths = _model_paths(config)
    ensure_upstream()
    from tnwf.mps_v import load_mps_v

    public, config_public = load_mps_v(str(paths["sygaldry_public_slice"]), device="cpu")
    return {
        "smooth_cfm_gmm": load_smooth_potential(paths["smooth_cfm_gmm"]),
        "sygaldry_public_slice": LearnedPotential(public, config_public, {}, paths["sygaldry_public_slice"]),
    }


def _public_forward_difference(model, case, t, table):
    import torch

    points = centered_grid(case["grid"], table.ndim, model.length)
    axis = centered_grid(case["grid"], 1, model.length).reshape(-1)
    tail = np.broadcast_to(axis[case["fixed_indices"]], (len(points), len(case["fixed_indices"])))
    inputs = torch.as_tensor(np.concatenate((points, tail), axis=1), dtype=torch.float32)
    times = torch.full((len(points), 1), t, dtype=torch.float32)
    with torch.no_grad():
        direct = model.model(inputs, times).numpy().reshape(-1)
    discrepancy = float(np.max(np.abs(direct - table.flat)))
    if discrepancy > 1e-5 * max(1, float(np.max(np.abs(table)))):
        raise AssertionError(f"Public slice disagrees with upstream float32 forward: {discrepancy}")
    return discrepancy


def _row(oracle, context, reference, budget, extras=None, intended=None):
    start = perf_counter()
    actual_phases = exact_monomial_phases(oracle.circuit, oracle.n_address)
    forward_seconds = perf_counter() - start
    certified = certified_phase_error(actual_phases, reference)
    implementation_error = None if intended is None else certified_phase_error(actual_phases, intended)
    if implementation_error is not None and implementation_error > Fraction(1, 10**10):
        raise AssertionError(f"Gate paths disagree with the intended arithmetic: {float(implementation_error)}")
    experimental = float(np.max(np.abs(
        np.exp(1j*np.asarray(actual_phases, dtype=float)) - np.exp(1j*np.asarray(reference, dtype=float))
    )))
    resources = estimate_resources(oracle.circuit, oracle.n_address, budget)
    metadata = {k: v for k, v in oracle.metadata.items() if k not in {
        "raw_intervals", "retained_intervals", "accumulator_absolute_bounds", "operator_error",
    }}
    total = certified + budget
    return {
        **context, **metadata, **resources, **(extras or {}),
        "native_operator_bound": _upward_float(certified),
        "total_bound_conditional_on_synthesis": _upward_float(total),
        "native_operator_empirical": experimental,
        "meets_target": total <= Fraction(str(context["tolerance"])),
        "clean_ancillas": resources["ancilla_qubits"], "dirty_ancillas": 0,
        "p_rz_gate_count": sum(resources["native_gate_counts"].get(g, 0) for g in ("p", "rz", "u1")),
        "exhaustive_exact_clean_branches": len(actual_phases),
        "encoded_vs_intended_phase_bound": None if implementation_error is None else _upward_float(implementation_error),
        "exact_path_seconds": forward_seconds,
    }


def _qualifying_baseline_bits(cores, values, beta, budget, maximum):
    for bits in range(maximum + 1):
        integers = integer_contract(quantize_cores(cores, bits)).flat
        scale = 1 << (len(cores) * bits)
        error = _max_difference((Fraction(int(x), scale) for x in integers), values)
        if abs(Fraction(beta)) * error < budget:
            return bits
    return None


def _candidate_plans(cores, values, beta, budget, around, maximum):
    from floworaclebench.precision_tt import plan_precision_tt

    dimension = len(cores)
    cap = max(0, math.ceil(math.log2(max(1.0, 4*abs(beta) / float(budget)))))
    center = around if around is not None else maximum
    visited, eligible = set(), defaultdict(list)
    for bits in range(max(0, center - 2), min(maximum, center + 2) + 1):
        profiles = {(bits,) * dimension}
        profiles.update(
            tuple(max(0, min(maximum, bits + (offset if j == edge else 0))) for j in range(dimension))
            for edge in (0, dimension-1) for offset in (-2, 2)
        )
        for profile in sorted(profiles):
            for stage_cap in (None, max(0, cap-2), cap):
                plan = plan_precision_tt(cores, profile, stage_cap)
                key = (plan.core_fraction_bits, plan.stage_fraction_bits)
                if key in visited:
                    continue
                visited.add(key)
                integers = plan.evaluate_exact().flat
                scale = 1 << plan.final_fraction_bits
                arithmetic_error = abs(Fraction(beta)) * _max_difference((Fraction(int(x), scale) for x in integers), values)
                if arithmetic_error >= budget:
                    continue
                # Select a bounded grid of measured candidates using an explicit
                # proxy for multiply-add size and allocated bits. These values
                # are never reported as gate counts or used as measured optima.
                effort = sum(sum(widths) * (core.shape[0] * (f + 3))
                             for widths, core, f in zip(plan.raw_widths[1:], cores[1:], profile[1:]))
                space = sum(map(sum, plan.raw_widths)) + max(map(max, plan.raw_widths))
                family = "rounded" if any(plan.shifts) else "full"
                eligible[family].append((effort, space, arithmetic_error, key, plan))
    winners = []
    for candidates in eligible.values():
        fastest = min(candidates, key=lambda item: item[:4])
        smallest = min(candidates, key=lambda item: (item[1], item[0], item[2], item[3]))
        winners.append(fastest[-1])
        if smallest is not fastest:
            winners.append(smallest[-1])
    return winners, len(visited), sum(map(len, eligible.values()))


def _analytic_plan(cores, beta, budget, residual, maximum):
    """First uniform coefficient bits and smallest cap meeting the proof bound."""
    from floworaclebench.precision_tt import plan_precision_tt

    if abs(Fraction(beta)) * residual >= budget:
        return None
    for bits in range(maximum + 1):
        unrounded = plan_precision_tt(cores, bits)
        if unrounded.phase_error_bound(beta, tt_error=residual) > budget:
            continue
        for cap in range(bits, unrounded.final_fraction_bits):
            capped = plan_precision_tt(cores, bits, cap)
            if capped.phase_error_bound(beta, tt_error=residual) <= budget:
                return capped
        return unrounded
    return None


def _precision_row(plan, beta, workspace, context, reference, budget, residual, extra=None):
    from floworaclebench.precision_tt import build_precision_tt_oracle

    oracle = build_precision_tt_oracle(plan, beta, workspace_strategy=workspace, materialize_values=False)
    analytic = plan.phase_error_bound(beta, tt_error=residual)
    exact_scale = Fraction(beta) / (1 << plan.final_fraction_bits)
    intended = tuple(exact_scale * int(x) for x in plan.evaluate_exact().flat)
    return _row(oracle, context, reference, budget, {
        "rank_residual": _upward_float(residual),
        "analytic_native_operator_bound": _upward_float(analytic),
        "analytic_total_bound_conditional_on_synthesis": _upward_float(min(Fraction(2), analytic + budget)),
        "analytic_meets_target": analytic + budget <= Fraction(str(context["tolerance"])),
        **(extra or {}),
    }, intended=intended), oracle


def _lookup_candidates(table, beta, budget, config):
    numerical_budget = float(budget) * (1 - 1e-10)
    yield build_walsh_oracle(table, beta, approximation_budget=numerical_budget)
    phase_bits = select_phase_bits(table, beta, numerical_budget)
    yield build_qrom_oracle(table, beta, phase_bits)
    n_address = table.size.bit_length() - 1
    for block in range(1, n_address + 1):
        qubits = n_address + (1 << block)*phase_bits + max(0, n_address-block-1)
        if qubits <= max(config["logical_qubit_caps"]):
            yield build_selectswap_oracle(table, beta, phase_bits, block)


def _match(rows, config):
    groups = defaultdict(list)
    for row in rows:
        if row["meets_target"] and row["experiment"] == "matched":
            groups[row["case"], row["tolerance"]].append(row)
    metrics = ("total_bound_conditional_on_synthesis", "t_count_estimate", "logical_qubits")
    frontier, by_cap = [], []
    for (case, tolerance), group in sorted(groups.items()):
        for row in group:
            point = tuple(row[k] for k in metrics)
            if any(all(other[k] <= x for k, x in zip(metrics, point))
                   and any(other[k] < x for k, x in zip(metrics, point)) for other in group):
                continue
            frontier.append({"case": case, "tolerance": tolerance, "method": row["method"],
                             "candidate": row["candidate"], **dict(zip(metrics, point))})
        families = {row["method"] for row in group}
        for cap in config["logical_qubit_caps"]:
            cap_candidates = [row for row in group if row["logical_qubits"] <= cap]
            for family in ["overall", *sorted(families)]:
                eligible = [row for row in cap_candidates if family == "overall" or row["method"] == family]
                if not eligible:
                    continue
                best = min(eligible, key=lambda row: (row["t_count_estimate"], row["logical_qubits"],
                                                       row["total_bound_conditional_on_synthesis"]))
                by_cap.append({"case": case, "tolerance": tolerance, "qubit_cap": cap, "family": family,
                               "method": best["method"], "candidate": best["candidate"],
                               **{k: best[k] for k in metrics}})
    return frontier, by_cap


def _verify_examples(examples):
    verified = []
    for row, oracle in examples.values():
        exact_forward = exact_monomial_phases(oracle.circuit, oracle.n_address)
        exact_backward = exact_monomial_phases(oracle.circuit.inverse(), oracle.n_address)
        inverse_bound = certified_phase_error(exact_backward, (-p for p in exact_forward))
        expected = np.asarray(exact_forward, dtype=float)
        result = validate_oracle(oracle.circuit, oracle.n_address, expected)
        if not all(result[key] for key in ("exhaustive", "address_preserved", "ancillas_clean", "inverse_ancillas_clean")):
            raise AssertionError(result)
        if max(result[key] for key in ("basis_phase_error", "superposition_error", "inverse_error")) > 3e-10:
            raise AssertionError(result)
        verified.append({"candidate": row["candidate"], "case": row["case"],
                         "inverse_exact_operator_bound": _upward_float(inverse_bound), **result})
    return verified


def _run_case(models, case, config, examples):
    table, native, t, beta, table_seconds, balance_seconds = _case_inputs(models, case, config)
    values = tuple(Fraction(float(value)) for value in table.flat)
    reference = tuple(Fraction(beta)*value for value in values)
    base = {
        **case, "time": t, "beta": beta, "time_steps": config["steps"],
        "table_entries": table.size, "dense_table_seconds": table_seconds,
        "time_balance_preprocessing_seconds": balance_seconds,
        "phase_target": "exact beta times represented binary64 grid values",
        "preprocessing": "materialized grid and balanced sequential SVD" if native is None else
            "public checkpoint resampling, fixed-coordinate slice and balanced sequential SVD",
    }
    if native is not None:
        base["public_upstream_float32_forward_max_difference"] = _public_forward_difference(models[case["dataset"]], case, t, table)
    rows, searches = [], []

    def add(row, oracle):
        rows.append(row)
        example_key = row["method"], row.get("workspace_strategy"), bool(any(row.get("shifts", [])))
        if row["meets_target"] and oracle.n_address <= 6 and example_key not in examples:
            examples[example_key] = (row, oracle)

    fitted = []
    for rank in case["rank_caps"]:
        start = perf_counter()
        cores = _dense_tt(table, rank)
        fitted.append((f"rank{rank}", cores, perf_counter() - start))
    if case.get("compare_native_rank") and native is not None:
        fitted.append(("native-rank", native, 0.0))

    for tolerance in config["tolerances"]:
        budget = Fraction(str(tolerance)) / 2
        context = {**base, "experiment": "matched", "tolerance": tolerance}
        label = f"{case['case']}/eps{tolerance:g}"
        for oracle in _lookup_candidates(table.reshape(-1), beta, budget, config):
            identity = f"{label}/{oracle.metadata['method']}-block{oracle.metadata.get('block_bits', 0)}"
            add(_row(oracle, {**context, "candidate": identity}, reference, budget), oracle)
        if native is None:
            from floworaclebench.spectral import build_fourier_oracle

            oracle = build_fourier_oracle(models[case["dataset"]], t, case["grid"], beta)
            add(_row(oracle, {**context, "candidate": f"{label}/direct_fourier"}, reference, budget), oracle)
        for rank_name, cores, svd_seconds in fitted:
            residual = _max_difference(exact_tt_values(cores), values)
            rank_context = {
                **context, "rank_name": rank_name, "bond_dimensions": [core.shape[-1] for core in cores[:-1]],
                "rank_residual": _upward_float(residual), "tt_core_entries": sum(core.size for core in cores),
                "svd_seconds": svd_seconds,
            }
            maximum = config["max_core_fraction_bits"]
            start = perf_counter()
            bits = _qualifying_baseline_bits(cores, values, beta, budget, maximum)
            baseline_search_seconds = perf_counter() - start
            if bits is not None:
                oracle = build_tt_oracle(cores, beta, bits, materialize_values=False)
                scale = Fraction(beta) / (1 << (len(cores)*bits))
                intended = tuple(int(x) * scale for x in integer_contract(oracle.quantized_cores).flat)
                add(_row(oracle, {**rank_context, "candidate": f"{label}/{rank_name}/v02-b{bits}"}, reference, budget,
                         {"precision_search_seconds": baseline_search_seconds}, intended=intended), oracle)
            if bits is None and abs(Fraction(beta))*residual > 2*budget:
                searches.append({"case": case["case"], "tolerance": tolerance, "rank_name": rank_name,
                                 "examined_v03": 0, "status": "rank residual exceeded twice the approximation allocation"})
                continue
            variants = [("svd", cores, 0.0)]
            if case.get("gauge_ablation"):
                start = perf_counter()
                gauged = _balance_bonds(cores)
                if exact_tt_values(gauged) != exact_tt_values(cores):
                    raise AssertionError("The diagonal gauge changed the represented potential")
                variants.append(("dyadic-balance", gauged, perf_counter()-start))
            for gauge, variant, gauge_seconds in variants:
                start = perf_counter()
                plans, examined, eligible = _candidate_plans(variant, values, beta, budget, bits, maximum)
                analytic = _analytic_plan(variant, beta, budget, residual, maximum)
                if analytic is not None and not any(
                    (plan.core_fraction_bits, plan.stage_fraction_bits) == (analytic.core_fraction_bits, analytic.stage_fraction_bits)
                    for plan in plans
                ):
                    plans.append(analytic)
                planning_seconds = perf_counter()-start
                searches.append({"case": case["case"], "tolerance": tolerance, "rank_name": rank_name, "gauge": gauge,
                                 "examined_v03": examined, "eligible_exact_arithmetic": eligible,
                                 "measured_plans": len(plans), "analytic_plan_found": analytic is not None,
                                 "planning_seconds": planning_seconds})
                for plan in plans:
                    schedules = ("retain", "recompute") if any(plan.shifts[1:-1]) else ("retain",)
                    plan_name = f"f{','.join(map(str, plan.core_fraction_bits))}-s{','.join(map(str, plan.stage_fraction_bits))}"
                    for workspace in schedules:
                        identity = f"{label}/{rank_name}/{gauge}/{plan_name}/{workspace}"
                        row, oracle = _precision_row(plan, beta, workspace, {**rank_context, "candidate": identity},
                                                     reference, budget, residual,
                                                     {"gauge": gauge, "gauge_seconds": gauge_seconds,
                                                      "precision_search_seconds": planning_seconds})
                        add(row, oracle)
        print(f"  {label}: {sum(r['tolerance'] == tolerance for r in rows)} built circuits", flush=True)

    if case.get("precision_sweep_bits"):
        from floworaclebench.precision_tt import plan_precision_tt

        tolerance = config["tolerances"][0]
        budget = Fraction(str(tolerance)) / 2
        rank_name, cores, svd_seconds = fitted[-1]
        residual = _max_difference(exact_tt_values(cores), values)
        context = {**base, "experiment": "precision_sweep", "tolerance": tolerance,
                   "rank_name": rank_name, "bond_dimensions": [core.shape[-1] for core in cores[:-1]],
                   "svd_seconds": svd_seconds, "rank_residual": _upward_float(residual)}
        for bits in case["precision_sweep_bits"]:
            baseline = build_tt_oracle(cores, beta, bits, materialize_values=False)
            add(_row(baseline, {**context, "candidate": f"{case['case']}/sweep/v02-b{bits}"}, reference, budget), baseline)
            for cap in (None, bits):
                plan = plan_precision_tt(cores, bits, cap)
                suffix = "full" if cap is None else "capped"
                row, oracle = _precision_row(plan, beta, "retain",
                    {**context, "candidate": f"{case['case']}/sweep/v03-b{bits}-{suffix}"},
                    reference, budget, residual, {"gauge": "svd"})
                add(row, oracle)
    return rows, searches


def _workspace_ablation(config):
    """Compare the schedules while holding the rounded mathematical map fixed."""
    from floworaclebench.precision_tt import plan_precision_tt

    rng = np.random.default_rng(config["seed"])
    dimension, grid, rank = (config[key] for key in ("dimension", "grid", "rank"))
    ranks = [1] + [rank] * (dimension-1) + [1]
    scale = config["scale"]
    cores = [rng.uniform(-scale, scale, (left, grid, right))
             for left, right in zip(ranks[:-1], ranks[1:], strict=True)]
    plan = plan_precision_tt(cores, config["core_fraction_bits"], config["stage_fraction_cap"])
    beta = config["beta"]
    reference = tuple(Fraction(beta) * value for value in exact_tt_values(cores))
    budget = Fraction(str(config["synthesis_budget"]))
    context = {**config, "experiment": "workspace_ablation", "case": "synthetic-workspace",
               "dataset": "seeded_synthetic_tt", "rank_name": f"rank{rank}", "bond_dimensions": ranks[1:-1],
               "table_entries": grid**dimension, "phase_target": "exact beta times unrounded binary64 TT"}
    rows, phases = [], []
    for workspace in ("retain", "recompute"):
        row, oracle = _precision_row(plan, beta, workspace, {**context, "candidate": f"synthetic-workspace/{workspace}"},
                                     reference, budget, Fraction(0))
        rows.append(row)
        phases.append(exact_monomial_phases(oracle.circuit, oracle.n_address))
    if phases[0] != phases[1]:
        raise AssertionError("Workspace ablation changed the encoded phases")
    return rows


def _plot(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {"reversible_tt": ("#ac4d20", "s", "Uniform precision TT"), "precision_tt": ("#275da3", "o", "Precision-aware TT"),
               "walsh": ("#424343", "^", "Walsh"), "qrom": ("#307d61", "v", "QROM"),
               "selectswap": ("#8367a9", "D", "SelectSwap"), "direct_fourier": ("#986c17", "P", "Direct Fourier")}
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6), layout="constrained")
    for method in sorted({row["method"] for row in rows}):
        color, marker, name = methods.get(method, ("#986c17", "P", method))
        matched = [row for row in rows if row["method"] == method and row["experiment"] == "matched" and row["meets_target"]]
        if matched:
            axes[0].scatter([r["logical_qubits"] for r in matched], [r["t_count_estimate"] for r in matched],
                            alpha=0.58, s=23, color=color, marker=marker, label=name)
        sweep = [row for row in rows if row["method"] == method and row["experiment"] == "precision_sweep"]
        if sweep:
            axes[1].scatter([r["native_operator_bound"] for r in sweep], [r["t_count_estimate"] for r in sweep],
                            alpha=0.82, s=30, color=color, marker=marker, label=name)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("Estimated T gates")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set(xlabel="Logical qubits", title="Candidates at matched error targets")
    axes[1].set(xlabel="Certified native operator-error bound", title="Fixed-rank precision sweep", xscale="log")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)


def run(config, out, plot):
    import platform
    import qiskit
    import torch

    torch.set_num_threads(1)
    models = _load_models(config)
    rows, searches, examples = [], [], {}
    start = perf_counter()
    for case in config["cases"]:
        print(f"Case {case['case']}", flush=True)
        measured, search = _run_case(models, case, config, examples)
        rows.extend(measured)
        searches.extend(search)
        write_rows(out / "oracles.csv", rows)
    if "workspace_ablation" in config:
        rows.extend(_workspace_ablation(config["workspace_ablation"]))
        write_rows(out / "oracles.csv", rows)
    frontier, caps = _match(rows, config)
    verified = _verify_examples(examples)
    write_rows(out / "oracles_pareto.csv", frontier)
    write_rows(out / "oracles_qubit_caps.csv", caps)
    _plot(rows, plot)
    summary = {
        "config": config, "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                           "numpy": np.__version__, "qiskit": qiskit.__version__, "torch": torch.__version__},
        "checkpoints_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in _model_paths(config).items()},
        "certification": "exact clean computational paths and rational phase-angle upper bounds modulo 2π; finite binary64 target table; all synthesis allocations conditional",
        "scope": "candidate frontier over the measured circuits; public eight-coordinate checkpoint evaluated on controlled slices",
        "resource_model": "all-to-all logical, clean-ancilla circuits; v0.2 7-T exact Toffoli and estimated 3 log2(1/delta)+10 per arbitrary rotation; no actual Clifford+T synthesis",
        "circuits": len(rows), "qualified_matched_circuits": sum(r["meets_target"] and r["experiment"] == "matched" for r in rows),
        "exact_clean_branch_cases": sum(r["exhaustive_exact_clean_branches"] for r in rows),
        "searches": searches, "independent_simulator_crosschecks": verified,
        "elapsed_seconds": perf_counter()-start,
    }
    dump_json(out / "oracles.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v03_oracles.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/v03")
    parser.add_argument("--plot", type=Path, default=ROOT / "figures/v03_oracle_frontier.png")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, args.plot)


if __name__ == "__main__":
    main()
