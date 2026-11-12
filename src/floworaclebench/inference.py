"""Gate-executed amplitude estimation with explicit preparation and bias costs.

The likelihood is Suzuki et al., Quantum Information Processing 19, 75 (2020),
arXiv:1904.10246, equations (3)--(6).  The finite-shot experiments below are
empirical; the asymptotic Fisher-information expression is not a confidence
guarantee.  Every amplified experiment uses the same compiled unitary and its
exact inverse.  Its preparation bias therefore does not disappear with shots.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from operator import index
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import xlogy, xlog1py
from scipy.stats import binomtest

from .baselines import build_walsh_oracle
from .evolution import (
    apply_product_formula,
    build_gaussian_preparation,
    build_kinetic_circuit,
    grid_initial_state,
    kinetic_matrix,
)
from .resources import estimate_resources


NATIVE_BASIS = ["h", "x", "z", "p", "rz", "cx", "ccx"]


class Potential(Protocol):
    dimension: int
    length: float
    sigma0: float

    def values(self, t: float, n: int) -> np.ndarray: ...


@dataclass
class CompiledPreparation:
    circuit: QuantumCircuit
    initial_state: np.ndarray
    block_state: np.ndarray
    metadata: dict[str, Any]


def _validate_binary_grid(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or n < 2 or n & (n - 1):
        raise ValueError("a power-of-two grid of size at least two is required")
    return int(n).bit_length() - 1


def compile_initial_state(potential: Potential, n: int) -> tuple[QuantumCircuit, np.ndarray, dict[str, Any]]:
    """Prepare the specified source with actual gates, exploiting separability.

    A potential can supply ``initial_state(n)`` to prescribe a periodized
    source.  It is never silently replaced by a truncated Gaussian.  A generic
    nonseparable source falls back to explicit, exponentially sized Qiskit
    StatePreparation; no efficient input-loading promise is made for it.
    """
    width = _validate_binary_grid(n)
    dimension = int(potential.dimension)
    if dimension < 1:
        raise ValueError("positive dimension required")
    initial_fn = getattr(potential, "initial_state", None)
    state = grid_initial_state(potential, n)
    if not callable(initial_fn):
        circuit = build_gaussian_preparation(n, dimension, potential.length, potential.sigma0)
        mode = "sampled separable Gaussian"
    else:
        probability = np.abs(state.reshape((n,) * dimension)) ** 2
        marginals = [probability.sum(axis=tuple(j for j in range(dimension) if j != i))
                     for i in range(dimension)]
        separable = np.ones(1, dtype=float)
        for marginal in marginals:
            separable = np.kron(separable, np.sqrt(marginal))
        circuit = QuantumCircuit(dimension * width, name="source_preparation")
        if np.max(np.abs(state - separable)) < 2e-13:
            for axis, marginal in enumerate(marginals):
                wires = list(range((dimension - axis - 1) * width, (dimension - axis) * width))
                circuit.append(StatePreparation(np.sqrt(marginal)), wires)
            mode = "specified separable source; per-coordinate StatePreparation"
        else:
            circuit.append(StatePreparation(state), range(dimension * width))
            mode = "generic dense StatePreparation; exponential loading cost retained"
        circuit = transpile(circuit, basis_gates=NATIVE_BASIS, optimization_level=1)
    actual = Statevector.from_instruction(circuit).data
    error = float(np.linalg.norm(actual - state))
    if error > 2e-12:
        raise AssertionError(f"source circuit failed: {error}")
    return circuit, state, {"source_compilation": mode, "source_circuit_l2_error": error}


def compile_flow_preparation(
    potential: Potential,
    n: int,
    steps: int,
    *,
    final_time: float = 1.0,
    balance_samples: int = 17,
) -> CompiledPreparation:
    """Compile all gates of the existing eight-factor commutator construction.

    Time is a classical schedule.  Exact native Walsh phase rotations are used
    here so a full learned circuit fits in a small statevector; this does not
    replace or claim execution of the separate wide reversible-TT circuits.
    Potential tables are materialized and their cost is recorded.  No opaque
    dense UnitaryGate appears in the result.
    """
    start = perf_counter()
    _validate_binary_grid(n)
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("positive integer time-step count required")
    if not 0 < final_time <= 1 or balance_samples < 2:
        raise ValueError("0 < final_time <= 1 and at least two balance samples required")
    circuit, initial, metadata = compile_initial_state(potential, n)
    dt = final_time / steps
    sampled_values = [np.asarray(potential.values(float(t), n), dtype=float).reshape(-1)
                      for t in np.linspace(0, final_time, balance_samples)]
    if any(v.size != n**potential.dimension or not np.isfinite(v).all() for v in sampled_values):
        raise ValueError("potential returned invalid finite-grid values")
    potential_half_range = max(float(np.ptp(v)) / 2 for v in sampled_values)
    kinetic_half_range = potential.dimension * np.pi**2 * n**2 / (4 * potential.length**2)
    if potential_half_range > 0:
        alpha = math.sqrt(dt * potential_half_range / (2 * kinetic_half_range))
        beta = math.sqrt(dt * kinetic_half_range / (2 * potential_half_range))
    else:
        alpha = beta = math.sqrt(dt / 2)
    kinetic_circuit = build_kinetic_circuit(n, potential.dimension, potential.length, alpha)
    kinetic_inverse = kinetic_circuit.inverse()
    kinetic = kinetic_matrix(n, potential.dimension, potential.length)
    kinetic_exact = expm(1j * alpha * kinetic)
    kinetic_error = float(np.linalg.norm(Operator(kinetic_circuit).data - kinetic_exact, ord=2))
    block_state = initial.copy()
    table_seconds = 0.0
    max_phase_error = 0.0
    for step in range(steps):
        t = (step + 0.5) * dt
        before = perf_counter()
        values = np.asarray(potential.values(t, n), dtype=float).reshape(-1)
        table_seconds += perf_counter() - before
        if values.shape != initial.shape or not np.isfinite(values).all():
            raise ValueError("potential returned invalid midpoint values")
        oracle = build_walsh_oracle(values, beta=beta)
        max_phase_error = max(max_phase_error, float(oracle.metadata["operator_error"]))
        plus, minus = oracle.circuit, oracle.circuit.inverse()
        # Chronological order of Layden et al. Eq. (36), alpha*beta=dt/2.
        for kinetic_sign, phase in [(1, plus), (-1, minus), (-1, minus), (1, plus)]:
            circuit.compose(kinetic_circuit if kinetic_sign == 1 else kinetic_inverse,
                            range(circuit.num_qubits), inplace=True)
            circuit.compose(phase, range(circuit.num_qubits), inplace=True)
        block_state = apply_product_formula(block_state, np.exp(1j * beta * values), kinetic_exact)
    circuit.name = f"learned_flow_n{n}_r{steps}"
    if set(circuit.count_ops()) - set(NATIVE_BASIS):
        raise AssertionError("the preparation contains undecomposed instructions")
    metadata.update({
        "grid": n, "dimension": potential.dimension, "length": float(potential.length),
        "steps": int(steps), "final_time": final_time, "alpha": alpha, "beta": beta,
        "alpha_times_beta": alpha * beta,
        "potential_half_range_on_balance_samples": potential_half_range,
        "balance_samples": balance_samples, "kinetic_half_range": kinetic_half_range,
        "balance_is_error_certificate": False,
        "potential_calls_per_preparation": 4 * steps, "kinetic_calls_per_preparation": 4 * steps,
        "potential_table_evaluation_seconds": table_seconds,
        "potential_table_entries_evaluated": (steps + balance_samples) * n**potential.dimension,
        "compile_seconds": perf_counter() - start,
        "kinetic_circuit_operator_error": kinetic_error,
        "maximum_native_walsh_operator_error": max_phase_error,
        "native_rotation_synthesis_executed": False,
        "implementation": "Gaussian/source gates, QFT kinetic gates, native Walsh phase gates",
    })
    return CompiledPreparation(circuit, initial, block_state, metadata)


def _marked_pattern(n_qubits: int, assignments: dict[int, int]) -> QuantumCircuit:
    """Phase-flip exactly the inputs satisfying specified binary bits."""
    circuit = QuantumCircuit(n_qubits)
    if not assignments:
        circuit.global_phase = np.pi
        return circuit
    wires = sorted(assignments)
    if any(w < 0 or w >= n_qubits or assignments[w] not in (0, 1) for w in wires):
        raise ValueError("invalid binary condition")
    negative = [w for w in wires if assignments[w] == 0]
    for w in negative:
        circuit.x(w)
    if len(wires) == 1:
        circuit.z(wires[0])
    else:
        circuit.h(wires[-1])
        circuit.mcx(wires[:-1], wires[-1])
        circuit.h(wires[-1])
    for w in reversed(negative):
        circuit.x(w)
    return circuit


def _prefix_cover(start: int, stop: int, width: int) -> list[dict[int, int]]:
    """Disjoint binary-prefix blocks covering [start, stop); classical compile step."""
    if not 0 <= start <= stop <= 2**width:
        raise ValueError("interval outside the register")
    output: list[dict[int, int]] = []
    while start < stop:
        trailing = width if start == 0 else (start & -start).bit_length() - 1
        free = min(trailing, (stop - start).bit_length() - 1)
        output.append({j: (start >> j) & 1 for j in range(free, width)})
        start += 2**free
    return output


def box_event(n: int, intervals: Sequence[tuple[int, int]]) -> tuple[np.ndarray, QuantumCircuit]:
    """Compile a rectangular event on C-ordered coordinate registers.

    Each interval is a half-open interval of grid indices.  A disjoint binary
    prefix cover gives exact phase flips without a free comparator oracle.
    Multi-controlled operations are decomposed into the declared native basis.
    """
    width = _validate_binary_grid(n)
    dimension = len(intervals)
    if dimension < 1:
        raise ValueError("at least one interval required")
    try:
        bounds = [(index(lo), index(hi)) for lo, hi in intervals]
    except TypeError as exc:
        raise ValueError("Event bounds must be integer grid indices") from exc
    local = [_prefix_cover(lo, hi, width) for lo, hi in bounds]
    circuit = QuantumCircuit(width * dimension, name="box_event")
    for blocks in product(*local):
        assignments: dict[int, int] = {}
        for axis, block in enumerate(blocks):
            offset = (dimension - axis - 1) * width
            assignments.update({offset + bit: value for bit, value in block.items()})
        circuit.compose(_marked_pattern(circuit.num_qubits, assignments), inplace=True)
    # This is an oracle on arbitrary data, not a state-preparation circuit.
    # Qiskit's default initial-zero assumption can otherwise borrow idle
    # address bits as clean auxiliaries and change the oracle's full operator.
    circuit = transpile(circuit, basis_gates=NATIVE_BASIS, optimization_level=1,
                        qubits_initially_zero=False)
    mask = np.ones((n,) * dimension, dtype=bool)
    for axis, (lo, hi) in enumerate(bounds):
        shape = [1] * dimension
        shape[axis] = n
        selected = (np.arange(n) >= lo) & (np.arange(n) < hi)
        mask &= selected.reshape(shape)
    return mask.reshape(-1), circuit


def lookup_event(mask: np.ndarray) -> QuantumCircuit:
    """An explicit Walsh synthesis of I-2 Pi_event for an arbitrary finite mask."""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return build_walsh_oracle(np.pi * mask.astype(float)).circuit


def zero_reflection(n_qubits: int) -> QuantumCircuit:
    if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
        raise ValueError("at least one qubit required")
    circuit = _marked_pattern(n_qubits, {j: 0 for j in range(n_qubits)})
    circuit.name = "zero_reflection"
    return transpile(circuit, basis_gates=NATIVE_BASIS, optimization_level=1,
                     qubits_initially_zero=False)


def amplified_circuit(preparation: QuantumCircuit, event: QuantumCircuit, k: int) -> QuantumCircuit:
    """Return Q^k A, Q=-A S0 A† S_event, with no controlled-A operations."""
    if preparation.num_qubits != event.num_qubits or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("matching circuit widths and a nonnegative integer k required")
    circuit = QuantumCircuit(preparation.num_qubits, name=f"amplified_k{k}")
    circuit.compose(preparation, inplace=True)
    inverse = preparation.inverse()
    reflection = zero_reflection(preparation.num_qubits)
    for _ in range(k):
        circuit.compose(event, inplace=True)
        circuit.compose(inverse, inplace=True)
        circuit.compose(reflection, inplace=True)
        circuit.compose(preparation, inplace=True)
        circuit.global_phase += np.pi
    return circuit


def probability(state: np.ndarray, mask: np.ndarray) -> float:
    state = np.asarray(state, dtype=complex).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if state.shape != mask.shape or not np.isfinite(state).all():
        raise ValueError("state and event must have the same finite dimension")
    return float(np.sum(np.abs(state[mask]) ** 2))


def execute_amplification(
    preparation: QuantumCircuit, event: QuantumCircuit, mask: np.ndarray, depths: Sequence[int]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Execute actual native gates, reusing trajectories at intermediate depths.

    This avoids repeatedly simulating an identical prefix.  It does not replace
    A, A† or Q by a dense precomputed matrix.  Physical repeated-shot resource
    accounting is separate from this classical simulation optimization.
    """
    depths = sorted(set(int(k) for k in depths))
    if not depths or depths[0] < 0 or preparation.num_qubits != event.num_qubits:
        raise ValueError("valid depths and equal circuit widths required")
    start = perf_counter()
    state = Statevector.from_instruction(preparation)
    initial_state = state.data.copy()
    p = probability(initial_state, mask)
    if not -1e-10 <= p <= 1 + 1e-10:
        raise AssertionError("statevector probability is not normalized")
    theta = math.asin(math.sqrt(float(np.clip(p, 0, 1))))
    inverse, reflection = preparation.inverse(), zero_reflection(preparation.num_qubits)
    output = []
    elapsed = perf_counter() - start
    for k in range(max(depths) + 1):
        if k:
            start = perf_counter()
            state = state.evolve(event).evolve(inverse).evolve(reflection).evolve(preparation)
            # Overall minus sign in Q changes phase only, but is retained.
            state = Statevector(-state.data)
            elapsed += perf_counter() - start
        if k in depths:
            measured = probability(state.data, mask)
            predicted = math.sin((2 * k + 1) * theta) ** 2
            output.append({
                "k": k, "prepared_event_probability": p,
                "gate_executed_event_probability": measured,
                "amplification_identity_prediction": predicted,
                "amplification_identity_abs_error": abs(measured - predicted),
                "state_norm_drift": float(abs(np.linalg.norm(state.data) - 1)),
                "cumulative_native_statevector_seconds": elapsed,
                "A_forward_calls_per_shot": k + 1, "A_inverse_calls_per_shot": k,
                "A_or_inverse_calls_per_shot": 2 * k + 1,
                "event_calls_per_shot": k, "zero_reflection_calls_per_shot": k,
                "controlled_A_calls_per_shot": 0,
            })
    return initial_state, output


def maximum_likelihood_amplitude(
    counts: Sequence[int], shots: Sequence[int], depths: Sequence[int], *, grid_points: int = 4097
) -> float:
    """Search the binomial likelihood on [0, pi/2] by grid and local refinement.

    Every local maximum detected on the coarse grid is refined. The grid resolution is checked
    against the largest oscillation frequency; finite-shot global ambiguity is
    evaluated empirically, not suppressed by choosing a preferred local mode.
    There is no exact finite-shot success/coverage theorem asserted for this
    numerical MLE implementation.
    """
    h = np.asarray(counts, dtype=int)
    n = np.asarray(shots, dtype=int)
    k = np.asarray(depths, dtype=int)
    if h.ndim != 1 or h.size == 0 or h.shape != n.shape or h.shape != k.shape:
        raise ValueError("one-dimensional nonempty arrays of the same length required")
    if (h < 0).any() or (n < h).any() or (n <= 0).any() or (k < 0).any():
        raise ValueError("invalid binomial observations or depths")
    frequencies = 2 * k + 1
    if grid_points < max(257, 32 * int(max(frequencies))):
        raise ValueError("likelihood search grid is too small for the configured depths")

    def log_likelihood(theta: np.ndarray | float) -> np.ndarray:
        angles = np.asarray(theta)[..., None] * frequencies
        q = np.clip(np.sin(angles) ** 2, 0.0, 1.0)
        return np.sum(xlogy(h, q) + xlog1py(n - h, -q), axis=-1)

    grid = np.linspace(0, np.pi / 2, grid_points)
    logl = log_likelihood(grid)
    candidates: list[tuple[float, float]] = [(float(logl[0]), 0.0), (float(logl[-1]), np.pi / 2)]
    peaks = np.where((logl[1:-1] >= logl[:-2]) & (logl[1:-1] >= logl[2:]))[0] + 1
    for j in peaks:
        fit = minimize_scalar(lambda theta: -float(log_likelihood(theta)),
                              bounds=(float(grid[j - 1]), float(grid[j + 1])),
                              method="bounded", options={"xatol": 1e-13})
        if fit.success:
            candidates.append((-float(fit.fun), float(fit.x)))
    best = max(candidates, key=lambda pair: pair[0])[1]
    return float(math.sin(best) ** 2)


def _empirical_summary(estimates: np.ndarray, truth: float) -> dict[str, Any]:
    errors = np.asarray(estimates, dtype=float) - truth
    squared = errors**2
    rmse = float(np.sqrt(np.mean(squared)))
    successes = []
    for tolerance in [0.001, 0.002, 0.005, 0.01]:
        count = int(np.sum(abs(errors) <= tolerance))
        interval = binomtest(count, len(errors)).proportion_ci(confidence_level=0.95, method="exact")
        successes.append({
            "absolute_tolerance": tolerance, "successful_repetitions": count,
            "independent_repetitions": len(errors), "empirical_success_frequency": count / len(errors),
            "clopper_pearson_95_interval": [float(interval.low), float(interval.high)],
            "scope": "individual interval for success frequency in this noiseless shot experiment against the prepared probability; not an AE theorem or target-accuracy guarantee",
        })
    return {
        "mean_estimate": float(np.mean(estimates)), "empirical_bias": float(np.mean(errors)),
        "rmse": rmse,
        "rmse_standard_error_delta_method": float(np.std(squared, ddof=1) / math.sqrt(len(errors)) / (2 * rmse)) if rmse else 0.0,
        "median_absolute_error": float(np.median(abs(errors))),
        "absolute_error_q90": float(np.quantile(abs(errors), 0.9)),
        "absolute_error_q95": float(np.quantile(abs(errors), 0.95)),
        "fraction_absolute_error_over_0_1": float(np.mean(abs(errors) > 0.1)),
        "empirical_success_by_absolute_tolerance": successes,
    }


def sampling_benchmark(
    executed_probabilities: Sequence[float], depths: Sequence[int],
    *, shots_per_depth: int = 64, repetitions: int = 256, seed: int = 20261105,
    reference_probability: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Simulate repeated measured counts from gate-executed Born probabilities.

    Classical MC uses exactly the same number of draws as paid A/A† calls in
    each MLAE schedule.  This is a *query-budget* comparison, not a claim that
    one classical draw costs a fault-tolerant preparation.  Runtime and full
    circuit resources are reported separately by the experiment driver.
    """
    k = np.asarray(depths, dtype=int)
    q = np.asarray(executed_probabilities, dtype=float)
    if k.shape != q.shape or k.ndim != 1 or len(k) == 0 or k[0] != 0:
        raise ValueError("depth zero followed by matching probabilities required")
    if (q < -1e-9).any() or (q > 1 + 1e-9).any() or shots_per_depth < 1 or repetitions < 2:
        raise ValueError("invalid probabilities or sample counts")
    q = np.clip(q, 0, 1)
    p = float(q[0])
    rng = np.random.default_rng(seed)
    counts = rng.binomial(shots_per_depth, q, size=(repetitions, len(k)))
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"measured_binomial_counts": counts, "depths": k}
    for prefix in range(1, len(k) + 1):
        depths_used = k[:prefix]
        nshots = np.full(prefix, shots_per_depth, dtype=int)
        paid_queries = int(shots_per_depth * np.sum(2 * depths_used + 1))
        start = perf_counter()
        estimates = np.array([maximum_likelihood_amplitude(h[:prefix], nshots, depths_used) for h in counts])
        mle_seconds = perf_counter() - start
        start = perf_counter()
        classical = rng.binomial(paid_queries, p, size=repetitions) / paid_queries
        sampling_seconds = perf_counter() - start
        for method, samples in [("maximum_likelihood_AE", estimates), ("Monte_Carlo_equal_A_query_budget", classical)]:
            row = {
                "method": method, "schedule": depths_used.tolist(), "maximum_k": int(max(depths_used)),
                "shots_per_depth": shots_per_depth, "independent_repetitions": repetitions,
                "seed": seed, "total_circuit_shots": int(prefix * shots_per_depth) if method.startswith("maximum") else paid_queries,
                "paid_preparation_queries_per_repetition": paid_queries,
                "prepared_probability": p, **_empirical_summary(samples, p),
                "empirical_statistics_are_confidence_guarantees": False,
                "classical_processing_seconds_all_repetitions": mle_seconds if method.startswith("maximum") else sampling_seconds,
            }
            if reference_probability is not None:
                row["finite_grid_reference_probability"] = reference_probability
                row["preparation_probability_bias"] = p - reference_probability
                row["rmse_to_finite_grid_reference"] = float(np.sqrt(np.mean((samples - reference_probability)**2)))
            if method.startswith("maximum") and row["rmse"] > 0:
                row["MC_draws_matching_empirical_MLAE_rmse_variance_formula"] = int(math.ceil(p * (1 - p) / row["rmse"]**2))
            rows.append(row)
        arrays[f"mle_prefix{prefix}"] = estimates
        arrays[f"mc_prefix{prefix}"] = classical
    return rows, arrays


def repeated_circuit_resources(
    preparation: QuantumCircuit, event: QuantumCircuit,
    depths: Sequence[int], shots_per_depth: int,
    *, total_rotation_budget: float = 1e-3,
) -> dict[str, Any]:
    """Count every repeated native operation and project a declared T cost.

    The rotation budget is divided across all block occurrences in one complete
    experiment.  This is conservative prospective synthesis accounting, not
    executed Clifford+T synthesis, a physical runtime, or a model-error budget.
    """
    k = np.asarray(depths, dtype=int)
    if k.ndim != 1 or not len(k) or (k < 0).any() or shots_per_depth < 1:
        raise ValueError("valid schedule and positive shots required")
    reflection = zero_reflection(preparation.num_qubits)
    a_calls = int(shots_per_depth * np.sum(2 * k + 1))
    q_calls = int(shots_per_depth * np.sum(k))
    total_blocks = a_calls + 2 * q_calls
    exact_budget = Fraction(str(total_rotation_budget))
    budget = exact_budget / total_blocks
    a = estimate_resources(preparation, preparation.num_qubits, budget)
    e = estimate_resources(event, preparation.num_qubits, budget)
    z = estimate_resources(reflection, preparation.num_qubits, budget)
    summed_keys = ["gate_count", "native_depth", "cx", "ccx", "cnot_decomposed",
                   "arbitrary_rotations", "t_count_estimate", "t_depth_macro_estimate"]
    totals = {key: a_calls * a[key] + q_calls * (e[key] + z[key]) for key in summed_keys}
    counts: dict[str, int] = {}
    used_budget = Fraction()
    for item, multiple in [(a, a_calls), (e, q_calls), (z, q_calls)]:
        for name, count in item["native_gate_counts"].items():
            counts[name] = counts.get(name, 0) + multiple * count
        special = Fraction(item["special_rotation_replacement_error_bound_exact"])
        arbitrary = item["arbitrary_rotations"] * Fraction(item["per_rotation_error"])
        used_budget += multiple * (special + arbitrary)
    if used_budget > exact_budget:
        raise AssertionError("Aggregate synthesis allocation exceeded the declared budget")
    return {
        **totals, "logical_qubits": preparation.num_qubits, "ancilla_qubits": 0,
        "native_gate_counts": counts,
        "A_forward_calls": int(shots_per_depth * np.sum(k + 1)),
        "A_inverse_calls": q_calls, "A_or_inverse_calls": a_calls,
        "event_calls": q_calls, "zero_reflection_calls": q_calls, "controlled_A_calls": 0,
        "circuit_shots": int(shots_per_depth * len(k)),
        "maximum_coherent_preparation_calls": int(2 * max(k) + 1),
        "maximum_coherent_native_gates": int((2 * max(k) + 1) * a["gate_count"] + max(k) * (e["gate_count"] + z["gate_count"])),
        "total_rotation_synthesis_budget": total_rotation_budget,
        "per_block_rotation_budget": float(budget),
        "per_block_rotation_budget_exact": str(budget),
        "total_rotation_synthesis_budget_exact": str(exact_budget),
        "used_rotation_synthesis_allowance_exact": str(used_budget),
        "rotation_synthesis_executed": False,
        "physical_hardware_runtime_estimated": False,
        "depth_schedule": "sum of native block depths, sequential shots and block boundaries",
        "A_resources_at_allocated_budget": a, "event_resources_at_allocated_budget": e,
        "zero_reflection_resources_at_allocated_budget": z,
    }


def two_site_contraction(
    state: np.ndarray, n: int, intervals: Sequence[tuple[int, int]], ranks: Sequence[int]
) -> list[dict[str, Any]]:
    """Exact/truncated two-site amplitude MPS contraction of a product event.

    Dense-to-MPS SVD setup is paid and recorded.  This is a small two-dimensional
    diagnostic, not a scalable high-dimensional state-construction algorithm.
    The event contraction is Tr[(L† E_x L)(R E_y R†)] after normalization.
    """
    if len(intervals) != 2:
        raise ValueError("two coordinate intervals required")
    state = np.asarray(state, dtype=complex).reshape(n, n)
    input_norm = float(np.linalg.norm(state))
    if not np.isclose(input_norm, 1, atol=1e-10, rtol=0):
        raise ValueError("normalized state required")
    # Remove measured floating-point norm drift before applying a pure-state
    # trace-distance formula.  Do not interpret that drift as discarded rank.
    state = state / input_norm
    start = perf_counter()
    u, singular, vh = np.linalg.svd(state, full_matrices=False)
    setup_seconds = perf_counter() - start
    ex = ((np.arange(n) >= intervals[0][0]) & (np.arange(n) < intervals[0][1])).astype(float)
    ey = ((np.arange(n) >= intervals[1][0]) & (np.arange(n) < intervals[1][1])).astype(float)
    exact = float(np.sum(np.abs(state)**2 * ex[:, None] * ey[None, :]))
    rows: list[dict[str, Any]] = []
    for rank in sorted(set(int(r) for r in ranks)):
        if not 1 <= rank <= n:
            raise ValueError("rank outside the two-site matrix range")
        kept_weight = float(np.sum(singular[:rank]**2) / np.sum(singular**2))
        left = u[:, :rank] * singular[:rank]
        right = vh[:rank]
        start = perf_counter()
        a = left.conj().T @ (ex[:, None] * left)
        b = (right * ey[None, :]) @ right.conj().T
        value = float(np.trace(a @ b).real / kept_weight)
        contraction_seconds = perf_counter() - start
        reconstructed = left @ right / math.sqrt(kept_weight)
        measured = float(np.sum(np.abs(reconstructed)**2 * ex[:, None] * ey[None, :]))
        if abs(value - measured) > 2e-12:
            raise AssertionError("tensor contraction disagrees with the reconstructed state")
        bound = math.sqrt(max(0.0, 1 - kept_weight))
        if abs(value - exact) > bound + 1e-12:
            raise AssertionError("pure-state trace-distance event bound failed")
        rows.append({
            "rank": rank, "probability": value, "dense_probability": exact,
            "event_absolute_error": abs(value - exact), "retained_squared_singular_values": kept_weight,
            "pure_state_trace_distance_event_bound": bound,
            "amplitude_coefficients_stored": 2 * n * rank, "dense_amplitude_coefficients": n * n,
            "dense_SVD_setup_seconds": setup_seconds, "event_contraction_seconds": contraction_seconds,
            "singular_values": singular.tolist(), "exact_at_full_rank": rank == n,
            "input_state_norm": input_norm,
            "roundoff_norm_drift_removed_before_pure_state_bound": True,
            "high_dimensional_scalability_claimed": False,
        })
    return rows
