"""Independent event, amplification, likelihood, and complete-cost checks."""
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector

from floworaclebench.inference import (
    NATIVE_BASIS, amplified_circuit, box_event, compile_flow_preparation,
    execute_amplification, lookup_event, maximum_likelihood_amplitude,
    repeated_circuit_resources, two_site_contraction, zero_reflection,
)


class SmoothToyPotential:
    dimension = 2
    length = 8.0
    sigma0 = 1.0

    def values(self, t, n):
        axis = np.arange(n) * 2 * np.pi / n
        return ((0.2 + t) * np.cos(axis[:, None]) * np.sin(axis[None, :])).reshape(-1)


def test_full_native_flow_preparation_matches_independent_block_evolution():
    result = compile_flow_preparation(SmoothToyPotential(), 4, 3, final_time=0.25)
    assert set(result.circuit.count_ops()) <= set(NATIVE_BASIS)
    actual = Statevector.from_instruction(result.circuit)
    np.testing.assert_allclose(actual.data, result.block_state, atol=6e-13, rtol=0)
    restored = actual.evolve(result.circuit.inverse()).data
    np.testing.assert_allclose(restored, np.eye(1, 16, dtype=complex)[0], atol=8e-13, rtol=0)
    assert result.metadata["potential_calls_per_preparation"] == 12
    assert result.metadata["alpha_times_beta"] == pytest.approx(0.25 / 6)


@pytest.mark.parametrize("n,intervals", [
    (4, [(1, 4), (2, 4)]), (8, [(5, 8), (5, 8)]), (16, [(10, 16), (10, 16)]),
    (4, [(0, 4), (0, 4)]), (4, [(0, 0), (0, 4)]),
])
def test_compiled_event_matches_every_basis_input(n, intervals):
    mask, circuit = box_event(n, intervals)
    expected = np.diag(1 - 2 * mask.astype(float))
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=3e-13, rtol=0)
    np.testing.assert_allclose(Operator(lookup_event(mask)).data, expected, atol=3e-13, rtol=0)
    assert set(circuit.count_ops()) <= set(NATIVE_BASIS)


def test_zero_reflection_exactly_marks_zero_including_global_phase():
    matrix = Operator(zero_reflection(4)).data
    expected = np.eye(16, dtype=complex)
    expected[0, 0] = -1
    np.testing.assert_allclose(matrix, expected, atol=3e-13, rtol=0)


def test_box_event_rejects_fractional_grid_indices():
    with pytest.raises(ValueError, match="integer grid indices"):
        box_event(4, [(0.5, 3)])


def test_amplification_uses_same_preparation_inverse_and_correct_success_law():
    desired = np.sqrt(np.array([0.31, 0.16, 0.43, 0.10]))
    preparation = QuantumCircuit(2)
    preparation.append(StatePreparation(desired), [0, 1])
    preparation = transpile(preparation, basis_gates=NATIVE_BASIS, optimization_level=1)
    mask = np.array([False, False, False, True])
    event = lookup_event(mask)
    initial, rows = execute_amplification(preparation, event, mask, [0, 1, 2, 5])
    np.testing.assert_allclose(initial, desired, atol=3e-14, rtol=0)
    assert max(row["amplification_identity_abs_error"] for row in rows) < 2e-13
    for row in rows:
        state = Statevector.from_instruction(amplified_circuit(preparation, event, row["k"]))
        assert abs(state.data[-1])**2 == pytest.approx(row["gate_executed_event_probability"], abs=2e-13)
        assert row["A_or_inverse_calls_per_shot"] == 2 * row["k"] + 1
        assert row["controlled_A_calls_per_shot"] == 0


@pytest.mark.parametrize("p", [0.02, 0.23, 0.61, 0.94])
def test_global_likelihood_recovers_deterministic_binomial_means_and_grid_refines(p):
    depths = np.array([0, 1, 2, 4, 8])
    shots = np.full(5, 10000)
    theta = np.arcsin(np.sqrt(p))
    counts = np.rint(shots * np.sin((2 * depths + 1) * theta)**2).astype(int)
    estimate = maximum_likelihood_amplitude(counts, shots, depths)
    refined = maximum_likelihood_amplitude(counts, shots, depths, grid_points=8193)
    assert abs(estimate - p) < 1e-4
    assert abs(refined - estimate) < 1e-7


def test_single_depth_zero_mle_agrees_with_bernoulli_estimate_and_boundary_cases():
    assert maximum_likelihood_amplitude([3], [10], [0]) == pytest.approx(0.3, abs=1e-7)
    assert maximum_likelihood_amplitude([0], [10], [0]) == 0
    assert maximum_likelihood_amplitude([10], [10], [0]) == 1
    with pytest.raises(ValueError):
        maximum_likelihood_amplitude([11], [10], [0])


def test_costs_count_every_inverse_event_and_reflection():
    preparation = QuantumCircuit(2)
    preparation.h(0)
    preparation.cx(0, 1)
    preparation.rz(0.37, 0)
    event = lookup_event(np.array([False, False, True, True]))
    result = repeated_circuit_resources(preparation, event, [0, 1, 2], 7)
    assert result["A_or_inverse_calls"] == 63
    assert result["A_forward_calls"] == 42
    assert result["A_inverse_calls"] == 21
    assert result["event_calls"] == result["zero_reflection_calls"] == 21
    assert result["controlled_A_calls"] == 0
    expected = {}
    for k in [0, 1, 2]:
        for name, count in amplified_circuit(preparation, event, k).count_ops().items():
            expected[name] = expected.get(name, 0) + 7 * count
    assert result["native_gate_counts"] == expected
    assert result["gate_count"] == sum(expected.values())


def test_repeated_resources_allocate_exact_rational_block_budget():
    from fractions import Fraction

    preparation = QuantumCircuit(1)
    preparation.p(0.37, 0)
    event = lookup_event(np.array([False, True]))
    result = repeated_circuit_resources(preparation, event, [0], 3, total_rotation_budget=0.001)
    block_budget = Fraction(result["per_block_rotation_budget_exact"])
    assert block_budget == Fraction(1, 3000)
    assert Fraction(result["A_resources_at_allocated_budget"]["rotation_error_budget_exact"]) == block_budget
    assert Fraction(result["used_rotation_synthesis_allowance_exact"]) <= Fraction("0.001")


def test_tensor_contraction_and_trace_distance_bound_for_complex_state():
    rng = np.random.default_rng(534)
    state = rng.normal(size=64) + 1j * rng.normal(size=64)
    state /= np.linalg.norm(state)
    rows = two_site_contraction(state, 8, [(5, 8), (4, 8)], [1, 2, 4, 8])
    assert rows[-1]["event_absolute_error"] < 2e-13
    assert rows[-1]["pure_state_trace_distance_event_bound"] < 4e-8
    for row in rows:
        assert row["event_absolute_error"] <= row["pure_state_trace_distance_event_bound"] + 1e-13
