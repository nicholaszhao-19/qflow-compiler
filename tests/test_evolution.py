"""Independent signs, register layout, preparation and convergence checks."""
import numpy as np
import pytest
from scipy.linalg import expm
from qiskit.quantum_info import Operator, Statevector

from floworaclebench.evolution import (
    kinetic_matrix, gaussian_state, build_kinetic_circuit,
    build_gaussian_preparation, apply_product_formula, dense_reference, state_metrics,
)


@pytest.mark.parametrize("n,d", [(2, 1), (4, 2), (8, 2), (4, 3), (np.int64(4), 2)])
def test_kinetic_circuit_matches_dense_fourier_generator(n, d):
    length, alpha = 7.0, 0.137
    kinetic = kinetic_matrix(n, d, length)
    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=3e-14)
    circuit = build_kinetic_circuit(n, d, length, alpha)
    assert set(circuit.count_ops()) <= {"h", "p", "rz", "cx"}
    np.testing.assert_allclose(Operator(circuit).data, expm(1j * alpha * kinetic), atol=2e-13, rtol=0)
    np.testing.assert_allclose(Operator(circuit.inverse()).data @ Operator(circuit).data,
                               np.eye(n**d), atol=2e-13, rtol=0)


@pytest.mark.parametrize("n,d", [(2, 1), (4, 2), (8, 2), (4, 3), (np.int64(4), 2)])
def test_initial_qsample_is_prepared_by_actual_gates(n, d):
    circuit = build_gaussian_preparation(n, d, length=8, sigma=4 / 3)
    assert set(circuit.count_ops()) <= {"h", "x", "p", "rz", "cx"}
    np.testing.assert_allclose(Statevector.from_instruction(circuit).data,
                               gaussian_state(n, d), atol=2e-13, rtol=0)
    expected_probability = np.exp(-(np.arange(n) * 8 / n - 4)**2 / (2 * (4 / 3)**2))
    expected_probability /= expected_probability.sum()
    actual_probability = abs(gaussian_state(n, 1))**2
    np.testing.assert_allclose(actual_probability, expected_probability, atol=2e-15)


def test_commutator_formula_has_correct_sign_and_second_order_local_error():
    n, d = 4, 2
    kinetic = kinetic_matrix(n, d)
    rng = np.random.default_rng(581)
    potential = rng.normal(size=n**d)
    generator = kinetic * potential[None, :] - potential[:, None] * kinetic
    state = gaussian_state(n, d)
    errors = []
    for h in [0.01, 0.005, 0.0025]:
        alpha = beta = np.sqrt(h / 2)
        output = apply_product_formula(state, np.exp(1j * beta * potential), expm(1j * alpha * kinetic))
        errors.append(np.linalg.norm(output - expm(h * generator) @ state))
        assert np.linalg.norm(output - expm(-h * generator) @ state) > 50 * errors[-1]
    assert all(3.7 < errors[i] / errors[i + 1] < 4.3 for i in [0, 1])


def test_global_product_formula_converges_for_frozen_potential():
    n, d, final_time = 4, 1, 0.4
    kinetic = kinetic_matrix(n, d)
    values = np.array([0.3, -0.7, 0.1, 0.8])
    generator = kinetic * values[None, :] - values[:, None] * kinetic
    reference = expm(final_time * generator) @ gaussian_state(n, d)
    errors = []
    for steps in [16, 32, 64]:
        h = final_time / steps
        alpha = beta = np.sqrt(h / 2)
        phase, kplus = np.exp(1j * beta * values), expm(1j * alpha * kinetic)
        state = gaussian_state(n, d)
        for _ in range(steps):
            state = apply_product_formula(state, phase, kplus)
        errors.append(np.linalg.norm(state - reference))
    assert all(1.85 < errors[i] / errors[i + 1] < 2.15 for i in [0, 1])


class LinearPotential:
    dimension = 1
    length = 8.0
    sigma0 = 4 / 3
    config = {"N_t": 4}

    def values(self, t, n):
        return (1 + t) * np.cos(2 * np.pi * np.arange(n) / n)


def test_dense_solver_matches_independent_exact_commuting_time_integral():
    potential = LinearPotential()
    n, final_time = 8, 0.81
    actual, metadata = dense_reference(potential, n, final_time)
    kinetic = kinetic_matrix(n, 1)
    values = potential.values(0, n)
    generator = kinetic * values[None, :] - values[:, None] * kinetic
    expected = expm((final_time + final_time**2 / 2) * generator) @ gaussian_state(n, 1)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=0)
    assert metadata["segments"] == 3
    assert metadata["norm_drift"] < 2e-11
    assert metadata["nfev"] > 0


def test_dense_reference_honors_explicit_source_and_nonmapping_config():
    class PrescribedSource(LinearPotential):
        config = object()

        def initial_state(self, n):
            return np.eye(1, n, 1, dtype=complex)[0]

    potential = PrescribedSource()
    n, final_time = 4, 0.3
    actual, metadata = dense_reference(potential, n, final_time)
    kinetic = kinetic_matrix(n, 1)
    values = potential.values(0, n)
    generator = kinetic * (values[None, :] - values[:, None])
    expected = expm((final_time + final_time**2 / 2) * generator) @ potential.initial_state(n)
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=0)
    assert metadata["segments"] == 1


def test_metrics_separate_global_phase_from_distribution_error():
    reference = gaussian_state(4, 2)
    metrics = state_metrics(np.exp(0.7j) * reference, reference)
    assert metrics["fidelity"] == pytest.approx(1)
    assert metrics["tv_distance"] < 1e-15
    assert metrics["phase_aligned_l2_error"] < 1e-15
    assert metrics["state_l2_error"] > 0.6
    with pytest.raises(ValueError):
        state_metrics(np.zeros(2), np.ones(2))


def test_bad_phase_fails_instead_of_masking_ancilla_or_normalization_error():
    with pytest.raises(ValueError):
        apply_product_formula(np.ones(2), np.array([1, 0.9]), np.eye(2))
