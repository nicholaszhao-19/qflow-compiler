"""Full-statevector integration check including every arithmetic workspace wire.

This uses a controlled synthetic integer TT to keep the complete Hilbert space
small. Learned-model runs use the separately verified blockwise simulation;
this test independently checks that method across non-monomial kinetic gates.
"""
import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from floworaclebench.evolution import (
    apply_product_formula, build_gaussian_preparation, build_kinetic_circuit,
    gaussian_state, kinetic_matrix,
)
from floworaclebench.reversible import build_tt_oracle
from floworaclebench.simulation import simulate_paths


@pytest.mark.parametrize("steps", [1, 2])
def test_full_statevector_flow_matches_blockwise_and_uncomputes_workspace(steps):
    n, dimension, length = 2, 2, 8.0
    # Consecutive steps deliberately use different, exactly integral potentials.
    core_sets = [
        [np.array([0, 1]).reshape(1, 2, 1), np.array([0, 1]).reshape(1, 2, 1)],
        [np.array([1, 0]).reshape(1, 2, 1), np.array([0, 1]).reshape(1, 2, 1)],
    ]
    h = 0.025
    alphas = [0.2, 0.15]
    oracles = [build_tt_oracle(cores, beta=h / (2 * alpha), fraction_bits=0)
               for cores, alpha in zip(core_sets[:steps], alphas[:steps])]
    n_address = dimension
    width = max(oracle.circuit.num_qubits for oracle in oracles)
    assert width <= 18, "Keep this independent dense integration test bounded"
    full_circuit = QuantumCircuit(width)
    full_circuit.compose(build_gaussian_preparation(n, dimension, length), range(n_address), inplace=True)
    block_state = gaussian_state(n, dimension, length)
    kinetic = kinetic_matrix(n, dimension, length)

    for oracle, alpha in zip(oracles, alphas):
        kinetic_circuit = build_kinetic_circuit(n, dimension, length, alpha)
        plus = oracle.circuit
        minus = plus.inverse()
        # Established chronological order: K+,V+,K-,V-,K-,V-,K+,V+.
        for kinetic_sign, potential_circuit in [(1, plus), (-1, minus), (-1, minus), (1, plus)]:
            full_circuit.compose(kinetic_circuit if kinetic_sign == 1 else kinetic_circuit.inverse(),
                                 range(n_address), inplace=True)
            full_circuit.compose(potential_circuit, range(potential_circuit.num_qubits), inplace=True)
        paths = simulate_paths(oracle.circuit, n_address)
        assert not paths.bits[n_address:].any()
        block_state = apply_product_formula(block_state, paths.amplitudes, expm(1j * alpha * kinetic))

    actual = Statevector.from_instruction(full_circuit)
    expected = np.zeros(1 << width, dtype=complex)
    expected[:1 << n_address] = block_state
    np.testing.assert_allclose(actual.data, expected, atol=3e-13, rtol=0)
    # Ancillas are the high-order wires, so all nonzero-workspace amplitudes
    # lie outside the first four entries; check their total probability.
    leakage = float(np.vdot(actual.data[1 << n_address:], actual.data[1 << n_address:]).real)
    assert leakage < 1e-25

    restored = actual.evolve(full_circuit.inverse()).data
    zero = np.zeros(1 << width, dtype=complex)
    zero[0] = 1
    np.testing.assert_allclose(restored, zero, atol=4e-13, rtol=0)
