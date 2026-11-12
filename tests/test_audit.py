"""Adversarial checks added independently of the baseline implementation."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from floworaclebench.baselines import build_qrom_oracle, build_walsh_oracle
from floworaclebench.resources import estimate_resources
from floworaclebench.reversible import build_tt_oracle
from floworaclebench.simulation import simulate_paths, validate_oracle


@pytest.mark.parametrize("magnitude", [1.0, 1.0e4, 1.0e12])
def test_qrom_phase_wrapping_respects_rounding_budget(magnitude):
    phases = np.array([-magnitude, magnitude, magnitude + 1, -2 * np.pi - 1e-9])
    oracle = build_qrom_oracle(phases, phase_bits=20)
    measured = validate_oracle(oracle.circuit, oracle.n_address, phases)
    assert measured["basis_phase_error"] <= oracle.metadata["rounding_bound"] + 3e-14
    assert measured["address_preserved"] and measured["ancillas_clean"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_qrom_rejects_nonfinite_phases(invalid):
    with pytest.raises(ValueError):
        build_qrom_oracle([0.0, invalid])


def test_walsh_preserves_absolute_global_phase_even_large_constant():
    phases = np.full(4, 1.0e12)
    oracle = build_walsh_oracle(phases)
    result = validate_oracle(oracle.circuit, oracle.n_address, phases)
    assert result["basis_phase_error"] < 3e-14


@pytest.mark.parametrize("budget", [0.0, 1e-4, 0.01, 0.2, 1.0])
def test_pruned_walsh_bound_and_independent_dense_qiskit(budget):
    phases = np.random.default_rng(37).normal(size=16)
    oracle = build_walsh_oracle(phases, approximation_budget=budget)
    matrix = Operator(oracle.circuit).data
    err = np.linalg.norm(matrix - np.diag(np.exp(1j * phases)), ord=2)
    assert err <= budget + 2e-14
    assert abs(err - oracle.metadata["operator_error"]) < 2e-14
    assert np.max(np.abs(matrix - np.diag(np.exp(1j * oracle.phases)))) < 2e-14


def test_near_clifford_rotation_not_exact_at_tighter_error_budget():
    qc = QuantumCircuit(1)
    qc.p(np.pi / 4 + 1e-12, 0)
    costs = estimate_resources(qc, 1, synthesis_budget=1e-13)
    assert costs["arbitrary_rotations"] == 1
    assert costs["t_count_estimate"] > 1


def test_toffoli_cnot_and_t_depth_use_one_consistent_template():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    costs = estimate_resources(qc, 3)
    assert costs["t_count_estimate"] == 7
    # Standard six-CNOT circuit has T-depth four; the Amy et al. T-depth-three
    # circuit uses seven CNOTs. Mixing their best entries is not one template.
    assert (costs["cnot_decomposed"], costs["t_depth_macro_estimate"]) in {(6, 4), (7, 3)}


def test_t_depth_macro_parallel_and_dependent_toffolis():
    parallel = QuantumCircuit(6)
    parallel.ccx(0, 1, 2)
    parallel.ccx(3, 4, 5)
    dependent = QuantumCircuit(5)
    dependent.ccx(0, 1, 2)
    dependent.ccx(2, 3, 4)
    a, b = estimate_resources(parallel, 6), estimate_resources(dependent, 5)
    assert a["t_count_estimate"] == b["t_count_estimate"] == 14
    assert b["t_depth_macro_estimate"] == 2 * a["t_depth_macro_estimate"]


def test_sparse_relative_phase_simulation_matches_qiskit():
    qc = QuantumCircuit(3)
    qc.rccx(0, 1, 2)
    qc.p(-0.4, 2)
    qc.rccx(0, 1, 2)
    qc.rz(0.3, 0)
    qc.global_phase = 0.7
    paths = simulate_paths(qc, 3)
    matrix = Operator(qc).data
    for column in range(8):
        row = sum(int(paths.bits[j, column]) << j for j in range(3))
        assert abs(matrix[row, column] - paths.amplitudes[column]) < 2e-14
        assert np.sum(np.abs(matrix[:, column]) > 1e-14) == 1


def test_duplicate_paths_do_not_count_as_exhaustive_validation():
    circuit = QuantumCircuit(2)
    phases = np.zeros(4)
    assert validate_oracle(circuit, 2, phases, addresses=[3, 1, 0, 2])["exhaustive"]
    result = validate_oracle(circuit, 2, phases, addresses=[0, 0, 1, 2])
    assert result["validated_inputs"] == 4
    assert not result["exhaustive"]


def test_signed_phase_large_beta_uses_accurate_range_reduction():
    values = np.array([-2.0, -1.0, 1.0, 3.0])
    oracle = build_tt_oracle([values.reshape(1, 4, 1)], beta=1e12, fraction_bits=0)
    result = validate_oracle(oracle.circuit, 2, 1e12 * values)
    assert result["basis_phase_error"] < 3e-14
    assert result["ancillas_clean"] and result["inverse_ancillas_clean"]


def test_signed_accumulator_boundary_and_zero_cores():
    for values in (np.array([-8, -1, 0, 7]), np.zeros(4)):
        oracle = build_tt_oracle([values.reshape(1, 4, 1)], beta=0.7, fraction_bits=0)
        result = validate_oracle(oracle.circuit, 2, 0.7 * values)
        assert result["basis_phase_error"] < 2e-14
        assert result["address_preserved"] and result["ancillas_clean"]
