"""Exact independent references and gate-path checks for mixed precision TT."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from floworaclebench.precision_tt import build_precision_tt_oracle, plan_precision_tt
from floworaclebench.reversible import build_tt_oracle
from floworaclebench.simulation import validate_oracle


def exact_reference(cores, addresses, core_bits=None, stage_bits=None):
    vector = [Fraction(1)]
    prefixes = []
    for j, (core, address) in enumerate(zip(cores, addresses)):
        matrix = [[Fraction(float(value)) for value in row] for row in core[:, address, :]]
        if core_bits is not None:
            step = 1 << core_bits[j]
            matrix = [[Fraction(round(value * step), step) for value in row] for row in matrix]
        dot = [sum(a * b for a, b in zip(vector, col)) for col in zip(*matrix)]
        if stage_bits is not None:
            step = 1 << stage_bits[j]
            vector = [Fraction(value * step // 1, step) for value in dot]
        else:
            vector = dot
        prefixes.append((dot, vector))
    return vector[0], prefixes


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_certified_rounded_rectangular_tt_exhaustive(strategy):
    rng = np.random.default_rng(693)
    cores = [rng.normal(size=shape) for shape in ((1, 2, 2), (2, 4, 3), (3, 2, 2), (2, 2, 1))]
    # The second core includes a constant negative coefficient and a zero.
    cores[1][0, :, 0] = -0.5
    cores[1][1, :, 1] = 0.0
    core_bits, stage_bits = (4, 2, 3, 2), (2, 2, 1, 0)
    plan = plan_precision_tt(cores, core_bits, stage_bits)
    assert plan.shifts == (2, 2, 4, 3)
    actual = plan.evaluate_exact()
    for address in product(*(range(core.shape[1]) for core in cores)):
        reference, _ = exact_reference(cores, address)
        quantized, _ = exact_reference(cores, address, core_bits)
        rounded, prefixes = exact_reference(cores, address, core_bits, stage_bits)
        assert rounded == Fraction(actual[address], 1 << stage_bits[-1])
        assert abs(reference - quantized) <= plan.coefficient_error_bound
        assert abs(quantized - rounded) <= plan.rounding_error_bound
        assert abs(reference - rounded) <= plan.potential_error_bound
        for j, (before, after) in enumerate(prefixes):
            raw_scale = 1 << (stage_bits[j] + plan.shifts[j])
            retained_scale = 1 << stage_bits[j]
            for b, (raw, retained) in enumerate(zip(before, after)):
                lo, hi = plan.raw_intervals[j][b]
                assert lo <= raw * raw_scale <= hi
                assert -(1 << (plan.raw_widths[j][b] - 1)) <= lo <= hi < 1 << (plan.raw_widths[j][b] - 1)
                lo, hi = plan.retained_intervals[j][b]
                assert lo <= retained * retained_scale <= hi
                assert -(1 << (plan.retained_widths[j][b] - 1)) <= lo <= hi < 1 << (plan.retained_widths[j][b] - 1)
    beta = Fraction(-1, 7)
    compiled = build_precision_tt_oracle(plan, beta, workspace_strategy=strategy)
    assert plan.phase_parameter_error_bound(beta) > 0
    assert plan.phase_error_bound(beta) == abs(beta) * plan.potential_error_bound + plan.phase_parameter_error_bound(beta)
    result = validate_oracle(compiled.circuit, compiled.n_address, float(beta) * compiled.quantized_values)
    assert result["exhaustive"] and result["ancillas_clean"] and result["inverse_ancillas_clean"] and result["address_preserved"]
    assert max(result[key] for key in ("basis_phase_error", "superposition_error", "inverse_error")) < 2e-14


def test_homogeneous_exact_strategy_matches_v02():
    rng = np.random.default_rng(693)
    cores = [rng.uniform(-1.0, 1.0, size=shape) for shape in ((1, 4, 2), (2, 4, 3), (3, 4, 1))]
    beta = 0.5
    plan = plan_precision_tt(cores, 3)
    assert plan.stage_fraction_bits == (3, 6, 9)
    assert plan.rounding_error_bound == 0
    compiled = build_precision_tt_oracle(plan, beta)
    baseline = build_tt_oracle(cores, beta, 3)
    assert np.array_equal(compiled.quantized_values, baseline.quantized_values)
    assert compiled.circuit.num_qubits <= baseline.circuit.num_qubits
    assert plan.phase_parameter_error_bound(beta) == 0


def test_dense_two_coordinate_superposition_and_inverse():
    cores = [np.array([-1.0, 0.0]).reshape(1, 2, 1), np.array([-0.5, 0.5]).reshape(1, 2, 1)]
    plan = plan_precision_tt(cores, (0, 1), (0, 0))
    assert plan.retained_intervals[-1] == ((-1, 0),)
    compiled = build_precision_tt_oracle(plan, 0.25)
    state = np.zeros(1 << compiled.circuit.num_qubits, dtype=complex)
    state[:4] = np.array([1, 1j, -1, -1j]) / 2
    actual = Statevector(state).evolve(compiled.circuit).data
    expected = state.copy()
    expected[:4] *= np.exp(0.25j * np.array([0, -1, 0, 0]))
    assert np.allclose(actual, expected, rtol=0, atol=2e-14)
    assert np.allclose(Statevector(actual).evolve(compiled.circuit.inverse()).data, state, rtol=0, atol=2e-14)


def test_floor_discards_entire_raw_register_and_keeps_sign():
    # The raw integer dot is +/-1 with a scale of 2**8. Two raw bits suffice,
    # but an arithmetic shift by eight must still retain its negative sign.
    cores = [np.array([1, -1]).reshape(1, 2, 1) / 16, np.ones((1, 1, 1)) / 16]
    plan = plan_precision_tt(cores, (4, 4), (4, 0))
    assert plan.raw_widths[-1] == (2,)
    assert plan.shifts[-1] == 8
    assert plan.evaluate_exact().reshape(-1).tolist() == [0, -1]
    compiled = build_precision_tt_oracle(plan, 0.4)
    result = validate_oracle(compiled.circuit, compiled.n_address, 0.4 * compiled.quantized_values)
    assert result["basis_phase_error"] < 2e-14 and result["ancillas_clean"]


def test_widths_account_for_values_beyond_int64_and_zero_bit_addresses():
    cores = [np.full((1, 1, 1), -2.0) for _ in range(4)]
    plan = plan_precision_tt(cores, 17)
    assert plan.evaluate_exact().item() == 1 << 72
    assert plan.raw_widths[-1] == (74,)
    assert plan.potential_error_bound == 0
    compiled = build_precision_tt_oracle(plan, 0.01, materialize_values=False)
    assert compiled.quantized_values is None
    assert compiled.n_address == 0
    assert compiled.metadata["dirty_ancilla_qubits"] == 0
    result = validate_oracle(compiled.circuit, 0, [0.16])
    assert result["basis_phase_error"] < 2e-14 and result["ancillas_clean"]


def test_zero_and_nearest_even_and_input_validation():
    core = np.array([-1.5, -0.5, 0.5, 1.5]).reshape(1, 4, 1)
    plan = plan_precision_tt([core], 0)
    assert plan.evaluate_exact().tolist() == [-2, 0, 0, 2]
    assert plan.phase_error_bound(0) == 0
    assert plan.phase_error_bound(1, tt_error=Fraction(1, 8), synthesis_error=Fraction(1, 16)) == Fraction(11, 16)
    assert plan.phase_error_bound(4, tt_error=Fraction(1, 8)) == 2
    assert plan_precision_tt([core, core], (3, 4), 2).stage_fraction_bits == (2, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        plan_precision_tt([core], 0, [1])
    with pytest.raises(ValueError, match="nonnegative"):
        plan.phase_error_bound(1, synthesis_error=-0.1)
    with pytest.raises(ValueError, match="workspace_strategy"):
        build_precision_tt_oracle(plan, 1, workspace_strategy="unknown")
