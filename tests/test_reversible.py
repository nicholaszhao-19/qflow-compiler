"""Independent basis-path tests, plus dense Qiskit cross-checks on small circuits."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from floworaclebench.reversible import (
    build_tt_oracle,
    contract_cores,
    modular_add,
    multiply_accumulate,
    qrom_xor,
    quantization_bound,
    quantize_cores,
)


def basis_program(circuit):
    """Compile Qiskit gates into a separate integer/complex test interpreter."""
    program = []
    for item in circuit.data:
        name = item.operation.name
        wires = tuple(circuit.find_bit(q).index for q in item.qubits)
        if name not in ("x", "cx", "ccx", "p"):
            raise AssertionError(f"Unexpected non-elementary gate {name}")
        phase = complex(np.exp(1j * float(item.operation.params[0]))) if name == "p" else None
        program.append((name, tuple(1 << q for q in wires), phase))

    def apply(bits):
        value, amplitude = int(bits), complex(np.exp(1j * float(circuit.global_phase)))
        for name, masks, phase in program:
            if name == "x":
                value ^= masks[0]
            elif name == "cx":
                if value & masks[0]:
                    value ^= masks[1]
            elif name == "ccx":
                if value & masks[0] and value & masks[1]:
                    value ^= masks[2]
            elif value & masks[0]:
                amplitude *= phase
        return value, amplitude

    return apply


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_ripple_add_exhaustive(width):
    qc = QuantumCircuit(2 * width + 1)
    modular_add(qc, range(width), range(width, 2 * width), 2 * width)
    evaluate = basis_program(qc)
    for a in range(1 << width):
        for b in range(1 << width):
            output, phase = evaluate(a | (b << width))
            assert output == a | (((a + b) % (1 << width)) << width)
            assert phase == 1
    qc_inverse = QuantumCircuit(2 * width + 1)
    modular_add(qc_inverse, range(width), range(width, 2 * width), 2 * width, inverse=True)
    inverse = basis_program(qc_inverse)
    for a in range(1 << width):
        for b in range(1 << width):
            output, _ = inverse(a | (b << width))
            assert output == a | (((b - a) % (1 << width)) << width)


def test_ripple_add_against_dense_qiskit_operator():
    qc = QuantumCircuit(5)
    modular_add(qc, [0, 1], [2, 3], 4)
    matrix = Operator(qc).data
    for a in range(4):
        for b in range(4):
            column = a | b << 2
            row = a | ((a + b) % 4) << 2
            assert np.allclose(matrix[:, column], np.eye(32)[:, row], atol=1e-14)


@pytest.mark.parametrize("k,width", [(0, 3), (1, 3), (2, 4), (3, 5), (4, 3)])
def test_qrom_all_addresses_nonzero_output_and_inverse(k, width):
    rng = np.random.default_rng(130 + k)
    words = rng.integers(-(1 << (width - 1)), 1 << (width - 1), 1 << k).tolist()
    n_scratch = max(0, k - 1)
    qc = QuantumCircuit(k + width + n_scratch)
    qrom_xor(qc, range(k), range(k, k + width), words, range(k + width, qc.num_qubits))
    evaluate, inverse = basis_program(qc), basis_program(qc.inverse())
    for x in range(1 << k):
        for y in range(1 << width):
            output, phase = evaluate(x | y << k)
            expected = x | ((y ^ (words[x] % (1 << width))) << k)
            assert output == expected  # also ensures every scratch bit is zero
            assert phase == 1
            assert inverse(output) == (x | y << k, 1)


def test_qrom_constant_and_repeated_subtrees():
    qc = QuantumCircuit(9)
    qrom_xor(qc, [0, 1, 2], [3, 4, 5, 6], [9, 9, 9, 9, 3, 3, 5, 5], [7, 8])
    run = basis_program(qc)
    for x, word in enumerate([9, 9, 9, 9, 3, 3, 5, 5]):
        assert run(x)[0] == x | (word << 3)
    # Factored constant bits are actual X gates, not an uncharged lookup.
    assert qc.count_ops().get("x", 0) > 0


@pytest.mark.parametrize("constant", [None, -3, -1, 0, 1, 3])
def test_signed_multiply_accumulate_and_clean_workspace(constant):
    # 2-bit signed multiplicand, 2-bit coefficient, 4-bit modular output.
    qc = QuantumCircuit(13)
    multiply_accumulate(qc, [0, 1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11], 12, coefficient_constant=constant)
    run = basis_program(qc)
    for a in range(4):
        a_signed = a if a < 2 else a - 4
        for b in range(4):
            b_signed = (b if b < 2 else b - 4) if constant is None else constant
            for accumulator in range(16):
                output, phase = run(a | (b << 2) | (accumulator << 4))
                assert output == a | (b << 2) | (((accumulator + a_signed * b_signed) % 16) << 4)
                assert phase == 1


def test_one_dimensional_oracle_dense_statevector_and_inverse():
    core = np.array([-1.0, -0.25, 0.75, 1.5]).reshape(1, 4, 1)
    result = build_tt_oracle([core], beta=0.7, fraction_bits=2)
    state = np.zeros(1 << result.circuit.num_qubits, dtype=complex)
    state[:4] = np.array([1, 1j, -1, -1j]) / 2
    actual = Statevector(state).evolve(result.circuit).data
    expected = state.copy()
    expected[:4] *= np.exp(0.7j * core.reshape(-1))
    assert np.max(np.abs(actual - expected)) < 2e-14
    restored = Statevector(actual).evolve(result.circuit.inverse()).data
    assert np.max(np.abs(restored - state)) < 2e-14


@pytest.mark.parametrize("grid", [4, 8])
def test_tt_all_addresses_signed_phases_clean_ancillas_inverse(grid):
    rng = np.random.default_rng(846)
    cores = [rng.uniform(-1.0, 1.0, (1, grid, 2)), rng.uniform(-1.0, 1.0, (2, grid, 1))]
    result = build_tt_oracle(cores, beta=-0.4, fraction_bits=2)
    run = basis_program(result.circuit)
    inverse = basis_program(result.circuit.inverse())
    ideal_integer = contract_cores(quantize_cores(cores, 2)).reshape(-1)
    ideal_values = np.asarray(ideal_integer, dtype=float) / 16
    assert np.allclose(result.quantized_values, ideal_values)
    for x, value in enumerate(ideal_values):
        output, phase = run(x)
        assert output == x  # includes every allocated ancilla, not only data
        assert abs(phase - np.exp(-0.4j * value)) < 2e-13
        restored, inverse_phase = inverse(output)
        assert restored == x
        assert abs(phase * inverse_phase - 1) < 2e-13


def test_four_dimensional_and_constant_coefficient_optimization():
    rng = np.random.default_rng(312)
    cores = [rng.uniform(-0.5, 0.5, (1, 2, 2)), rng.uniform(-0.5, 0.5, (2, 2, 2)), np.ones((2, 2, 2)) / 2, rng.uniform(-0.5, 0.5, (2, 2, 1))]
    result = build_tt_oracle(cores, beta=0.3, fraction_bits=2)
    run = basis_program(result.circuit)
    for x, value in enumerate(result.quantized_values):
        output, phase = run(x)
        assert output == x
        assert abs(phase - np.exp(0.3j * value)) < 2e-13
    assert result.metadata["final_fraction_bits"] == 8


def test_quantization_bound_and_large_integer_arithmetic():
    rng = np.random.default_rng(32)
    cores = [rng.normal(size=(1, 4, 3)), rng.normal(size=(3, 4, 2)), rng.normal(size=(2, 4, 1))]
    integer = quantize_cores(cores, 24)
    exact = contract_cores(cores)
    approx = np.asarray(contract_cores(integer), dtype=float) * 2.0 ** -72
    bound = quantization_bound(cores, integer, 24)
    assert np.max(np.abs(exact - approx)) <= bound + 1e-14
    assert max(abs(int(v)) for v in contract_cores(integer).flat).bit_length() > 64


def test_compilation_does_not_require_exponential_table():
    cores = [np.ones((1, 4, 1)) / 2 for _ in range(8)]
    result = build_tt_oracle(cores, beta=0.3, fraction_bits=1, materialize_values=False)
    assert result.quantized_values is None
    assert result.n_address == 16
    assert result.metadata["coefficient_table_entries"] == 32


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        build_tt_oracle([np.ones((2, 4, 1))], 1, 2)
    with pytest.raises(ValueError):
        build_tt_oracle([np.ones((1, 3, 1))], 1, 2)
    with pytest.raises(ValueError):
        build_tt_oracle([np.full((1, 4, 1), np.nan)], 1, 2)
    with pytest.raises(ValueError):
        build_tt_oracle([np.ones((1, 4, 1))], 1, -1)
