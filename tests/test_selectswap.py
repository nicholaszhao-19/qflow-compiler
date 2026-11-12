import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from floworaclebench.baselines import build_qrom_oracle
from floworaclebench.selectswap import build_selectswap_oracle
from floworaclebench.simulation import simulate_paths, validate_oracle


@pytest.mark.parametrize("block_bits", [0,1,2])
def test_selectswap_full_qiskit_superposition_and_inverse(block_bits):
    # n=2 and b=3: even the largest bank has a tractable full statevector.
    values = np.array([-.7,1.8,3.2,6.0])
    oracle = build_selectswap_oracle(values, beta=.4, phase_bits=3, block_bits=block_bits)
    state = np.zeros(2**oracle.circuit.num_qubits, dtype=complex)
    state[:4] = np.array([1,1j,-1,-1j])/2
    expected = state.copy()
    expected[:4] *= np.exp(1j*oracle.phases)
    actual = Statevector(state).evolve(oracle.circuit).data
    np.testing.assert_allclose(actual,expected,atol=2e-13)
    restored = Statevector(actual).evolve(oracle.circuit.inverse()).data
    np.testing.assert_allclose(restored,state,atol=2e-13)
    sparse = simulate_paths(oracle.circuit,2,amplitudes=state[:4])
    np.testing.assert_allclose(sparse.amplitudes,actual[:4],atol=2e-13)
    assert not sparse.bits[2:].any()


@pytest.mark.parametrize("block_bits", [0,1,2,3,4])
def test_selectswap_exhaustive_four_bit_address(block_bits):
    rng = np.random.default_rng(100+block_bits)
    values = rng.normal(size=16)*10
    oracle = build_selectswap_oracle(values,beta=-.37,phase_bits=7,block_bits=block_bits)
    result = validate_oracle(oracle.circuit,4,oracle.phases)
    assert result["ancillas_clean"] and result["address_preserved"]
    assert result["basis_phase_error"] < 2e-13
    assert oracle.metadata["operator_error"] <= oracle.metadata["rounding_bound"] + 1e-13
    assert oracle.metadata["selection_gates"].get("ccx",0) == (2**block_bits-1)*7


def test_zero_block_bits_matches_qrom_exactly():
    values = np.linspace(-3,7,16)
    a = build_qrom_oracle(values,beta=.9,phase_bits=5)
    b = build_selectswap_oracle(values,beta=.9,phase_bits=5,block_bits=0)
    np.testing.assert_array_equal(a.phases,b.phases)
    assert a.circuit.count_ops() == b.circuit.count_ops()
    assert a.circuit.num_qubits == b.circuit.num_qubits


def test_invalid_selectswap_configuration_rejected():
    for kw in [{"block_bits":-1},{"block_bits":3},{"phase_bits":0},{"beta":np.nan}]:
        with pytest.raises(ValueError):
            build_selectswap_oracle([0,1,2,3],**kw)
