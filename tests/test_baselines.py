import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from floworaclebench.baselines import fwht, build_walsh_oracle, build_qrom_oracle
from floworaclebench.simulation import validate_oracle, simulate_paths


@pytest.mark.parametrize("n", [1,2,4,6])
def test_walsh_against_full_qiskit_operator(n):
    rng=np.random.default_rng(2+n)
    values=rng.normal(size=2**n)
    oracle=build_walsh_oracle(values,beta=.31)
    matrix=Operator(oracle.circuit).data
    np.testing.assert_allclose(matrix,np.diag(np.exp(.31j*values)),atol=2e-13)
    assert oracle.circuit.count_ops().get("cx",0)==2**n-2
    result=validate_oracle(oracle.circuit,n,.31*values)
    assert result["ancillas_clean"] and result["address_preserved"]
    assert result["basis_phase_error"]<2e-13


def test_walsh_pruning_error_budget():
    rng=np.random.default_rng(5)
    theta=rng.normal(size=64)
    oracle=build_walsh_oracle(theta,approximation_budget=.13)
    err=np.linalg.norm(Operator(oracle.circuit).data-np.diag(np.exp(1j*theta)),ord=2)
    assert err <= oracle.metadata["discarded_l1_bound"]+1e-13
    assert err <= .13+1e-13


def test_qrom_actual_statevector_and_inverse():
    values=np.array([-.5,1.3,2.9,6.4])
    oracle=build_qrom_oracle(values,phase_bits=5)
    state=np.zeros(2**oracle.circuit.num_qubits,complex)
    state[:4]=np.array([1,1j,-1,-1j])/2
    actual=Statevector(state).evolve(oracle.circuit).data
    expected=np.zeros_like(state); expected[:4]=state[:4]*np.exp(1j*oracle.phases)
    np.testing.assert_allclose(actual,expected,atol=2e-13)
    restored=Statevector(actual).evolve(oracle.circuit.inverse()).data
    np.testing.assert_allclose(restored,state,atol=2e-13)
    result=simulate_paths(oracle.circuit,2,amplitudes=state[:4])
    np.testing.assert_allclose(result.amplitudes,actual[:4],atol=2e-13)
    assert not result.bits[2:].any()


def test_fwht_involution():
    a=np.arange(32.)
    np.testing.assert_allclose(fwht(fwht(a))/len(a),a)

