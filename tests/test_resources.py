import numpy as np
from qiskit import QuantumCircuit
from floworaclebench.resources import estimate_resources


def test_exact_clifford_t_and_toffoli_costs():
    qc=QuantumCircuit(3)
    qc.ccx(0,1,2); qc.p(np.pi/2,0); qc.p(np.pi/4,1)
    result=estimate_resources(qc,1)
    assert result["t_count_estimate"]==8
    assert result["arbitrary_rotations"]==0
    assert result["cnot_decomposed"]==6
    assert result["ancilla_qubits"]==2


def test_tighter_budget_costs_more():
    qc=QuantumCircuit(1); qc.rz(.1234,0)
    assert estimate_resources(qc,1,1e-6)["t_count_estimate"]>estimate_resources(qc,1,1e-3)["t_count_estimate"]


def test_dyadic_special_angles_have_explicit_error_allowance():
    from fractions import Fraction
    qc=QuantumCircuit(1); qc.p(np.pi/4,0); qc.rz(np.pi/2,0); qc.p(.123,0)
    result=estimate_resources(qc,1)
    error=Fraction(result["special_rotation_replacement_error_bound_exact"])
    assert 0 < error < Fraction(1,10**14)
    assert Fraction(result["per_rotation_error"])*result["arbitrary_rotations"]+error <= Fraction(1e-4)


def test_tiny_budget_synthesizes_nonexact_special_angles():
    qc=QuantumCircuit(1); qc.p(np.pi/4,0)
    result=estimate_resources(qc,1,1e-20)
    assert result["arbitrary_rotations"]==1
    assert result["special_rotation_replacement_error_bound"]==0


def test_zero_angle_cost_remains_zero():
    qc=QuantumCircuit(1); qc.p(0,0)
    result=estimate_resources(qc,1,1e-20)
    assert result["t_count_estimate"]==0
    assert result["special_rotation_replacement_error_bound"]==0


def test_repeated_accounting_respects_exact_global_budget():
    from fractions import Fraction
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_v03_preparation import _resource_total
    qc=QuantumCircuit(1); qc.p(np.pi/4,0); qc.rz(.1234,0)
    r=estimate_resources(qc,1)
    result=_resource_total([(r,2240)],0.01)
    assert Fraction(result["used_rotation_synthesis_allowance_exact"]) <= Fraction("0.01")
    assert Fraction(result["allocated_rotation_synthesis_budget_exact"]) == Fraction("0.01")


def test_repeated_accounting_rejects_insufficient_special_budget():
    import pytest
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_v03_preparation import _resource_total
    qc=QuantumCircuit(1); qc.p(np.pi/4,0)
    r=estimate_resources(qc,1)
    with pytest.raises(ValueError,match="special-angle"):
        _resource_total([(r,100000)],1e-20)


def test_single_rotation_allocation_matches_decimal_certificate_budget():
    from fractions import Fraction
    qc=QuantumCircuit(1); qc.p(.1234,0)
    result=estimate_resources(qc,1,0.001)
    assert Fraction(result["per_rotation_error"]) <= Fraction("0.001")
    assert Fraction(result["rotation_error_budget_exact"]) == Fraction("0.001")


def test_rotation_projection_supports_subnormal_tolerances():
    import math

    circuit = QuantumCircuit(1)
    circuit.p(0.1234, 0)
    result = estimate_resources(circuit, 1, 1e-310)
    tolerance = result["per_rotation_error"]
    assert 0 < tolerance <= 1e-310
    assert result["t_count_estimate"] == math.ceil(-3 * math.log2(tolerance) + 10)
