"""Measured logical circuit counts and explicitly MODELLED Clifford+T costs.

Architecture: abstract all-to-all logical qubits; no routing, code distance,
magic-state factories, physical errors, or wall-clock runtime are included.
Exact CCX uses the standard ancilla-free 7-T / 6-CNOT / T-depth-4 template.
Arbitrary Rz/P uses the planning model ceil(3 log2(1/delta)+10) T gates.
The leading term follows Ross--Selinger; +10 is an explicit modelling choice,
NOT a rigorous finite-epsilon upper bound or an executed synthesis result.
We budget replacement of dyadic special angles by exact pi/4 multiples separately. T-depth is a conservative macro
schedule: each CCX monopolises its wires for 4 T layers; rotations use one
T layer per modelled T. Both the optimistic leading estimate and a sensitivity
model ceil(4 log2(1/delta)+10) can be requested.
"""
import math
from fractions import Fraction
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def _pi_over_four_interval():
    """Exact Machin alternating-series enclosure, independent of numpy.pi."""
    def atan(q):
        lo = sum(Fraction((-1)**k, (2*k+1)*q**(2*k+1)) for k in range(32))
        return lo, lo + Fraction(1, 65*q**65)
    a, b = atan(5), atan(239)
    return 4*a[0]-b[1], 4*a[1]-b[0]


def _upward(value):
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


@lru_cache(maxsize=8192)
def special_rotation_error(angle):
    """Bound replacing a stored dyadic angle by the suggested exact k*pi/4.

    A floating quotient equal to an integer does not prove exact equality to
    irrational pi. P uses the full angle bound; this also safely bounds Rz.
    None means the gate must be treated as an arbitrary rotation.
    """
    if not math.isfinite(angle):
        raise ValueError("rotation angles must be finite")
    quarter = angle / (np.pi/4)
    k = round(quarter)
    if quarter != k:
        return None
    lo, hi = _pi_over_four_interval()
    value = Fraction(angle)
    return max(abs(value-k*lo), abs(value-k*hi))


def _rotation_cost(angle, per_rotation_error, slope, special=True):
    if special and special_rotation_error(angle) is not None:
        return abs(round(angle/(np.pi/4))) % 2, False
    return math.ceil(-slope * math.log2(per_rotation_error) + 10), True


def estimate_resources(circuit, n_address, synthesis_budget=1e-4, slope=3.0):
    if not (0 < synthesis_budget < 1):
        raise ValueError("synthesis budget must lie in (0,1)")
    ops = dict(circuit.count_ops())
    allowed = {"x","cx","ccx","p","rz","u1","h","z","s","sdg","t","tdg",
               "rccx","rccx_dg","swap","id","barrier"}
    unknown = set(ops)-allowed
    if unknown:
        raise ValueError(f"decompose gates before costing: {unknown}")
    rotations = [float(item.operation.params[0]) for item in circuit.data
                 if item.operation.name in ("p", "rz", "u1")]
    errors = [special_rotation_error(angle) for angle in rotations]
    special_error = sum((error for error in errors if error is not None), Fraction(0))
    # Reserve part of the declared total budget for exact Clifford/T replacement.
    # If the special-angle approximation is too coarse, synthesize those gates.
    exact_budget = Fraction(str(synthesis_budget))  # Preserve rational suballocations and decimal inputs.
    use_special = special_error < exact_budget / 2
    if not use_special:
        special_error = Fraction(0)
    arbitrary = [angle for angle, error in zip(rotations, errors)
                 if error is None or not use_special]
    remainder = exact_budget - special_error
    delta = float(remainder / max(1, len(arbitrary)))
    if Fraction(delta) > remainder / max(1, len(arbitrary)):
        delta = math.nextafter(delta, 0.0)
    if arbitrary and delta == 0.0:
        raise ValueError("Per-rotation synthesis tolerance is below the positive binary64 range")
    layer=[0]*circuit.num_qubits
    qindex={q:i for i,q in enumerate(circuit.qubits)}
    tcount=0
    for item in circuit.data:
        name=item.operation.name
        q=[qindex[x] for x in item.qubits]
        if name=="barrier" or not q: continue
        cost=0; duration=0
        if name=="ccx": cost,duration=7,4
        elif name in ("rccx","rccx_dg"): cost,duration=4,4
        elif name in ("t","tdg"): cost,duration=1,1
        elif name in ("p","rz","u1"):
            cost,_=_rotation_cost(float(item.operation.params[0]),delta,slope,use_special)
            duration=cost
        tcount += cost
        finish=max(layer[j] for j in q)+duration
        for j in q: layer[j]=finish
    return {
        "logical_qubits": circuit.num_qubits,
        "address_qubits": n_address,
        "ancilla_qubits": circuit.num_qubits-n_address,
        "gate_count": len(circuit.data), "native_depth": circuit.depth(),
        "cx": ops.get("cx",0), "ccx": ops.get("ccx",0),
        "rccx": ops.get("rccx",0)+ops.get("rccx_dg",0),
        "single_qubit_gates": sum(v for k,v in ops.items() if k in {"x","p","rz","u1","h","z","s","sdg","t","tdg"}),
        "cnot_decomposed": ops.get("cx",0)+6*ops.get("ccx",0)+3*(ops.get("rccx",0)+ops.get("rccx_dg",0))+3*ops.get("swap",0),
        "arbitrary_rotations": len(arbitrary),
        "t_count_estimate": tcount, "t_depth_macro_estimate": max(layer,default=0),
        "special_rotation_replacement_error_bound": _upward(special_error),
        "special_rotation_replacement_error_bound_exact": str(special_error),
        "rotation_error_budget_exact": str(exact_budget),
        "rotation_error_budget": float(exact_budget), "per_rotation_error": delta,
        "rotation_model_slope": slope, "rotation_model_offset": 10,
        "native_gate_counts": ops,
    }

