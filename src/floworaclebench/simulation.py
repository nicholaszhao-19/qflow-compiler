"""Gate-level verification for wide reversible arithmetic circuits.

X, CX, CCX and diagonal gates are monomial: every computational-basis input
follows exactly one basis path. We propagate a vector of such paths and their
complex amplitudes. This is exact gate simulation (up to float phases), not
replacement by the classically contracted TT. It avoids allocating 2**q
amplitudes for clean-workspace reversible circuits. It is NOT a general
quantum-circuit simulator and gives no claim of quantum speedup.
"""
from dataclasses import dataclass
import numpy as np
from qiskit.quantum_info import Operator


@dataclass
class PathResult:
    bits: np.ndarray
    amplitudes: np.ndarray


def simulate_paths(circuit, n_address, addresses=None, amplitudes=None):
    if addresses is None:
        addresses = np.arange(2**n_address, dtype=np.int64)
    addresses = np.asarray(addresses, dtype=np.int64)
    if np.any(addresses < 0) or np.any(addresses >= 2**n_address):
        raise ValueError("address outside register")
    bits = np.zeros((circuit.num_qubits, len(addresses)), dtype=np.bool_)
    for j in range(n_address):
        bits[j] = ((addresses >> j)&1).astype(bool)
    if amplitudes is None:
        amps = np.ones(len(addresses), dtype=complex)
    else:
        amps = np.array(amplitudes, dtype=complex, copy=True)
        if amps.shape != addresses.shape:
            raise ValueError("one amplitude per path required")
    amps *= np.exp(1j*float(circuit.global_phase))
    qindex = {q:i for i,q in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        op = instruction.operation
        q = [qindex[b] for b in instruction.qubits]
        if op.name == "x":
            bits[q[0]] ^= True
        elif op.name == "cx":
            bits[q[1]] ^= bits[q[0]]
        elif op.name == "ccx":
            bits[q[2]] ^= bits[q[0]] & bits[q[1]]
        elif op.name in ("p", "u1", "rz"):
            angle = float(op.params[0])
            if op.name == "rz":
                amps *= np.exp(1j*angle*(bits[q[0]].astype(float)-0.5))
            else:
                amps *= np.exp(1j*angle*bits[q[0]])
        elif op.name in ("z", "s", "sdg", "t", "tdg"):
            angle = {"z":np.pi,"s":np.pi/2,"sdg":-np.pi/2,
                     "t":np.pi/4,"tdg":-np.pi/4}[op.name]
            amps *= np.exp(1j*angle*bits[q[0]])
        elif op.name in ("rccx", "rccx_dg"):
            mat = Operator(op).data
            local = sum((bits[b].astype(np.int64)<<j) for j,b in enumerate(q))
            rows = np.argmax(np.abs(mat), axis=0)[local]
            amps *= mat[rows, local]
            for j,b in enumerate(q):
                bits[b] = ((rows>>j)&1).astype(bool)
        elif op.name in ("barrier", "id"):
            continue
        else:
            raise ValueError(f"Unsupported non-monomial or undecomposed gate: {op.name}")
    return PathResult(bits, amps)


def validate_oracle(circuit, n_address, target_phases, addresses=None, seed=17):
    if addresses is None:
        addresses = np.arange(2**n_address)
    addresses = np.asarray(addresses, dtype=np.int64)
    target_phases = np.asarray(target_phases)[addresses]
    result = simulate_paths(circuit, n_address, addresses)
    expected_bits = ((addresses[None,:] >> np.arange(n_address)[:,None])&1).astype(bool)
    address_ok = np.array_equal(result.bits[:n_address], expected_bits)
    clean = not result.bits[n_address:].any()
    basis_error = float(np.max(np.abs(result.amplitudes-np.exp(1j*target_phases))))
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=len(addresses))+1j*rng.normal(size=len(addresses))
    initial /= np.linalg.norm(initial)
    superposed = simulate_paths(circuit, n_address, addresses, initial)
    superposition_error = float(np.linalg.norm(superposed.amplitudes-initial*np.exp(1j*target_phases)))
    # Independently run the inverse, rather than trusting cancellation in a
    # concatenated circuit. For a clean diagonal isometry it must give -phase.
    backward = simulate_paths(circuit.inverse(), n_address, addresses)
    inverse_error = float(np.max(np.abs(result.amplitudes*backward.amplitudes-1)))
    inverse_clean = not backward.bits[n_address:].any()
    return {
        "validated_inputs": len(addresses),
        "exhaustive": len(addresses) == 2**n_address and np.unique(addresses).size == len(addresses),
        "address_preserved": bool(address_ok), "ancillas_clean": bool(clean),
        "inverse_ancillas_clean": bool(inverse_clean),
        "basis_phase_error": basis_error, "superposition_error": superposition_error,
        "inverse_error": inverse_error,
    }

