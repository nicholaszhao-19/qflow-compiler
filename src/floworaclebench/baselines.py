"""Established diagonal synthesis and table lookup baselines.

Walsh synthesis: Welch et al., NJP 16 033040 (2014), arXiv:1306.3991.
The dense implementation has 2**n-2 CNOTs, 2**n-1 Rz rotations, and no
ancillas. Approximation removes a certified l1 budget of Walsh coefficients.
It is a heuristic sparse approximation, not a claim of globally optimal
diagonal synthesis. Lookup uses coherent shared-prefix QROM, never QRAM.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from qiskit import QuantumCircuit


@dataclass
class Oracle:
    circuit: QuantumCircuit
    n_address: int
    phases: np.ndarray
    metadata: dict


def fwht(values):
    """Unnormalised fast Walsh-Hadamard transform, binary bit-mask order."""
    a = np.array(values, dtype=float, copy=True).reshape(-1)
    if len(a) == 0 or len(a) & (len(a)-1):
        raise ValueError("length must be a positive power of two")
    h = 1
    while h < len(a):
        blocks = a.reshape(-1, 2*h)
        lo, hi = blocks[:, :h].copy(), blocks[:, h:].copy()
        blocks[:, :h], blocks[:, h:] = lo + hi, lo - hi
        h *= 2
    return a


def phase_error(phases, reference_phases):
    """Exact spectral norm of a difference of diagonal unitaries."""
    return float(np.max(np.abs(np.exp(1j*np.asarray(phases))
                               - np.exp(1j*np.asarray(reference_phases)))))


def select_phase_bits(values, beta, budget):
    """Smallest phase-word precision meeting the same exhaustive grid test as TT."""
    if not (0 < budget < 2):
        raise ValueError("phase-error budget must lie in (0,2)")
    theta=beta*np.asarray(values,dtype=float).reshape(-1)
    if not np.isfinite(theta).all():
        raise ValueError("finite phases required")
    principal=np.remainder(np.angle(np.exp(1j*theta)),2*np.pi)
    upper=max(1,int(np.ceil(np.log2(np.pi/budget))))
    if upper>50:
        raise ValueError("requested phase precision exceeds float64 audit range")
    for bits in range(1,upper+1):
        words=np.rint(principal*2**bits/(2*np.pi)).astype(np.int64)%(2**bits)
        if phase_error(2*np.pi*words/2**bits,theta)<=budget:
            return bits
    raise RuntimeError("floating-point phase rounding did not meet the target")


def build_walsh_oracle(values, beta=1.0, approximation_budget=0.0):
    start = perf_counter()
    theta = beta * np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(theta).all() or approximation_budget < 0:
        raise ValueError("finite phases and nonnegative approximation budget required")
    n = len(theta).bit_length()-1
    if n < 1 or 2**n != len(theta):
        raise ValueError("at least two, and a power-of-two number of values required")
    coeff = fwht(theta)/len(theta)
    dropped = 0.0
    # Never discard the global phase: operator error here includes global phase.
    for j in np.argsort(np.abs(coeff[1:]), kind="stable")+1:
        candidate = abs(float(coeff[j]))
        if dropped + candidate <= approximation_budget:
            dropped += candidate
            coeff[j] = 0.0
        else:
            break
    qc = QuantumCircuit(n, name="Walsh_phase")
    qc.global_phase = float(np.angle(np.exp(1j*coeff[0])))
    for pivot in range(n):
        current = 0
        for k in range(2**pivot):
            gray = k ^ (k >> 1)
            c = float(coeff[(1 << pivot) | gray])
            if c == 0.0:
                continue
            changes = current ^ gray
            while changes:
                bit = (changes & -changes).bit_length()-1
                qc.cx(bit, pivot)
                changes &= changes-1
            qc.rz(-2*c, pivot)
            current = gray
        while current:
            bit = (current & -current).bit_length()-1
            qc.cx(bit, pivot)
            current &= current-1
    phases = fwht(coeff)
    return Oracle(qc, n, phases, {
        "method": "walsh", "compile_seconds": perf_counter()-start,
        "discarded_l1_bound": dropped, "approximation_budget": approximation_budget,
        "operator_error": phase_error(phases, theta),
        "retained_walsh_terms": int(np.count_nonzero(coeff[1:])),
        "dense_table_bytes": theta.nbytes,
    })


def build_qrom_oracle(values, beta=1.0, phase_bits=12):
    """Round angles modulo 2pi; lookup, phase the word, uncompute lookup.

    Nearest-word rounding gives operator error <= pi*2**(-phase_bits).
    No value-amplitude oracle or free data access is assumed.
    """
    from .reversible import qrom_xor
    start = perf_counter()
    theta = beta*np.asarray(values, dtype=float).reshape(-1)
    n = len(theta).bit_length()-1
    if (2**n != len(theta) or n < 1 or not isinstance(phase_bits,(int,np.integer))
        or not (1 <= phase_bits <= 50) or not np.isfinite(theta).all()):
        raise ValueError("power-of-two input and 1 <= phase_bits <= 50 required")
    principal = np.angle(np.exp(1j*theta))
    words = np.rint(np.remainder(principal, 2*np.pi)*(2**phase_bits)/(2*np.pi))
    words = words.astype(np.int64) % (2**phase_bits)
    scratch_count = max(0, n-1)
    total = n+phase_bits+scratch_count
    compute = QuantumCircuit(total, name="QROM_lookup")
    address = list(range(n))
    data = list(range(n, n+phase_bits))
    scratch = list(range(n+phase_bits, total))
    qrom_xor(compute, address, data, words.tolist(), scratch)
    qc = QuantumCircuit(total, name="QROM_phase")
    qc.compose(compute, inplace=True)
    for j, qubit in enumerate(data):
        qc.p(2*np.pi*2.0**(j-phase_bits), qubit)
    qc.compose(compute.inverse(), inplace=True)
    phases = 2*np.pi*words.astype(float)/(2**phase_bits)
    return Oracle(qc, n, phases, {
        "method": "qrom", "compile_seconds": perf_counter()-start,
        "phase_bits": phase_bits, "rounding_bound": np.pi*2.0**(-phase_bits),
        "operator_error": phase_error(phases, theta),
        "lookup_gates": dict(compute.count_ops()), "lookup_depth": compute.depth(),
        "dense_table_bytes": theta.nbytes,
    })

