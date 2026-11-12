"""Direct spatial Fourier phase compilation for the smooth two-site model."""

from collections import defaultdict
import math
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit

from .baselines import Oracle, build_walsh_oracle
from .reversible import modular_add
from .smooth_flow import SmoothPotential, frequency_indices


def _projection(circuit, target, other, slope, carry):
    """Add slope times the other address into the target modulo its size."""
    for shift in range(len(target)):
        if (slope >> shift) & 1:
            width = len(target) - shift
            if width == 1:
                circuit.cx(other[0], target[shift])
            else:
                modular_add(circuit, other[:width], target[shift:], carry)


def build_fourier_oracle(potential: SmoothPotential, time: float, n: int, beta: float) -> Oracle:
    """Group Fourier modes by modular direction, shear, apply a 1D phase, undo.

    The spatial grid is centered and periodic. Time is fixed classically. The
    compiler uses the learned mode weights directly and does not construct an
    n-by-n table. Values returned for validation are reconstructed afterwards.
    """
    start = perf_counter()
    if not isinstance(n, (int, np.integer)) or n < 2 or n & (n - 1) or not math.isfinite(beta):
        raise ValueError("Expected a power-of-two grid and a finite phase scale")
    n = int(n)
    weights = potential._coefficients(time) / potential.wavevector_norm
    frequencies = frequency_indices(potential.config.max_mode).astype(int)
    directions = defaultdict(list)
    constant = 0.0
    for (k, l), weight in zip(frequencies, weights, strict=True):
        centered_weight = -float(weight) if (k + l) % 2 else float(weight)
        scale = math.gcd(int(k), int(l), n)
        modulus = n // scale
        if modulus == 1:
            constant += centered_weight
            continue
        a, b = int(k) // scale % modulus, int(l) // scale % modulus
        axis = 0 if a % 2 else 1
        odd, other = (a, b) if axis == 0 else (b, a)
        slope = other * pow(odd, -1, modulus) % modulus
        directions[modulus, axis, slope].append((odd, centered_weight))

    bits = n.bit_length() - 1
    n_address = 2 * bits
    uses_carry = any(slope % (modulus // 2) != 0 for modulus, _, slope in directions)
    circuit = QuantumCircuit(n_address + int(uses_carry), name="direct_fourier_phase")
    circuit.global_phase = beta * constant
    diagonals = []
    for (modulus, axis, slope), tones in sorted(directions.items()):
        width = modulus.bit_length() - 1
        wires = [list(range(bits, bits + width)), list(range(width))]
        angles = np.zeros(modulus)
        indices = np.arange(modulus)
        for tone, weight in tones:
            angles += weight * np.cos(2 * np.pi * (tone * indices % modulus) / modulus)
        phase = build_walsh_oracle(angles, beta)
        shear = QuantumCircuit(circuit.num_qubits)
        _projection(shear, wires[axis], wires[1 - axis], slope, n_address)
        circuit.compose(shear, inplace=True)
        circuit.compose(phase.circuit, wires[axis], inplace=True)
        circuit.compose(shear.inverse(), inplace=True)
        diagonals.append((modulus, axis, slope, phase.phases))
    compile_seconds = perf_counter() - start

    first, second = np.indices((n, n))
    phases = np.full((n, n), beta * constant)
    for modulus, axis, slope, angles in diagonals:
        target, other = (first, second) if axis == 0 else (second, first)
        phases += angles[(target + slope * other) % modulus]
    return Oracle(circuit, n_address, phases.ravel(), {
        "method": "direct_fourier",
        "compile_seconds": compile_seconds,
        "fourier_modes": len(frequencies),
        "modular_directions": len(directions),
        "one_dimensional_phase_words": sum(modulus for modulus, _, _ in directions),
    })
