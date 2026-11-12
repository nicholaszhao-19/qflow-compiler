"""Clean reversible tensor-train evaluation using explicit elementary gates.

All arithmetic is integer arithmetic modulo a power of two.  Register bounds
ensure that the *final* interpreted signed values equal the exact integer TT
contraction; intermediate modular wraparound is harmless.  There is no hidden
floating-point oracle or truncation operation in the quantum circuit.

References: Cuccaro et al., arXiv:quant-ph/0410184, Secs. 2 and 4.1;
Vedral et al., arXiv:quant-ph/9511018; Babbush et al., arXiv:1805.03662.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class TTOracle:
    circuit: QuantumCircuit
    n_address: int
    quantized_values: np.ndarray | None
    metadata: dict
    quantized_cores: list[np.ndarray]


def qrom_xor(
    circuit: QuantumCircuit,
    address: Sequence[int],
    data: Sequence[int],
    words: Sequence[int],
    scratch: Sequence[int],
) -> None:
    """Append |x>|y>|0> -> |x>|y XOR words[x]>|0>.

    Address and data wires are little endian.  ``words`` has exactly
    ``2**len(address)`` integer entries, encoded modulo ``2**len(data)``.
    Scratch needs ``max(0, len(address)-1)`` clean qubits, disjoint from the
    other registers.  All scratch is restored.  Constant columns and identical
    subtrees are removed classically.  A clean shared-prefix traversal needs
    at most ``2*(2**k - 2)`` CCX gates when k >= 1, independent of word width.

    This is a coherent prefix-tree QROM, not the measurement-assisted 4T
    temporary-AND variant.  Its actual X/CX/CCX gates are counted.
    """
    address, data, scratch = list(address), list(data), list(scratch)
    k, width = len(address), len(data)
    if len(words) != 1 << k:
        raise ValueError("QROM table length must equal 2**number_of_address_bits")
    if len(scratch) < max(0, k - 1):
        raise ValueError("QROM needs k-1 clean scratch bits")
    used = address + data + scratch[: max(0, k - 1)]
    if len(used) != len(set(used)):
        raise ValueError("QROM registers must be disjoint")
    if width == 0:
        return
    mask = (1 << width) - 1
    table = [int(word) & mask for word in words]
    constant = mask
    for word in table:
        constant &= word
    for bit in range(width):
        if (constant >> bit) & 1:
            circuit.x(data[bit])
    table = [word ^ constant for word in table]
    if not any(table):
        return
    if k == 0:
        raise AssertionError("A one-entry table must have been constant")

    def leaf(word: int, control: int) -> None:
        while word:
            lowbit = word & -word
            circuit.cx(control, data[lowbit.bit_length() - 1])
            word ^= lowbit

    def visit(start: int, length: int, depth: int, control: int) -> None:
        block = table[start : start + length]
        if not any(block):
            return
        if all(word == block[0] for word in block):
            leaf(block[0], control)
            return
        # First form control AND NOT address_bit; switching the sibling
        # branch by XOR with control avoids rebuilding the prefix.
        bit = address[k - depth - 1]
        flag = scratch[depth - 1]
        circuit.x(bit)
        circuit.ccx(control, bit, flag)
        circuit.x(bit)
        visit(start, length // 2, depth + 1, flag)
        circuit.cx(control, flag)
        visit(start + length // 2, length // 2, depth + 1, flag)
        circuit.ccx(control, bit, flag)

    high = address[-1]
    circuit.x(high)
    visit(0, 1 << (k - 1), 1, high)
    circuit.x(high)
    visit(1 << (k - 1), 1 << (k - 1), 1, high)


@lru_cache(maxsize=None)
def _adder_operations(width: int) -> tuple:
    """MAJ/UMA ripple adder with the unused highest carry removed."""
    if width < 1:
        raise ValueError("Adder width must be positive")
    if width == 1:
        return (("cx", (0, 1)),)
    a, b, carry = list(range(width)), list(range(width, 2 * width)), 2 * width
    ops = []
    for j in range(width - 1):
        c = carry if j == 0 else a[j - 1]
        ops.extend((("cx", (a[j], b[j])), ("cx", (a[j], c)), ("ccx", (c, b[j], a[j]))))
    ops.extend((("cx", (a[-1], b[-1])), ("cx", (a[-2], b[-1]))))
    for j in reversed(range(width - 1)):
        c = carry if j == 0 else a[j - 1]
        ops.extend((("ccx", (c, b[j], a[j])), ("cx", (a[j], c)), ("cx", (c, b[j]))))
    return tuple(ops)


def modular_add(
    circuit: QuantumCircuit,
    addend: Sequence[int],
    target: Sequence[int],
    carry: int,
    *,
    inverse: bool = False,
) -> None:
    """Add (or subtract) addend into target modulo 2**width, clean carry.

    Uses the elementary MAJ/UMA construction of Cuccaro et al. with the
    top carry omitted.  Addend is restored.  Carry must start at zero and is
    restored; the circuit is also defined reversibly on every other input.
    """
    if len(addend) != len(target):
        raise ValueError("Adder register widths must match")
    mapping = list(addend) + list(target) + [carry]
    if len(mapping) != len(set(mapping)):
        raise ValueError("Adder registers must be disjoint")
    operations = _adder_operations(len(addend))
    for name, indices in reversed(operations) if inverse else operations:
        getattr(circuit, name)(*[mapping[i] for i in indices])


def multiply_accumulate(
    circuit: QuantumCircuit,
    multiplicand: Sequence[int],
    coefficient: Sequence[int],
    target: Sequence[int],
    workspace: Sequence[int],
    carry: int,
    *,
    coefficient_constant: int | None = None,
) -> None:
    """target += signed(multiplicand)*signed(coefficient), modulo target size.

    Both inputs are little-endian two's complement.  Scratch workspace needs
    len(target) clean bits.  Shifted partial products are computed into scratch,
    accumulated with a ripple adder, and uncomputed.  The sign bit of a quantum
    coefficient is subtracted.  For a known classical coefficient, ordinary
    constant shift-and-add omits zero bits and uses CNOT rather than CCX.
    """
    a, b, out, work = map(list, (multiplicand, coefficient, target, workspace))
    if not a or not out or (not b and coefficient_constant is None):
        raise ValueError("Arithmetic input and output registers must be nonempty")
    if len(work) < len(out):
        raise ValueError("Multiplier workspace is too small")
    if coefficient_constant is None:
        shifts = [(j, j == len(b) - 1, b[j]) for j in range(min(len(b), len(out)))]
    else:
        value = int(coefficient_constant)
        shifts = [(j, value < 0, None) for j in range(min(abs(value).bit_length(), len(out))) if (abs(value) >> j) & 1]
    for shift, subtract, control in shifts:
        size = len(out) - shift
        # Sign extension is performed by reusing the sign bit as a control;
        # it needs no uncharged extension register.
        sources = [a[min(j, len(a) - 1)] for j in range(size)]
        for source, dest in zip(sources, work[:size]):
            if control is None:
                circuit.cx(source, dest)
            else:
                circuit.ccx(control, source, dest)
        modular_add(circuit, work[:size], out[shift:], carry, inverse=subtract)
        for source, dest in reversed(list(zip(sources, work[:size]))):
            if control is None:
                circuit.cx(source, dest)
            else:
                circuit.ccx(control, source, dest)


def signed_phase(circuit: QuantumCircuit, data: Sequence[int], scale: float) -> None:
    """Append exp(i*scale*signed(data)) using single-qubit P rotations."""
    data = list(data)
    for bit, qubit in enumerate(data):
        weight = -(1 << bit) if bit == len(data) - 1 else 1 << bit
        # A remainder against float(2*pi) loses O(|angle|*machine_epsilon)
        # phase accuracy for large inputs. libm's trig range reduction instead
        # returns the principal phase of the actual represented argument.
        raw_angle = float(scale) * weight
        if not math.isfinite(raw_angle):
            raise ValueError("Phase angle overflow; rescale beta or the potential")
        angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))
        if angle != 0.0:
            circuit.p(angle, qubit)


def validate_cores(cores: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not cores:
        raise ValueError("A tensor train must contain at least one core")
    arrays = [np.asarray(core, dtype=float) for core in cores]
    for j, core in enumerate(arrays):
        if core.ndim != 3 or not np.all(np.isfinite(core)):
            raise ValueError("Each TT core must be a finite real rank-three array")
        if min(core.shape) < 1 or core.shape[1] & (core.shape[1] - 1):
            raise ValueError("Each coordinate grid size must be a positive power of two")
        if j and arrays[j - 1].shape[2] != core.shape[0]:
            raise ValueError("Neighboring TT bond dimensions do not match")
    if arrays[0].shape[0] != 1 or arrays[-1].shape[2] != 1:
        raise ValueError("Open-boundary TT ranks must be one")
    return arrays


def contract_cores(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Dense tensor in C-order; intended for validation, not quantum access."""
    value = np.asarray(cores[0])[0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=([-1], [0]))
    return value[..., 0]


def quantize_cores(cores: Sequence[np.ndarray], fraction_bits: int) -> list[np.ndarray]:
    if not isinstance(fraction_bits, (int, np.integer)) or not 0 <= fraction_bits <= 30:
        raise ValueError("fraction_bits must be an integer between 0 and 30")
    arrays = validate_cores(cores)
    scale = 1 << int(fraction_bits)
    # Python arbitrary-precision integers prevent classical overflow when
    # exact contraction widths exceed 64 bits in a scaling experiment.
    return [np.vectorize(lambda x: int(np.rint(x * scale)), otypes=[object])(core) for core in arrays]


def quantization_bound(cores: Sequence[np.ndarray], quantized: Sequence[np.ndarray], fraction_bits: int) -> float:
    """Telescoping induced-infinity-norm bound for core quantization error.

    This evaluates an analytic bound in float64; it is not interval arithmetic.
    The suprema here range over the supplied finite coordinate grids.
    """
    scale = float(1 << fraction_bits)
    exact = [np.asarray(c, dtype=float) for c in cores]
    approx = [np.asarray(c, dtype=float) / scale for c in quantized]
    norm = lambda c: float(np.max(np.sum(np.abs(c), axis=2)))
    a, b, e = [norm(c) for c in exact], [norm(c) for c in approx], [norm(c - q) for c, q in zip(exact, approx)]
    total = 0.0
    for j in range(len(exact)):
        total += math.prod(b[:j]) * e[j] * math.prod(a[j + 1 :])
    return float(total)


def _signed_width(bound: int) -> int:
    return max(1, int(bound).bit_length() + 1)


def build_tt_oracle(
    cores: Sequence[np.ndarray],
    beta: float,
    fraction_bits: int,
    *,
    materialize_values: bool = True,
) -> TTOracle:
    """Compile a clean phase oracle for an integer-quantized tensor train.

    Quantum address wires come first.  The final coordinate occupies the least
    significant address bits, so integer x indexes the C-order flattened grid.
    ``materialize_values=False`` avoids exponential *classical validation*
    output; circuit compilation itself never enumerates the full product grid.
    Coefficient QROM tables contain one coordinate at a time.
    """
    start = time.perf_counter()
    arrays = validate_cores(cores)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    integer_cores = quantize_cores(arrays, fraction_bits)
    dimensions = len(arrays)
    grid_bits = [int(c.shape[1]).bit_length() - 1 for c in arrays]
    n_address = sum(grid_bits)
    shapes = [c.shape for c in arrays]
    # Bounds include the sum of absolute products along every bond.  Modular
    # intermediate arithmetic is exact; these bounds disallow final overflow.
    previous = [1]
    bounds, widths = [], []
    for core in integer_cores:
        current = [sum(previous[a] * max(abs(int(v)) for v in core[a, :, b]) for a in range(core.shape[0])) for b in range(core.shape[2])]
        bounds.append(current)
        widths.append(_signed_width(max(current)))
        previous = current
    coefficient_widths = [_signed_width(max(abs(int(v)) for v in c.flat)) for c in integer_cores]
    specifications, max_coefficient_bits = [], 0
    for j, core in enumerate(integer_cores):
        spec, offset = {}, 0
        for a in range(core.shape[0]):
            for b in range(core.shape[2]):
                column = [int(v) for v in core[a, :, b]]
                if all(v == column[0] for v in column):
                    spec[a, b] = ("constant", column[0])
                else:
                    spec[a, b] = ("quantum", offset)
                    offset += coefficient_widths[j]
        specifications.append((spec, offset))
        if j:
            max_coefficient_bits = max(max_coefficient_bits, offset)
    cursor = n_address
    stages = []
    for width, core in zip(widths, integer_cores):
        stage = []
        for _ in range(core.shape[2]):
            stage.append(list(range(cursor, cursor + width)))
            cursor += width
        stages.append(stage)
    coefficient_bank = list(range(cursor, cursor + max_coefficient_bits))
    cursor += max_coefficient_bits
    max_width = max(widths[1:], default=0)
    workspace = list(range(cursor, cursor + max_width))
    cursor += max_width
    carry = cursor if dimensions > 1 else None
    cursor += int(dimensions > 1)
    scratch = list(range(cursor, cursor + max(0, max(grid_bits) - 1)))
    cursor += len(scratch)
    compute = QuantumCircuit(cursor, name="tt_compute")

    def coordinate_address(j: int) -> list[int]:
        offset = sum(grid_bits[j + 1 :])
        return list(range(offset, offset + grid_bits[j]))

    # The first contraction is simply a vector-valued coefficient lookup.
    first_width = widths[0]
    first_words = []
    for x in range(shapes[0][1]):
        word = sum((int(integer_cores[0][0, x, b]) % (1 << first_width)) << (b * first_width) for b in range(shapes[0][2]))
        first_words.append(word)
    qrom_xor(compute, coordinate_address(0), [q for reg in stages[0] for q in reg], first_words, scratch)

    for j in range(1, dimensions):
        core = integer_cores[j]
        spec, bank_size = specifications[j]
        width = coefficient_widths[j]
        words = []
        for x in range(core.shape[1]):
            word = 0
            for (a, b), (kind, value) in spec.items():
                if kind == "quantum":
                    word |= (int(core[a, x, b]) % (1 << width)) << value
            words.append(word)
        # Load all variable core coefficients in one shared selection tree.
        # This bank is reused between coordinates; explicit unloading is paid.
        loader = QuantumCircuit(cursor, name=f"core_{j}_lookup")
        qrom_xor(loader, coordinate_address(j), coefficient_bank[:bank_size], words, scratch)
        compute.compose(loader, inplace=True)
        for a in range(core.shape[0]):
            for b in range(core.shape[2]):
                kind, value = spec[a, b]
                if kind == "constant":
                    multiply_accumulate(compute, stages[j - 1][a], [], stages[j][b], workspace, carry, coefficient_constant=value)
                else:
                    multiply_accumulate(compute, stages[j - 1][a], coefficient_bank[value : value + width], stages[j][b], workspace, carry)
        compute.compose(loader.inverse(), inplace=True)
    compile_seconds = time.perf_counter() - start
    circuit = compute.copy(name="tt_phase")
    signed_phase(circuit, stages[-1][0], math.ldexp(float(beta), -dimensions * fraction_bits))
    circuit.compose(compute.inverse(), inplace=True)
    compile_seconds = time.perf_counter() - start
    materialize_start = time.perf_counter()
    values = None
    if materialize_values:
        values = np.asarray(contract_cores(integer_cores), dtype=float).reshape(-1) * math.ldexp(1.0, -dimensions * fraction_bits)
    metadata = {
        "method": "reversible_tt",
        "dimensions": dimensions,
        "grid_sizes": [int(c.shape[1]) for c in arrays],
        "ranks": [1] + [int(c.shape[2]) for c in arrays],
        "core_fraction_bits": int(fraction_bits),
        "final_fraction_bits": dimensions * int(fraction_bits),
        "coefficient_widths": coefficient_widths,
        "accumulator_widths": widths,
        "accumulator_absolute_bounds": [[str(x) for x in bound] for bound in bounds],
        "coefficient_bank_qubits": max_coefficient_bits,
        "workspace_qubits": len(workspace) + int(carry is not None) + len(scratch),
        "intermediate_register_qubits": sum(len(reg) for stage in stages for reg in stage),
        "coefficient_table_entries": sum(int(c.size) for c in arrays),
        "core_quantization_bound": quantization_bound(arrays, integer_cores, fraction_bits),
        "compile_seconds": compile_seconds,
        "dense_validation_preprocess_seconds": time.perf_counter() - materialize_start,
        "arithmetic": "exact signed integer contraction; no intermediate truncation",
        "qrom": "coherent shared-prefix X/CX/CCX; constant-column/subtree simplification",
        "uncomputation": "entire compute circuit inverted after phase; reusable coefficient bank unloaded per stage",
        "phase_scale": math.ldexp(float(beta), -dimensions * fraction_bits),
    }
    return TTOracle(circuit, n_address, values, metadata, integer_cores)
