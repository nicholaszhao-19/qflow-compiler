"""Exact dyadic precision planning and clean, mixed-width TT phase oracles."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from time import perf_counter
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit

from .reversible import TTOracle, multiply_accumulate, qrom_xor, validate_cores


def _signed_width(low: int, high: int) -> int:
    return max(0, high, -low - 1).bit_length() + 1


def _fraction(value: float | int | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    return Fraction.from_float(float(value))


def _precision(value: int) -> int:
    if not isinstance(value, Integral) or value < 0:
        raise ValueError("Fractional bit counts must be nonnegative integers")
    return int(value)


def _rounded_view(register: Sequence[int], drop: int, width: int) -> Sequence[int]:
    """View the arithmetic right shift of a two's-complement register."""
    if drop >= len(register):
        return register[-1:]
    return register[drop : drop + width]


@dataclass(frozen=True)
class PrecisionPlan:
    """Certificate for the finite-grid TT whose inputs are exact binary64 data."""

    integer_cores: tuple[np.ndarray, ...]
    core_fraction_bits: tuple[int, ...]
    stage_fraction_bits: tuple[int, ...]
    shifts: tuple[int, ...]
    raw_intervals: tuple[tuple[tuple[int, int], ...], ...]
    retained_intervals: tuple[tuple[tuple[int, int], ...], ...]
    raw_widths: tuple[tuple[int, ...], ...]
    retained_widths: tuple[tuple[int, ...], ...]
    coefficient_error_bound: Fraction
    rounding_error_bound: Fraction

    @property
    def final_fraction_bits(self) -> int:
        return self.stage_fraction_bits[-1]

    @property
    def potential_error_bound(self) -> Fraction:
        return self.coefficient_error_bound + self.rounding_error_bound

    def evaluate_exact(self) -> np.ndarray:
        """Materialize the rounded integer contraction using Python integers."""
        value = self.integer_cores[0][0] >> self.shifts[0]
        for core, drop in zip(self.integer_cores[1:], self.shifts[1:]):
            value = np.tensordot(value, core, axes=([-1], [0])) >> drop
        return value[..., 0]

    def _phase_angles(self, beta: float | int | Fraction) -> tuple[tuple[float, ...], Fraction]:
        scale = _fraction(beta) / (1 << self.final_fraction_bits)
        width = self.retained_widths[-1][0]
        exact = tuple(scale * ((1 << bit) if bit < width - 1 else -(1 << bit)) for bit in range(width))
        angles = tuple(float(angle) for angle in exact)
        error = sum((abs(_fraction(emitted) - intended) for emitted, intended in zip(angles, exact)), Fraction())
        return angles, error

    def phase_parameter_error_bound(self, beta: float | int | Fraction) -> Fraction:
        """Bound the sum of errors incurred when encoding the ideal P angles."""
        return self._phase_angles(beta)[1]

    def phase_error_bound(
        self,
        beta: float | int | Fraction,
        *,
        tt_error: float | int | Fraction = 0,
        synthesis_error: float | int | Fraction = 0,
    ) -> Fraction:
        """Operator-error bound, with supplied TT residual and synthesis bounds."""
        residual, synthesis = _fraction(tt_error), _fraction(synthesis_error)
        if min(residual, synthesis) < 0:
            raise ValueError("Error budgets must be nonnegative")
        return min(
            Fraction(2),
            abs(_fraction(beta)) * (residual + self.potential_error_bound)
            + self.phase_parameter_error_bound(beta) + synthesis,
        )


def plan_precision_tt(
    cores: Sequence[np.ndarray],
    core_fraction_bits: int | Sequence[int],
    stage_fraction_bits: int | Sequence[int] | None = None,
) -> PrecisionPlan:
    """Plan nearest-even coefficient quantization and signed floor rounding.

    A scalar core precision broadcasts. A scalar stage precision caps every
    stage, an explicit sequence specifies every stage, and None retains all
    available fractional bits. Certification uses exact rational arithmetic
    and coordinate-local tables. It does not enumerate the product grid.
    """
    arrays = validate_cores(cores)
    count = len(arrays)
    if isinstance(core_fraction_bits, Integral):
        core_bits = (_precision(core_fraction_bits),) * count
    else:
        core_bits = tuple(map(_precision, core_fraction_bits))
    if len(core_bits) != count:
        raise ValueError("Provide one coefficient precision per core")
    if stage_fraction_bits is None or isinstance(stage_fraction_bits, Integral):
        cap = None if stage_fraction_bits is None else _precision(stage_fraction_bits)
        stage_bits = []
        for bits in core_bits:
            available = (stage_bits[-1] if stage_bits else 0) + bits
            stage_bits.append(available if cap is None else min(cap, available))
    else:
        stage_bits = list(map(_precision, stage_fraction_bits))
    if len(stage_bits) != count:
        raise ValueError("Provide one retained precision per stage")
    raw_bits = tuple(a + b for a, b in zip((0, *stage_bits[:-1]), core_bits))
    shifts = tuple(raw - retained for raw, retained in zip(raw_bits, stage_bits))
    if min(shifts) < 0:
        raise ValueError("Retained stage precision cannot exceed the available fractional bits")

    previous_ranges = ((1, 1),)
    exact_prefix = (Fraction(1),)
    coefficient_error = rounding_error = (Fraction(0),)
    integer_cores, raw_intervals, retained_intervals = [], [], []
    for j, array in enumerate(arrays):
        exact = np.vectorize(_fraction, otypes=[object])(array)
        coefficient_scale = 1 << core_bits[j]
        integers = np.vectorize(lambda value: round(value * coefficient_scale), otypes=[object])(exact)
        integers.setflags(write=False)
        integer_cores.append(integers)
        rank_in, grid, rank_out = integers.shape
        drop = shifts[j]
        floor_error = Fraction((1 << drop) - 1, 1 << raw_bits[j])
        ranges, next_prefix, next_coefficient, next_rounding = [], [], [], []
        for b in range(rank_out):
            lows, highs, prefixes, coefficients, roundings = [], [], [], [], []
            for x in range(grid):
                low = high = 0
                prefix = coefficient = rounding = Fraction(0)
                for a in range(rank_in):
                    integer = integers[a, x, b]
                    lo, hi = previous_ranges[a]
                    low += min(lo * integer, hi * integer)
                    high += max(lo * integer, hi * integer)
                    source = exact[a, x, b]
                    quantized = Fraction(integer, coefficient_scale)
                    prefix += exact_prefix[a] * abs(source)
                    coefficient += coefficient_error[a] * abs(quantized) + exact_prefix[a] * abs(source - quantized)
                    rounding += rounding_error[a] * abs(quantized)
                # The first-stage dot is a single exact table value.
                residue = Fraction(low % (1 << drop), 1 << raw_bits[j]) if j == 0 else floor_error
                lows.append(low)
                highs.append(high)
                prefixes.append(prefix)
                coefficients.append(coefficient)
                roundings.append(rounding + residue)
            ranges.append((min(lows), max(highs)))
            next_prefix.append(max(prefixes))
            next_coefficient.append(max(coefficients))
            next_rounding.append(max(roundings))
        previous_ranges = tuple((lo >> drop, hi >> drop) for lo, hi in ranges)
        raw_intervals.append(tuple(ranges))
        retained_intervals.append(previous_ranges)
        exact_prefix = tuple(next_prefix)
        coefficient_error, rounding_error = tuple(next_coefficient), tuple(next_rounding)
    return PrecisionPlan(
        tuple(integer_cores), core_bits, tuple(stage_bits), shifts,
        tuple(raw_intervals), tuple(retained_intervals),
        tuple(tuple(_signed_width(*interval) for interval in stage) for stage in raw_intervals),
        tuple(tuple(_signed_width(*interval) for interval in stage) for stage in retained_intervals),
        coefficient_error[0], rounding_error[0],
    )


def build_precision_tt_oracle(
    plan: PrecisionPlan,
    beta: float | int | Fraction,
    *,
    workspace_strategy: str = "retain",
    materialize_values: bool = True,
) -> TTOracle:
    """Emit an elementary clean circuit implementing the certified rounded TT.

    Retain keeps low residues and passes only high wires to the next stage.
    Recompute temporarily forms each rounded intermediate dot, XORs its high
    bits into its retained register, and reverses the dot to clean scratch.
    Both policies compute, phase, and fully uncompute the contraction.
    """
    if workspace_strategy not in ("retain", "recompute"):
        raise ValueError("workspace_strategy must be 'retain' or 'recompute'")
    start = perf_counter()
    cores = plan.integer_cores
    dimensions = len(cores)
    grid_bits = tuple(core.shape[1].bit_length() - 1 for core in cores)
    n_address = sum(grid_bits)
    recomputed = tuple(
        workspace_strategy == "recompute" and 0 < j < dimensions - 1 and bool(drop)
        for j, drop in enumerate(plan.shifts)
    )
    cursor = n_address

    def allocate(size: int) -> list[int]:
        nonlocal cursor
        register = list(range(cursor, cursor + size))
        cursor += size
        return register

    stages = [
        [allocate(width) for width in (kept if j == 0 or recomputed[j] else raw)]
        for j, (kept, raw) in enumerate(zip(plan.retained_widths, plan.raw_widths))
    ]
    intermediate_qubits = cursor - n_address
    specs, bank_sizes = [], []
    for core in cores[1:]:
        offset, constants, variable = 0, {}, {}
        for a in range(core.shape[0]):
            for b in range(core.shape[2]):
                words = core[a, :, b]
                low, high = min(words), max(words)
                if low == high:
                    constants[a, b] = low
                else:
                    width = _signed_width(low, high)
                    variable[a, b] = (offset, width)
                    offset += width
        specs.append((constants, variable))
        bank_sizes.append(offset)
    bank = allocate(max(bank_sizes, default=0))
    workspace = allocate(max((max(widths) for widths in plan.raw_widths[1:]), default=0))
    raw_scratch = allocate(max((max(widths) for widths, use in zip(plan.raw_widths, recomputed) if use), default=0))
    carry = allocate(1)[0] if dimensions > 1 else None
    selection = allocate(max(0, max(grid_bits) - 1))
    compute = QuantumCircuit(cursor, name="precision_tt_compute")

    def address(j: int) -> list[int]:
        offset = sum(grid_bits[j + 1 :])
        return list(range(offset, offset + grid_bits[j]))

    def retained(j: int, b: int) -> Sequence[int]:
        register = stages[j][b]
        if j == 0 or recomputed[j]:
            return register
        return _rounded_view(register, plan.shifts[j], plan.retained_widths[j][b])

    first = cores[0][0] >> plan.shifts[0]
    first_words = []
    for row in first:
        word = offset = 0
        for value, register in zip(row, stages[0]):
            word |= (value % (1 << len(register))) << offset
            offset += len(register)
        first_words.append(word)
    qrom_xor(compute, address(0), [q for register in stages[0] for q in register], first_words, selection)

    for j, (core, (constants, variable), bank_size) in enumerate(zip(cores[1:], specs, bank_sizes), start=1):
        words = [
            sum((core[a, x, b] % (1 << width)) << offset for (a, b), (offset, width) in variable.items())
            for x in range(core.shape[1])
        ]
        loader = QuantumCircuit(cursor, name=f"precision_tt_lookup_{j}")
        qrom_xor(loader, address(j), bank[:bank_size], words, selection)
        compute.compose(loader, inplace=True)
        for b, register in enumerate(stages[j]):
            target = raw_scratch[:plan.raw_widths[j][b]] if recomputed[j] else register
            dot = QuantumCircuit(cursor, name="rounded_dot") if recomputed[j] else compute
            for a in range(core.shape[0]):
                if (a, b) in constants:
                    multiply_accumulate(dot, retained(j - 1, a), [], target, workspace, carry, coefficient_constant=constants[a, b])
                else:
                    offset, width = variable[a, b]
                    multiply_accumulate(dot, retained(j - 1, a), bank[offset : offset + width], target, workspace, carry)
            if recomputed[j]:
                compute.compose(dot, inplace=True)
                for source, dest in zip(_rounded_view(target, plan.shifts[j], len(register)), register):
                    compute.cx(source, dest)
                compute.compose(dot.inverse(), inplace=True)
        compute.compose(loader.inverse(), inplace=True)

    angles, encoding_error = plan._phase_angles(beta)
    circuit = compute.copy(name="precision_tt_phase")
    for angle, qubit in zip(angles, retained(dimensions - 1, 0)):
        if angle:
            circuit.p(angle, qubit)
    circuit.compose(compute.inverse(), inplace=True)
    compile_seconds = perf_counter() - start
    values = None
    start_materialize = perf_counter()
    if materialize_values:
        scale = 1 << plan.final_fraction_bits
        values = np.fromiter((float(Fraction(value, scale)) for value in plan.evaluate_exact().flat), dtype=float)
    metadata = {
        "method": "precision_tt",
        "workspace_strategy": workspace_strategy,
        "core_fraction_bits": plan.core_fraction_bits,
        "stage_fraction_bits": plan.stage_fraction_bits,
        "final_fraction_bits": plan.final_fraction_bits,
        "shifts": plan.shifts,
        "raw_widths": plan.raw_widths,
        "retained_widths": plan.retained_widths,
        "raw_intervals": plan.raw_intervals,
        "retained_intervals": plan.retained_intervals,
        "coefficient_error_bound": str(plan.coefficient_error_bound),
        "rounding_error_bound": str(plan.rounding_error_bound),
        "potential_error_bound": str(plan.potential_error_bound),
        "phase_parameter_error_bound": str(encoding_error),
        "coefficient_table_entries": sum(int(core.size) for core in cores),
        "coefficient_bank_qubits": len(bank),
        "intermediate_register_qubits": intermediate_qubits,
        "workspace_qubits": len(workspace) + len(raw_scratch) + len(selection) + int(carry is not None),
        "clean_ancilla_qubits": cursor - n_address,
        "dirty_ancilla_qubits": 0,
        "compile_seconds": compile_seconds,
        "dense_validation_preprocess_seconds": perf_counter() - start_materialize,
    }
    return TTOracle(circuit, n_address, values, metadata, list(cores))
