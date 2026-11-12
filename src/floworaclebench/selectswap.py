"""Coherent SelectSwap phase-table lookup with an explicit space/time tradeoff.

This uses the established SelectSwap construction (Low, Kliuchnikov and
Schaeffer, arXiv:1812.00954): select a block of phase words using a QROM, select
one word with a controlled swap tree, apply its binary phase, then uncompute.
Our implementation uses clean banks and exact CCX gates throughout.  It does
not claim the constants of measurement-assisted QROM or dirty-ancilla variants.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit

from .baselines import Oracle, phase_error
from .reversible import qrom_xor


def build_selectswap_oracle(values, beta=1.0, phase_bits=12, block_bits=1) -> Oracle:
    """Compile an angle-table oracle using a bank of ``2**block_bits`` words.

    Address qubits are little endian and occupy the first ``n`` circuit wires.
    Low address bits select the word within a block; high bits select a block
    through QROM.  Every data and prefix-scratch qubit starts and ends at zero.
    ``block_bits=0`` is the ordinary coherent QROM baseline.
    """
    start = perf_counter()
    theta = float(beta) * np.asarray(values, dtype=float).reshape(-1)
    n = len(theta).bit_length() - 1
    if n < 1 or 2**n != len(theta) or not np.isfinite(theta).all():
        raise ValueError("a finite power-of-two phase table with at least two entries is required")
    if not isinstance(phase_bits, (int, np.integer)) or not 1 <= phase_bits <= 50:
        raise ValueError("phase_bits must be an integer in [1,50]")
    if not isinstance(block_bits, (int, np.integer)) or not 0 <= block_bits <= n:
        raise ValueError("block_bits must be an integer between zero and the address width")
    phase_bits, block_bits = int(phase_bits), int(block_bits)
    bank_size = 1 << block_bits
    high_bits = n - block_bits
    principal = np.angle(np.exp(1j*theta))
    words = np.rint(np.remainder(principal, 2*np.pi) * (2**phase_bits) / (2*np.pi))
    words = words.astype(np.int64) % (2**phase_bits)

    bank = [list(range(n + j*phase_bits, n + (j+1)*phase_bits)) for j in range(bank_size)]
    scratch = list(range(n + bank_size*phase_bits,
                         n + bank_size*phase_bits + max(0, high_bits-1)))
    total = n + bank_size*phase_bits + len(scratch)
    lookup = QuantumCircuit(total, name="SelectSwap_block_QROM")
    # One selected QROM word contains all phase words from its contiguous block.
    # Python ints are required because a packed word can exceed 64 bits.
    packed = [sum(int(words[h*bank_size+j]) << (j*phase_bits) for j in range(bank_size))
              for h in range(1 << high_bits)]
    qrom_xor(lookup, list(range(block_bits,n)), [q for word in bank for q in word], packed, scratch)

    selection = QuantumCircuit(total, name="SelectSwap_word_selection")
    # Only the candidate word of each subtree has to move.  This tree uses
    # (bank_size-1)*phase_bits Fredkin gates, not a full sorting network.
    for level in range(block_bits):
        stride = 1 << (level+1)
        for first in range(0, bank_size, stride):
            second = first + (1 << level)
            for a,b in zip(bank[first], bank[second]):
                selection.cx(a,b)
                selection.ccx(level,b,a)
                selection.cx(a,b)

    qc = QuantumCircuit(total, name="SelectSwap_phase")
    qc.compose(lookup, inplace=True)
    qc.compose(selection, inplace=True)
    for j,q in enumerate(bank[0]):
        qc.p(2*np.pi*2.0**(j-phase_bits), q)
    qc.compose(selection.inverse(), inplace=True)
    qc.compose(lookup.inverse(), inplace=True)

    phases = 2*np.pi*words.astype(float)/(2**phase_bits)
    return Oracle(qc, n, phases, {
        "method": "selectswap",
        "compile_seconds": perf_counter()-start,
        "phase_bits": phase_bits,
        "block_bits": block_bits,
        "bank_size": bank_size,
        "bank_qubits": bank_size*phase_bits,
        "prefix_scratch_qubits": len(scratch),
        "rounding_bound": np.pi*2.0**(-phase_bits),
        "operator_error": phase_error(phases,theta),
        "lookup_gates": dict(lookup.count_ops()),
        "lookup_depth": lookup.depth(),
        "selection_gates": dict(selection.count_ops()),
        "selection_depth": selection.depth(),
        "fredkin_count_compute": (bank_size-1)*phase_bits,
        "fredkin_count_complete": 2*(bank_size-1)*phase_bits,
        "dense_table_bytes": theta.nbytes,
        "packed_table_bits": len(words)*phase_bits,
        "construction": "coherent clean-bank SelectSwap; shared-prefix QROM; exact CX/CCX Fredkin decomposition",
        "uncomputation": "inverse selection followed by inverse block QROM",
    })
