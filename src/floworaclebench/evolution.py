"""Independent Fourier-grid evolution and established commutator circuits.

The eight-letter formula is Layden et al., arXiv:2510.08462 Eq. (36),
with the sign convention H=i[K,V] and K=-Laplacian/2. Arrays use C order;
the last spatial axis is the least significant quantum register.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, StatePreparation

from .baselines import build_walsh_oracle


def _validate_grid(n: int, d: int, length: float, *, binary: bool = False) -> None:
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    if binary and n & (n - 1):
        raise ValueError("quantum circuit grid size must be a power of two")


def kinetic_matrix(n: int, d: int, length: float = 8.0) -> np.ndarray:
    """Dense Fourier-collocation kinetic matrix for a periodic side length L.

    Uses standard FFT frequency order, including the negative Nyquist mode.
    This is unitarily equivalent to the centered spectrum/sign-mask convention
    of Layden Eqs. (26)--(29). No finite-difference stencil is substituted.
    """
    _validate_grid(n, d, length)
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    grid = np.arange(n)
    fourier = np.exp(2j * np.pi * np.outer(grid, grid) / n) / np.sqrt(n)
    one_axis = (fourier * (0.5 * frequency**2)) @ fourier.conj().T
    total = np.zeros((n**d, n**d), dtype=complex)
    for axis in range(d):
        total += np.kron(np.eye(n**axis), np.kron(one_axis, np.eye(n**(d - axis - 1))))
    return total


def gaussian_state(n: int, d: int, length: float = 8.0, sigma: float = 4 / 3) -> np.ndarray:
    """Normalized sampled product-Gaussian QSAMPLE on the centered grid.

    The supplied sigma is the *probability distribution's* standard deviation:
    amplitudes are proportional to exp(-x^2/(4 sigma^2)). This is a normalized
    grid restriction, not a claim that finite-box continuum tails are absent.
    """
    _validate_grid(n, d, length)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    x = np.arange(n) * length / n - length / 2
    amplitudes = np.exp(-x**2 / (4 * sigma**2))
    amplitudes /= np.linalg.norm(amplitudes)
    state = amplitudes
    for _ in range(d - 1):
        state = np.kron(state, amplitudes)
    return state.astype(complex)


def grid_initial_state(potential, n: int) -> np.ndarray:
    """Return the specified normalized source, or the legacy grid Gaussian."""
    dimension = int(potential.dimension)
    initial_state = getattr(potential, "initial_state", None)
    if not callable(initial_state):
        return gaussian_state(n, dimension, getattr(potential, "length", 8.0),
                              getattr(potential, "sigma0", 4 / 3))
    state = np.asarray(initial_state(n), dtype=complex).reshape(-1)
    if state.size != n**dimension or not np.isfinite(state).all():
        raise ValueError("initial_state returned an invalid array")
    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-12, rtol=0):
        raise ValueError("initial source must already be normalized")
    return state


def _axis_wires(n: int, d: int, axis: int) -> list[int]:
    width = int(n).bit_length() - 1
    return list(range((d - axis - 1) * width, (d - axis) * width))


def build_kinetic_circuit(n: int, d: int, length: float = 8.0, alpha: float = 1.0) -> QuantumCircuit:
    """Compile exp(i alpha K) as axis-local inverse-QFT, phase, QFT.

    All controlled phases and swaps in the established QFT are decomposed to
    H/P/CX; the diagonal spectrum uses the same established Walsh synthesizer
    as the baseline. There are no ancillas and no opaque unitary matrices.
    """
    _validate_grid(n, d, length, binary=True)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    width = int(n).bit_length() - 1
    qft = QuantumCircuit(width)
    qft.append(QFTGate(width), range(width))
    qft = transpile(qft, basis_gates=["h", "p", "cx"], optimization_level=0)
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    diagonal = build_walsh_oracle(0.5 * frequency**2, beta=alpha).circuit
    circuit = QuantumCircuit(width * d, name="kinetic")
    for axis in range(d):
        wires = _axis_wires(n, d, axis)
        circuit.compose(qft.inverse(), wires, inplace=True)
        circuit.compose(diagonal, wires, inplace=True)
        circuit.compose(qft, wires, inplace=True)
    return circuit


def build_gaussian_preparation(n: int, d: int, length: float = 8.0,
                               sigma: float = 4 / 3) -> QuantumCircuit:
    """Compile the product-Gaussian QSAMPLE, including all preparation gates.

    Each spatial axis is prepared by Qiskit's established StatePreparation
    decomposition, then expanded into the native benchmark gate basis. No
    amplitude-loading black box or free state preparation is assumed.
    """
    _validate_grid(n, d, length, binary=True)
    width = int(n).bit_length() - 1
    one_axis = QuantumCircuit(width)
    one_axis.append(StatePreparation(gaussian_state(n, 1, length, sigma)), range(width))
    one_axis = transpile(one_axis, basis_gates=["h", "x", "p", "rz", "cx"], optimization_level=1)
    circuit = QuantumCircuit(d * width, name="Gaussian_QSAMPLE")
    for axis in range(d):
        circuit.compose(one_axis, _axis_wires(n, d, axis), inplace=True)
    return circuit


def apply_product_formula(state: np.ndarray, vplus: np.ndarray,
                          kinetic_plus: np.ndarray) -> np.ndarray:
    """Apply one existing eight-letter commutator step in chronological order.

    ``vplus`` contains unit-modulus exp(i beta V) entries obtained from an
    executed compiled potential circuit; ``kinetic_plus`` is exp(i alpha K).
    Clean ancillas must already have been verified by the oracle simulator.
    This function does not replace the resource cost of either circuit.
    """
    state = np.asarray(state, dtype=complex).reshape(-1)
    phase = np.asarray(vplus, dtype=complex).reshape(-1)
    kinetic = np.asarray(kinetic_plus, dtype=complex)
    if phase.shape != state.shape or kinetic.shape != (len(state), len(state)):
        raise ValueError("state, potential phases and kinetic dimensions differ")
    if not np.isfinite(phase).all() or not np.allclose(np.abs(phase), 1, atol=1e-12, rtol=0):
        raise ValueError("potential entries must be finite unit-modulus phases")
    inverse_kinetic = kinetic.conj().T
    inverse_phase = phase.conj()
    # Matrix factors act right-to-left in Layden Eq. (36).
    out = kinetic @ state
    out = phase * out
    out = inverse_kinetic @ out
    out = inverse_phase * out
    out = inverse_kinetic @ out
    out = inverse_phase * out
    out = kinetic @ out
    return phase * out


def dense_reference(potential, n: int, final_time: float = 1.0,
                    rtol: float = 1e-10, atol: float = 1e-12):
    """DOP853 solution of dpsi/dt=[K,V_t]psi, segmented at model time knots.

    The public MPS-V time site represents a piecewise-linear function.
    Interpolation of its stored-weight endpoint tables in float64 is exact for
    the represented piecewise-linear function, up to floating-point roundoff.
    This avoids repeatedly invoking Torch during the ODE solve. The independent
    upstream float32 evaluation is validated separately. Generic potentials
    are called directly.
    The returned state is not renormalized, so norm drift remains inspectable.
    """
    start = perf_counter()
    if not np.isfinite(final_time) or not 0 <= final_time <= 1:
        raise ValueError("final_time must lie in [0,1]")
    if not (0 < rtol < 1 and 0 < atol < 1):
        raise ValueError("positive solver tolerances below one required")
    dimension = int(potential.dimension)
    length = float(getattr(potential, "length", 8.0))
    sigma = float(getattr(potential, "sigma0", 4 / 3))
    kinetic = kinetic_matrix(n, dimension, length)
    state = grid_initial_state(potential, n)
    from .potentials import LearnedPotential
    is_learned = isinstance(potential, LearnedPotential)
    config = getattr(potential, "config", {})
    time_count = int(config.get("N_t", config.get("Nt", 1))) if isinstance(config, Mapping) else 1
    all_knots = np.linspace(0.0, 1.0, time_count) if time_count >= 2 else np.array([0.0, 1.0])
    endpoints = np.unique(np.r_[0.0, all_knots[(all_knots > 0) & (all_knots < final_time)], final_time])
    knot_values = None
    if is_learned:
        knot_values = [np.asarray(potential.values(float(t), n), dtype=float) for t in all_knots]

    def values(t):
        if knot_values is None:
            return np.asarray(potential.values(float(t), n), dtype=float).reshape(-1)
        segment = min(len(all_knots) - 2, max(0, np.searchsorted(all_knots, t, side="right") - 1))
        weight = (t - all_knots[segment]) / (all_knots[segment + 1] - all_knots[segment])
        return (1 - weight) * knot_values[segment] + weight * knot_values[segment + 1]

    def rhs(t, y):
        diagonal = values(t)
        if diagonal.shape != y.shape or not np.isfinite(diagonal).all():
            raise ValueError("potential returned invalid grid values")
        return kinetic @ (diagonal * y) - diagonal * (kinetic @ y)

    setup_done = perf_counter()
    nfev = 0
    solver_steps = 0
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        result = solve_ivp(rhs, (float(left), float(right)), state,
                           method="DOP853", rtol=rtol, atol=atol)
        if not result.success:
            raise RuntimeError(f"dense ODE reference failed: {result.message}")
        state = result.y[:, -1]
        nfev += result.nfev
        solver_steps += len(result.t) - 1
    metadata = {
        "method": "segmented_DOP853", "n": n, "dimension": dimension,
        "length": length, "sigma0": sigma, "final_time": final_time,
        "rtol": rtol, "atol": atol, "nfev": nfev,
        "solver_steps": solver_steps, "segments": max(0, len(endpoints) - 1),
        "segment_endpoints": endpoints.tolist(), "seconds": perf_counter() - start,
        "setup_seconds": setup_done - start, "ode_seconds": perf_counter() - setup_done,
        "norm_drift": float(abs(np.linalg.norm(state) - 1)),
        "time_evaluation": ("float64 linear interpolation of stored-weight model knot values"
                            if is_learned else "potential.values(t,n)"),
    }
    return state, metadata


def state_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    """State fidelity and distribution TV, with explicit norm diagnostics.

    Fidelity and TV use normalized vectors. The raw vector error includes norm
    drift and global phase; the additional aligned metric removes global phase.
    """
    actual = np.asarray(actual, dtype=complex).reshape(-1)
    reference = np.asarray(reference, dtype=complex).reshape(-1)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("finite states of identical dimension required")
    norm_actual, norm_reference = np.linalg.norm(actual), np.linalg.norm(reference)
    if min(norm_actual, norm_reference) <= 0:
        raise ValueError("zero vector is not a quantum state")
    a, b = actual / norm_actual, reference / norm_reference
    overlap = np.vdot(b, a)
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1
    return {
        "fidelity": float(np.clip(abs(overlap)**2, 0, 1)),
        "tv_distance": float(0.5 * np.sum(np.abs(np.abs(a)**2 - np.abs(b)**2))),
        "state_l2_error": float(np.linalg.norm(actual - reference)),
        "phase_aligned_l2_error": float(np.linalg.norm(a / phase - b)),
        "actual_norm": float(norm_actual), "reference_norm": float(norm_reference),
    }
