"""Check the cross-grid diagnostic convention and separable dense reference."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from floworaclebench.evolution import kinetic_matrix

spec = importlib.util.spec_from_file_location(
    "reference_diagnostics", Path(__file__).resolve().parents[1] / "scripts/run_reference_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_spectral_interpolation_preserves_resolved_fourier_mode():
    old_n, new_n = 4, 8
    x, y = np.meshgrid(np.arange(old_n), np.arange(old_n), indexing="ij")
    old = (np.exp(2j * np.pi * (x - y) / old_n) / old_n).ravel()
    x, y = np.meshgrid(np.arange(new_n), np.arange(new_n), indexing="ij")
    expected = (np.exp(2j * np.pi * (x - y) / new_n) / new_n).ravel()
    actual = diagnostics.interpolate_state(old, old_n, new_n, 2)
    np.testing.assert_allclose(actual, expected, atol=5e-16, rtol=0)
    assert abs(np.linalg.norm(actual) - 1) < 2e-15


def test_separable_kinetic_exponential_matches_full_dense_exponential():
    actual = diagnostics.kinetic_exponential(4, 2, 8, 0.123)
    expected = expm(1j * 0.123 * kinetic_matrix(4, 2, 8))
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)


def test_periodized_mixture_is_normalized_and_respects_opposite_axis_symmetry():
    n, length = 16, 8
    x = np.arange(n) * length / n - length / 2
    points = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    centers = np.array([[3, 0], [-3, 0], [0, 3], [0, -3]])
    probability = diagnostics.grid_target_probability(points, centers, 0.5, length, image_radius=1)
    assert abs(probability.sum() - 1) < 8 * np.finfo(float).eps
    array = probability.reshape(n, n)
    reflected = (-np.arange(n)) % n
    np.testing.assert_allclose(array, array[reflected, :][:, reflected], atol=2e-16, rtol=0)
    # Wrapping matters at the boundary for modes near the two box edges.
    unwrapped = diagnostics.grid_target_probability(points, centers, 0.5, length)
    assert 0.5 * np.sum(abs(probability - unwrapped)) > 0.01
