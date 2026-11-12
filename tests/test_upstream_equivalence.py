"""Cross-check the independent matrix code with pinned Sygaldry FFT code."""
import numpy as np
import pytest
from scipy.linalg import expm

from floworaclebench.evolution import kinetic_matrix, apply_product_formula
from floworaclebench.potentials import ensure_upstream


@pytest.mark.parametrize("n,dimension,length", [(4, 1, 8.0), (4, 2, 12.0), (8, 2, 12.0)])
def test_product_formula_matches_pinned_public_implementation(n, dimension, length):
    ensure_upstream()
    from tnwf.dense.evolution import apply_product_formula as upstream_product

    rng = np.random.default_rng(42)
    state = rng.normal(size=n**dimension) + 1j * rng.normal(size=n**dimension)
    state /= np.linalg.norm(state)
    potential = rng.normal(size=n**dimension)
    alpha, beta = 0.13, 0.07
    actual = apply_product_formula(
        state, np.exp(1j * beta * potential),
        expm(1j * alpha * kinetic_matrix(n, dimension, length)),
    )
    reference = upstream_product(state, potential, alpha, beta, n, dimension, length)
    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=0)
