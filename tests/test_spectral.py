from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from floworaclebench.simulation import validate_oracle
from floworaclebench.smooth_flow import load_smooth_potential
from floworaclebench.spectral import build_fourier_oracle


@pytest.fixture(scope="module")
def potential():
    return load_smooth_potential(Path(__file__).resolve().parents[1] / "data/checkpoints/smooth_gmm.pt")


@pytest.mark.parametrize("n,time,beta", [(2, 0.125, 0.23), (4, 0.5, -0.31), (np.int64(8), 0.875, 0.11)])
def test_fourier_oracle_matches_learned_phase(potential, n, time, beta):
    oracle = build_fourier_oracle(potential, time, n, beta)
    reference = beta * potential.values(time, n)
    record = validate_oracle(oracle.circuit, oracle.n_address, reference)
    assert record["address_preserved"] and record["ancillas_clean"] and record["inverse_ancillas_clean"]
    for key in ("basis_phase_error", "superposition_error", "inverse_error"):
        assert record[key] < 2e-12
    if n == 4:
        block = Operator(oracle.circuit).data[:, :n**2]
        expected = np.zeros_like(block)
        expected[:n**2] = np.diag(np.exp(1j * reference))
        np.testing.assert_allclose(block, expected, rtol=0, atol=2e-12)
