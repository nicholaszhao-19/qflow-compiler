"""Independent model-forward/extracted-core checks, including coordinate shift."""
import csv
import hashlib
import json

import numpy as np
import pytest

from floworaclebench.potentials import DEFAULT_CHECKPOINT, contract_cores, load_potential


@pytest.fixture(scope="module")
def learned():
    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("train the small public MPS-V checkpoint first")
    return load_potential()


@pytest.mark.parametrize("n", [4, 8, 16, 10])
@pytest.mark.parametrize("t", [0.0, 0.13, 0.5, 1.0])
def test_extracted_tt_matches_independent_pytorch_forward(learned, n, t):
    np.testing.assert_allclose(learned.values(t, n), learned.direct_values(t, n), rtol=2e-5, atol=3e-6)


def test_native_resolution_matches_upstream_provider(learned):
    from tnwf.mps_v import make_mps_v_provider
    n = learned.config["N"]
    provider = make_mps_v_provider(learned.model, N=n, L=learned.length)
    for t in [0.0, 3/7, 1.0]:
        for a, b in zip(learned.cores(t, n), provider(t)):
            np.testing.assert_allclose(a, b.real, atol=2e-6, rtol=2e-6)


def test_float64_time_interpolation_is_piecewise_linear(learned):
    # The reference must not inherit float32 stepwise time-weight rounding.
    t0,t1 = 2/7,3/7
    a = learned.values(t0,8)
    b = learned.values(t1,8)
    for weight in [0.123456789,0.5,0.987654321]:
        middle = learned.values((1-weight)*t0+weight*t1,8)
        np.testing.assert_allclose(middle,(1-weight)*a+weight*b,rtol=2e-14,atol=2e-14)


def test_loading_preserves_training_source(learned):
    assert learned.dimension == 2
    assert learned.sigma0 == pytest.approx(4 / 3)
    assert learned.training_args["stopped_at"] >= 1000


def test_bad_time_is_rejected(learned):
    for t in [-0.01, 1.01, np.nan]:
        with pytest.raises(ValueError):
            learned.cores(t, 4)


def test_contract_validates_boundaries():
    with pytest.raises(ValueError):
        contract_cores([np.zeros((2, 4, 1))])
    with pytest.raises(ValueError):
        contract_cores([np.zeros((1, 4, 2)), np.zeros((3, 4, 1))])


def test_released_checkpoint_has_real_training_provenance(learned):
    metadata = json.loads(learned.path.with_suffix(".metadata.json").read_text())
    assert hashlib.sha256(learned.path.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
    with learned.path.with_suffix(".losses.csv").open() as file:
        losses = [float(row["loss"]) for row in csv.DictReader(file) if row["kind"] == "train"]
    assert len(losses) == learned.training_args["stopped_at"]
    assert np.mean(losses[-100:]) < np.mean(losses[:100]) - 1.0


@pytest.mark.parametrize("name", ["gmm_2d_rank4", "gmm_4d_rank2"])
def test_additional_learned_models_match_forward(name):
    path = DEFAULT_CHECKPOINT.with_name(name + ".pt")
    if not path.is_file():
        pytest.skip("optional training run is absent")
    model = load_potential(path)
    for n in [4, 8]:
        for t in [0.13, 0.5, 0.87]:
            np.testing.assert_allclose(model.values(t,n), model.direct_values(t,n), atol=3e-5, rtol=2e-5)
