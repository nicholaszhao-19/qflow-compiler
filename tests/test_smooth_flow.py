"""Independent mathematical checks for smooth learned flow validation.

These tests verify derivatives, conditional Gaussian identities, Fourier signs,
reference density normalization and step-size convergence. They do not encode
one trained checkpoint's favorable empirical outcome as a unit-test oracle.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from floworaclebench.evolution import apply_product_formula, kinetic_matrix
from floworaclebench.potentials import centered_grid, contract_cores
from floworaclebench.smooth_flow import (
    PotentialNet, SmoothConfig, SmoothPotential, boundary_diagnostics,
    exact_cfm_velocity, kinetic_spectrum, load_smooth_potential, particle_flow,
    product_formula, reverse_density, save_model, wavefunction_reference,
)


def nonzero_model(config):
    torch.manual_seed(83)
    model = PotentialNet(config)
    with torch.no_grad():
        model.time_net[-1].weight.normal_(std=0.04)
        model.time_net[-1].bias.normal_(std=0.03)
    return model


def test_exact_potential_derivatives_match_pytorch_autograd():
    config = replace(SmoothConfig(), max_mode=2, hidden_width=8)
    model = nonzero_model(config)
    x = torch.tensor([[0.2, -0.7], [1.4, 0.5], [-0.3, 1.7]], dtype=torch.float64,
                     requires_grad=True)
    t = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    gradient = torch.autograd.grad(model(x, t).sum(), x, create_graph=True)[0]
    divergence = sum(torch.autograd.grad(gradient[:, i].sum(), x, retain_graph=True)[0][:, i]
                     for i in range(2))
    torch.testing.assert_close(model.velocity(x, t), gradient, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(model.divergence(x, t), divergence, rtol=1e-13, atol=1e-13)


def test_periodicity_of_values_first_and_second_derivatives():
    config = replace(SmoothConfig(), max_mode=3, hidden_width=8)
    model = nonzero_model(config)
    x = torch.tensor([[0.2, -0.7], [1.4, 0.5]], dtype=torch.float64)
    for shift in ([config.length, 0], [0, config.length]):
        shifted = x + torch.tensor(shift, dtype=torch.float64)
        for function in (model, model.velocity, model.divergence):
            torch.testing.assert_close(function(x, 0.37), function(shifted, 0.37),
                                       atol=3e-14, rtol=1e-12)


def test_analytic_cfm_velocity_boundary_times_and_single_gaussian_limit():
    config = replace(SmoothConfig(), mean_offset=0, sigma0=1.2, target_std=0.7)
    x = np.array([[0.2, -0.4], [1.7, 2.0]])
    for t in [0, 0.3, 1]:
        variance = (1 - t)**2 * config.sigma0**2 + t*t*config.target_std**2
        c = t*config.target_std**2 - (1-t)*config.sigma0**2
        np.testing.assert_allclose(exact_cfm_velocity(config, t, x), c / variance * x, atol=1e-15)
    config = SmoothConfig()
    np.testing.assert_allclose(exact_cfm_velocity(config, 0, x), -x, atol=2e-15)
    np.testing.assert_allclose(exact_cfm_velocity(config, 1, x), x, atol=2e-15)


def test_cfm_conditional_mean_from_direct_gaussian_conditioning():
    config = replace(SmoothConfig(), sigma0=1.1, target_std=0.7)
    x = np.array([[0.2, -0.4], [1.7, 2.0]])
    t = np.array([0.2, 0.81])
    # Per-component Gaussian conditional E[x0|xt] and E[x1|xt].
    expected = []
    for time, point in zip(t, x):
        variance = (1-time)**2*config.sigma0**2 + time*time*config.target_std**2
        delta = point - time*config.centers
        post = np.asarray(config.weights)*np.exp(-np.sum(delta**2,axis=1)/(2*variance))
        post /= post.sum()
        e0 = (1-time)*config.sigma0**2/variance*delta
        e1 = config.centers + time*config.target_std**2/variance*delta
        expected.append(post@(e1-e0))
    np.testing.assert_allclose(exact_cfm_velocity(config,t,x),expected,atol=2e-15)


def test_numpy_adapter_and_full_rank_tt_match_model():
    config = replace(SmoothConfig(), max_mode=2, hidden_width=8)
    model = nonzero_model(config)
    potential = SmoothPotential(model, config)
    points = centered_grid(8, 2, config.length)
    x = torch.tensor(points, dtype=torch.float64)
    with torch.no_grad():
        values = model(x, 0.37).numpy()
        velocity = model.velocity(x, 0.37).numpy()
        divergence = model.divergence(x, 0.37).numpy()
    np.testing.assert_allclose(potential.values(0.37, 8), values, atol=1e-14)
    v, div = potential.velocity_divergence(0.37, points)
    np.testing.assert_allclose(v, velocity, atol=1e-14)
    np.testing.assert_allclose(div, divergence, atol=1e-14)
    np.testing.assert_allclose(contract_cores(potential.cores(0.37, 8)).reshape(-1), values, atol=3e-14)


def test_wrapped_gaussian_is_periodic_and_grid_normalized():
    config = replace(SmoothConfig(), max_mode=1, hidden_width=4)
    potential = SmoothPotential(PotentialNet(config), config)
    points = centered_grid(64, 2, config.length)
    np.testing.assert_allclose(potential.source_density(points),
                               potential.source_density(points + [config.length, 0]), atol=1e-15)
    assert potential.source_density(points).sum()*(config.length/64)**2 == pytest.approx(1,abs=1e-14)
    assert np.linalg.norm(potential.initial_state(64)) == pytest.approx(1,abs=1e-15)
    assert boundary_diagnostics(potential)["source_outside_euclidean_box_probability"] < 4e-9


def test_zero_velocity_classical_and_quantum_identity():
    config = replace(SmoothConfig(), max_mode=1, hidden_width=4)
    potential = SmoothPotential(PotentialNet(config), config)
    points = centered_grid(8, 2, config.length)
    np.testing.assert_array_equal(particle_flow(potential, points, 4), points)
    density, _ = reverse_density(potential, points, 4)
    np.testing.assert_allclose(density, potential.source_density(points), atol=0)
    state, metadata = wavefunction_reference(potential,8)
    np.testing.assert_allclose(state,potential.initial_state(8),atol=1e-15)
    assert metadata["norm_drift"] < 1e-15


def test_fft_generator_matches_dense_collocation_and_is_antihermitian():
    config = replace(SmoothConfig(), max_mode=2, hidden_width=8)
    potential = SmoothPotential(nonzero_model(config), config)
    n = 8
    kinetic = kinetic_matrix(n, 2, config.length)
    values = potential.values(0.37, n)
    generator = kinetic * values[None,:] - values[:,None] * kinetic
    np.testing.assert_allclose(generator + generator.conj().T,0,atol=1e-14)
    rng = np.random.default_rng(17)
    state = rng.normal(size=n*n)+1j*rng.normal(size=n*n)
    spectrum = kinetic_spectrum(n,config.length)
    def k(v):
        return np.fft.ifft2(spectrum*np.fft.fft2(v.reshape(n,n))).reshape(-1)
    np.testing.assert_allclose(k(values*state)-values*k(state),generator@state,atol=5e-14)


def test_reverse_density_jacobian_sign_for_exact_periodic_flow():
    # v(x)=a sin(x), torus length 2*pi. tan(x(t)/2)=exp(a*t)tan(x0/2).
    # Uniform source gives p1(x)=1/[2*pi*(cosh(a)+sinh(a)*cos(x))].
    class SinePotential:
        def velocity_divergence(self,t,x):
            return 0.4*np.sin(x),0.4*np.cos(x).sum(axis=1)
        def source_density(self,x):
            return np.full(len(x),1/(2*np.pi)**2)
    axis = (np.arange(24)+0.5)*2*np.pi/24-np.pi
    points = np.stack(np.meshgrid(axis,axis,indexing="ij"),axis=-1).reshape(-1,2)
    density,_=reverse_density(SinePotential(),points,32)
    exact=np.prod(1/(2*np.pi*(np.cosh(0.4)+np.sinh(0.4)*np.cos(points))),axis=1)
    np.testing.assert_allclose(density,exact,rtol=2e-9,atol=1e-12)


def test_product_formula_matches_independent_dense_eight_factor_step():
    config=replace(SmoothConfig(),max_mode=1,hidden_width=4)
    potential=SmoothPotential(nonzero_model(config),config)
    n=4
    state, _=product_formula(potential,n,1)
    values=potential.values(0.5,n)
    kinetic=kinetic_matrix(n,2,config.length)
    alpha=np.sqrt(0.5*(np.ptp(values)/2)/np.max(kinetic_spectrum(n,config.length)))
    beta=1/(2*alpha)
    reference=apply_product_formula(potential.initial_state(n),np.exp(1j*beta*values),expm(1j*alpha*kinetic))
    np.testing.assert_allclose(state,reference,atol=2e-14)


def test_checkpoint_roundtrip_preserves_exact_learned_function(tmp_path):
    config=replace(SmoothConfig(),max_mode=2,hidden_width=8)
    model=nonzero_model(config)
    original=SmoothPotential(model,config)
    path=tmp_path/'model.pt'
    save_model(model,config,path,{"executed_test":True})
    restored=load_smooth_potential(path)
    np.testing.assert_array_equal(original.values(0.37,8),restored.values(0.37,8))


def test_low_rank_svd_error_matches_discarded_singular_values():
    config=replace(SmoothConfig(),max_mode=2,hidden_width=8)
    potential=SmoothPotential(nonzero_model(config),config)
    values=potential.values(0.37,8).reshape(8,8)
    low=contract_cores(potential.cores(0.37,8,max_rank=2))
    singular=np.linalg.svd(values,compute_uv=False)
    assert np.linalg.norm(values-low)==pytest.approx(np.linalg.norm(singular[2:]),abs=2e-14)


def test_time_independent_wave_reference_matches_exact_matrix_exponential():
    config=replace(SmoothConfig(),max_mode=1,hidden_width=4)
    model=PotentialNet(config)
    with torch.no_grad():
        model.time_net[-1].bias.copy_(torch.tensor([0.2,-0.1,0.07,0.05],dtype=torch.float64))
    potential=SmoothPotential(model,config)
    n=8
    values=potential.values(0,n)
    kinetic=kinetic_matrix(n,2,config.length)
    generator=kinetic*values[None,:]-values[:,None]*kinetic
    expected=expm(generator)@potential.initial_state(n)
    actual, metadata=wavefunction_reference(potential,n,rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(actual,expected,atol=8e-12)
    assert metadata['norm_drift']<1e-12
