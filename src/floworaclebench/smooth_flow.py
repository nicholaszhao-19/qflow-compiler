"""Smooth learned CFM potentials and independent continuum/grid diagnostics.

This extends the earlier periodic PotentialNet experiment using finite Fourier
spatial features and a PyTorch time MLP. The scalar potential is periodic and
C-infinity; its exact derivatives avoid numerical differentiation in validation.
CFM uses genuinely sampled independent Gaussian/GMM pairs, not a teacher field.

Arrays use the repository's centered periodic grid and NumPy C order. Results
from this module are classical numerical simulations, not quantum executions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import logsumexp, ndtr
from torch import Tensor, nn

from .evolution import gaussian_state, kinetic_matrix, state_metrics
from .potentials import centered_grid

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SmoothConfig:
    seed: int = 42
    steps: int = 2500
    batch_size: int = 1024
    learning_rate: float = 0.002
    hidden_width: int = 48
    max_mode: int = 5
    length: float = 12.0
    sigma0: float = 1.0
    target_std: float = 0.55
    mean_offset: float = 1.25
    weights: tuple[float, ...] = (0.4, 0.1, 0.1, 0.4)
    coefficient_penalty: float = 1e-5
    threads: int = 1
    samples: int = 4096
    grids: tuple[int, ...] = (8, 16, 32, 64)
    particle_steps: int = 128
    density_steps: int = 128
    formula_steps: tuple[int, ...] = (32, 64, 128, 256, 512)
    formula_grid: int = 16

    @property
    def centers(self) -> FloatArray:
        a = self.mean_offset
        return np.array([[-a, -a], [-a, a], [a, -a], [a, a]], dtype=float)

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 2 or self.hidden_width < 1 or self.max_mode < 1:
            raise ValueError("positive training dimensions required")
        if self.length <= 0 or self.sigma0 <= 0 or self.target_std <= 0:
            raise ValueError("positive spatial scales required")
        w = np.asarray(self.weights)
        if w.shape != (4,) or np.any(w <= 0) or not np.isclose(w.sum(), 1):
            raise ValueError("four strictly positive weights summing to one required")
        if not np.isclose(w[0], w[3]) or not np.isclose(w[1], w[2]):
            raise ValueError("cosine-only potential uses the target's inversion symmetry")


def frequency_indices(max_mode: int) -> FloatArray:
    """One representative of each +/- pair on a square Fourier lattice."""
    if max_mode < 1:
        raise ValueError("max_mode must be positive")
    return np.array([(0, j) for j in range(1, max_mode + 1)] +
                    [(i, j) for i in range(1, max_mode + 1)
                     for j in range(-max_mode, max_mode + 1)], dtype=float)


class PotentialNet(nn.Module):
    """V_theta(t,x)=sum_k a_theta,k(t) cos(k.x)/|k|, with real Fourier k.

    The velocity is exactly grad_x V. A small nonlinear network models time;
    all spatial modes, including mixed modes, are learned. The central-inversion
    symmetry is shared by the source and non-product mixture and is imposed
    deliberately. Coefficients are initialized to zero (zero initial velocity).
    """

    def __init__(self, config: SmoothConfig):
        super().__init__()
        config.validate()
        wavevectors = 2 * np.pi / config.length * frequency_indices(config.max_mode)
        self.register_buffer("wavevectors", torch.tensor(wavevectors, dtype=torch.float64))
        self.register_buffer("wavevector_norm", torch.tensor(np.linalg.norm(wavevectors, axis=1),
                                                           dtype=torch.float64))
        width = config.hidden_width
        self.time_net = nn.Sequential(nn.Linear(3, width, dtype=torch.float64), nn.Tanh(),
                                      nn.Linear(width, width, dtype=torch.float64), nn.Tanh(),
                                      nn.Linear(width, len(wavevectors), dtype=torch.float64))
        nn.init.zeros_(self.time_net[-1].weight)
        nn.init.zeros_(self.time_net[-1].bias)

    def coefficients(self, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t, dtype=self.wavevectors.dtype, device=self.wavevectors.device)
        if t.ndim == 0:
            t = t[None]
        features = torch.stack((t, torch.sin(np.pi * t), torch.cos(np.pi * t)), dim=-1)
        return self.time_net(features)

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        phases = x @ self.wavevectors.T
        return (torch.cos(phases) / self.wavevector_norm * self.coefficients(t)).sum(dim=-1)

    def velocity(self, x: Tensor, t: Tensor | float) -> Tensor:
        phases = x @ self.wavevectors.T
        factors = -torch.sin(phases) * self.coefficients(t) / self.wavevector_norm
        return factors @ self.wavevectors

    def divergence(self, x: Tensor, t: Tensor | float) -> Tensor:
        return (-torch.cos(x @ self.wavevectors.T) * self.coefficients(t)
                * self.wavevector_norm).sum(dim=-1)


def cfm_batch(config: SmoothConfig, n: int, generator: torch.Generator
              ) -> tuple[Tensor, Tensor, Tensor]:
    """Independent linear-path CFM: x_t=(1-t)x0+t*x1 and u=x1-x0."""
    x0 = config.sigma0 * torch.randn(n, 2, generator=generator, dtype=torch.float64)
    labels = torch.multinomial(torch.tensor(config.weights), n, replacement=True,
                               generator=generator)
    means = torch.tensor(config.centers, dtype=torch.float64)[labels]
    x1 = means + config.target_std * torch.randn(n, 2, generator=generator, dtype=torch.float64)
    t = torch.rand(n, generator=generator, dtype=torch.float64)
    return (1 - t[:, None]) * x0 + t[:, None] * x1, t, x1 - x0


def exact_cfm_velocity(config: SmoothConfig, t: FloatArray | float,
                       x: FloatArray) -> FloatArray:
    """Exact conditional mean E[x1-x0 | x_t=x] for this Euclidean GMM path.

    This is used only for held-out diagnostics. It is not the learned velocity
    and is not the exact velocity of the periodized model away from the box.
    """
    x = np.asarray(x, dtype=float)
    times = np.broadcast_to(np.asarray(t, dtype=float), x.shape[:-1])
    variance = (1 - times)**2 * config.sigma0**2 + times**2 * config.target_std**2
    cross = times * config.target_std**2 - (1 - times) * config.sigma0**2
    difference = x[..., None, :] - times[..., None, None] * config.centers
    logits = np.log(config.weights) - np.sum(difference**2, axis=-1) / (2 * variance[..., None])
    weights = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    mean = weights @ config.centers
    return cross[..., None] / variance[..., None] * x + (
        (1 - times) * config.sigma0**2 / variance)[..., None] * mean


def train(config: SmoothConfig) -> tuple[PotentialNet, list[dict[str, float]], dict[str, Any]]:
    """Train by sampled CFM with a disclosed small coefficient regularizer."""
    config.validate()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(config.seed + 1)
    model = PotentialNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps, eta_min=config.learning_rate / 10)
    rows: list[dict[str, float]] = []
    start = perf_counter()
    for step in range(1, config.steps + 1):
        x, t, u = cfm_batch(config, config.batch_size, generator)
        predicted = model.velocity(x, t)
        cfm_loss = torch.mean(torch.sum((predicted - u)**2, dim=-1))
        penalty = config.coefficient_penalty * torch.mean(model.coefficients(t)**2)
        loss = cfm_loss + penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 1 or step % max(1, config.steps // 100) == 0:
            rows.append({"step": step, "cfm_loss": float(cfm_loss.detach()),
                         "coefficient_penalty": float(penalty.detach()),
                         "learning_rate": float(optimizer.param_groups[0]["lr"])})
    elapsed = perf_counter() - start
    return model, rows, {"seconds": elapsed, "train_pairs": config.steps * config.batch_size,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "objective": "sampled independent-pair linear CFM + coefficient L2 penalty",
                        "uses_analytic_teacher_for_training": False}


def save_model(model: PotentialNet, config: SmoothConfig, path: str | Path,
               training_metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "smooth_fourier_time_mlp_v1", "config": asdict(config),
                "state_dict": model.state_dict(), "training": training_metadata}, path)


class SmoothPotential:
    """NumPy-facing adapter for frozen learned smooth periodic potential."""

    dimension = 2

    def __init__(self, model: PotentialNet, config: SmoothConfig):
        self.config = config
        self.model = model.eval().double()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.length = config.length
        self.sigma0 = config.sigma0
        self.wavevectors = self.model.wavevectors.detach().numpy().copy()
        self.wavevector_norm = self.model.wavevector_norm.detach().numpy().copy()
        self._grid_features: dict[int, FloatArray] = {}

    def _coefficients(self, t: float) -> FloatArray:
        if not np.isfinite(t) or t < 0 or t > 1:
            # Allow floating-point RK endpoint noise, not out-of-range extrapolation.
            if not np.isfinite(t) or t < -1e-12 or t > 1 + 1e-12:
                raise ValueError("t must lie in [0,1]")
            t = float(np.clip(t, 0, 1))
        with torch.no_grad():
            return self.model.coefficients(float(t)).numpy()[0]

    def values(self, t: float, n: int) -> FloatArray:
        if n not in self._grid_features:
            points = centered_grid(n, 2, self.length)
            self._grid_features[n] = np.cos(points @ self.wavevectors.T) / self.wavevector_norm
        return self._grid_features[n] @ self._coefficients(t)

    def velocity_numpy(self, t: float, points: FloatArray) -> FloatArray:
        x = np.asarray(points, dtype=float)
        return (-np.sin(x @ self.wavevectors.T)
                * (self._coefficients(t) / self.wavevector_norm)) @ self.wavevectors

    def velocity_divergence(self, t: float, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        phases = np.asarray(points, dtype=float) @ self.wavevectors.T
        coefficients = self._coefficients(t)
        velocity = (-np.sin(phases) * (coefficients / self.wavevector_norm)) @ self.wavevectors
        divergence = -np.cos(phases) @ (coefficients * self.wavevector_norm)
        return velocity, divergence

    def cores(self, t: float, n: int, max_rank: int | None = None) -> list[FloatArray]:
        """Two-site TT-SVD with balanced singular-value factors, on a dense grid.

        This is classical small-grid preprocessing. The singular values are
        measured; no scalable high-dimensional compression claim is made.
        """
        values = self.values(t, n).reshape(n, n)
        u, singular_values, vh = np.linalg.svd(values, full_matrices=False)
        rank = n if max_rank is None else min(n, int(max_rank))
        if rank < 1:
            raise ValueError("positive TT rank required")
        root = np.sqrt(singular_values[:rank])
        return [(u[:, :rank] * root).reshape(1, n, rank),
                (root[:, None] * vh[:rank]).reshape(rank, n, 1)]

    def source_density(self, points: FloatArray) -> FloatArray:
        """Product periodized N(0,sigma0^2), image truncation j=-2,...,2.

        Arguments are wrapped before evaluation. For the recorded L=12,sigma=1,
        omitted one-dimensional image tails start at |x|>=30.
        """
        x = (np.asarray(points, dtype=float) + self.length / 2) % self.length - self.length / 2
        shifts = np.arange(-2, 3) * self.length
        density_axis = (np.exp(-(x[..., None] + shifts)**2 / (2 * self.sigma0**2))
                        .sum(axis=-1) / (np.sqrt(2 * np.pi) * self.sigma0))
        return np.prod(density_axis, axis=-1)

    def target_density(self, points: FloatArray) -> FloatArray:
        """Periodized target GMM; all 25 nearest spatial images are included."""
        x = (np.asarray(points, dtype=float) + self.length / 2) % self.length - self.length / 2
        sigma = self.config.target_std
        centers = self.config.centers
        weights = np.asarray(self.config.weights)
        density = np.zeros(x.shape[:-1])
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                delta = x[..., None, :] + self.length * np.array([ix, iy]) - centers
                density += np.exp(-np.sum(delta**2, axis=-1) / (2 * sigma**2)) @ weights
        return density / (2 * np.pi * sigma**2)

    def initial_state(self, n: int) -> NDArray[np.complex128]:
        density = self.source_density(centered_grid(n, 2, self.length))
        return np.sqrt(density / density.sum()).astype(complex)


def load_smooth_potential(path: str | Path) -> SmoothPotential:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_type") != "smooth_fourier_time_mlp_v1":
        raise ValueError("unsupported smooth potential checkpoint")
    config = SmoothConfig(**payload["config"])
    model = PotentialNet(config)
    model.load_state_dict(payload["state_dict"])
    return SmoothPotential(model, config)


def heldout_diagnostics(potential: SmoothPotential, n: int = 16384,
                        seed: int = 1907) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    x, t, u = cfm_batch(potential.config, n, generator)
    with torch.no_grad():
        learned = potential.model.velocity(x.double(), t.double()).numpy()
    truth = exact_cfm_velocity(potential.config, t.numpy(), x.numpy())
    observed = u.numpy()
    return {"n": n, "seed": seed,
            "learned_cfm_mse": float(np.mean(np.sum((learned - observed)**2, axis=1))),
            "zero_velocity_cfm_mse": float(np.mean(np.sum(observed**2, axis=1))),
            "analytic_conditional_cfm_mse": float(np.mean(np.sum((truth - observed)**2, axis=1))),
            "conditional_velocity_rmse": float(np.sqrt(np.mean(np.sum((learned - truth)**2, axis=1)))),
            "conditional_velocity_relative_rms": float(np.sqrt(np.mean(np.sum((learned-truth)**2,axis=1))
                                                         / np.mean(np.sum(truth**2,axis=1))))}


def sample_source(config: SmoothConfig, n: int, seed: int) -> FloatArray:
    return np.random.default_rng(seed).normal(scale=config.sigma0, size=(n, 2))


def sample_target(config: SmoothConfig, n: int, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    labels = rng.choice(4, size=n, p=config.weights)
    return config.centers[labels] + rng.normal(scale=config.target_std, size=(n, 2))


def particle_flow(potential: SmoothPotential, source: FloatArray, steps: int = 128) -> FloatArray:
    """RK4 particle pushforward of the same learned velocity, no particle noise."""
    if steps < 1:
        raise ValueError("steps must be positive")
    x = np.array(source, dtype=float, copy=True)
    h = 1 / steps
    for j in range(steps):
        t = j * h
        k1 = potential.velocity_numpy(t, x)
        k2 = potential.velocity_numpy(t + h / 2, x + h * k1 / 2)
        k3 = potential.velocity_numpy(t + h / 2, x + h * k2 / 2)
        k4 = potential.velocity_numpy(t + h, x + h * k3)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def reverse_density(potential: SmoothPotential, points: FloatArray,
                    steps: int = 128) -> tuple[FloatArray, dict[str, float]]:
    """Density by backward characteristics and the exact divergence of V.

    With y(1)=x and J(1)=0, integrate y'=v and J'=div(v) backwards.
    Then log p_1(x)=log p_0(y(0))+J(0). This is an independent classical
    reference; it does not evolve a grid wavefunction or use target densities.
    """
    start = perf_counter()
    if steps < 1:
        raise ValueError("steps must be positive")
    x = np.array(points, dtype=float, copy=True)
    log_jacobian = np.zeros(x.shape[:-1])
    h = -1 / steps
    for j in range(steps):
        t = 1 + j * h
        v1, d1 = potential.velocity_divergence(t, x)
        v2, d2 = potential.velocity_divergence(t + h / 2, x + h * v1 / 2)
        v3, d3 = potential.velocity_divergence(t + h / 2, x + h * v2 / 2)
        v4, d4 = potential.velocity_divergence(t + h, x + h * v3)
        x += h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        log_jacobian += h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
    density = potential.source_density(x) * np.exp(log_jacobian)
    if not np.all(np.isfinite(density)):
        raise FloatingPointError("nonfinite reverse-flow density")
    return density, {"seconds": perf_counter() - start, "steps": steps,
                     "maximum_absolute_log_jacobian": float(np.max(np.abs(log_jacobian)))}


def kinetic_spectrum(n: int, length: float) -> FloatArray:
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    return 0.5 * (frequency[:, None]**2 + frequency[None, :]**2)


def wavefunction_reference(potential: SmoothPotential, n: int, rtol: float = 1e-10,
                          atol: float = 1e-12) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """DOP853 for dpsi/dt=[K,V_t]psi, K applied by periodic Fourier FFT.

    No normalization is performed after integration; norm drift is reported.
    Adaptive tolerance is a numerical setting, not a certified error bound.
    """
    start = perf_counter()
    spectrum = kinetic_spectrum(n, potential.length)
    potential.values(0, n)  # Precompute spatial basis outside integrator timing.

    def kinetic(y: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.fft.ifft2(spectrum * np.fft.fft2(y.reshape(n, n))).reshape(-1)

    def rhs(t: float, y: NDArray[np.complex128]) -> NDArray[np.complex128]:
        values = potential.values(t, n)
        return kinetic(values * y) - values * kinetic(y)

    result = solve_ivp(rhs, (0, 1), potential.initial_state(n), method="DOP853",
                       rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    state = result.y[:, -1]
    return state, {"method": "DOP853_FFT", "seconds": perf_counter() - start,
                   "n": n, "rtol": rtol, "atol": atol, "nfev": result.nfev,
                   "solver_steps": len(result.t) - 1,
                   "norm_drift": float(abs(np.linalg.norm(state) - 1))}


def product_formula(potential: SmoothPotential, n: int, steps: int
                    ) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Layden's established eight-factor commutator formula, midpoint in time.

    Chronological order is K+,V+,K-,V-,K-,V-,K+,V+ and alpha*beta=dt/2.
    Frozen-H local O(dt^2), global O(dt); no second-order global claim.
    Alpha balances the finite-grid spectral norms using half the V range.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    start = perf_counter()
    state = potential.initial_state(n).reshape(n, n)
    spectrum = kinetic_spectrum(n, potential.length)
    kinetic_norm = np.max(spectrum)
    h = 1 / steps
    maximum_drift = 0.0
    alphas, betas = [], []
    for j in range(steps):
        values = potential.values((j + 0.5) * h, n).reshape(n, n)
        potential_norm = max(1e-12, np.ptp(values) / 2)
        alpha = np.sqrt(h / 2 * potential_norm / kinetic_norm)
        beta = h / (2 * alpha)
        alphas.append(float(alpha))
        betas.append(float(beta))
        vplus = np.exp(1j * beta * values)
        kplus = np.exp(1j * alpha * spectrum)
        for sign in (1, -1, -1, 1):
            state = np.fft.ifft2((kplus if sign == 1 else kplus.conj()) * np.fft.fft2(state))
            state *= vplus if sign == 1 else vplus.conj()
        maximum_drift = max(maximum_drift, abs(np.linalg.norm(state) - 1))
    return state.reshape(-1), {"seconds": perf_counter() - start, "n": n, "steps": steps,
                              "potential_blocks": 4 * steps, "kinetic_blocks": 4 * steps,
                              "maximum_norm_drift": float(maximum_drift),
                              "alpha_range": [min(alphas), max(alphas)],
                              "beta_range": [min(betas), max(betas)],
                              "native_rotations": "exact floating-point phase exponentials; not Clifford+T synthesis"}


def sliced_wasserstein_1(x: FloatArray, y: FloatArray, directions: int = 128,
                         seed: int = 51) -> float:
    """Empirical mean 1D W1 over seeded unit projections; equal sample sizes."""
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("matching point matrices required")
    rng = np.random.default_rng(seed)
    unit = rng.normal(size=(x.shape[1], directions))
    unit /= np.linalg.norm(unit, axis=0, keepdims=True)
    return float(np.mean(np.abs(np.sort(x @ unit, axis=0) - np.sort(y @ unit, axis=0))))


def hermiticity_error(potential: SmoothPotential, n: int = 8, t: float = 0.37) -> float:
    kinetic = kinetic_matrix(n, 2, potential.length)
    values = potential.values(t, n)
    hamiltonian = 1j * (kinetic * values[None, :] - values[:, None] * kinetic)
    return float(np.linalg.norm(hamiltonian - hamiltonian.conj().T, ord="fro"))


def boundary_diagnostics(potential: SmoothPotential) -> dict[str, Any]:
    config = potential.config
    gaussian_tail_1d = 2 * ndtr(-config.length / (2 * config.sigma0))
    source_tail = 2 * gaussian_tail_1d - gaussian_tail_1d**2
    means = config.centers
    sigma = config.target_std
    mass_axes = ndtr((config.length / 2 - means) / sigma) - ndtr((-config.length / 2 - means) / sigma)
    target_tail = 1 - float(np.asarray(config.weights) @ np.prod(mass_axes, axis=1))
    comparisons = {str(n): float(np.linalg.norm(potential.initial_state(n) -
                         gaussian_state(n, 2, config.length, config.sigma0))) for n in config.grids}
    # Cancellation-free union bound is useful when target tail rounds to zero.
    union = float(np.asarray(config.weights) @ np.sum(
        ndtr((-config.length / 2 - means) / sigma) + ndtr((means - config.length / 2) / sigma), axis=1))
    return {"domain": [-config.length / 2, config.length / 2],
            "periodic_potential": True, "smoothness": "finite Fourier features and tanh time MLP, C-infinity",
            "source_outside_euclidean_box_probability": float(source_tail),
            "target_outside_euclidean_box_probability_float": float(target_tail),
            "target_outside_euclidean_box_union_bound": union,
            "periodized_vs_plain_grid_initial_state_l2": comparisons,
            "periodic_images_per_axis": [-2, -1, 0, 1, 2],
            "omitted_source_images_minimum_distance": 2.5 * config.length,
            "omitted_source_images_probability_upper_bound": float(4*ndtr(-2.5*config.length/config.sigma0)),
            "interpretation": "Training is Euclidean CFM; continuum validation is the learned periodic flow of a wrapped Gaussian. Model error is measured separately."}
