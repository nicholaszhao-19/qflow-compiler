"""Load the public Sygaldry MPS-V model without changing its learned function.

The benchmark stores grid values in NumPy C order: the last spatial coordinate
is the least significant register.  All coordinates are centered, on the
periodic grid ``-L/2 + j*L/n``.  This is the same half-box shift performed by
``tnwf.mps_v.make_mps_v_provider`` at the model's native resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "11358d3dca52193f93f037046229797296a2063e"
DEFAULT_CHECKPOINT = ROOT / "data" / "checkpoints" / "gmm_2d_rank2.pt"


def ensure_upstream() -> None:
    """Make the vendored, pinned, MIT-licensed upstream package importable."""
    source = ROOT / "vendor" / "tnwf-paper" / "src"
    if not (source / "tnwf" / "mps_v" / "model.py").is_file():
        raise FileNotFoundError(
            f"Missing pinned tnwf source at {source}; run scripts/fetch_upstream.py"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def centered_grid(n: int, d: int, length: float = 8.0) -> np.ndarray:
    """Return all centered periodic grid points in C order."""
    if n < 2 or d < 1 or not np.isfinite(length) or length <= 0:
        raise ValueError("n>=2, d>=1 and finite positive length are required")
    axis = np.arange(n, dtype=np.float64) * (length / n) - length / 2
    return np.stack(np.meshgrid(*([axis] * d), indexing="ij"), axis=-1).reshape(-1, d)


def contract_cores(cores: list[np.ndarray]) -> np.ndarray:
    """Contract real TT cores to an N-dimensional array (only small grids)."""
    if not cores:
        raise ValueError("at least one TT core is required")
    checked = [np.asarray(c, dtype=np.float64) for c in cores]
    if checked[0].ndim != 3 or checked[0].shape[0] != 1:
        raise ValueError("left boundary rank must equal one")
    value = checked[0][0]
    for i, core in enumerate(checked[1:], 1):
        if core.ndim != 3 or value.shape[-1] != core.shape[0]:
            raise ValueError(f"incompatible TT core at site {i}")
        value = np.tensordot(value, core, axes=(-1, 0))
    if value.shape[-1] != 1:
        raise ValueError("right boundary rank must equal one")
    if not np.isfinite(value).all():
        raise ValueError("TT has nonfinite values")
    return value[..., 0]


@dataclass
class LearnedPotential:
    """A trained public-code MPS-V model with extraction validation helpers."""

    model: object
    config: dict
    training_args: dict
    path: Path

    @property
    def dimension(self) -> int:
        return int(self.config["d"])

    @property
    def length(self) -> float:
        return float(self.config["L"])

    @property
    def sigma0(self) -> float:
        return float(self.training_args["sigma_0"])

    def cores(self, t: float, n: int) -> list[np.ndarray]:
        """Extract V_t on an n-point centered grid using exact interpolation.

        Stored float32 weights are interpreted as exact real numbers and all
        interpolation/contraction is evaluated in float64.  This follows the
        upstream ``get_mps_cores`` algebra, but avoids its float32 time-site
        rounding jitter in adaptive ODE references.  Interpolating the local
        grid axis of each core exactly reproduces the model's multilinear
        spatial interpolation.  This is resampling, not TT fitting/truncation.
        """
        if not np.isfinite(t) or not 0 <= t <= 1:
            raise ValueError("t must lie in [0,1]")
        if n < 2:
            raise ValueError("grid size must be at least two")
        if self.model.boundary_mode != "periodic" or self.model.linear_baseline:
            raise ValueError("only periodic MPS-V without a linear baseline is supported")
        raw = [c.detach().cpu().numpy().astype(np.float64) for c in self.model.cores]
        nx = int(self.model.d_eff_x)
        nt = int(self.model.N_t)
        position = float(t) * (nt-1)
        t0 = min(nt-2, int(np.floor(position)))
        fraction = position - t0
        time_core = (1-fraction)*raw[nx][:,t0,:] + fraction*raw[nx][:,t0+1,:]
        last = np.einsum("anb,bc->anc",raw[nx-1],time_core)
        native = raw[:nx-1] + [last]
        size = int(self.config["N"])
        x = np.arange(n, dtype=np.float64) * self.length / n - self.length / 2
        u = np.mod(x / (self.length / size), size)
        lower = np.floor(u).astype(np.int64) % size
        upper = (lower + 1) % size
        alpha = (u - np.floor(u))[None, :, None]
        result = [(1 - alpha) * c[:, lower, :] + alpha * c[:, upper, :] for c in native]
        if any(not np.isfinite(c).all() for c in result):
            raise ValueError("checkpoint contains nonfinite core coefficients")
        return result

    def values(self, t: float, n: int) -> np.ndarray:
        return contract_cores(self.cores(t, n)).reshape(-1)

    def direct_values(self, t: float, n: int) -> np.ndarray:
        """Independent evaluation through upstream PyTorch forward (float32)."""
        import torch

        points = torch.tensor(centered_grid(n, self.dimension, self.length), dtype=torch.float32)
        times = torch.full((len(points), 1), float(t), dtype=torch.float32)
        with torch.no_grad():
            return self.model(points, times).detach().cpu().numpy().reshape(-1).astype(np.float64)


def load_potential(path: str | Path = DEFAULT_CHECKPOINT) -> LearnedPotential:
    """Load a checkpoint using upstream's own loader and validate its schema."""
    import torch

    ensure_upstream()
    from tnwf.mps_v import load_mps_v

    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = checkpoint.get("args", {})
    if checkpoint.get("model_type") != "mps_potential_time_site":
        raise ValueError("expected a public MPS-V time-site checkpoint")
    if "sigma_0" not in args:
        raise ValueError("source width must be recorded explicitly")
    model, config = load_mps_v(str(path), device="cpu")
    result = LearnedPotential(model=model, config=config, training_args=args, path=path)
    result.cores(0.5, 4)
    return result


def get_cores(t: float, grid_size: int, checkpoint: str | Path = DEFAULT_CHECKPOINT) -> list[np.ndarray]:
    """Convenience entry point; cache ``load_potential`` for repeated calls."""
    return load_potential(checkpoint).cores(t, grid_size)
