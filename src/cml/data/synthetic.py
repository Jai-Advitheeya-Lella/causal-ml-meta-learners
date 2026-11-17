"""Simulated observational data with known individual treatment effects.

Covariates ``X ~ U(-1, 1)^d``.  The potential outcomes are

    mu0(x) = sin(pi x1 x2) + 2 (x3 - 1/2)^2 + x4 + x5 / 2,
    tau(x) = 1 + 2 x1 + 2 * 1[x2 > 0],                       (ATE = 2)
    Y(a)   = mu0(x) + a tau(x) + sigma * eps,                 eps ~ N(0, 1).

Treatment is assigned with ``logit e(x) = s * (rho * z_prog(x) + sqrt(1 - rho^2) * z_other(x))``
where ``z_prog`` is the standardised prognostic score ``mu0(x)`` and ``z_other = sqrt(3) x6`` is
independent of the outcome.

* ``confounding = rho`` in ``[0, 1]``: how strongly treatment follows the prognostic score
  (``0`` = no confounding, ``1`` = treatment is a deterministic-direction function of ``mu0``).
* ``overlap_scale = s``: sharpness of the assignment; large ``s`` pushes ``e(x)`` towards 0 and 1
  and therefore destroys overlap (positivity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CausalData


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 2000
    d: int = 10
    confounding: float = 0.7
    overlap_scale: float = 1.0
    noise: float = 1.0
    seed: int = 0


def mu0_fn(X: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]


def tau_fn(X: np.ndarray) -> np.ndarray:
    return 1.0 + 2.0 * X[:, 0] + 2.0 * (X[:, 1] > 0)


def _prognostic_stats(d: int) -> tuple[float, float]:
    """Mean and sd of mu0(X) (computed once on a large fixed sample)."""
    X = np.random.default_rng(12345).uniform(-1, 1, size=(200_000, max(d, 6)))
    m = mu0_fn(X)
    return float(m.mean()), float(m.std())


def propensity_fn(X: np.ndarray, confounding: float, overlap_scale: float) -> np.ndarray:
    mean, sd = _prognostic_stats(X.shape[1])
    z_prog = (mu0_fn(X) - mean) / sd
    z_other = np.sqrt(3.0) * X[:, 5]
    logit = overlap_scale * (confounding * z_prog + np.sqrt(1 - confounding**2) * z_other)
    return 1.0 / (1.0 + np.exp(-logit))


def make_synthetic(cfg: SyntheticConfig = SyntheticConfig()) -> CausalData:
    rng = np.random.default_rng(cfg.seed)
    X = rng.uniform(-1, 1, size=(cfg.n, cfg.d))
    e = propensity_fn(X, cfg.confounding, cfg.overlap_scale)
    t = (rng.random(cfg.n) < e).astype(int)
    mu0, tau = mu0_fn(X), tau_fn(X)
    y = mu0 + t * tau + cfg.noise * rng.standard_normal(cfg.n)
    return CausalData(X=X, t=t, y=y, tau=tau, mu0=mu0, mu1=mu0 + tau, e=e)
