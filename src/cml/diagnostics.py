"""Overlap and covariate-balance diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def standardized_mean_difference(X: ArrayLike, t: ArrayLike, w: ArrayLike | None = None) -> Array:
    """Per-covariate ``(mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)`` with optional unit weights.

    Following Rosenbaum & Rubin, an absolute SMD below 0.1 is usually regarded as balanced.
    Variances are the *unweighted* ones so that SMDs before and after weighting are comparable.
    """
    X, t = np.asarray(X, dtype=float), np.asarray(t).ravel()
    w = np.ones(len(t)) if w is None else np.asarray(w, dtype=float)
    m1 = np.average(X[t == 1], axis=0, weights=w[t == 1]) if w[t == 1].sum() > 0 else 0 * X[0]
    m0 = np.average(X[t == 0], axis=0, weights=w[t == 0]) if w[t == 0].sum() > 0 else 0 * X[0]
    pooled = np.sqrt((X[t == 1].var(0, ddof=1) + X[t == 0].var(0, ddof=1)) / 2)
    pooled[pooled < 1e-12] = 1.0
    return (m1 - m0) / pooled


def effective_sample_size(w: ArrayLike) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2``."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())


def overlap_summary(e: ArrayLike, t: ArrayLike, eps: float = 0.05) -> dict[str, float]:
    """Share of units with propensity outside ``[eps, 1-eps]`` and ESS of the ATE weights."""
    e, t = np.asarray(e, dtype=float), np.asarray(t).ravel()
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    return {
        "frac_extreme": float(np.mean((e < eps) | (e > 1 - eps))),
        "min_e": float(e.min()),
        "max_e": float(e.max()),
        "ess_treated": effective_sample_size(w[t == 1]),
        "ess_control": effective_sample_size(w[t == 0]),
    }
