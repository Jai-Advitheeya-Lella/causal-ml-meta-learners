"""Evaluation metrics against ground-truth effects."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def sqrt_pehe(tau_hat: ArrayLike, tau: ArrayLike) -> float:
    """Root of the precision in estimation of heterogeneous effects (Hill, 2011)."""
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau)) ** 2)))


def ate_error(tau_hat: ArrayLike, tau: ArrayLike) -> float:
    """Absolute error of the average effect, ``|mean(tau_hat) - mean(tau)|``."""
    return float(abs(np.mean(tau_hat) - np.mean(tau)))
