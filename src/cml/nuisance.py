"""Cross-fitted nuisance functions (propensity score and outcome regressions).

Cross-fitting (Chernozhukov et al., 2018) splits the sample into ``K`` folds and
predicts every observation with models that were trained on the *other* folds, so
that the nuisance estimates are independent of the observation they are applied to.
This removes the own-observation overfitting bias that plain plug-in estimators
inherit from flexible machine-learning models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone

from .models import make_classifier, make_regressor

Array = NDArray[np.float64]
ModelSpec = "str | BaseEstimator"


def as_regressor(model: "str | BaseEstimator", seed: int = 0) -> BaseEstimator:
    return make_regressor(model, seed) if isinstance(model, str) else clone(model)


def as_classifier(model: "str | BaseEstimator", seed: int = 0) -> BaseEstimator:
    return make_classifier(model, seed) if isinstance(model, str) else clone(model)


def fit_predict(model: BaseEstimator, X: Array, y: Array, X_new: Array,
                sample_weight: Array | None = None) -> Array:
    """Fit a (cloned) model and return ``predict`` (regression) or ``P(y=1|x)`` (classification)."""
    m = clone(model)
    if sample_weight is None:
        m.fit(X, y)
    else:
        m.fit(X, y, sample_weight=sample_weight)
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X_new)[:, 1]
    return np.asarray(m.predict(X_new), dtype=float)


def make_folds(t: ArrayLike, n_folds: int, rng: np.random.Generator) -> Array:
    """Fold index per observation, stratified by treatment so every training split has both arms."""
    t = np.asarray(t)
    fold = np.empty(len(t), dtype=int)
    for arm in (0, 1):
        idx = np.flatnonzero(t == arm)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % n_folds
    return fold


@dataclass
class Nuisances:
    """Out-of-fold estimates: propensity ``e``, arm-specific means ``mu0``/``mu1``, marginal mean ``m``."""

    e: Array
    mu0: Array
    mu1: Array
    m: Array | None = None


def cross_fit_nuisances(
    X: ArrayLike,
    t: ArrayLike,
    y: ArrayLike,
    outcome_model: "str | BaseEstimator" = "gbm",
    propensity_model: "str | BaseEstimator" = "gbm",
    n_folds: int = 5,
    seed: int = 0,
    clip: float = 0.01,
    marginal: bool = False,
) -> Nuisances:
    """Cross-fitted ``e(x) = P(T=1|x)``, ``mu_a(x) = E[Y|x, T=a]`` (and ``m(x) = E[Y|x]`` if asked).

    Propensities are clipped to ``[clip, 1 - clip]`` to keep inverse weights finite.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=int).ravel()
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    fold = make_folds(t, n_folds, rng)
    n = len(y)
    e, mu0, mu1 = np.empty(n), np.empty(n), np.empty(n)
    m = np.empty(n) if marginal else None
    prop = as_classifier(propensity_model, seed)
    reg0, reg1 = as_regressor(outcome_model, seed), as_regressor(outcome_model, seed)
    regm = as_regressor(outcome_model, seed) if marginal else None
    for k in range(n_folds):
        te, tr = fold == k, fold != k
        Xtr, ttr, ytr, Xte = X[tr], t[tr], y[tr], X[te]
        e[te] = fit_predict(prop, Xtr, ttr, Xte)
        mu0[te] = fit_predict(reg0, Xtr[ttr == 0], ytr[ttr == 0], Xte)
        mu1[te] = fit_predict(reg1, Xtr[ttr == 1], ytr[ttr == 1], Xte)
        if marginal:
            m[te] = fit_predict(regm, Xtr, ytr, Xte)
    return Nuisances(np.clip(e, clip, 1 - clip), mu0, mu1, m)
