"""Average-treatment-effect estimators with uncertainty quantification.

Estimands: ``"ate"`` = ``E[Y(1) - Y(0)]`` and ``"att"`` = ``E[Y(1) - Y(0) | T = 1]``.

* difference in means, regression adjustment (g-computation)
* inverse-propensity weighting (Horvitz-Thompson or Hajek-normalised)
* augmented IPW (AIPW) with cross-fitted nuisances -- doubly robust
* double/debiased ML, partially linear model (Robinson partialling-out with cross-fitting)
* propensity-score matching (nearest neighbour on the logit, with replacement and caliper)

Each estimator returns an :class:`Estimate`; analytic standard errors come from the efficient
influence function where one exists, and :func:`bootstrap` gives percentile intervals for any
estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from .nuisance import as_classifier, as_regressor, cross_fit_nuisances, fit_predict, make_folds

Array = NDArray[np.float64]
Z95 = 1.959963984540054


@dataclass
class Estimate:
    """Point estimate with an (optional) analytic standard error."""

    value: float
    se: float = float("nan")
    method: str = ""
    estimand: str = "ate"

    @property
    def ci(self) -> tuple[float, float]:
        return self.value - Z95 * self.se, self.value + Z95 * self.se


def _arrays(X: ArrayLike, t: ArrayLike, y: ArrayLike) -> tuple[Array, Array, Array]:
    return (np.asarray(X, dtype=float), np.asarray(t, dtype=int).ravel(),
            np.asarray(y, dtype=float).ravel())


def _check(estimand: str) -> None:
    if estimand not in ("ate", "att"):
        raise ValueError("estimand must be 'ate' or 'att'")


# ------------------------------------------------------------------ simple baselines
def difference_in_means(t: ArrayLike, y: ArrayLike) -> Estimate:
    """Naive contrast; unbiased only for randomised treatment.  Welch standard error."""
    t, y = np.asarray(t).ravel(), np.asarray(y, dtype=float).ravel()
    y1, y0 = y[t == 1], y[t == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return Estimate(float(y1.mean() - y0.mean()), float(se), "difference in means", "ate")


def regression_adjustment(X, t, y, outcome_model: "str | BaseEstimator" = "ridge",
                          estimand: str = "att", seed: int = 0) -> Estimate:
    """G-computation: fit ``mu_0, mu_1`` and average ``mu_1 - mu_0`` (or ``Y - mu_0`` for the ATT)."""
    _check(estimand)
    X, t, y = _arrays(X, t, y)
    mu0 = as_regressor(outcome_model, seed).fit(X[t == 0], y[t == 0]).predict(X)
    if estimand == "att":
        return Estimate(float(np.mean(y[t == 1] - mu0[t == 1])), method="regression adjustment",
                        estimand=estimand)
    mu1 = as_regressor(outcome_model, seed).fit(X[t == 1], y[t == 1]).predict(X)
    return Estimate(float(np.mean(mu1 - mu0)), method="regression adjustment", estimand=estimand)


# ------------------------------------------------------------------------------ IPW
def propensity_scores(X, t, model: "str | BaseEstimator" = "logistic", clip: float = 0.01,
                      cross_fit: bool = False, n_folds: int = 5, seed: int = 0) -> Array:
    """Estimated ``P(T=1|x)``, optionally cross-fitted, clipped to ``[clip, 1-clip]``."""
    X, t = np.asarray(X, dtype=float), np.asarray(t, dtype=int).ravel()
    prop = as_classifier(model, seed)
    if not cross_fit:
        e = fit_predict(prop, X, t, X)
    else:
        fold = make_folds(t, n_folds, np.random.default_rng(seed))
        e = np.empty(len(t))
        for k in range(n_folds):
            e[fold == k] = fit_predict(prop, X[fold != k], t[fold != k], X[fold == k])
    return np.clip(e, clip, 1 - clip)


def ipw(X, t, y, propensity_model: "str | BaseEstimator" = "logistic", estimand: str = "ate",
        normalize: bool = True, clip: float = 0.01, e: Array | None = None, seed: int = 0) -> Estimate:
    """Inverse-propensity weighting.

    ATE weights ``T/e`` and ``(1-T)/(1-e)``; ATT weights treated units by 1 and controls by
    ``e/(1-e)``.  ``normalize=True`` divides by the weight sums (Hajek estimator), which is
    far more stable when some propensities are small.  The standard error treats ``e`` as known
    (ignoring the estimation of the propensity), so prefer :func:`bootstrap` for intervals.
    """
    _check(estimand)
    X, t, y = _arrays(X, t, y)
    if e is None:
        e = propensity_scores(X, t, propensity_model, clip, seed=seed)
    if estimand == "ate":
        w1, w0 = t / e, (1 - t) / (1 - e)
    else:
        w1, w0 = t.astype(float), (1 - t) * e / (1 - e)
    if normalize:
        m1, m0 = (w1 * y).sum() / w1.sum(), (w0 * y).sum() / w0.sum()
        theta = m1 - m0
        n = len(y)
        psi = n * (w1 * (y - m1) / w1.sum() - w0 * (y - m0) / w0.sum())  # linearisation (propensity known)
    else:
        n = len(y)
        if estimand == "ate":
            psi = w1 * y - w0 * y
            theta = psi.mean()
        else:
            p = t.mean()
            psi = (w1 * y - w0 * y) / p
            theta = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    return Estimate(float(theta), float(se), f"IPW{' (Hajek)' if normalize else ''}", estimand)


# ----------------------------------------------------------------------------- AIPW
def aipw(X, t, y, outcome_model: "str | BaseEstimator" = "gbm",
         propensity_model: "str | BaseEstimator" = "gbm", estimand: str = "ate",
         n_folds: int = 5, clip: float = 0.01, seed: int = 0) -> Estimate:
    """Augmented IPW with cross-fitted nuisances (a.k.a. double ML with the AIPW score).

    ATE score:  ``psi = mu_1 - mu_0 + T (Y - mu_1)/e - (1-T)(Y - mu_0)/(1-e)``.
    ATT score:  ``psi = [T (Y - mu_0) - (1-T) e/(1-e) (Y - mu_0)] / P(T=1)``.
    The estimate is the sample mean of ``psi`` and the standard error ``sd(psi)/sqrt(n)``
    (the empirical efficient influence function).  Consistent if *either* nuisance is.
    """
    _check(estimand)
    X, t, y = _arrays(X, t, y)
    nu = cross_fit_nuisances(X, t, y, outcome_model, propensity_model, n_folds, seed, clip)
    if estimand == "ate":
        psi = nu.mu1 - nu.mu0 + t * (y - nu.mu1) / nu.e - (1 - t) * (y - nu.mu0) / (1 - nu.e)
        theta = psi.mean()
        infl = psi - theta
    else:
        p = t.mean()
        score = t * (y - nu.mu0) - (1 - t) * nu.e / (1 - nu.e) * (y - nu.mu0)
        theta = score.sum() / t.sum()
        infl = (score - t * theta) / p
    return Estimate(float(theta), float(infl.std(ddof=1) / np.sqrt(len(y))), "AIPW (cross-fit)", estimand)


# ----------------------------------------------------------------------------- DML
def dml_plr(X, t, y, outcome_model: "str | BaseEstimator" = "gbm",
            propensity_model: "str | BaseEstimator" = "gbm", n_folds: int = 5,
            clip: float = 0.01, n_repeats: int = 1, seed: int = 0) -> Estimate:
    """Double/debiased ML for the partially linear model ``Y = theta T + g(X) + eps`` (Robinson).

    Cross-fit ``m(x) = E[Y|x]`` and ``e(x) = E[T|x]``; with residuals ``\\tilde Y = Y - m``,
    ``\\tilde T = T - e`` the Neyman-orthogonal estimator is
    ``theta = sum(\\tilde T \\tilde Y) / sum(\\tilde T^2)`` with sandwich variance.  With a
    heterogeneous effect it estimates a variance-weighted average of ``tau(x)``.  Estimates are
    medians over ``n_repeats`` independent fold splits.
    """
    X, t, y = _arrays(X, t, y)
    thetas, ses = [], []
    for r in range(n_repeats):
        nu = cross_fit_nuisances(X, t, y, outcome_model, propensity_model, n_folds, seed + r,
                                 clip, marginal=True)
        rt, ry = t - nu.e, y - nu.m
        theta = (rt * ry).sum() / (rt**2).sum()
        eps = ry - theta * rt
        var = np.mean((rt * eps) ** 2) / np.mean(rt**2) ** 2 / len(y)
        thetas.append(theta)
        ses.append(np.sqrt(var))
    k = int(np.argsort(thetas)[len(thetas) // 2])
    return Estimate(float(thetas[k]), float(ses[k]), "DML (PLR)", "ate")


# ----------------------------------------------------------------------------- PSM
def match_indices(score: Array, t: Array, n_neighbors: int = 1, caliper: float | None = None,
                  source: int = 1) -> tuple[list[Array], Array]:
    """For every unit with ``t == source``, indices of the nearest units of the other arm in ``score``.

    Matching is with replacement.  Units whose nearest neighbour is farther than ``caliper`` get no
    match and are dropped.  Returns (list of match-index arrays, boolean mask over source units).
    """
    src = np.flatnonzero(t == source)
    pool = np.flatnonzero(t != source)
    dist = np.abs(score[src][:, None] - score[pool][None, :])
    k = min(n_neighbors, len(pool))
    nn = np.argpartition(dist, k - 1, axis=1)[:, :k]
    matches, keep = [], np.ones(len(src), dtype=bool)
    for i in range(len(src)):
        cand = nn[i]
        if caliper is not None:
            cand = cand[dist[i, cand] <= caliper]
        if len(cand) == 0:
            keep[i] = False
            matches.append(np.empty(0, dtype=int))
        else:
            matches.append(pool[cand])
    return matches, keep


def psm(X, t, y, propensity_model: "str | BaseEstimator" = "logistic", estimand: str = "att",
        n_neighbors: int = 1, caliper_sd: float | None = 0.25, clip: float = 1e-4,
        e: Array | None = None, seed: int = 0, return_weights: bool = False):
    """Propensity-score matching on the logit of the score, with replacement.

    ``caliper_sd`` is the caliper in standard deviations of the logit propensity (Rosenbaum &
    Rubin recommend 0.2-0.25).  ATT: each treated unit is compared with its matched controls.
    ATE: every unit is compared with matches from the opposite arm.  Standard errors are not
    provided (the bootstrap is not valid for matching with replacement; Abadie & Imbens, 2008),
    use :func:`bootstrap` only as a rough guide.
    """
    _check(estimand)
    X, t, y = _arrays(X, t, y)
    if e is None:
        e = propensity_scores(X, t, propensity_model, clip, seed=seed)
    logit = np.log(e / (1 - e))
    cal = None if caliper_sd is None else caliper_sd * logit.std()
    effects = []
    weights = np.zeros(len(y))  # how often each unit is used (for balance diagnostics)
    for src in ([1] if estimand == "att" else [1, 0]):
        matches, keep = match_indices(logit, t, n_neighbors, cal, source=src)
        idx = np.flatnonzero(t == src)
        for i, m in zip(idx[keep], [mm for mm, k in zip(matches, keep) if k]):
            diff = y[i] - y[m].mean()
            effects.append(diff if src == 1 else -diff)
            weights[i] += 1
            weights[m] += 1.0 / len(m)
    est = Estimate(float(np.mean(effects)), method="PSM", estimand=estimand)
    return (est, weights, int(len(effects))) if return_weights else est


# ------------------------------------------------------------------------ bootstrap
def bootstrap(estimator: Callable[..., float | Estimate], X, t, y, n_boot: int = 200,
              alpha: float = 0.05, seed: int = 0, **kw) -> dict[str, float]:
    """Nonparametric bootstrap of ``estimator(X, t, y, **kw)`` (resample units, refit everything).

    Returns the point estimate on the full sample, the bootstrap standard error and the
    percentile interval ``[lo, hi]`` at level ``1 - alpha``.  Resampling is stratified by arm so that
    every replicate contains treated and control units.
    """
    X, t, y = _arrays(X, t, y)
    rng = np.random.default_rng(seed)

    def val(*a, **k) -> float:
        r = estimator(*a, **k)
        return float(r.value if isinstance(r, Estimate) else r)

    point = val(X, t, y, **kw)
    i1, i0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        draws[b] = val(X[idx], t[idx], y[idx], **kw)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": point, "se": float(draws.std(ddof=1)), "lo": float(lo), "hi": float(hi),
            "draws": draws}
