"""Meta-learners for heterogeneous treatment effects (CATE).

With potential outcomes ``Y(0), Y(1)`` and unconfoundedness, the conditional average
treatment effect is ``tau(x) = E[Y(1) - Y(0) | X = x]``.  Every learner below turns
arbitrary supervised regressors into an estimate of ``tau``:

* **S-learner**  -- one model ``mu(x, t)``; ``tau(x) = mu(x, 1) - mu(x, 0)``.
* **T-learner**  -- two models ``mu_0, mu_1``; ``tau = mu_1 - mu_0``.
* **X-learner**  (Kunzel et al., 2019) -- impute individual effects with the *other* arm's
  model, regress them on ``x`` per arm and blend with the propensity score.
* **R-learner**  (Nie & Wager, 2021) -- minimise the residual-on-residual (Robinson) loss.
* **DR-learner** (Kennedy, 2020) -- regress the doubly robust (AIPW) pseudo-outcome on ``x``.

All learners share ``fit(X, t, y)``, ``predict(X)`` (CATE), ``ate()`` and ``att()``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from .nuisance import as_classifier, as_regressor, cross_fit_nuisances

Array = NDArray[np.float64]


class MetaLearner:
    """Common interface; subclasses implement ``_fit`` and ``predict``."""

    name = "base"

    def __init__(
        self,
        outcome_model: "str | BaseEstimator" = "gbm",
        propensity_model: "str | BaseEstimator" = "gbm",
        effect_model: "str | BaseEstimator | None" = None,
        n_folds: int = 5,
        clip: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.effect_model = effect_model if effect_model is not None else outcome_model
        self.n_folds = n_folds
        self.clip = clip
        self.seed = seed

    def fit(self, X: ArrayLike, t: ArrayLike, y: ArrayLike) -> "MetaLearner":
        self.X_ = np.asarray(X, dtype=float)
        self.t_ = np.asarray(t, dtype=int).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self._fit(self.X_, self.t_, y)
        return self

    def _fit(self, X: Array, t: Array, y: Array) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: ArrayLike) -> Array:  # pragma: no cover
        raise NotImplementedError

    def ate(self) -> float:
        """Average of the estimated CATE over the training sample."""
        return float(self.predict(self.X_).mean())

    def att(self) -> float:
        """Average of the estimated CATE over the treated units of the training sample."""
        return float(self.predict(self.X_[self.t_ == 1]).mean())

    # shared helper: fit a regressor and keep it
    def _reg(self, model: "str | BaseEstimator", X: Array, y: Array, w: Array | None = None):
        m = as_regressor(model, self.seed)
        m.fit(X, y) if w is None else m.fit(X, y, sample_weight=w)
        return m


class SLearner(MetaLearner):
    """Single model on ``[x, t]``; the treatment is just another feature."""

    name = "S"

    def _fit(self, X: Array, t: Array, y: Array) -> None:
        self.model_ = self._reg(self.outcome_model, np.column_stack([X, t]), y)

    def predict(self, X: ArrayLike) -> Array:
        X = np.asarray(X, dtype=float)
        one, zero = np.ones((len(X), 1)), np.zeros((len(X), 1))
        return self.model_.predict(np.hstack([X, one])) - self.model_.predict(np.hstack([X, zero]))


class TLearner(MetaLearner):
    """Separate outcome regressions for control and treated units."""

    name = "T"

    def _fit(self, X: Array, t: Array, y: Array) -> None:
        self.mu0_ = self._reg(self.outcome_model, X[t == 0], y[t == 0])
        self.mu1_ = self._reg(self.outcome_model, X[t == 1], y[t == 1])

    def predict(self, X: ArrayLike) -> Array:
        X = np.asarray(X, dtype=float)
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class XLearner(MetaLearner):
    """Kunzel et al. (2019).  Stage 1: ``mu_0, mu_1``.  Stage 2: imputed effects

    ``D1_i = Y_i - mu_0(x_i)`` (treated), ``D0_i = mu_1(x_i) - Y_i`` (controls) are regressed on
    ``x`` giving ``tau_1, tau_0``.  Final: ``tau(x) = g(x) tau_0(x) + (1 - g(x)) tau_1(x)`` with
    ``g`` the propensity score, so the estimate leans on the arm that is better covered at ``x``.
    """

    name = "X"

    def _fit(self, X: Array, t: Array, y: Array) -> None:
        mu0 = self._reg(self.outcome_model, X[t == 0], y[t == 0])
        mu1 = self._reg(self.outcome_model, X[t == 1], y[t == 1])
        d1 = y[t == 1] - mu0.predict(X[t == 1])
        d0 = mu1.predict(X[t == 0]) - y[t == 0]
        self.tau1_ = self._reg(self.effect_model, X[t == 1], d1)
        self.tau0_ = self._reg(self.effect_model, X[t == 0], d0)
        prop = as_classifier(self.propensity_model, self.seed).fit(X, t)
        self.prop_ = prop

    def predict(self, X: ArrayLike) -> Array:
        X = np.asarray(X, dtype=float)
        g = np.clip(self.prop_.predict_proba(X)[:, 1], self.clip, 1 - self.clip)
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)


class RLearner(MetaLearner):
    """Nie & Wager (2021).  With cross-fitted ``m(x) = E[Y|x]`` and ``e(x)``, minimise

    ``sum_i ((Y_i - m(x_i)) - (T_i - e(x_i)) tau(x_i))^2``.

    This is a weighted regression of the pseudo-outcome ``(Y - m)/(T - e)`` on ``x`` with weights
    ``(T - e)^2``; any regressor that accepts ``sample_weight`` can serve as ``tau`` model.
    """

    name = "R"

    def _fit(self, X: Array, t: Array, y: Array) -> None:
        nu = cross_fit_nuisances(X, t, y, self.outcome_model, self.propensity_model,
                                 self.n_folds, self.seed, self.clip, marginal=True)
        resid_t = t - nu.e
        self.pseudo_ = (y - nu.m) / resid_t
        self.weights_ = resid_t**2
        self.model_ = self._reg(self.effect_model, X, self.pseudo_, self.weights_)

    def predict(self, X: ArrayLike) -> Array:
        return self.model_.predict(np.asarray(X, dtype=float))


class DRLearner(MetaLearner):
    """Kennedy (2020).  Regress the doubly robust pseudo-outcome

    ``phi = mu_1 - mu_0 + (T - e)/(e (1 - e)) (Y - mu_T)``

    on ``x``.  ``E[phi | x] = tau(x)`` if either the outcome models or the propensity are correct,
    and the second-order bias is the *product* of the two nuisance errors.
    """

    name = "DR"

    def _fit(self, X: Array, t: Array, y: Array) -> None:
        nu = cross_fit_nuisances(X, t, y, self.outcome_model, self.propensity_model,
                                 self.n_folds, self.seed, self.clip)
        mu_t = np.where(t == 1, nu.mu1, nu.mu0)
        self.pseudo_ = nu.mu1 - nu.mu0 + (t - nu.e) / (nu.e * (1 - nu.e)) * (y - mu_t)
        self.model_ = self._reg(self.effect_model, X, self.pseudo_)

    def predict(self, X: ArrayLike) -> Array:
        return self.model_.predict(np.asarray(X, dtype=float))


LEARNERS: dict[str, Callable[..., MetaLearner]] = {
    "S": SLearner, "T": TLearner, "X": XLearner, "R": RLearner, "DR": DRLearner,
}


def make_learner(name: str, **kw) -> MetaLearner:
    return LEARNERS[name](**kw)
