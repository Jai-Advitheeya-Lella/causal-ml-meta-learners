"""Base learners (nuisance and effect models) used by the causal estimators.

Two small estimators are implemented from scratch with NumPy -- ridge regression
(closed form) and L2-regularised logistic regression (Newton / IRLS) -- so that the
whole pipeline can run on a purely hand-written stack.  Tree ensembles and the
lasso are taken from scikit-learn and are used as interchangeable *base learners*;
every estimator in this package only needs ``fit(X, y[, sample_weight])`` and
``predict`` (``predict_proba`` for classifiers).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

Array = NDArray[np.float64]


class Ridge(BaseEstimator, RegressorMixin):
    """Weighted ridge regression with an unpenalised intercept and standardised inputs.

    Solves ``min_b  sum_i w_i (y_i - b0 - z_i' b)^2 + alpha * ||b||^2`` where ``z`` are
    the standardised features, via the normal equations.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> "Ridge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        self.mean_ = np.average(X, axis=0, weights=w)
        self.scale_ = np.sqrt(np.average((X - self.mean_) ** 2, axis=0, weights=w))
        self.scale_[self.scale_ < 1e-12] = 1.0
        Z = (X - self.mean_) / self.scale_
        ybar = np.average(y, weights=w)
        A = Z.T @ (Z * w[:, None]) + self.alpha * np.eye(Z.shape[1])
        b = Z.T @ (w * (y - ybar))
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = ybar
        return self

    def predict(self, X: ArrayLike) -> Array:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return self.intercept_ + Z @ self.coef_


class Logistic(BaseEstimator, ClassifierMixin):
    """L2-regularised logistic regression fitted by Newton's method (IRLS).

    Minimises ``-sum_i w_i log p(y_i | z_i) + (alpha / 2) ||b||^2`` on standardised
    features (the intercept is not penalised).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> "Logistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.array([0, 1])
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        Z = np.hstack([np.ones((len(y), 1)), (X - self.mean_) / self.scale_])
        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        beta = np.zeros(Z.shape[1])
        beta[0] = np.log((np.average(y, weights=w) + 1e-6) / (1 - np.average(y, weights=w) + 1e-6))
        prev = np.inf
        for _ in range(self.max_iter):
            eta = Z @ beta
            p = 1.0 / (1.0 + np.exp(-eta))
            nll = -np.sum(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            nll += 0.5 * beta @ reg @ beta
            grad = Z.T @ (w * (p - y)) + reg @ beta
            H = Z.T @ (Z * (w * p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, grad)
            # backtracking line search keeps Newton stable under (near-)separation
            t = 1.0
            while t > 1e-6:
                b_new = beta - t * step
                p_new = 1.0 / (1.0 + np.exp(-(Z @ b_new)))
                nll_new = -np.sum(w * (y * np.log(p_new + 1e-12) + (1 - y) * np.log(1 - p_new + 1e-12)))
                nll_new += 0.5 * b_new @ reg @ b_new
                if nll_new <= nll + 1e-12:
                    break
                t *= 0.5
            beta = b_new
            if abs(prev - nll_new) < self.tol * (1 + abs(nll_new)):
                break
            prev = nll_new
        self.beta_ = beta
        return self

    def decision_function(self, X: ArrayLike) -> Array:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return self.beta_[0] + Z @ self.beta_[1:]

    def predict_proba(self, X: ArrayLike) -> Array:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X: ArrayLike) -> Array:
        return (self.decision_function(X) > 0).astype(int)


# --------------------------------------------------------------------------- factories
REGRESSORS = ("ridge", "lasso", "rf", "gbm")
CLASSIFIERS = ("logistic", "rf", "gbm")


def make_regressor(name: str, seed: int = 0, n_jobs: int = 1) -> BaseEstimator:
    """Fresh regression base learner by name: ``ridge``, ``lasso``, ``rf`` or ``gbm``."""
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "lasso":
        from sklearn.linear_model import LassoCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return _Pipe(make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=seed, n_jobs=n_jobs)))
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5,
                                     random_state=seed, n_jobs=n_jobs)
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=10, l2_regularization=1.0,
                                             random_state=seed)
    raise ValueError(f"unknown regressor {name!r}; choose from {REGRESSORS}")


def make_classifier(name: str, seed: int = 0, n_jobs: int = 1) -> BaseEstimator:
    """Fresh classification base learner by name: ``logistic``, ``rf`` or ``gbm``."""
    if name == "logistic":
        return Logistic(alpha=1.0)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_features=0.5,
                                      random_state=seed, n_jobs=n_jobs)
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=seed)
    raise ValueError(f"unknown classifier {name!r}; choose from {CLASSIFIERS}")


class _Pipe(BaseEstimator, RegressorMixin):
    """Pipeline wrapper that forwards ``sample_weight`` to the final step."""

    def __init__(self, pipe) -> None:
        self.pipe = pipe

    def fit(self, X, y, sample_weight=None):
        from sklearn.base import clone

        self.pipe_ = clone(self.pipe)
        if sample_weight is None:
            self.pipe_.fit(X, y)
        else:
            last = self.pipe_.steps[-1][0]
            self.pipe_.fit(X, y, **{f"{last}__sample_weight": sample_weight})
        return self

    def predict(self, X):
        return self.pipe_.predict(X)
