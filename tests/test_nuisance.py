import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from cml.nuisance import cross_fit_nuisances, make_folds


def test_folds_partition_and_stratify():
    t = np.r_[np.ones(23), np.zeros(77)].astype(int)
    fold = make_folds(t, 5, np.random.default_rng(0))
    assert set(fold) == set(range(5))
    for k in range(5):
        assert 4 <= (t[fold == k] == 1).sum() <= 5  # 23 treated over 5 folds


class _Memoriser(BaseEstimator, RegressorMixin):
    """Predicts the training label if the point was seen in training, else the training mean."""

    def fit(self, X, y, sample_weight=None):
        self.table_ = {float(x[0]): float(v) for x, v in zip(X, y)}
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.table_.get(float(x[0]), self.mean_) for x in X])


def test_cross_fitting_never_uses_own_observation():
    rng = np.random.default_rng(0)
    n = 200
    X = np.arange(n, dtype=float)[:, None]  # unique ids
    t = rng.integers(0, 2, n)
    y = rng.normal(size=n) + 100.0 * t
    nu = cross_fit_nuisances(X, t, y, outcome_model=_Memoriser(), propensity_model="logistic",
                             n_folds=4)
    # a memorising model would return y exactly if it saw the point; out-of-fold it must not
    mu_t = np.where(t == 1, nu.mu1, nu.mu0)
    assert not np.any(np.isclose(mu_t, y))
    assert np.all((nu.e >= 0.01) & (nu.e <= 0.99))
