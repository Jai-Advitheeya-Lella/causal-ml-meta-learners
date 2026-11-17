import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.linear_model import Ridge as SkRidge

from cml.models import Logistic, Ridge, make_classifier, make_regressor


def test_ridge_matches_sklearn_on_standardised_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * [1, 3, 0.2, 10, 1]
    y = X @ rng.normal(size=5) + 0.3 * rng.normal(size=200)
    ours = Ridge(alpha=2.0).fit(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    ref = SkRidge(alpha=2.0).fit(Z, y)
    assert np.allclose(ours.predict(X), ref.predict(Z), atol=1e-8)


def test_ridge_sample_weights_equal_duplication():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(60, 3)), rng.normal(size=60)
    w = rng.integers(1, 4, size=60)
    a = Ridge(0.5).fit(X, y, sample_weight=w)
    # weights are normalised to mean one, so the equivalent penalty on the duplicated data is alpha * mean(w)
    b = Ridge(0.5 * w.mean()).fit(np.repeat(X, w, axis=0), np.repeat(y, w))
    assert np.allclose(a.predict(X), b.predict(X), atol=1e-8)


def test_logistic_matches_sklearn():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4))
    p = 1 / (1 + np.exp(-(X @ [1.0, -2.0, 0.5, 0.0] + 0.3)))
    y = (rng.random(500) < p).astype(int)
    ours = Logistic(alpha=1.0).fit(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    ref = SkLogistic(C=1.0, tol=1e-10, max_iter=1000).fit(Z, y)  # C=1/alpha, sum-loss objective
    assert np.allclose(ours.predict_proba(X)[:, 1], ref.predict_proba(Z)[:, 1], atol=1e-5)


def test_logistic_survives_separable_data():
    X = np.linspace(-1, 1, 40)[:, None]
    y = (X[:, 0] > 0).astype(int)
    p = Logistic(alpha=1.0).fit(X, y).predict_proba(X)[:, 1]
    assert np.all(np.isfinite(p)) and p[0] < 0.2 and p[-1] > 0.8


@pytest.mark.parametrize("name", ["ridge", "lasso", "rf", "gbm"])
def test_regressor_factory_accepts_sample_weight(name):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] + 0.1 * rng.normal(size=120)
    m = make_regressor(name).fit(X, y, sample_weight=np.ones(120))
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.9


@pytest.mark.parametrize("name", ["logistic", "rf", "gbm"])
def test_classifier_factory(name):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    p = make_classifier(name).fit(X, y).predict_proba(X)
    assert p.shape == (300, 2) and np.allclose(p.sum(1), 1)
    assert ((p[:, 1] > 0.5) == y).mean() > 0.8
