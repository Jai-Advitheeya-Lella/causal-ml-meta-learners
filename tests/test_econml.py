"""Agreement with EconML's reference X-learner (optional dependency, skipped on CI)."""

import os

import numpy as np
import pytest

econml = pytest.importorskip("econml")
pytestmark = pytest.mark.skipif(bool(os.environ.get("CI")), reason="optional heavy dependency")


def test_x_learner_matches_econml():
    from econml.metalearners import XLearner as EXLearner
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from cml import XLearner, SyntheticConfig, make_synthetic

    d = make_synthetic(SyntheticConfig(n=800, confounding=0.5, seed=3))
    reg = lambda: RandomForestRegressor(100, min_samples_leaf=5, random_state=0)  # noqa: E731
    clf = lambda: RandomForestClassifier(100, min_samples_leaf=10, random_state=0)  # noqa: E731
    ours = XLearner(outcome_model=reg(), propensity_model=clf(), effect_model=reg(), clip=1e-6, seed=0)
    ours.fit(d.X, d.t, d.y)
    theirs = EXLearner(models=reg(), propensity_model=clf(), cate_models=reg())
    theirs.fit(d.y, d.t, X=d.X)
    a, b = ours.predict(d.X), theirs.effect(d.X)
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert np.sqrt(np.mean((a - b) ** 2)) < 0.15
