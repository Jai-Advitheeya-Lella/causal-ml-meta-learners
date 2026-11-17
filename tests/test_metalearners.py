import numpy as np
import pytest

from cml import (DRLearner, RLearner, SLearner, TLearner, XLearner, SyntheticConfig, make_synthetic,
                 sqrt_pehe)
from cml.nuisance import cross_fit_nuisances


def linear_dgp(n=3000, seed=0, confounded=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(0.8 * X[:, 0]))) if confounded else np.full(n, 0.5)
    t = (rng.random(n) < e).astype(int)
    tau = 1.0 + 2.0 * X[:, 1]
    y = X[:, 0] + 0.5 * X[:, 2] + t * tau + 0.3 * rng.normal(size=n)
    return X, t, y, tau


@pytest.mark.parametrize("cls", [TLearner, XLearner, RLearner, DRLearner])
def test_learners_recover_linear_cate_with_ridge(cls):
    X, t, y, tau = linear_dgp()
    m = cls(outcome_model="ridge", propensity_model="logistic").fit(X, t, y)
    assert sqrt_pehe(m.predict(X), tau) < 0.25
    assert abs(m.ate() - tau.mean()) < 0.15
    assert abs(m.att() - tau[t == 1].mean()) < 0.25


def test_s_learner_with_a_linear_model_collapses_to_a_constant_effect():
    """The treatment enters additively, so tau_hat(x) is constant (the regularisation bias of S)."""
    X, t, y, tau = linear_dgp()
    m = SLearner(outcome_model="ridge").fit(X, t, y)
    tau_hat = m.predict(X)
    assert tau_hat.std() < 1e-9
    assert abs(tau_hat.mean() - tau.mean()) < 0.15  # but the ATE is still right


def test_t_learner_is_difference_of_arm_regressions():
    X, t, y, _ = linear_dgp(500)
    m = TLearner(outcome_model="ridge").fit(X, t, y)
    assert np.allclose(m.predict(X), m.mu1_.predict(X) - m.mu0_.predict(X))


def test_r_learner_minimises_the_r_loss():
    """The weighted pseudo-outcome regression must give the same tau as directly minimising
    sum_i ((y - m) - (t - e) * (a + b'x))^2 by least squares (ridge penalty -> 0)."""
    X, t, y, _ = linear_dgp(1500, seed=3)
    r = RLearner(outcome_model="ridge", propensity_model="logistic", effect_model="ridge", seed=0)
    r.effect_model = __import__("cml.models", fromlist=["Ridge"]).Ridge(alpha=1e-9)
    r.fit(X, t, y)
    nu = cross_fit_nuisances(X, t, y, "ridge", "logistic", 5, 0, 0.01, marginal=True)
    rt, ry = t - nu.e, y - nu.m
    D = np.column_stack([rt, rt[:, None] * X])
    beta = np.linalg.lstsq(D, ry, rcond=None)[0]
    tau_direct = beta[0] + X @ beta[1:]
    assert np.allclose(r.predict(X), tau_direct, atol=1e-4)


def test_dr_pseudo_outcome_is_unbiased_for_tau_with_known_nuisances():
    # With oracle mu_a and e, E[phi | x] = tau(x); check the sample mean over a large draw.
    d = make_synthetic(SyntheticConfig(n=60_000, confounding=0.7, seed=0))
    mu_t = np.where(d.t == 1, d.mu1, d.mu0)
    phi = d.mu1 - d.mu0 + (d.t - d.e) / (d.e * (1 - d.e)) * (d.y - mu_t)
    assert abs(phi.mean() - d.tau.mean()) < 0.03


def test_dr_and_r_beat_naive_on_confounded_synthetic():
    d = make_synthetic(SyntheticConfig(n=1500, confounding=0.9, overlap_scale=1.0, seed=2))
    naive = d.y[d.t == 1].mean() - d.y[d.t == 0].mean()
    assert abs(naive - d.tau.mean()) > 0.3  # confounding really biases the naive contrast
    for cls in (RLearner, DRLearner, XLearner):
        m = cls(outcome_model="rf", propensity_model="logistic").fit(d.X, d.t, d.y)
        assert abs(m.ate() - d.tau.mean()) < abs(naive - d.tau.mean())
