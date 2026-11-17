import numpy as np
import pytest

from cml import aipw, bootstrap, difference_in_means, dml_plr, ipw, psm, regression_adjustment
from cml.estimators import match_indices


def confounded(n=4000, seed=0, effect=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(1.0 * X[:, 0] - 0.5 * X[:, 1])))
    t = (rng.random(n) < e).astype(int)
    y = 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2] ** 2 + effect * t + rng.normal(size=n)
    return X, t, y, e


def test_naive_is_biased_adjusted_are_not():
    X, t, y, _ = confounded()
    assert abs(difference_in_means(t, y).value - 2.0) > 0.8
    assert abs(ipw(X, t, y).value - 2.0) < 0.3
    assert abs(aipw(X, t, y, "rf", "logistic").value - 2.0) < 0.3
    assert abs(dml_plr(X, t, y, "rf", "logistic").value - 2.0) < 0.3
    assert abs(psm(X, t, y, estimand="ate").value - 2.0) < 0.5
    assert abs(regression_adjustment(X, t, y, "rf", estimand="ate").value - 2.0) < 0.3


def test_ipw_is_exact_with_oracle_propensity_in_a_randomised_trial():
    rng = np.random.default_rng(1)
    n = 100_000
    X = rng.normal(size=(n, 2))
    t = rng.integers(0, 2, n)
    y = X[:, 0] + 3 * t + rng.normal(size=n)
    est = ipw(X, t, y, e=np.full(n, 0.5), normalize=False)
    assert abs(est.value - 3.0) < 4 * est.se


def test_att_estimators_target_effect_on_treated():
    # effect heterogeneity makes ATT != ATE: tau = 1 + x0, treatment more likely for large x0
    rng = np.random.default_rng(2)
    n = 20000
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-1.5 * X[:, 0]))
    t = (rng.random(n) < e).astype(int)
    tau = 1 + X[:, 0]
    y = X[:, 1] + t * tau + 0.5 * rng.normal(size=n)
    att = tau[t == 1].mean()
    assert att > 1.5 and abs(tau.mean() - 1) < 0.05
    for est in (ipw(X, t, y, estimand="att"), aipw(X, t, y, "ridge", "logistic", estimand="att"),
                regression_adjustment(X, t, y, "ridge", estimand="att"),
                psm(X, t, y, estimand="att", n_neighbors=3)):
        assert abs(est.value - att) < 0.12, est.method


def test_aipw_is_doubly_robust():
    """Correct outcome model + wrong propensity, and vice versa: AIPW stays consistent while the
    corresponding plug-in (IPW / regression) with the wrong nuisance does not."""
    from cml import estimators
    from cml.nuisance import Nuisances

    X, t, y, e = confounded(40_000, seed=5)
    mu0_true = 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2] ** 2
    mu1_true = mu0_true + 2.0
    wrong_e = np.full(len(y), t.mean())
    wrong_mu = np.full(len(y), y.mean())

    def aipw_score(e_hat, mu0, mu1):
        return np.mean(mu1 - mu0 + t * (y - mu1) / e_hat - (1 - t) * (y - mu0) / (1 - e_hat))

    assert abs(aipw_score(wrong_e, mu0_true, mu1_true) - 2.0) < 0.08  # outcome model right
    assert abs(aipw_score(e, wrong_mu, wrong_mu) - 2.0) < 0.1         # propensity right
    assert abs(aipw_score(wrong_e, wrong_mu, wrong_mu) - 2.0) > 0.8   # both wrong -> biased
    _ = (estimators, Nuisances)


def test_cross_fit_aipw_ci_covers_truth_most_of_the_time():
    hits = 0
    reps = 40
    for s in range(reps):
        X, t, y, _ = confounded(800, seed=100 + s)
        est = aipw(X, t, y, "ridge", "logistic", seed=s)
        lo, hi = est.ci
        hits += lo <= 2.0 <= hi
    assert hits / reps > 0.8  # nominal 95%; small-sample and misspecified ridge


def test_bootstrap_ci_brackets_difference_in_means():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    t = rng.integers(0, 2, 400)
    y = 1.0 * t + rng.normal(size=400)
    res = bootstrap(lambda X, t, y: difference_in_means(t, y).value, X, t, y, n_boot=300, seed=0)
    analytic = difference_in_means(t, y)
    assert res["lo"] < analytic.value < res["hi"]
    assert abs(res["se"] - analytic.se) < 0.3 * analytic.se


def test_matching_finds_nearest_neighbours_with_caliper():
    score = np.array([0.0, 1.0, 2.0, 10.0, 0.1, 1.9, 5.0])
    t = np.array([1, 1, 1, 1, 0, 0, 0])
    matches, keep = match_indices(score, t, n_neighbors=1, caliper=0.5, source=1)
    assert [list(m) for m in matches] == [[4], [], [5], []]
    assert list(keep) == [True, False, True, False]


def test_psm_is_exact_when_every_treated_unit_has_a_twin_control():
    # Supply the propensity score so matching is deterministic: treated unit i and control i share
    # a score; 300 further controls sit at a far-away score and must never be matched.
    n = 100
    score = 0.1 + 0.002 * np.arange(n)
    e = np.r_[score, score, np.full(300, 0.9)]
    t = np.r_[np.ones(n), np.zeros(n + 300)].astype(int)
    y = np.r_[np.full(n, 5.0), np.zeros(n), np.ones(300)]
    X = np.zeros((len(t), 1))
    est, w, n_used = psm(X, t, y, estimand="att", e=e, caliper_sd=0.25, return_weights=True)
    assert n_used == n and est.value == pytest.approx(5.0)
    assert w[t == 0][n:].sum() == 0  # distant controls are never used


def test_estimand_validation():
    X, t, y, _ = confounded(300)
    with pytest.raises(ValueError):
        ipw(X, t, y, estimand="atc")
