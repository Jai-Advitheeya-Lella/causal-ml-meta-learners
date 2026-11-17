import numpy as np

from cml.data import SyntheticConfig, make_synthetic
from cml.data.synthetic import mu0_fn, propensity_fn, tau_fn
from cml.diagnostics import effective_sample_size, overlap_summary, standardized_mean_difference


def test_synthetic_ground_truth_is_consistent():
    d = make_synthetic(SyntheticConfig(n=500, seed=1))
    assert np.allclose(d.mu1 - d.mu0, d.tau)
    assert np.allclose(d.tau, tau_fn(d.X)) and np.allclose(d.mu0, mu0_fn(d.X))
    # observed outcome = potential outcome of the realised arm + noise
    resid = d.y - np.where(d.t == 1, d.mu1, d.mu0)
    assert abs(resid.mean()) < 0.15 and abs(resid.std() - 1.0) < 0.15


def test_synthetic_ate_is_two():
    d = make_synthetic(SyntheticConfig(n=200_000, seed=0))
    assert abs(d.tau.mean() - 2.0) < 0.02


def test_confounding_zero_gives_independent_treatment_and_prognosis():
    d = make_synthetic(SyntheticConfig(n=20_000, confounding=0.0, overlap_scale=1.0, seed=0))
    assert abs(np.corrcoef(d.e, d.mu0)[0, 1]) < 0.03
    d2 = make_synthetic(SyntheticConfig(n=20_000, confounding=0.9, seed=0))
    assert np.corrcoef(d2.e, d2.mu0)[0, 1] > 0.5


def test_overlap_scale_controls_extremity_of_propensities():
    X = np.random.default_rng(0).uniform(-1, 1, size=(5000, 10))
    frac = [np.mean((p < 0.05) | (p > 0.95))
            for p in (propensity_fn(X, 0.7, s) for s in (0.5, 1.5, 3.0))]
    assert frac[0] < frac[1] < frac[2]
    assert frac[0] < 0.01 and frac[2] > 0.3


def test_smd_and_ess():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 4000)
    X = rng.normal(size=(4000, 2))
    X[:, 0] += 0.5 * t
    smd = standardized_mean_difference(X, t)
    assert 0.4 < smd[0] < 0.6 and abs(smd[1]) < 0.1
    assert effective_sample_size(np.ones(100)) == 100
    assert effective_sample_size(np.r_[np.ones(99), 1000.0]) < 5
    s = overlap_summary(np.full(10, 0.5), np.array([0, 1] * 5))
    assert s["frac_extreme"] == 0.0
