"""Loaders for IHDP and LaLonde (need network access: skipped on CI)."""

import os

import numpy as np
import pytest

from cml import aipw, difference_in_means, load_ihdp, load_lalonde
from cml.data.lalonde import design_matrix

pytestmark = pytest.mark.skipif(bool(os.environ.get("CI")), reason="downloads data")


def test_ihdp_shapes_and_ground_truth():
    tr, te = load_ihdp(0, "train"), load_ihdp(0, "test")
    assert tr.X.shape == (672, 25) and te.X.shape == (75, 25)
    assert np.allclose(tr.tau, tr.mu1 - tr.mu0)
    assert 3.0 < tr.ate < 5.0  # response surface B is calibrated to an ATE of about 4
    # factual outcome equals the potential outcome of the realised arm plus noise
    resid = tr.y - np.where(tr.t == 1, tr.mu1, tr.mu0)
    assert abs(resid.mean()) < 0.2 and 0.5 < resid.std() < 1.5
    # confounded by design: treated units are a non-random subset
    assert 0.1 < tr.t.mean() < 0.25


def test_lalonde_matches_published_summary_statistics():
    d = load_lalonde()
    nsw, psid = d["nsw"], d["psid"]
    assert (nsw.treat.sum(), (1 - nsw.treat).sum()) == (185, 260)
    assert (psid.treat.sum(), (1 - psid.treat).sum()) == (185, 2490)
    est = difference_in_means(nsw.treat.to_numpy(int), nsw.re78.to_numpy())
    assert abs(est.value - 1794.34) < 0.1  # Dehejia & Wahba (1999), Table 3
    naive = difference_in_means(psid.treat.to_numpy(int), psid.re78.to_numpy())
    assert abs(naive.value - (-15204.78)) < 1.0  # the classic "PSID-1" bias
    assert design_matrix(psid).shape == (2675, 16)


def test_aipw_moves_psid_estimate_towards_the_experiment():
    psid = load_lalonde()["psid"]
    X, t, y = design_matrix(psid), psid.treat.to_numpy(int), psid.re78.to_numpy()
    est = aipw(X, t, y, "ridge", "logistic", estimand="att", clip=1e-3)
    assert abs(est.value - 1794) < 4000  # naive is off by 17,000
