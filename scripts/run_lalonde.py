"""LaLonde / Dehejia-Wahba: can observational estimators recover the experimental benchmark?

The NSW randomised experiment gives the benchmark effect on the treated (difference in mean 1978
earnings).  We then replace the 260 experimental controls with 2,490 PSID-1 controls and run
every estimator on that observational sample, with covariates age, education, race, marital and
degree status, 1974/1975 earnings and the usual Dehejia-Wahba squares / zero-earnings dummies.
All estimands are the ATT; intervals are 95% percentile bootstrap intervals (B resamples,
stratified by arm, with every nuisance model refitted on each resample).

    python scripts/run_lalonde.py --boot 200 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cml import (aipw, bootstrap, difference_in_means, dml_plr, ipw, make_learner,
                 propensity_scores, psm, regression_adjustment)
from cml.data.lalonde import design_matrix, load_lalonde
from cml.diagnostics import effective_sample_size, overlap_summary, standardized_mean_difference
from cml.models import Ridge


def _ols():
    return Ridge(alpha=1e-8)


ESTIMATORS = {
    # name: (callable(X, t, y) -> float, family)
    "Difference in means": (lambda X, t, y: difference_in_means(t, y).value, "naive"),
    "OLS regression adjustment": (lambda X, t, y: regression_adjustment(X, t, y, _ols(), "att").value, "outcome"),
    "GBM regression adjustment": (lambda X, t, y: regression_adjustment(X, t, y, "gbm", "att").value, "outcome"),
    "IPW (logistic)": (lambda X, t, y: ipw(X, t, y, "logistic", "att", clip=1e-4).value, "propensity"),
    "PSM (1-NN, caliper)": (lambda X, t, y: psm(X, t, y, "logistic", "att", 1, 0.25).value, "propensity"),
    "PSM (5-NN, caliper)": (lambda X, t, y: psm(X, t, y, "logistic", "att", 5, 0.25).value, "propensity"),
    "AIPW (ridge + logistic)": (lambda X, t, y: aipw(X, t, y, _ols(), "logistic", "att", clip=1e-3).value, "doubly robust"),
    "AIPW (GBM + GBM)": (lambda X, t, y: aipw(X, t, y, "gbm", "gbm", "att", clip=1e-3).value, "doubly robust"),
    "T-learner (GBM)": (lambda X, t, y: make_learner("T", outcome_model="gbm").fit(X, t, y).att(), "meta-learner"),
    "X-learner (GBM)": (lambda X, t, y: make_learner("X", outcome_model="gbm", propensity_model="gbm").fit(X, t, y).att(), "meta-learner"),
    "R-learner (GBM)": (lambda X, t, y: make_learner("R", outcome_model="gbm", propensity_model="gbm").fit(X, t, y).att(), "meta-learner"),
    "DR-learner (GBM)": (lambda X, t, y: make_learner("DR", outcome_model="gbm", propensity_model="gbm").fit(X, t, y).att(), "meta-learner"),
    "DML, partially linear (GBM)": (lambda X, t, y: dml_plr(X, t, y, "gbm", "gbm").value, "doubly robust"),
}

_DATA: dict = {}


def _init(data_dir: str) -> None:
    warnings.filterwarnings("ignore")
    d = load_lalonde(data_dir)
    _DATA["psid"] = (design_matrix(d["psid"]), d["psid"].treat.to_numpy(int), d["psid"].re78.to_numpy(float))


def run(job: tuple[str, int]) -> dict:
    name, n_boot = job
    X, t, y = _DATA["psid"]
    fn, family = ESTIMATORS[name]
    t0 = time.perf_counter()
    res = bootstrap(fn, X, t, y, n_boot=n_boot, seed=0)
    return {"name": name, "family": family, "value": res["value"], "se": res["se"], "lo": res["lo"],
            "hi": res["hi"], "time": time.perf_counter() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", default="data")
    p.add_argument("--out", type=Path, default=Path("results/lalonde.json"))
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    d = load_lalonde(args.data)
    nsw, psid = d["nsw"], d["psid"]

    # experimental benchmark
    ex = difference_in_means(nsw.treat.to_numpy(int), nsw.re78.to_numpy(float))
    Xn = design_matrix(nsw)
    adj = bootstrap(lambda X, t, y: regression_adjustment(X, t, y, _ols(), "att").value, Xn,
                    nsw.treat.to_numpy(int), nsw.re78.to_numpy(float), n_boot=args.boot, seed=0)
    benchmark = {"diff_in_means": ex.value, "se": ex.se, "ci": list(ex.ci), "ols_adjusted": adj["value"],
                 "ols_adjusted_ci": [adj["lo"], adj["hi"]], "n_treated": int(nsw.treat.sum()),
                 "n_control": int((1 - nsw.treat).sum())}
    print("benchmark", benchmark, flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run, [(k, args.boot) for k in ESTIMATORS])):
            results.append(r)
            print(f"[{i + 1}/{len(ESTIMATORS)}] {r['name']}: {r['value']:.0f} [{r['lo']:.0f}, {r['hi']:.0f}] "
                  f"({r['time']:.0f}s)", flush=True)
    order = {k: i for i, k in enumerate(ESTIMATORS)}
    results.sort(key=lambda r: order[r["name"]])

    # diagnostics on the observational sample
    X, t, y = design_matrix(psid), psid.treat.to_numpy(int), psid.re78.to_numpy(float)
    X_raw = design_matrix(psid, expand=False)
    e = propensity_scores(X, t, "logistic", clip=1e-4)
    w_ipw = np.where(t == 1, 1.0, e / (1 - e))
    _, w_psm, n_used = psm(X, t, y, "logistic", "att", 1, 0.25, return_weights=True)
    diag = {
        "overlap": overlap_summary(e, t),
        "n_treated": int(t.sum()), "n_control": int((1 - t).sum()),
        "ess_control_ipw": effective_sample_size(w_ipw[t == 0]),
        "psm_matched_treated": n_used,
        "psm_distinct_controls": int(((w_psm > 0) & (t == 0)).sum()),
        "smd_columns": ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75"],
        "smd_unadjusted": standardized_mean_difference(X_raw, t).tolist(),
        "smd_ipw": standardized_mean_difference(X_raw, t, w_ipw).tolist(),
        "smd_psm": standardized_mean_difference(X_raw, t, w_psm).tolist(),
        "nsw_smd": standardized_mean_difference(design_matrix(nsw, expand=False),
                                                nsw.treat.to_numpy(int)).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"benchmark": benchmark, "estimates": results, "diagnostics": diag,
                                    "boot": args.boot, "wall_time_s": time.time() - t0}, indent=1))
    np.savez_compressed(args.out.with_name("lalonde_scores.npz"), e=e, t=t)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
