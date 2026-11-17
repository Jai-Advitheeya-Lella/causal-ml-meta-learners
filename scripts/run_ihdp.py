"""IHDP benchmark (Hill 2011): sqrt(PEHE) and ATE error over the 100 NPCI replications.

For every replication the learners are fitted on the 672 training units; sqrt(PEHE) is reported
within-sample (the same 672 units) and out-of-sample (the 75 held-out units).  ``eps_ate`` is the
absolute error of the estimated average effect on the training units.

    python scripts/run_ihdp.py --reps 100 --workers 4 --out results/ihdp.jsonl
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


from cml import (aipw, ate_error, difference_in_means, dml_plr, ipw, load_ihdp, make_learner,
                 sqrt_pehe)
from cml.models import make_classifier, make_regressor

LEARNERS = ["S", "T", "X", "R", "DR"]
BASES = {"ridge": "logistic", "rf": "rf", "gbm": "gbm"}  # outcome model -> propensity model


def econml_baselines(tr, te, seed):
    """EconML reference implementations with the same random-forest base learners."""
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    from econml.metalearners import XLearner

    reg = lambda: make_regressor("rf", seed)  # noqa: E731
    clf = lambda: make_classifier("rf", seed)  # noqa: E731
    out = {}
    t0 = time.perf_counter()
    m = XLearner(models=reg(), propensity_model=clf(), cate_models=reg())
    m.fit(tr.y, tr.t, X=tr.X)
    out["econml-X"] = (m.effect(tr.X), m.effect(te.X), time.perf_counter() - t0)
    t0 = time.perf_counter()
    m = DRLearner(model_regression=reg(), model_propensity=clf(), model_final=reg(), cv=5,
                  random_state=seed)
    m.fit(tr.y, tr.t, X=tr.X)
    out["econml-DR"] = (m.effect(tr.X), m.effect(te.X), time.perf_counter() - t0)
    t0 = time.perf_counter()
    m = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True, n_estimators=200,
                        cv=5, random_state=seed)
    m.fit(tr.y, tr.t, X=tr.X)
    out["econml-CausalForest"] = (m.effect(tr.X), m.effect(te.X), time.perf_counter() - t0)
    return out


def run_rep(job: tuple[int, bool]) -> list[dict]:
    rep, use_econml = job
    warnings.filterwarnings("ignore")
    tr, te = load_ihdp(rep, "train"), load_ihdp(rep, "test")
    rows: list[dict] = []

    def add(name, base, tau_in, tau_out, t0):
        rows.append({"rep": rep, "method": name, "base": base,
                     "pehe_in": sqrt_pehe(tau_in, tr.tau), "pehe_out": sqrt_pehe(tau_out, te.tau),
                     "eps_ate": ate_error(tau_in, tr.tau), "time": time.perf_counter() - t0})

    for base, prop in BASES.items():
        for name in LEARNERS:
            t0 = time.perf_counter()
            m = make_learner(name, outcome_model=base, propensity_model=prop, seed=rep).fit(tr.X, tr.t, tr.y)
            add(name, base, m.predict(tr.X), m.predict(te.X), t0)
    if use_econml:
        for name, (a, b, secs) in econml_baselines(tr, te, rep).items():
            add(name, "rf", a, b, time.perf_counter() - secs)
    # ATE-only estimators (no CATE): error of the sample ATE
    true_ate = tr.ate
    for name, fn in {
        "naive": lambda: difference_in_means(tr.t, tr.y).value,
        "IPW": lambda: ipw(tr.X, tr.t, tr.y, "logistic").value,
        "AIPW-gbm": lambda: aipw(tr.X, tr.t, tr.y, "gbm", "gbm", seed=rep).value,
        "DML-PLR-gbm": lambda: dml_plr(tr.X, tr.t, tr.y, "gbm", "gbm", seed=rep).value,
    }.items():
        t0 = time.perf_counter()
        rows.append({"rep": rep, "method": name, "base": "-", "eps_ate": abs(fn() - true_ate),
                     "time": time.perf_counter() - t0})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-econml", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/ihdp.jsonl"))
    args = p.parse_args()
    load_ihdp(0, "train"), load_ihdp(0, "test")  # download once before forking
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():  # resume
        done = {json.loads(l)["rep"] for l in args.out.read_text().splitlines() if l}
    todo = [(r, not args.no_econml) for r in range(args.reps) if r not in done]
    t0 = time.time()
    with Pool(args.workers) as pool, open(args.out, "a") as f:
        for i, rows in enumerate(pool.imap_unordered(run_rep, todo)):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
