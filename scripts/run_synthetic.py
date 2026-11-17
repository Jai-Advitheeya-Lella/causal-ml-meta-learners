"""Synthetic study with known CATE: vary confounding strength and overlap.

For every (confounding, overlap_scale) cell and seed we simulate ``n`` units, fit the five
meta-learners with gradient-boosting base learners, and also run the ATE estimators.  We record
sqrt(PEHE) and the signed error of the average effect (estimate minus the *sample* ATE, so that
bias and variance over seeds are directly comparable).

    python scripts/run_synthetic.py --seeds 25 --n 1500 --workers 4
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cml import (SyntheticConfig, aipw, difference_in_means, dml_plr, ipw, make_learner,
                 make_synthetic, psm, regression_adjustment, sqrt_pehe)
from cml.diagnostics import overlap_summary

CONFOUNDING = [0.0, 0.5, 0.9]
OVERLAP = [0.5, 1.5, 3.0]
LEARNERS = ["S", "T", "X", "R", "DR"]


def run_cell(job: tuple[float, float, int, int, str]) -> tuple[list[dict], dict]:
    rho, s, seed, n, base = job
    warnings.filterwarnings("ignore")
    d = make_synthetic(SyntheticConfig(n=n, confounding=rho, overlap_scale=s, seed=seed))
    true_ate = d.tau.mean()
    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {"tau": d.tau, "X": d.X, "t": d.t, "e": d.e}

    def add(method: str, ate_hat: float, pehe: float | None = None, t0: float = 0.0) -> None:
        rows.append({"rho": rho, "s": s, "seed": seed, "method": method, "ate_err": ate_hat - true_ate,
                     "pehe": pehe, "time": time.perf_counter() - t0})

    for name in LEARNERS:
        t0 = time.perf_counter()
        m = make_learner(name, outcome_model=base, propensity_model=base, seed=seed).fit(d.X, d.t, d.y)
        tau_hat = m.predict(d.X)
        preds[name] = tau_hat
        add(f"{name}-learner", tau_hat.mean(), sqrt_pehe(tau_hat, d.tau), t0)
    for method, fn in {
        "naive": lambda: difference_in_means(d.t, d.y).value,
        "IPW (logistic)": lambda: ipw(d.X, d.t, d.y, "logistic").value,
        "IPW (gbm)": lambda: ipw(d.X, d.t, d.y, "gbm", seed=seed).value,
        "AIPW": lambda: aipw(d.X, d.t, d.y, base, base, seed=seed).value,
        "DML (PLR)": lambda: dml_plr(d.X, d.t, d.y, base, base, seed=seed).value,
        "PSM": lambda: psm(d.X, d.t, d.y, "logistic", estimand="ate", n_neighbors=1).value,
        "regression adj.": lambda: regression_adjustment(d.X, d.t, d.y, base, estimand="ate").value,
    }.items():
        t0 = time.perf_counter()
        add(method, fn(), None, t0)
    info = {"rho": rho, "s": s, "seed": seed, **overlap_summary(d.e, d.t)}
    return rows, {"info": info, "preds": preds}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=25)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--base", default="gbm")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/synthetic.jsonl"))
    args = p.parse_args()
    jobs = [(r, s, k, args.n, args.base) for k in range(args.seeds) for r, s in product(CONFOUNDING, OVERLAP)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep: dict[str, np.ndarray] = {}
    infos: list[dict] = []
    t0 = time.time()
    with Pool(args.workers) as pool, open(args.out, "w") as f:
        for i, (rows, extra) in enumerate(pool.imap_unordered(run_cell, jobs)):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            infos.append(extra["info"])
            if extra["info"]["seed"] == 0:  # keep one dataset per cell for the scatter plots
                tag = f"rho{extra['info']['rho']}_s{extra['info']['s']}"
                keep.update({f"{tag}/{k}": v for k, v in extra["preds"].items()})
            print(f"[{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(args.out.with_name("synthetic_preds.npz"), **keep)
    args.out.with_name("synthetic_overlap.json").write_text(json.dumps(infos))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
