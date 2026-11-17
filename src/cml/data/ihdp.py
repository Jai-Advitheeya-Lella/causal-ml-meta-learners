"""IHDP semi-synthetic benchmark (Hill, 2011) in the 100-replication NPCI release.

Covariates come from the Infant Health and Development Program randomised experiment
(25 covariates, 747 children); Hill removed a non-random subset of the treated group to
create confounding and simulated outcomes from "response surface B" (non-linear, with
ground-truth potential outcomes ``mu0, mu1``).  The files are the ones used by Johansson et al.
(2016) and Shalit et al. (2017): ``ihdp_npci_1-100.{train,test}.npz`` with 672 / 75 units
per replication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import CausalData, download

BASE = "https://www.fredjo.com/files/"
FILES = {"train": "ihdp_npci_1-100.train.npz", "test": "ihdp_npci_1-100.test.npz"}
N_REPLICATIONS = 100


def _load(split: str, data_dir: Path) -> dict[str, np.ndarray]:
    path = download(BASE + FILES[split], Path(data_dir) / "ihdp" / FILES[split])
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load_ihdp(rep: int = 0, split: str = "train", data_dir: str | Path = "data") -> CausalData:
    """One replication (``0 <= rep < 100``) of the ``train`` (672 units) or ``test`` (75) split."""
    d = _load(split, Path(data_dir))
    X = d["x"][:, :, rep].astype(float)
    mu0, mu1 = d["mu0"][:, rep].astype(float), d["mu1"][:, rep].astype(float)
    return CausalData(X=X, t=d["t"][:, rep].astype(int), y=d["yf"][:, rep].astype(float),
                      tau=mu1 - mu0, mu0=mu0, mu1=mu1)
