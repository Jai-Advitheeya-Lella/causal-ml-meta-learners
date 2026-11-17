"""Shared containers and download helper."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class CausalData:
    """Observed data ``(X, t, y)`` plus, for simulated / semi-synthetic data, the ground truth."""

    X: Array
    t: Array
    y: Array
    tau: Array | None = None   # true individual treatment effect mu1 - mu0
    mu0: Array | None = None
    mu1: Array | None = None
    e: Array | None = None     # true propensity score, when known

    def __len__(self) -> int:
        return len(self.y)

    @property
    def ate(self) -> float:
        assert self.tau is not None
        return float(self.tau.mean())

    def subset(self, idx) -> "CausalData":
        f = lambda a: None if a is None else a[idx]  # noqa: E731
        return CausalData(self.X[idx], self.t[idx], self.y[idx], f(self.tau), f(self.mu0),
                          f(self.mu1), f(self.e))


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cml-data-loader"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dest)
    return dest
