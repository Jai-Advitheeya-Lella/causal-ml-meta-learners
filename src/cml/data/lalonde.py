"""LaLonde (1986) National Supported Work data with the Dehejia-Wahba (1999) samples.

* ``nsw``  : randomised experiment, 185 treated + 260 control men (the Dehejia-Wahba subsample
  with two years of pre-treatment earnings; file ``nsw_dw.dta``).  The difference in mean 1978 earnings is the
  experimental benchmark for the effect of the programme on participants.
* ``psid`` : the same 185 treated men combined with 2,490 non-experimental PSID-1 controls
  (male household heads under 55, not retired).  Estimators applied to this sample should recover
  the experimental benchmark if the covariates make treatment ignorable.

Columns: ``treat, age, educ, black, hisp, married, nodegr, re74, re75, re78`` (earnings in USD).
Files are fetched from Rajeev Dehejia's public data page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import download

BASE = "https://users.nber.org/~rdehejia/data/"
COLS = ["treat", "age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", "re78"]
COVARIATES = COLS[1:-1]


def _read_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_lalonde(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """Return ``{"nsw": experimental sample, "psid": treated + PSID-1 controls}``."""
    d = Path(data_dir) / "lalonde"
    nsw = pd.read_stata(download(BASE + "nsw_dw.dta", d / "nsw_dw.dta"))
    nsw = nsw.rename(columns={"education": "educ", "hispanic": "hisp", "nodegree": "nodegr"})
    nsw = nsw[COLS].astype(float).reset_index(drop=True)
    psid_controls = _read_txt(download(BASE + "psid_controls.txt", d / "psid_controls.txt"))
    psid = pd.concat([nsw[nsw.treat == 1], psid_controls], ignore_index=True)
    return {"nsw": nsw, "psid": psid}


def design_matrix(df: pd.DataFrame, expand: bool = True) -> np.ndarray:
    """Covariates; ``expand=True`` adds the Dehejia-Wahba terms (squares and zero-earnings dummies)."""
    X = df[COVARIATES].to_numpy(float)
    if not expand:
        return X
    extra = np.column_stack([
        df["age"] ** 2, df["educ"] ** 2, df["re74"] ** 2 / 1e6, df["re75"] ** 2 / 1e6,
        (df["re74"] == 0).astype(float), (df["re75"] == 0).astype(float),
        df["black"] * (df["re74"] == 0), df["educ"] * df["re74"] / 1e3,
    ])
    return np.hstack([X, extra])
