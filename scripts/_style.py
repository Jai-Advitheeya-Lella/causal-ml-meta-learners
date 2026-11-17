"""Shared matplotlib styling for all figures (light surface, thin marks)."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed categorical order (validated for colour-vision deficiency on adjacent pairs)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# colour follows the learner / estimator family in every figure
LEARNER_COLORS = {"S": CAT[0], "T": CAT[1], "X": CAT[2], "R": CAT[3], "DR": CAT[6]}
BASE_COLORS = {"ridge": CAT[3], "rf": CAT[2], "gbm": CAT[0]}


def blues(reverse: bool = False) -> LinearSegmentedColormap:
    ramp = BLUE_RAMP[::-1] if reverse else BLUE_RAMP
    return LinearSegmentedColormap.from_list("blues_seq", ramp)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "font.family": "DejaVu Sans",
        "text.color": INK,
    })
