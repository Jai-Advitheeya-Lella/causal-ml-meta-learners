"""Figures and Markdown tables from the saved experiment results.

    python scripts/make_figures.py synthetic
    python scripts/make_figures.py ihdp
    python scripts/make_figures.py lalonde
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style as S

ASSETS, RESULTS = Path("assets"), Path("results")
LEARNERS = ["S", "T", "X", "R", "DR"]
CONF = [0.0, 0.5, 0.9]
OVERLAP = [0.5, 1.5, 3.0]


# --------------------------------------------------------------------------- synthetic
def synthetic() -> None:
    P = np.load(RESULTS / "synthetic_preds.npz")
    df = pd.read_json(RESULTS / "synthetic.jsonl", lines=True)
    info = pd.DataFrame(json.loads((RESULTS / "synthetic_overlap.json").read_text()))
    n_seeds = df.seed.nunique()

    # 1. CATE recovery scatter (hero)
    rho, s = 0.5, 1.5
    tag = f"rho{rho}_s{s}"
    tau = P[f"{tag}/tau"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharex=True, sharey=True)
    lim = (tau.min() - 0.5, tau.max() + 0.5)
    for ax, L in zip(axes, LEARNERS):
        est = P[f"{tag}/{L}"]
        ax.scatter(tau, est, s=5, alpha=0.35, color=S.LEARNER_COLORS[L], lw=0)
        ax.plot(lim, lim, color=S.INK, lw=1)
        rmse = np.sqrt(np.mean((est - tau) ** 2))
        corr = np.corrcoef(est, tau)[0, 1]
        ax.set_title(f"{L}-learner", loc="left")
        ax.text(0.04, 0.95, rf"$\sqrt{{\mathrm{{PEHE}}}}$ = {rmse:.2f}" + f"\ncorr = {corr:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=S.INK2)
        ax.set_xlabel(r"true $\tau(x)$")
        ax.set_xlim(lim), ax.set_ylim(lim)
    axes[0].set_ylabel(r"estimated $\hat\tau(x)$")
    fig.suptitle(f"CATE recovery on simulated data (n = 2000, confounding = {rho}, overlap scale = {s}; "
                 "gradient-boosting base learners, one dataset)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "cate_recovery.png", dpi=130)
    plt.close(fig)

    # 2. overlap diagnostics: propensity histograms by overlap scale (+ LaLonde PSID if available)
    panels = [(f"rho{0.5}_s{sv}", f"overlap scale {sv}") for sv in OVERLAP]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=False)
    bins = np.linspace(0, 1, 26)
    for ax, (tg, title) in zip(axes[:3], panels):
        e, t = P[f"{tg}/e"], P[f"{tg}/t"]
        ax.hist(e[t == 1], bins=bins, color=S.CAT[1], alpha=0.75, label="treated")
        ax.hist(e[t == 0], bins=bins, color=S.CAT[0], alpha=0.6, label="control")
        frac = np.mean((e < 0.05) | (e > 0.95))
        ax.set_title(f"simulated, {title}", loc="left")
        ax.text(0.5, 0.95, f"{100 * frac:.0f}% of units outside [0.05, 0.95]", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color=S.INK2)
        ax.set_xlabel(r"true propensity $e(x)$")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 4)
    axes[0].set_ylabel("units (log scale)")
    axes[0].legend(loc="upper left", fontsize=8)
    sc = RESULTS / "lalonde_scores.npz"
    if sc.exists():
        z = np.load(sc)
        e, t = z["e"], z["t"]
        ax = axes[3]
        ax.hist(e[t == 1], bins=bins, color=S.CAT[1], alpha=0.75, label="NSW treated")
        ax.hist(e[t == 0], bins=bins, color=S.CAT[0], alpha=0.6, label="PSID controls")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 4)
        ax.set_title("LaLonde NSW vs PSID-1", loc="left")
        ax.set_xlabel(r"estimated propensity $\hat e(x)$ (logistic)")
        ax.text(0.5, 0.95, f"{100 * np.mean((e < 0.05) | (e > 0.95)):.0f}% outside [0.05, 0.95]",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=S.INK2)
    fig.suptitle("Overlap diagnostics: propensity-score distributions by treatment arm", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "overlap_diagnostics.png", dpi=130)
    plt.close(fig)

    # 3. ATE error: bias and spread per estimator in every cell
    methods = ["naive", "regression adj.", "IPW (logistic)", "IPW (gbm)", "PSM", "AIPW", "DML (PLR)",
               "S-learner", "T-learner", "X-learner", "R-learner", "DR-learner"]
    colors = {"naive": S.MUTED, "regression adj.": S.CAT[3], "IPW (logistic)": S.CAT[4], "IPW (gbm)": S.CAT[7],
              "PSM": S.CAT[5], "AIPW": S.CAT[6], "DML (PLR)": S.CAT[2]}
    for L in LEARNERS:
        colors[f"{L}-learner"] = S.LEARNER_COLORS[L]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=False)
    rows = []
    for i, rho in enumerate(CONF):
        for j, sv in enumerate(OVERLAP):
            ax = axes[i, j]
            sub = df[(df.rho == rho) & (df.s == sv)]
            for k, m in enumerate(methods):
                err = sub[sub.method == m].ate_err.to_numpy()
                bias, sd = err.mean(), err.std(ddof=1)
                rmse = np.sqrt(np.mean(err**2))
                ax.errorbar(bias, k, xerr=sd, fmt="o", color=colors[m], ms=5, lw=1.6, capsize=0)
                rows.append({"confounding": rho, "overlap_scale": sv, "method": m, "bias": bias,
                             "sd": sd, "rmse": rmse})
            ax.axvline(0, color=S.INK, lw=1)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods if j == 0 else [], fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlim(-1.6, 2.2) if rho < 0.9 else ax.set_xlim(-1.6, 2.6)
            e_cell = info[(info.rho == rho) & (info.s == sv)]
            ax.set_title(f"confounding {rho}, overlap scale {sv}", loc="left", fontsize=9.5)
            ax.text(0.98, 0.03, f"{100 * e_cell.frac_extreme.mean():.0f}% extreme e", transform=ax.transAxes,
                    ha="right", fontsize=8, color=S.MUTED)
            if i == 2:
                ax.set_xlabel("error of ATE estimate (mean ± sd over seeds)")
    fig.suptitle(f"ATE estimation: bias and variability over {n_seeds} simulated datasets "
                 "(true ATE = 2, n = 2000; zero line = unbiased)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ASSETS / "ate_bias_variance.png", dpi=120)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "synthetic_ate_summary.csv", index=False)

    # 4. sqrt(PEHE) heatmap: learners x cells
    sub = df[df.method.isin([f"{L}-learner" for L in LEARNERS])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    tab = []
    for ax, rho in zip(axes, CONF):
        M = np.array([[sub[(sub.rho == rho) & (sub.s == sv) & (sub.method == f"{L}-learner")].pehe.mean()
                       for sv in OVERLAP] for L in LEARNERS])
        ax.imshow(M, cmap=S.blues(), aspect="auto", vmin=0.3, vmax=max(1.6, M.max()))
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[a, b] > 1.0 else S.INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"s = {v}" for v in OVERLAP])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{L}-learner" for L in LEARNERS])
        ax.set_title(f"confounding {rho}", loc="left")
        ax.grid(False)
        for a, L in enumerate(LEARNERS):
            for b, sv in enumerate(OVERLAP):
                tab.append({"learner": L, "confounding": rho, "overlap_scale": sv, "sqrt_pehe": M[a, b]})
    fig.suptitle(rf"$\sqrt{{\mathrm{{PEHE}}}}$ of the CATE estimate (mean over {n_seeds} datasets); "
                 "larger overlap scale = poorer overlap", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ASSETS / "synthetic_pehe.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(tab).to_csv(RESULTS / "synthetic_pehe_summary.csv", index=False)
    print(pd.DataFrame(tab).pivot_table(index="learner", columns=["confounding", "overlap_scale"],
                                        values="sqrt_pehe").round(2).to_string())


# -------------------------------------------------------------------------------- IHDP
def ihdp() -> None:
    df = pd.read_json(RESULTS / "ihdp.jsonl", lines=True)
    n = df.rep.nunique()
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))  # noqa: E731
    cate = df[df.pehe_in.notna()]
    order = [(b, L) for b in ["ridge", "rf", "gbm"] for L in LEARNERS] + \
        [("rf", "econml-X"), ("rf", "econml-DR"), ("rf", "econml-CausalForest")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    lines = ["| Base learner | Method | sqrt(PEHE) within-sample | sqrt(PEHE) out-of-sample | median within-sample | eps_ATE |",
             "|---|---|---|---|---|---|"]
    for ax, col, ttl in zip(axes, ["pehe_in", "pehe_out"], ["within-sample (672 units)", "out-of-sample (75 units)"]):
        for k, (b, L) in enumerate(order):
            x = cate[(cate.base == b) & (cate.method == L)][col]
            c = S.LEARNER_COLORS.get(L, S.MUTED)
            ax.errorbar(x.mean(), k, xerr=se(x), fmt="o", color=c, ms=6, lw=1.8)
            ax.plot(x.median(), k, marker="|", color=c, ms=12, mew=2, ls="none")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{L} ({b})" if not L.startswith("econml") else L.replace("econml-", "EconML ") + " (rf)"
                            for b, L in order], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(rf"$\sqrt{{\mathrm{{PEHE}}}}$, {ttl}")
        ax.set_title(f"{ttl}: mean ± s.e. (dot), median (tick)", loc="left", fontsize=9.5)
    for b, L in order:
        x = cate[(cate.base == b) & (cate.method == L)]
        lines.append(f"| {b} | {L} | {x.pehe_in.mean():.2f} ± {se(x.pehe_in):.2f} | "
                     f"{x.pehe_out.mean():.2f} ± {se(x.pehe_out):.2f} | {x.pehe_in.median():.2f} | "
                     f"{x.eps_ate.mean():.3f} ± {se(x.eps_ate):.3f} |")
    fig.suptitle(f"IHDP, {n} replications: error of the CATE estimate", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ASSETS / "ihdp_pehe.png", dpi=125)
    plt.close(fig)

    # ATE-only table
    ate = df[df.base == "-"]
    lines += ["", "| ATE estimator | eps_ATE (mean ± s.e.) | median |", "|---|---|---|"]
    for m in ["naive", "IPW", "AIPW-gbm", "DML-PLR-gbm"]:
        x = ate[ate.method == m].eps_ate
        lines.append(f"| {m} | {x.mean():.3f} ± {se(x):.3f} | {x.median():.3f} |")
    (RESULTS / "ihdp_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # distribution of per-replication sqrt(PEHE) for the best config of each learner (gbm)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    data = [cate[(cate.base == "gbm") & (cate.method == L)].pehe_in.to_numpy() for L in LEARNERS]
    bp = ax.boxplot(data, orientation="vertical", widths=0.55, showfliers=True, patch_artist=True,
                    flierprops={"marker": "o", "ms": 3, "mfc": S.MUTED, "mec": "none"})
    for patch, L in zip(bp["boxes"], LEARNERS):
        patch.set(facecolor=S.LEARNER_COLORS[L], alpha=0.5, edgecolor=S.INK2)
    for el in ("whiskers", "caps", "medians"):
        for ln in bp[el]:
            ln.set(color=S.INK2)
    ax.set_xticklabels([f"{L}-learner" for L in LEARNERS])
    ax.set_yscale("log")
    ax.set_ylabel(r"within-sample $\sqrt{\mathrm{PEHE}}$ (log scale)")
    ax.set_title(f"IHDP: spread over {n} replications (gradient-boosting base learners)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / "ihdp_replications.png", dpi=125)
    plt.close(fig)


# ----------------------------------------------------------------------------- LaLonde
def usd(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def lalonde() -> None:
    res = json.loads((RESULTS / "lalonde.json").read_text())
    b, est, dg = res["benchmark"], res["estimates"], res["diagnostics"]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fam_c = {"naive": S.MUTED, "outcome": S.CAT[3], "propensity": S.CAT[4], "doubly robust": S.CAT[6],
             "meta-learner": S.CAT[0]}
    lo_b, hi_b = b["ci"]
    ax.axvspan(lo_b, hi_b, color=S.CAT[2], alpha=0.15, lw=0)
    ax.axvline(b["diff_in_means"], color=S.CAT[2], lw=2)
    ax.text(b["diff_in_means"], -0.9, f"  experimental benchmark: {usd(b['diff_in_means'])}", color=S.CAT[2],
            fontsize=9, va="center", fontweight="bold")
    ax.axvline(0, color=S.INK, lw=0.8)
    names = [e["name"] for e in est]
    for k, e in enumerate(est):
        c = fam_c[e["family"]]
        ax.plot([e["lo"], e["hi"]], [k, k], color=c, lw=2.2, solid_capstyle="butt")
        ax.plot(e["value"], k, "o", color=c, ms=7)
        ax.text(max(e["hi"], e["value"]) + 350, k, usd(e["value"]), va="center", fontsize=8.5, color=S.INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_ylim(len(names) - 0.5, -1.5)
    ax.set_xlim(-18500, 14500)
    ax.set_xlabel("estimated effect of NSW on 1978 earnings, treated units (USD; 95% bootstrap interval)")
    ax.set_title("LaLonde: NSW treated vs PSID-1 controls, observational estimators vs the experiment",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "lalonde_estimates.png", dpi=130)
    plt.close(fig)

    # balance (Love plot) and overlap
    cols = dg["smd_columns"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(cols))
    ax.axvspan(-0.1, 0.1, color=S.GRID, alpha=0.6, lw=0)
    ax.plot(dg["smd_unadjusted"], y, "o", color=S.MUTED, label="unadjusted (PSID)")
    ax.plot(dg["smd_ipw"], y, "s", color=S.CAT[4], label="after IPW (ATT weights)")
    ax.plot(dg["smd_psm"], y, "D", color=S.CAT[6], label="after 1-NN matching")
    ax.plot(dg["nsw_smd"], y, "^", color=S.CAT[2], label="experimental NSW")
    ax.axvline(0, color=S.INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(cols)
    ax.invert_yaxis()
    ax.set_xlabel("standardised mean difference (treated - control)")
    ax.set_title("Covariate balance: shaded band is |SMD| < 0.1", loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "lalonde_balance.png", dpi=130)
    plt.close(fig)

    lines = ["| Family | Estimator | ATT estimate (USD) | 95% bootstrap interval | covers benchmark |", "|---|---|---|---|---|"]
    for e in est:
        cover = "yes" if e["lo"] <= b["diff_in_means"] <= e["hi"] else "no"
        lines.append(f"| {e['family']} | {e['name']} | {e['value']:,.0f} | [{e['lo']:,.0f}, {e['hi']:,.0f}] | {cover} |")
    (RESULTS / "lalonde_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", choices=["synthetic", "ihdp", "lalonde"])
    args = p.parse_args()
    S.apply()
    {"synthetic": synthetic, "ihdp": ihdp, "lalonde": lalonde}[args.which]()


if __name__ == "__main__":
    main()
