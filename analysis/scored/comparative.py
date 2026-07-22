"""Comparative analysis of vMF and WiC scoring (``vmf_scores.csv`` + ``wic_scores.csv``).

vMF (``vmf_kappa``) and WiC (``wic_p_diff_mean``) each get their own scored-mode
analysis (:mod:`analysis.scored.vmf_scored`, :mod:`analysis.scored.wic_scored`), but
neither puts the two side by side. This module does two comparative things:

* how well each method's score tracks the two ground-truth design properties (sense
  entropy, Zipfian slope), conditional on k -- one Spearman rho per (method, k,
  predictor), reusing :func:`~analysis.scored.stats.correlation_table` for each
  method in turn. vMF concentration and WiC P(diff) move in opposite directions
  relative to sense diversity (high kappa = low diversity, high P(diff) = high
  diversity), so rho signs are expected to be opposite; an ``expected_sign`` column
  is attached rather than transforming either score;
* WiC's own "performance penalty" -- the gap between its empirical rho
  (``wic_p_diff_mean`` vs. a predictor) and its theoretical ceiling rho
  (``p_diff_theoretical`` vs. the same predictor). vMF has no probability output and
  so has no theoretical-ceiling analogue; this half of the analysis is WiC-only.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_table
from analysis.scored.stats import correlation_table, merge_entropy

logger = logging.getLogger("div")

VMF_SCORE_COL = "vmf_kappa"
WIC_SCORE_COL = "wic_p_diff_mean"
THEORETICAL_COL = "p_diff_theoretical"
PREDICTORS = ["entropy_bits", "applied_slope"]

# vMF concentration falls where WiC P(diff) rises (and vice versa) for the same
# ground-truth predictor, since kappa measures concentration (low diversity) while
# P(diff) measures diversity directly.
_EXPECTED_SIGN = {
    ("vMF", "entropy_bits"): -1,
    ("vMF", "applied_slope"): 1,
    ("WiC (empirical)", "entropy_bits"): 1,
    ("WiC (empirical)", "applied_slope"): -1,
    ("WiC (empirical)", THEORETICAL_COL): 1,
    ("WiC (theoretical)", "entropy_bits"): 1,
    ("WiC (theoretical)", "applied_slope"): -1,
}

_METHOD_ORDER = ["vMF", "WiC (empirical)", "WiC (theoretical)"]


def _method_palette() -> dict[str, tuple]:
    """Fixed colour per method, shared across the comparative figures."""
    return dict(zip(_METHOD_ORDER, sns.color_palette("colorblind", len(_METHOD_ORDER))))


def comparative_correlation_table(
    vmf_df: pd.DataFrame | None, wic_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Spearman rho of each method's score vs. entropy/slope, conditional on k.

    One row per ``(method, k_senses, predictor)``: vMF's ``vmf_kappa``, WiC's
    empirical ``wic_p_diff_mean``, and WiC's theoretical ``p_diff_theoretical`` (its
    ceiling, since a perfect WiC model would predict exactly that value). The WiC
    empirical row set also includes ``p_diff_theoretical`` as a predictor (matching
    :mod:`analysis.scored.wic_scored`'s existing calibration angle), even though it
    is near-collinear with ``entropy_bits`` -- both are reported rather than
    collapsed, consistent with the existing per-method tables.
    """
    parts = []

    if vmf_df is not None:
        vmf_corr = correlation_table(vmf_df, VMF_SCORE_COL, PREDICTORS)
        vmf_corr.insert(0, "method", "vMF")
        parts.append(vmf_corr)

    if wic_df is not None:
        wic_emp = correlation_table(wic_df, WIC_SCORE_COL, PREDICTORS + [THEORETICAL_COL])
        wic_emp.insert(0, "method", "WiC (empirical)")
        parts.append(wic_emp)

        wic_theo = correlation_table(wic_df, THEORETICAL_COL, PREDICTORS)
        wic_theo.insert(0, "method", "WiC (theoretical)")
        parts.append(wic_theo)

    if not parts:
        return pd.DataFrame(
            columns=["method", "k_senses", "predictor", "expected_sign",
                     "spearmanr", "ci_low", "ci_high", "n"]
        )

    corr = pd.concat(parts, ignore_index=True)
    corr["expected_sign"] = [
        _EXPECTED_SIGN.get((m, p)) for m, p in zip(corr["method"], corr["predictor"])
    ]
    return corr[["method", "k_senses", "predictor", "expected_sign",
                 "spearmanr", "ci_low", "ci_high", "n"]]


def penalty_table(corr: pd.DataFrame) -> pd.DataFrame:
    """WiC's empirical-vs-theoretical rho gap: the model's performance penalty.

    For each ``(k_senses, predictor)`` present in both the "WiC (empirical)" and
    "WiC (theoretical)" rows of ``corr``, ``penalty = rho_theoretical -
    rho_empirical``: how much rank-correlation the model's imperfection costs
    relative to what a perfectly-calibrated WiC model would achieve. Both rows' own
    bootstrap CIs are carried alongside rather than composed into a new CI for the
    difference.
    """
    emp = corr[corr["method"] == "WiC (empirical)"]
    theo = corr[corr["method"] == "WiC (theoretical)"]
    merged = emp.merge(
        theo, on=["k_senses", "predictor"], suffixes=("_empirical", "_theoretical")
    )
    if merged.empty:
        return pd.DataFrame(
            columns=["k_senses", "predictor", "spearmanr_empirical", "ci_low_empirical",
                     "ci_high_empirical", "spearmanr_theoretical", "ci_low_theoretical",
                     "ci_high_theoretical", "penalty"]
        )
    merged["penalty"] = merged["spearmanr_theoretical"] - merged["spearmanr_empirical"]
    return merged[
        ["k_senses", "predictor", "spearmanr_empirical", "ci_low_empirical",
         "ci_high_empirical", "spearmanr_theoretical", "ci_low_theoretical",
         "ci_high_theoretical", "penalty"]
    ]


def _plot_rho_vs_k(corr: pd.DataFrame, predictor: str, figures_dir: Path, name: str) -> None:
    """Dot plot of Spearman rho against k, one colour per method, CI error bars.

    A small categorical/jittered dot plot (not a heatmap): x = k_senses, y = rho,
    colour = method, matching the errorbar convention used for WiC performance
    plots (:func:`analysis.scored.wic_scored._plot_performance`).
    """
    sub = corr[corr["predictor"] == predictor].copy()
    if sub.empty:
        return
    palette = _method_palette()
    methods = [m for m in _METHOD_ORDER if m in sub["method"].unique()]

    fig, ax = plt.subplots()
    ks = sorted(sub["k_senses"].unique())
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        grp = sub[sub["method"] == method].sort_values("k_senses")
        xs = [ks.index(k) + (i - (len(methods) - 1) / 2) * width for k in grp["k_senses"]]
        yerr = [grp["spearmanr"] - grp["ci_low"], grp["ci_high"] - grp["spearmanr"]]
        ax.errorbar(xs, grp["spearmanr"], yerr=yerr, marker="o", linestyle="none",
                    capsize=3, color=palette[method], label=method)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of senses (k)")
    ax.set_ylabel("Spearman's rho")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(title="method", fontsize="small")
    save_fig(fig, figures_dir, name)


def analyse_comparative(scores_dir: Path, sim_dir: Path, out_root: Path) -> None:
    """Compare vMF and WiC scores in ``scores_dir`` against the corpora in ``sim_dir``."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    vmf_path = scores_dir / "vmf" / "vmf_scores.csv"
    wic_path = scores_dir / "wic" / "wic_scores.csv"

    vmf_df = None
    if vmf_path.exists():
        vmf_df = merge_entropy(pd.read_csv(vmf_path), sim_dir)
    else:
        logger.warning(
            "No vmf_scores.csv at %s; comparative analysis will be WiC-only. "
            "Run score_data.py vmf first.",
            vmf_path,
        )

    wic_df = None
    if wic_path.exists():
        wic_df = merge_entropy(pd.read_csv(wic_path), sim_dir)
    else:
        logger.warning(
            "No wic_scores.csv at %s; comparative analysis will be vMF-only. "
            "Run score_data.py wic first.",
            wic_path,
        )

    if vmf_df is None and wic_df is None:
        logger.warning("Neither vmf_scores.csv nor wic_scores.csv found; nothing to analyse.")
        return

    corr = comparative_correlation_table(vmf_df, wic_df)
    write_table(corr, tables_dir, "comparative_correlations", convert_col_names=True)

    penalty = penalty_table(corr)
    write_table(penalty, tables_dir, "comparative_wic_penalty", convert_col_names=True)

    _plot_rho_vs_k(corr, "entropy_bits", figures_dir, "comparative_rho_vs_k_entropy")
    _plot_rho_vs_k(corr, "applied_slope", figures_dir, "comparative_rho_vs_k_slope")

    logger.info(
        "comparative: %d correlation rows across %d method(s)",
        len(corr),
        corr["method"].nunique(),
    )
