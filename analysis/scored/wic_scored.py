"""Analysis of WiC scoring output (``wic_scores.csv`` + ``wic_pair_scores.csv``).

Three things:

* how the WiC diversity signal (``wic_p_diff_mean``, mean P(different sense)) tracks
  sense entropy and Zipfian slope, conditional on k: per-corpus dot plots against
  both properties, plus the (slope, k) grid table;
* calibration -- the model's mean P(diff) against the design's theoretical diff rate,
  with the optimal diagonal and a pooled OLS fit of the observed points;
* WiC *model performance* -- accuracy and F1 of the per-pair predictions against the
  gold labels -- per (k, offset) cell and overall, with bootstrap confidence
  intervals, plotted against both the cell's mean slope and its mean entropy.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import linregress  # type: ignore
from sklearn.metrics import accuracy_score, f1_score  # type: ignore

from analysis.io import save_fig, write_csv, write_table
from analysis.scored.stats import (
    correlation_table,
    k_palette,
    merge_entropy,
    metric_with_ci,
    score_grid,
    score_scatter,
)

logger = logging.getLogger("div")

SCORE_COL = "wic_p_diff_mean"
PREDICTORS = ["entropy_bits", "applied_slope", "p_diff_theoretical"]


def _performance_row(pairs: pd.DataFrame, **keys) -> dict:
    """Accuracy and F1 (with bootstrap CIs) over one set of per-pair rows."""
    y_true = pairs["label"].to_numpy()
    y_pred = pairs["pred"].to_numpy()
    acc, acc_lo, acc_hi = metric_with_ci(y_true, y_pred, accuracy_score)
    # Binary F1 on the "same sense" positive class (label 1), matching the task.
    f1, f1_lo, f1_hi = metric_with_ci(
        y_true, y_pred, lambda a, b: f1_score(a, b, pos_label=1, zero_division=0)
    )
    return {
        **keys,
        "accuracy": acc,
        "acc_ci_low": acc_lo,
        "acc_ci_high": acc_hi,
        "F1": f1,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "n_pairs": len(pairs),
    }


def _performance_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-(k, offset) accuracy/F1 with CIs, plus an overall row.

    Cells are grouped on the offset (the regular design grid); the mean applied slope
    and mean sense entropy over the cell's corpora are reported too, since both are
    lemma-specific and so vary within a cell. The offset and both cell means are thus
    available in the table (and serve as x-positions for the performance plots).
    """
    rows = [
        _performance_row(
            grp,
            k_senses=k,
            offset=round(off, 4),
            mean_applied_slope=round(float(grp["applied_slope"].mean()), 4),
            mean_entropy_bits=round(float(grp["entropy_bits"].mean()), 4),
        )
        for (k, off), grp in pairs.groupby(["k_senses", "offset"])
    ]
    rows.append(
        _performance_row(
            pairs,
            k_senses="all",
            offset="all",
            mean_applied_slope="all",
            mean_entropy_bits="all",
        )
    )
    return pd.DataFrame(rows)


def _plot_performance(
    perf: pd.DataFrame, x_col: str, x_label: str, x_tag: str, figures_dir: Path
) -> None:
    """Accuracy and F1 against one design property, hued by k, with CI error bars.

    Each point is one (k, offset) cell positioned at the cell's mean value of the
    property (applied slope or sense entropy). Because both properties are
    lemma-specific, cells need not be evenly spaced along the x-axis.
    """
    import matplotlib.pyplot as plt

    cells = perf[perf["offset"] != "all"].copy()
    cells[x_col] = cells[x_col].astype(float)
    palette = k_palette(cells)

    specs = [("accuracy", "acc_ci_low", "acc_ci_high", "Accuracy"),
             ("F1", "f1_ci_low", "f1_ci_high", "F1")]
    for value_col, lo_col, hi_col, label in specs:
        fig, ax = plt.subplots()
        for k, grp in cells.sort_values(x_col).groupby("k_senses"):
            yerr = [grp[value_col] - grp[lo_col], grp[hi_col] - grp[value_col]]
            ax.errorbar(grp[x_col], grp[value_col], yerr=yerr, marker="o",
                        capsize=3, color=palette[k], label=f"k={k}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.legend(title="senses")
        save_fig(fig, figures_dir, f"wic_{value_col.lower()}_vs_{x_tag}")


def _plot_p_diff_calibration(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Model P(diff) against the theoretical design P(diff), one point per corpus.

    Points on the diagonal mean the model's mean predicted P(diff) matches the
    Zipfian design's own diff-rate target exactly. A pooled OLS fit of the observed
    points is drawn alongside, so systematic over-/under-shoot is visible as its
    deviation from that optimal line. The legend reports the coefficients with their
    standard errors (so slope != 1 / intercept != 0 can be judged against them), and
    a +-1 SE band of the mean prediction is shaded around the fit.
    """
    grid = sns.relplot(
        data=per_corpus,
        x="p_diff_theoretical",
        y=SCORE_COL,
        hue="k_senses",
        palette=k_palette(per_corpus),
        kind="scatter",
    )

    fit = per_corpus[["p_diff_theoretical", SCORE_COL]].dropna()
    for ax in grid.axes.flat:
        # Legend from the line handles only; the hue legend (k) is the figure-level
        # one seaborn already places outside the axes.
        lines = ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
                        label="optimal ($y = x$)")
        if len(fit) >= 2:
            x = fit["p_diff_theoretical"].to_numpy()
            y = fit[SCORE_COL].to_numpy()
            res = linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = res.slope * xs + res.intercept
            # +-1 SE of the mean prediction: s * sqrt(1/n + (x - x_bar)^2 / Sxx).
            resid_se = np.sqrt(((y - (res.slope * x + res.intercept)) ** 2).sum()
                               / (len(x) - 2))
            se_mean = resid_se * np.sqrt(
                1 / len(x) + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
            )
            lines += [ax.fill_between(xs, ys - se_mean, ys + se_mean, color="black",
                                      alpha=0.15, linewidth=0,
                                      label="$\\pm 1$ SE of the fit")]
            lines += ax.plot(
                xs, ys, color="black", linewidth=1,
                label=(f"OLS: $y = {res.slope:.2f}(\\pm{res.stderr:.2f})\\,x "
                       f"{res.intercept:+.2f}(\\pm{res.intercept_stderr:.2f})$ "
                       f"($R^2 = {res.rvalue ** 2:.2f}$)"),
            )
        ax.legend(handles=lines, loc="lower right", fontsize="small")
    grid.set_axis_labels("Theoretical P(diff)", "Mean predicted P(diff)")
    save_fig(grid.figure, figures_dir, "wic_p_diff_calibration")


def analyse_wic_scored(scores_dir: Path, sim_dir: Path, out_root: Path) -> None:
    """Analyse WiC scores in ``scores_dir`` against the corpora in ``sim_dir``."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    wic_dir = scores_dir / "wic"
    scores_path = wic_dir / "wic_scores.csv"
    pairs_path = wic_dir / "wic_pair_scores.csv"
    if not scores_path.exists() or not pairs_path.exists():
        logger.warning(
            "Missing wic_scores.csv / wic_pair_scores.csv under %s; nothing to "
            "analyse. Run score_data.py wic first.",
            wic_dir,
        )
        return

    per_corpus = merge_entropy(pd.read_csv(scores_path), sim_dir)
    per_corpus["lemma_pos"] = per_corpus["word"] + "_" + per_corpus["pos"]
    per_corpus = per_corpus.sort_values(["lemma_pos", "k_senses", "offset"])

    write_csv(per_corpus, tables_dir, "wic_per_corpus")
    per_lemma_dir = tables_dir / "per_lemma_pos"
    for lemma_pos, group in per_corpus.groupby("lemma_pos"):
        write_table(group.drop(columns="lemma_pos"), per_lemma_dir, str(lemma_pos))

    # (1) score-vs-property correlations, conditional on k
    corr = correlation_table(per_corpus, SCORE_COL, PREDICTORS)
    write_table(corr, tables_dir, "wic_correlations", convert_col_names=True)

    # (2) score vs design properties: (slope, k) grid table + per-corpus dot plots
    grid = score_grid(per_corpus, SCORE_COL)
    write_table(grid, tables_dir, "wic_p_diff_grid", index=True)
    score_scatter(per_corpus, "applied_slope", "Applied Zipfian slope",
                  SCORE_COL, "Mean P(diff)", figures_dir, "wic_p_diff_vs_slope")
    score_scatter(per_corpus, "entropy_bits", "Sense entropy (bits)",
                  SCORE_COL, "Mean P(diff)", figures_dir, "wic_p_diff_vs_entropy")
    _plot_p_diff_calibration(per_corpus, figures_dir)

    # (3) model performance: accuracy + F1 with bootstrap CIs, vs slope and entropy.
    # The entropy merge gives each pair its corpus's entropy, so cell means come out
    # pair-weighted -- consistent with how mean_applied_slope is computed.
    pairs = merge_entropy(pd.read_csv(pairs_path), sim_dir)
    perf = _performance_table(pairs)
    write_table(perf, tables_dir, "wic_performance", convert_col_names=True)
    _plot_performance(perf, "mean_applied_slope", "Applied Zipfian slope", "slope",
                      figures_dir)
    _plot_performance(perf, "mean_entropy_bits", "Sense entropy (bits)", "entropy",
                      figures_dir)

    logger.info(
        "wic_scored: %d scored corpora, %d pairs; overall acc=%.3f F1=%.3f",
        len(per_corpus),
        len(pairs),
        float(perf.iloc[-1]["accuracy"]),
        float(perf.iloc[-1]["F1"]),
    )
