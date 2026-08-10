"""Comparative analysis of the shift-in-diversity scores.

Each method writes a per-pair log-ratio score (``vmf_pair_scores.csv``,
``wic_pair_scores.csv``, ``cosine_pair_scores.csv``), all oriented so that a
positive value means the target corpus is more diverse than the source. This module
correlates each method's log-ratio against the ground-truth diversity *shift*
``log(qD(T)/qD(S))`` for the three Hill orders q in {0, 1, 2} (richness, Shannon,
Simpson) and against the evenness shift ``log(E(T)/E(S))``, using Spearman's rho with
bootstrap CIs, grouped by comparison scheme.

The richness (q=0) and evenness targets are the two dimensions the simulation varies
independently, so the pair of them says which dimension a single-valued method is
actually tracking.

Because all methods and all ground-truth shifts share the same orientation
(positive = target more diverse), the expected rho sign is ``+1`` throughout -- no
per-method sign bookkeeping is needed (unlike the earlier absolute-score analysis).

This is the *per-dataset* mode: one invocation sees one simulated vocabulary, so its
tables are conditional on a single part of speech. :mod:`analysis.scored.pooled` runs
the same comparison over every PoS at once.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

from analysis.io import save_fig, write_table
from analysis.scored.methods import (
    GT_COLS,
    METHOD_ORDER,
    CorpusIterator,
    dot_plot_by_group,
    load_all_methods,
    method_palette,
    n_sensitivity_table,
    plot_err_vs_n,
    score_col,
)
from analysis.scored.stats import GT_SHIFT_COLS, MEASURE_LABELS, correlation_table, score_scatter
from data_processing.simulation_loading import load_sim_corpora

logger = logging.getLogger("div")

# Re-exported: n_sensitivity_table moved to analysis.scored.methods so the pooled mode
# can reach it without importing this module, but it is still part of this module's
# published surface (run_analysis and the tests import it from here).
__all__ = ["analyse_comparative", "n_sensitivity_table", "shift_correlation_table"]


def shift_correlation_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman rho of each method's log-ratio vs each ground-truth shift, by scheme.

    One row per ``(method, scheme, gt_shift_q)`` with rho, bootstrap CI, and n.
    Reuses :func:`~analysis.scored.stats.correlation_table` per method (grouping on
    the comparison ``scheme`` and treating the three ``gt_shift_q`` columns as the
    predictors).
    """
    parts = []
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        corr = correlation_table(df, score_col(method), GT_COLS, group_col="scheme")
        corr.insert(0, "method", method)
        parts.append(corr)

    if not parts:
        return pd.DataFrame(
            columns=[
                "method",
                "scheme",
                "predictor",
                "spearmanr",
                "ci_low",
                "ci_high",
                "n",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _plot_rho_by_scheme(
    corr_df: pd.DataFrame, gt_col: str, figures_dir: Path, name: str
) -> None:
    """Dot plot of the SRC against comparison scheme, one colour per method.

    x = scheme, y = SRC (Spearman's rank correlation), colour = method, CI error
    bars -- the dot-plot convention used across the scored figures (not a heatmap).
    """
    sub = corr_df[corr_df["predictor"] == gt_col].copy()
    if sub.empty:
        return
    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    schemes = sorted(sub["scheme"].unique())

    fig, ax = plt.subplots()
    dot_plot_by_group(
        ax, sub, "scheme", methods, method_palette(), group_order=schemes
    )
    ax.set_xlabel("Comparison scheme")
    ax.set_ylabel("SRC (Spearman's rank correlation)")
    ax.legend(title="Method", fontsize="small")
    save_fig(fig, figures_dir, name)


def analyse_comparative(
    scores_dir: Path,
    sim_dir: Path,
    out_root: Path,
    iter_fn: CorpusIterator = load_sim_corpora,
) -> None:
    """Compare the methods' shift scores against the ground-truth diversity shifts.

    ``iter_fn`` selects how ``sim_dir`` is walked for the corpora's ``.meta.json``
    sidecars -- the simulated layout by default, or
    :func:`~data_processing.dwug_loading.load_dwug_corpora` for the diachronic
    evaluation, where the single ``diachronic`` scheme makes the per-scheme grouping
    collapse to one row per (method, Hill order).
    """
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    loaded = load_all_methods(scores_dir, sim_dir, iter_fn)
    if not loaded:
        logger.warning(
            "No method pair-scores found under %s; nothing to analyse.", scores_dir
        )
        return

    corr_df = shift_correlation_table(loaded)
    write_table(corr_df, tables_dir, "shift_correlations", convert_col_names=True)

    # Sample-size diagnostic: each rho above is computed at a particular n_used, and
    # the vMF estimator's bias depends on it. This table and its figures say whether a
    # method's error actually tracks n over the range present in the data.
    n_sens = n_sensitivity_table(loaded)
    if not n_sens.empty:
        write_table(n_sens, tables_dir, "n_sensitivity", convert_col_names=True)

    # Per-pair scatter of each method's log-ratio against each ground-truth shift.
    # Figure stems keep the measure suffix used in the column name ("q0" ... ,
    # "evenness") so a figure pairs up unambiguously with its table row.
    for method, df in loaded.items():
        for measure, gt_col in GT_SHIFT_COLS.items():
            suffix = gt_col.removeprefix("gt_shift_")
            score_scatter(
                df,
                gt_col,
                f"GT shift, {MEASURE_LABELS[measure]}",
                score_col(method),
                f"{method} log-ratio",
                figures_dir / method,
                f"shift_vs_gt_{suffix}",
                hue_col="scheme",
            )

    for gt_col in GT_SHIFT_COLS.values():
        suffix = gt_col.removeprefix("gt_shift_")
        _plot_rho_by_scheme(
            corr_df,
            gt_col,
            figures_dir / "rho_by_scheme",
            f"rho_by_scheme_{suffix}",
        )
        plot_err_vs_n(
            loaded, gt_col, figures_dir / "error_n", f"comparative_err_vs_n_{suffix}"
        )

    logger.info(
        "comparative: %d correlation rows across %d method(s)",
        len(corr_df),
        corr_df["method"].nunique(),
    )
    if not n_sens.empty:
        spans = n_sens[n_sens["note"] == ""]
        logger.info(
            "n-sensitivity: %d/%d cells had a varying n_used (range %s-%s overall)",
            len(spans),
            len(n_sens),
            f"{n_sens['n_used_min'].min():.0f}" if len(n_sens) else "-",
            f"{n_sens['n_used_max'].max():.0f}" if len(n_sens) else "-",
        )
