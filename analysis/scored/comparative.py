"""Comparative analysis of the shift-in-diversity scores.

Each method writes a per-pair log-ratio score (``vmf_pair_scores.csv``,
``wic_pair_scores.csv``, ``cosine_pair_scores.csv``), all oriented so that a
positive value means the target corpus is more diverse than the source. This module
correlates each method's log-ratio against the ground-truth diversity *shift*
``log(qD(T)/qD(S))`` for the three Hill orders q in {0, 1, 2} (richness, Shannon,
Simpson), using Spearman's rho with bootstrap CIs, grouped by comparison scheme.

Because all methods and all ground-truth shifts share the same orientation
(positive = target more diverse), the expected rho sign is ``+1`` throughout -- no
per-method sign bookkeeping is needed (unlike the earlier absolute-score analysis).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_table
from analysis.scored.stats import (
    GT_SHIFT_COLS,
    CorpusIterator,
    correlation_table,
    pair_ground_truth,
    score_scatter,
)
from data_processing.simulation_loading import load_sim_corpora

logger = logging.getLogger("div")

# Method name -> (scoring subdir, pair-scores filename, log-ratio column). The
# subdir/filename must match score_data.py's output layout; keeping all three in one
# place stops the subdir drifting out of sync with the filename.
_METHODS = {
    "cosine": ("cosine", "cosine_pair_scores.csv", "cosine_log_ratio"),
    "vMF": ("vmf", "vmf_pair_scores.csv", "vmf_log_ratio"),
    "WiC": ("wic", "wic_pair_scores.csv", "wic_log_ratio"),
}
_METHOD_ORDER = ["cosine", "vMF", "WiC"]
_GT_COLS = list(GT_SHIFT_COLS.values())


def _method_palette() -> dict[str, tuple]:
    """Fixed colour per method, shared across the comparative figures."""
    return dict(zip(_METHOD_ORDER, sns.color_palette("colorblind", len(_METHOD_ORDER))))


def _load_method(
    scores_dir: Path,
    method: str,
    sim_dir: Path,
    iter_fn: CorpusIterator = load_sim_corpora,
) -> pd.DataFrame | None:
    """Load one method's pair scores with ground-truth shift columns attached.

    Returns ``None`` (with a warning) if the method's pair-scores CSV is absent, so
    the comparative analysis degrades gracefully when only some methods have run.
    """
    subdir, filename, _ = _METHODS[method]
    path = scores_dir / subdir / filename
    if not path.exists():
        logger.warning("No %s at %s; %s excluded from comparison.", filename, path, method)
        return None
    return pair_ground_truth(pd.read_csv(path), sim_dir, iter_fn)


def shift_correlation_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman rho of each method's log-ratio vs each ground-truth shift, by scheme.

    One row per ``(method, scheme, gt_shift_q)`` with rho, bootstrap CI, and n.
    Reuses :func:`~analysis.scored.stats.correlation_table` per method (grouping on
    the comparison ``scheme`` and treating the three ``gt_shift_q`` columns as the
    predictors).
    """
    parts = []
    for method in _METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        *_, score_col = _METHODS[method]
        corr = correlation_table(df, score_col, _GT_COLS, group_col="scheme")
        corr.insert(0, "method", method)
        parts.append(corr)

    if not parts:
        return pd.DataFrame(
            columns=["method", "scheme", "predictor", "spearmanr", "ci_low", "ci_high", "n"]
        )
    return pd.concat(parts, ignore_index=True)


def _plot_rho_by_scheme(corr: pd.DataFrame, gt_col: str, figures_dir: Path, name: str) -> None:
    """Dot plot of Spearman rho against comparison scheme, one colour per method.

    x = scheme, y = rho, colour = method, CI error bars -- the dot-plot convention
    used across the scored figures (not a heatmap).
    """
    sub = corr[corr["predictor"] == gt_col].copy()
    if sub.empty:
        return
    palette = _method_palette()
    methods = [m for m in _METHOD_ORDER if m in sub["method"].unique()]
    schemes = sorted(sub["scheme"].unique())

    fig, ax = plt.subplots()
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        grp = sub[sub["method"] == method].set_index("scheme").reindex(schemes).reset_index()
        xs = [schemes.index(s) + (i - (len(methods) - 1) / 2) * width for s in grp["scheme"]]
        yerr = [grp["spearmanr"] - grp["ci_low"], grp["ci_high"] - grp["spearmanr"]]
        ax.errorbar(xs, grp["spearmanr"], yerr=yerr, marker="o", linestyle="none",
                    capsize=3, color=palette[method], label=method)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(schemes)
    ax.set_xlabel("Comparison scheme")
    ax.set_ylabel("Spearman's rho")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(title="method", fontsize="small")
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

    loaded = {m: _load_method(scores_dir, m, sim_dir, iter_fn) for m in _METHOD_ORDER}
    loaded = {m: df for m, df in loaded.items() if df is not None}
    if not loaded:
        logger.warning("No method pair-scores found under %s; nothing to analyse.", scores_dir)
        return

    corr = shift_correlation_table(loaded)
    write_table(corr, tables_dir, "shift_correlations", convert_col_names=True)

    # Per-pair scatter of each method's log-ratio against each ground-truth shift.
    for method, df in loaded.items():
        *_, score_col = _METHODS[method]
        for q, gt_col in GT_SHIFT_COLS.items():
            score_scatter(
                df, gt_col, f"Ground-truth shift log(qD_T/qD_S), q={q}",
                score_col, f"{method} log-ratio", figures_dir,
                f"{method}_shift_vs_gt_q{q}", hue_col="scheme",
            )

    for q, gt_col in GT_SHIFT_COLS.items():
        _plot_rho_by_scheme(corr, gt_col, figures_dir, f"comparative_rho_by_scheme_q{q}")

    logger.info(
        "comparative: %d correlation rows across %d method(s)",
        len(corr), corr["method"].nunique(),
    )
