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
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_table
from analysis.scored.stats import (
    GT_SHIFT_COLS,
    MEASURE_LABELS,
    N_USED_COL,
    CorpusIterator,
    correlation_table,
    pair_ground_truth,
    score_scatter,
    spearman_with_ci,
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
        logger.warning(
            "No %s at %s; %s excluded from comparison.", filename, path, method
        )
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


def n_sensitivity_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How each method's per-pair error depends on the corpus size it was scored at.

    Every scorer down-samples a pair to ``n_used = min(n_S, n_T)`` before measuring, so
    a pair's score is computed on however many usages the smaller corpus supplied. The
    vMF estimator's attenuation is a known function of that n (the resultant length is
    biased upward at small n and converges as n grows), while the cosine baseline does
    not share that dependence. If vMF trails the baseline, this table separates "the
    approach is weaker" from "the approach was run below the sample size it needs":
    the former shows a flat error-vs-n relation, the latter a negative one.

    One row per ``(method, scheme, gt_shift_q)``: Spearman rho of ``|score - gt|``
    against ``n_used``, plus the n range it spans. A negative rho means larger corpora
    gave smaller errors. A constant n (the simulation can fix corpus size within a
    scheme) or fewer than three pairs is reported with a ``note`` rather than dropped,
    matching :func:`~analysis.scored.stats.correlation_table`'s convention.
    """
    rows = []
    for method in _METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        *_, score_col = _METHODS[method]
        for scheme, sub in df.groupby("scheme"):
            for gt_col in _GT_COLS:
                pair = sub[[score_col, gt_col, N_USED_COL]].dropna()
                # Absolute deviation from the ground-truth shift: the per-pair error
                # whose size, not direction, is expected to shrink with n.
                err = (pair[score_col] - pair[gt_col]).abs().to_numpy()
                ns = pair[N_USED_COL].to_numpy(dtype=float)
                note = ""
                if len(ns) < 3:
                    rho = lo = hi = float("nan")
                    note = "n<3"
                elif np.ptp(ns) == 0:
                    rho = lo = hi = float("nan")
                    note = "constant n_used"
                elif np.ptp(err) == 0:
                    rho = lo = hi = float("nan")
                    note = "constant error"
                else:
                    rho, lo, hi, _ = spearman_with_ci(ns, err)
                rows.append(
                    {
                        "method": method,
                        "scheme": scheme,
                        "predictor": gt_col,
                        "rho_err_vs_n": rho,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_pairs": len(ns),
                        "n_used_min": float(ns.min()) if len(ns) else float("nan"),
                        "n_used_max": float(ns.max()) if len(ns) else float("nan"),
                        "note": note,
                    }
                )
    return pd.DataFrame(rows)


def _plot_err_vs_n(
    loaded: dict[str, pd.DataFrame], gt_col: str, figures_dir: Path, name: str
) -> None:
    """Per-pair absolute error against ``n_used``, one colour per method.

    The visual companion to :func:`n_sensitivity_table`: a downward trend for vMF but
    not the cosine baseline is the signature of a sample-size-driven deficit rather
    than a weakness of the approach.
    """
    palette = _method_palette()
    fig, ax = plt.subplots()
    plotted = False
    for method in _METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        *_, score_col = _METHODS[method]
        pair = df[[score_col, gt_col, N_USED_COL]].dropna()
        if pair.empty:
            continue
        ax.scatter(
            pair[N_USED_COL],
            (pair[score_col] - pair[gt_col]).abs(),
            s=12,
            alpha=0.6,
            color=palette[method],
            label=method,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("n used (post-downsample corpus size)")
    ax.set_ylabel(f"|score - {gt_col}|")
    ax.legend(title="method", fontsize="small")
    save_fig(fig, figures_dir, name)


def _plot_rho_by_scheme(
    corr_df: pd.DataFrame, gt_col: str, figures_dir: Path, name: str
) -> None:
    """Dot plot of Spearman rho against comparison scheme, one colour per method.

    x = scheme, y = rho, colour = method, CI error bars -- the dot-plot convention
    used across the scored figures (not a heatmap).
    """
    sub = corr_df[corr_df["predictor"] == gt_col].copy()
    if sub.empty:
        return
    palette = _method_palette()
    methods = [m for m in _METHOD_ORDER if m in sub["method"].unique()]
    schemes = sorted(sub["scheme"].unique())

    fig, ax = plt.subplots()
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        by_scheme_df = (
            sub[sub["method"] == method]
            .set_index("scheme")
            .reindex(schemes)
            .reset_index()
        )
        xs = [
            schemes.index(s) + (i - (len(methods) - 1) / 2) * width
            for s in by_scheme_df["scheme"]
        ]
        yerr = [
            by_scheme_df["spearmanr"] - by_scheme_df["ci_low"],
            by_scheme_df["ci_high"] - by_scheme_df["spearmanr"],
        ]
        ax.errorbar(
            xs,
            by_scheme_df["spearmanr"],
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=3,
            color=palette[method],
            label=method,
        )
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
        *_, score_col = _METHODS[method]
        for measure, gt_col in GT_SHIFT_COLS.items():
            suffix = gt_col.removeprefix("gt_shift_")
            score_scatter(
                df,
                gt_col,
                f"Ground-truth shift, {MEASURE_LABELS[measure]}",
                score_col,
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
        _plot_err_vs_n(
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
