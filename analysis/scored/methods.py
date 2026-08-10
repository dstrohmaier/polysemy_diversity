"""The scored methods: where their pair scores live, and how to draw them.

Everything in this module is knowledge of :mod:`score_data`'s output layout -- which
subdirectory and filename each method writes, and which column holds its log-ratio --
plus the loading and drawing primitives built on top of that layout. It is kept apart
from :mod:`analysis.scored.stats`, whose remit is the statistics themselves and which
knows nothing about scoring paths.

Both comparative modes import from here and neither imports the other, which is what
keeps the per-PoS analysis (:mod:`analysis.scored.comparative`) and the pooled all-PoS
analysis (:mod:`analysis.scored.pooled`) free of a circular dependency. The import
direction across the whole output stack is::

    naming -> io -> stats -> methods -> {comparative, pooled}
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import human_col_name, save_fig
from analysis.scored.stats import (
    GT_SHIFT_COLS,
    N_USED_COL,
    CorpusIterator,
    pair_ground_truth,
    spearman_with_ci,
)
from data_processing.simulation_loading import load_sim_corpora

logger = logging.getLogger("div")

# Re-exported so a caller loading pair scores needs only this module: the iterator
# selects the on-disk corpus layout and is passed straight through to load_method.
__all__ = [
    "CorpusIterator",
    "GT_COLS",
    "METHODS",
    "METHOD_ORDER",
    "dot_plot_by_group",
    "load_all_methods",
    "load_method",
    "method_palette",
    "n_sensitivity_table",
    "plot_err_vs_n",
    "score_col",
]

# Method name -> (scoring subdir, pair-scores filename, log-ratio column). The
# subdir/filename must match score_data.py's output layout; keeping all three in one
# place stops the subdir drifting out of sync with the filename.
METHODS = {
    "cosine": ("cosine", "cosine_pair_scores.csv", "cosine_log_ratio"),
    "vMF": ("vmf", "vmf_pair_scores.csv", "vmf_log_ratio"),
    "WiC": ("wic", "wic_pair_scores.csv", "wic_log_ratio"),
}
METHOD_ORDER = ["cosine", "vMF", "WiC"]
GT_COLS = list(GT_SHIFT_COLS.values())


def score_col(method: str) -> str:
    """The log-ratio column ``method`` writes into its pair-scores CSV."""
    *_, col = METHODS[method]
    return col


def method_palette() -> dict[str, tuple]:
    """Fixed colour per method, shared across the comparative figures."""
    return dict(zip(METHOD_ORDER, sns.color_palette("colorblind", len(METHOD_ORDER))))


def load_method(
    scores_dir: Path,
    method: str,
    sim_dir: Path,
    iter_fn: CorpusIterator = load_sim_corpora,
) -> pd.DataFrame | None:
    """Load one method's pair scores with ground-truth shift columns attached.

    Returns ``None`` (with a warning) if the method's pair-scores CSV is absent, so
    the comparative analysis degrades gracefully when only some methods have run.
    """
    subdir, filename, _ = METHODS[method]
    path = scores_dir / subdir / filename
    if not path.exists():
        logger.warning(
            "No %s at %s; %s excluded from comparison.", filename, path, method
        )
        return None
    return pair_ground_truth(pd.read_csv(path), sim_dir, iter_fn)


def load_all_methods(
    scores_dir: Path, sim_dir: Path, iter_fn: CorpusIterator = load_sim_corpora
) -> dict[str, pd.DataFrame]:
    """Every method with pair scores under ``scores_dir``, ground truth attached.

    Methods whose CSV is absent are omitted rather than mapped to ``None``, so
    callers can iterate the result without a per-entry emptiness check.
    """
    loaded = {
        m: load_method(scores_dir, m, sim_dir, iter_fn) for m in METHOD_ORDER
    }
    return {m: df for m, df in loaded.items() if df is not None}


def n_sensitivity_table(
    loaded: dict[str, pd.DataFrame],
    group_col: str | list[str] = "scheme",
) -> pd.DataFrame:
    """How each method's per-pair error depends on the corpus size it was scored at.

    Every scorer down-samples a pair to ``n_used = min(n_S, n_T)`` before measuring, so
    a pair's score is computed on however many usages the smaller corpus supplied. The
    vMF estimator's attenuation is a known function of that n (the resultant length is
    biased upward at small n and converges as n grows), while the cosine baseline does
    not share that dependence. If vMF trails the baseline, this table separates "the
    approach is weaker" from "the approach was run below the sample size it needs":
    the former shows a flat error-vs-n relation, the latter a negative one.

    One row per ``(method, group, gt_shift_q)``: Spearman rho of ``|score - gt|``
    against ``n_used``, plus the n range it spans. A negative rho means larger corpora
    gave smaller errors. A constant n (the simulation can fix corpus size within a
    scheme) or fewer than three pairs is reported with a ``note`` rather than dropped,
    matching :func:`~analysis.scored.stats.correlation_table`'s convention.

    ``group_col`` may name one column or a list of them; with a list the table gains
    one output column per key, as the pooled analysis's PoS breakdown needs.
    """
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    rows = []
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        col = score_col(method)
        # One grouping key or several: pandas yields a scalar key for a string
        # groupby and a tuple for a list groupby (even a one-element list), so pass
        # the bare string when there is one key and normalise the key to a dict here.
        by = group_cols if len(group_cols) > 1 else group_cols[0]
        for group_val, sub in df.groupby(by):
            keys = group_val if isinstance(group_val, tuple) else (group_val,)
            group_key = dict(zip(group_cols, keys))
            for gt_col in GT_COLS:
                pair = sub[[col, gt_col, N_USED_COL]].dropna()
                # Absolute deviation from the ground-truth shift: the per-pair error
                # whose size, not direction, is expected to shrink with n.
                err = (pair[col] - pair[gt_col]).abs().to_numpy()
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
                        **group_key,
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


def plot_err_vs_n(
    loaded: dict[str, pd.DataFrame],
    gt_col: str,
    figures_dir: Path,
    name: str,
    point_size: float = 12,
    alpha: float = 0.6,
) -> None:
    """Per-pair absolute error against ``n_used``, one colour per method.

    The visual companion to :func:`n_sensitivity_table`: a downward trend for vMF but
    not the cosine baseline is the signature of a sample-size-driven deficit rather
    than a weakness of the approach.

    ``point_size`` and ``alpha`` default to the per-PoS settings; the pooled analysis
    passes sparser values, since pooling four datasets multiplies the point count to
    the order of thousands per method and the defaults would render as a solid block.
    """
    palette = method_palette()
    fig, ax = plt.subplots()
    plotted = False
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        col = score_col(method)
        pair = df[[col, gt_col, N_USED_COL]].dropna()
        if pair.empty:
            continue
        ax.scatter(
            pair[N_USED_COL],
            (pair[col] - pair[gt_col]).abs(),
            s=point_size,
            alpha=alpha,
            color=palette[method],
            label=method,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("n used (post-downsample corpus size)")
    ax.set_ylabel(f"|score - {human_col_name(gt_col)}|")
    ax.legend(title="Method", fontsize="small")
    save_fig(fig, figures_dir, name)


def dot_plot_by_group(
    ax,
    corr_df: pd.DataFrame,
    group_col: str,
    methods: list[str],
    palette: dict[str, tuple],
    group_order: list[str] | None = None,
    value_col: str = "spearmanr",
    ci_cols: tuple[str, str] = ("ci_low", "ci_high"),
) -> list[str]:
    """Dodged dot plot with CI error bars: x = ``group_col``, one colour per method.

    Draws onto a caller-supplied ``ax`` rather than owning a figure, so the
    single-panel figures and the faceted PoS x scheme grid share one drawing routine.
    Passing an explicit ``group_order`` pins the x-axis vocabulary across panels, so a
    group absent from one panel still occupies its slot; the resolved order is
    returned for callers that need to label or annotate the same positions.

    The caller must filter to a single predictor first (and, when faceting, a single
    panel's worth of rows), so that each (method, group) cell is one row -- the
    reindex below depends on it.

    Cells with a NaN CI (the ``constant predictor`` rows) are left as NaN and skipped
    by matplotlib rather than substituted with 0, which would draw a spurious point
    at rho=0.
    """
    assert not corr_df.duplicated(["method", group_col]).any(), (
        f"dot plot needs one row per (method, {group_col}); "
        "filter to a single predictor first"
    )
    groups = (
        list(group_order)
        if group_order is not None
        else sorted(corr_df[group_col].unique())
    )
    lo_col, hi_col = ci_cols
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        by_group = (
            corr_df[corr_df["method"] == method]
            .set_index(group_col)
            .reindex(groups)
            .reset_index()
        )
        xs = [
            groups.index(g) + (i - (len(methods) - 1) / 2) * width
            for g in by_group[group_col]
        ]
        yerr = [
            by_group[value_col] - by_group[lo_col],
            by_group[hi_col] - by_group[value_col],
        ]
        ax.errorbar(
            xs,
            by_group[value_col],
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=3,
            color=palette[method],
            label=method,
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([human_col_name(g) for g in groups])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    return groups
