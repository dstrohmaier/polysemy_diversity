"""Shared statistics and grid helpers for the score-analysis modes.

Both scored modes (vMF, WiC) relate a per-corpus score to the corpus's known
ground-truth properties (sense entropy, Zipfian slope) and summarise scores over the
(slope, k) design grid. This module holds the pieces they share: the entropy lookup
that joins score rows back to the corpus metadata, Spearman correlations and metric
estimates with bootstrap confidence intervals, and the score-grid pivot used for both
the heatmap figure and its companion table.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import bootstrap, spearmanr  # type: ignore

from analysis.io import save_fig
from data_processing.wic_conversion import iter_corpora

logger = logging.getLogger("div")

# Offsets/slopes are rounded before use as a merge/grid key; the simulation grid is
# defined to 4 dp (see _offset_grid in simulation/corpus_simulation.py), so float
# equality only holds after matching that precision.
_ROUND_DP = 4

# Fixed generators so bootstrap CIs are reproducible across runs.
_BOOT_KW = dict(n_resamples=1000, vectorized=False, paired=True, method="percentile")


def entropy_lookup(sim_dir: Path) -> dict[tuple[str, str, int, float], float]:
    """Map ``(word, pos, k_senses, offset)`` -> theoretical ``entropy_bits``.

    The score CSVs do not carry entropy, so we read it from each corpus's
    ``.meta.json`` sidecar under ``sim_dir``. The offset key is derived the same way
    the scorer derives it (``applied_slope - baseline_slope``) and rounded so it joins
    cleanly to the score rows.
    """
    lookup: dict[tuple[str, str, int, float], float] = {}
    for corpus in iter_corpora(sim_dir):
        if not corpus.meta_path.exists():
            continue
        meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
        offset = round(meta["applied_slope"] - meta["baseline_slope"], _ROUND_DP)
        key = (meta["lemma"], meta["pos"], int(meta["k_senses"]), offset)
        lookup[key] = float(meta["entropy_bits"])
    return lookup


def merge_entropy(scores: pd.DataFrame, sim_dir: Path) -> pd.DataFrame:
    """Add an ``entropy_bits`` column to a score frame via :func:`entropy_lookup`."""
    lookup = entropy_lookup(sim_dir)
    keys = zip(
        scores["word"],
        scores["pos"],
        scores["k_senses"].astype(int),
        scores["offset"].round(_ROUND_DP),
    )
    out = scores.copy()
    out["entropy_bits"] = [lookup.get(k, np.nan) for k in keys]
    missing = int(out["entropy_bits"].isna().sum())
    if missing:
        logger.warning(
            "%d/%d score rows had no matching corpus .meta.json (entropy NaN)",
            missing,
            len(out),
        )
    return out


def _spearman_stat(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, seed: int = 0
) -> tuple[float, float, float, int]:
    """Spearman rho with a bootstrap CI. Returns ``(rho, ci_low, ci_high, n)``.

    Needs at least three paired points; below that the correlation (and its CI) is
    undefined, so we return NaNs with the sample size for the caller to report.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    rho = _spearman_stat(x, y)
    res = bootstrap(
        (x, y), _spearman_stat, random_state=np.random.default_rng(seed), **_BOOT_KW
    )
    return rho, float(res.confidence_interval.low), float(res.confidence_interval.high), n


def metric_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    seed: int = 0,
) -> tuple[float, float, float]:
    """A classification metric with a paired bootstrap CI. Returns ``(value, lo, hi)``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    value = float(metric_fn(y_true, y_pred))
    if len(y_true) < 2:
        return value, float("nan"), float("nan")
    res = bootstrap(
        (y_true, y_pred),
        lambda a, b: float(metric_fn(a, b)),
        random_state=np.random.default_rng(seed),
        **_BOOT_KW,
    )
    return value, float(res.confidence_interval.low), float(res.confidence_interval.high)


def correlation_table(
    df: pd.DataFrame,
    score_col: str,
    predictors: list[str],
    group_col: str = "k_senses",
) -> pd.DataFrame:
    """Spearman correlation of ``score_col`` vs each predictor, conditional on ``group_col``.

    One row per ``(group, predictor)`` with the rho, its bootstrap CI, and the group
    size. Predictor columns with NaNs (e.g. unmatched entropy) are dropped pairwise.
    """
    rows = []
    for group_val, sub in df.groupby(group_col):
        for predictor in predictors:
            pair = sub[[score_col, predictor]].dropna()
            rho, lo, hi, n = spearman_with_ci(
                pair[score_col].to_numpy(), pair[predictor].to_numpy()
            )
            if np.isnan(rho):
                logger.warning(
                    "%s=%s vs %s: only %d points, correlation undefined",
                    group_col,
                    group_val,
                    predictor,
                    n,
                )
            rows.append(
                {
                    group_col: group_val,
                    "predictor": predictor,
                    "spearmanr": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def score_grid(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Pivot mean ``score_col`` onto the (k_senses x applied_slope) design grid.

    Slope column labels are formatted to strings so the grid has ordinary text
    headers (float-valued column labels confuse downstream LaTeX styling).
    """
    grid = df.copy()
    grid["applied_slope"] = grid["applied_slope"].round(_ROUND_DP).map("{:.2f}".format)
    return grid.pivot_table(
        index="k_senses", columns="applied_slope", values=score_col, aggfunc="mean"
    )


def score_heatmap(grid: pd.DataFrame, figures_dir: Path, name: str, cbar_label: str) -> None:
    """Render a :func:`score_grid` pivot as a heatmap PDF."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sns.heatmap(grid, annot=True, fmt=".2f", cbar_kws={"label": cbar_label}, ax=ax)
    ax.set_xlabel("Applied Zipfian slope")
    ax.set_ylabel("Number of senses (k)")
    save_fig(fig, figures_dir, name)
