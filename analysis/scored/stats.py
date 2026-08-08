"""Shared statistics helpers for the shift-in-diversity comparative analysis.

Builds the ground-truth diversity shifts for each corpus pair from the stored
``sense_probs`` sidecars, correlates each method's per-pair log-ratio against them
with Spearman's rho and bootstrap CIs, and draws the per-pair score scatter.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import bootstrap, spearmanr  # type: ignore

from analysis.io import save_fig
from data_processing.shared_loading import CorpusHandle
from data_processing.simulation_loading import load_sim_corpora
from simulation.diversity import STANDARD_ORDERS, diversity_shift

logger = logging.getLogger("div")

# Ground-truth shift column per Hill order, e.g. "gt_shift_q0". These are the
# targets each method's log-ratio score is correlated against.
GT_SHIFT_COLS = {q: f"gt_shift_q{q}" for q in STANDARD_ORDERS}

# How a dataset directory is walked to reach the corpora's .meta.json sidecars. Both
# evaluations store the ground-truth distribution under the same ``sense_probs`` key,
# so swapping the iterator is all the diachronic evaluation needs.
CorpusIterator = Callable[[Path], Iterable[CorpusHandle]]

# Fixed generators so bootstrap CIs are reproducible across runs.
_BOOT_KW = dict(n_resamples=1000, vectorized=False, paired=True, method="percentile")


def _sense_probs_lookup(
    sim_dir: Path, iter_fn: CorpusIterator = load_sim_corpora
) -> dict[tuple[str, str], dict[str, float]]:
    """Map ``(lemma_pos, variant_stem)`` -> that corpus's ``sense_probs``.

    Read from each corpus's ``.meta.json`` sidecar; the diversity ground truth is a
    pure function of these probabilities (no re-simulation, nothing persisted).

    For the simulation these are the Zipfian *design* probabilities; for DWUG
    (``iter_fn=load_dwug_corpora``) they are the grouping's empirical cluster
    distribution, taken over the full grouping before any downsampling.
    """
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for corpus in iter_fn(sim_dir):
        if not corpus.meta_path.exists():
            continue
        meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
        lookup[(corpus.lemma_pos, corpus.csv_path.stem)] = meta["sense_probs"]
    return lookup


def pair_ground_truth(
    pair_scores: pd.DataFrame, sim_dir: Path, iter_fn: CorpusIterator = load_sim_corpora
) -> pd.DataFrame:
    """Attach ground-truth diversity shifts to a method's pair-score rows.

    For each row -- keyed by ``(lemma_pos, source_variant, target_variant)`` -- adds
    one ``gt_shift_q{q}`` column per Hill order (q in {0, 1, 2}), each equal to
    ``log(qD(target) / qD(source))`` computed from the two corpora's stored
    ``sense_probs``. Rows whose variants lack a ``.meta.json`` get NaN targets and
    are logged.

    ``iter_fn`` selects the dataset layout: the simulated ``k*_offset_*`` grid by
    default, or :func:`~data_processing.dwug_loading.load_dwug_corpora` for the
    diachronic evaluation, whose variant stems are ``g1``/``g2``.
    """
    lookup = _sense_probs_lookup(sim_dir, iter_fn)
    out = pair_scores.copy()
    for q, col in GT_SHIFT_COLS.items():
        values = []
        missing = 0
        for lemma_pos, s_var, t_var in zip(
            out["lemma_pos"], out["source_variant"], out["target_variant"]
        ):
            probs_s = lookup.get((lemma_pos, s_var))
            probs_t = lookup.get((lemma_pos, t_var))
            if probs_s is None or probs_t is None:
                values.append(np.nan)
                missing += 1
            else:
                values.append(diversity_shift(probs_s, probs_t, q))
        out[col] = values
        if missing:
            logger.warning(
                "%d/%d pair rows had no matching .meta.json for q=%d (NaN target)",
                missing, len(out), q,
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


def correlation_table(
    df: pd.DataFrame,
    score_col: str,
    predictors: list[str],
    group_col: str = "k_senses",
) -> pd.DataFrame:
    """Spearman correlation of ``score_col`` vs each predictor, conditional on ``group_col``.

    One row per ``(group, predictor)`` with the rho, its bootstrap CI, the group size,
    and a ``note`` explaining any NaN rho. Predictor columns with NaNs (e.g. an
    unmatched ground-truth shift) are dropped pairwise.

    Some (group, predictor) cells are undefined *by construction* rather than for
    lack of data: e.g. the q=0 (richness) shift is identically 0 for every pair that
    shares k, so its column is constant and Spearman is undefined. Those are detected
    up front -- a constant predictor never reaches ``spearmanr`` (which would emit a
    scipy DegenerateDataWarning) -- and marked ``note="constant predictor"`` so the
    table and figures distinguish them from the small-sample ``note="n<3"`` case.
    """
    rows = []
    for group_val, sub in df.groupby(group_col):
        for predictor in predictors:
            pair = sub[[score_col, predictor]].dropna()
            xs = pair[score_col].to_numpy()
            ys = pair[predictor].to_numpy()
            note = ""
            if len(xs) < 3:
                rho = lo = hi = float("nan")
                n = len(xs)
                note = "n<3"
            elif np.ptp(ys) == 0 or np.ptp(xs) == 0:
                # A constant score or predictor makes the rank correlation undefined;
                # skip it explicitly rather than let spearmanr return NaN with a warning.
                rho = lo = hi = float("nan")
                n = len(xs)
                note = "constant predictor" if np.ptp(ys) == 0 else "constant score"
            else:
                rho, lo, hi, n = spearman_with_ci(xs, ys)
            if note:
                logger.warning(
                    "%s=%s vs %s: correlation undefined (%s, n=%d)",
                    group_col, group_val, predictor, note, n,
                )
            rows.append(
                {
                    group_col: group_val,
                    "predictor": predictor,
                    "spearmanr": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def k_palette(df: pd.DataFrame) -> dict[int, tuple]:
    """Fixed colour per sense count k (ascending), shared across the scored figures.

    A dict palette forces seaborn's discrete hue mapping (a bare numeric hue column
    gets a continuous colormap) and pins each k to one colour in every figure. The
    ``colorblind`` palette passes CVD-separation checks for the small k grids used
    here.
    """
    ks = sorted(df["k_senses"].unique())
    return dict(zip(ks, sns.color_palette("colorblind", len(ks))))


def score_scatter(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    score_col: str,
    score_label: str,
    figures_dir: Path,
    name: str,
    hue_col: str = "k_senses",
) -> None:
    """One dot per row: ``score_col`` against ``x_col``, hued by ``hue_col``.

    Per-corpus modes hue by sense count k (the default); the shift-comparison mode
    passes ``hue_col="scheme"`` to colour by comparison scheme instead. A numeric
    hue (k) gets the fixed k-palette; a categorical hue falls back to seaborn's
    default discrete mapping.
    """
    palette = k_palette(df) if hue_col == "k_senses" else None
    grid = sns.relplot(
        data=df,
        x=x_col,
        y=score_col,
        hue=hue_col,
        palette=palette,
        kind="scatter",
    )
    grid.set_axis_labels(x_label, score_label)
    save_fig(grid.figure, figures_dir, name)
