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

from analysis.io import human_col_name, save_fig
from data_processing.shared_loading import CorpusHandle
from data_processing.simulation_loading import load_sim_corpora
from simulation.diversity import (
    EVENNESS_KEY,
    STANDARD_ORDERS,
    diversity_shift,
    evenness_shift,
)

logger = logging.getLogger("div")

# Ground-truth shift column per measure: one per Hill order ("gt_shift_q0" ...) plus
# the evenness ratio E = 1D/0D ("gt_shift_evenness"). These are the targets each
# method's log-ratio score is correlated against. Keys are the integer q for the Hill
# orders and EVENNESS_KEY for evenness; anything iterating these must not assume int.
GT_SHIFT_COLS = {q: f"gt_shift_q{q}" for q in STANDARD_ORDERS} | {
    EVENNESS_KEY: f"gt_shift_{EVENNESS_KEY}"
}

# Human-readable measure names for figure labels, keyed as GT_SHIFT_COLS.
MEASURE_LABELS = {
    0: "richness (q=0)",
    1: "Shannon (q=1)",
    2: "Simpson (q=2)",
    EVENNESS_KEY: "evenness (E=1D/0D)",
}

# How a dataset directory is walked to reach the corpora's .meta.json sidecars. Both
# evaluations store the ground-truth distribution under the same ``sense_probs`` key,
# so swapping the iterator is all the diachronic evaluation needs.
CorpusIterator = Callable[[Path], Iterable[CorpusHandle]]

# Fixed generators so bootstrap CIs are reproducible across runs.
_BOOT_KW = dict(n_resamples=1000, vectorized=False, paired=True, method="percentile")

# The post-downsample corpus size each scorer records per pair (equal for source and
# target by construction -- see simulation.pairing.equalise_indices).
N_USED_COL = "n_used"

# Normalised PoS tag per raw suffix of a ``lemma_pos`` identifier. The simulation
# writes uppercase Universal-style tags ("act_NOUN") and DWUG writes lowercase
# Penn-style ones ("graft_nn"), so a pooled analysis needs one vocabulary. Uppercase
# is the target because analysis.naming.human_col_name passes an all-caps token
# through unchanged, whereas "noun" would be title-cased to "Noun" in the tables.
POS_NORMALISATION = {
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "nn": "NOUN",
    "vb": "VERB",
    "jj": "ADJ",
    "rb": "ADV",
}

# Order the PoS appear in tables and on figure x-axes: descending vocabulary size
# (noun 100, verb 100, adj 35, adv 30), which is how the readme frames the study.
# Alphabetical order would lead with the two smallest vocabularies.
POS_ORDER = ["NOUN", "VERB", "ADJ", "ADV"]

UNKNOWN_POS = "UNKNOWN"


def pos_from_lemma(lemma_pos: str) -> str:
    """Normalised PoS tag of a ``lemma_pos`` identifier.

    Splits on the final underscore -- the same parse used by
    :mod:`data_processing.dwug_conversion` -- because a lemma may itself contain one
    ("take_off_VERB"). The suffix is then mapped onto the shared uppercase vocabulary
    via :data:`POS_NORMALISATION`.

    An unrecognised tag returns :data:`UNKNOWN_POS` rather than raising: a pooled run
    over four datasets must not abort because one lemma directory was named oddly, and
    an ``UNKNOWN`` row stays visible in the output (its own x-axis slot and table rows)
    instead of being silently merged into a real PoS.
    """
    if not isinstance(lemma_pos, str) or "_" not in lemma_pos:
        return UNKNOWN_POS
    return POS_NORMALISATION.get(lemma_pos.rsplit("_", 1)[1], UNKNOWN_POS)


def add_pos_column(df: pd.DataFrame, col: str = "pos") -> pd.DataFrame:
    """Return a copy of ``df`` with a normalised PoS column derived from ``lemma_pos``.

    Logs once per unrecognised tag actually seen, so a mis-tagged dataset is loud
    without emitting one warning per row.
    """
    assert "lemma_pos" in df.columns, f"pair scores lack 'lemma_pos': {list(df.columns)}"
    out = df.copy()
    out[col] = out["lemma_pos"].map(pos_from_lemma)
    unknown = sorted(
        {
            s.rsplit("_", 1)[-1] if isinstance(s, str) else s
            for s in out.loc[out[col] == UNKNOWN_POS, "lemma_pos"]
        }
    )
    if unknown:
        logger.warning("Unrecognised PoS tag(s) %s mapped to %s", unknown, UNKNOWN_POS)
    return out


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
    ``sense_probs``, plus ``gt_shift_evenness`` for ``log(E(T)/E(S))``. Rows whose
    variants lack a ``.meta.json`` get NaN targets and are logged.

    ``iter_fn`` selects the dataset layout: the simulated ``k*_offset_*`` grid by
    default, or :func:`~data_processing.dwug_loading.load_dwug_corpora` for the
    diachronic evaluation, whose variant stems are ``g1``/``g2``.
    """
    lookup = _sense_probs_lookup(sim_dir, iter_fn)
    out = pair_scores.copy()
    for measure, col in GT_SHIFT_COLS.items():
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
            elif measure == EVENNESS_KEY:
                values.append(evenness_shift(probs_s, probs_t))
            else:
                values.append(diversity_shift(probs_s, probs_t, measure))
        out[col] = values
        if missing:
            logger.warning(
                "%d/%d pair rows had no matching .meta.json for %s (NaN target)",
                missing, len(out), col,
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
    group_col: str | list[str] = "k_senses",
) -> pd.DataFrame:
    """Spearman correlation of ``score_col`` vs each predictor, conditional on ``group_col``.

    One row per ``(group, predictor)`` with the rho, its bootstrap CI, the group size,
    and a ``note`` explaining any NaN rho. Predictor columns with NaNs (e.g. an
    unmatched ground-truth shift) are dropped pairwise.

    ``group_col`` may name one column or a list of them. With a list the table gains
    one output column per key -- ``pos`` and ``scheme``, say -- and one row per (key
    combination, predictor); this is what the pooled analysis's two-way PoS x scheme
    breakdown groups on.

    ``n_used`` (the post-downsample corpus size) is summarised alongside each rho over
    the rows that survived the dropna: the vMF bias is a function of n, so a rho is
    only interpretable next to the n it was computed at.

    Some (group, predictor) cells are undefined *by construction* rather than for
    lack of data: e.g. the q=0 (richness) shift is identically 0 for every pair that
    shares k, so its column is constant and Spearman is undefined. Those are detected
    up front -- a constant predictor never reaches ``spearmanr`` (which would emit a
    scipy DegenerateDataWarning) -- and marked ``note="constant predictor"`` so the
    table and figures distinguish them from the small-sample ``note="n<3"`` case.
    """
    assert N_USED_COL in df.columns, f"pair scores lack {N_USED_COL!r}: {list(df.columns)}"
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    missing = [c for c in group_cols if c not in df.columns]
    assert not missing, f"grouping column(s) absent from the frame: {missing}"

    rows = []
    # pandas yields a scalar key for a string groupby and a tuple for a list groupby
    # (even a one-element list), so pass the bare string when there is a single key.
    # That keeps the single-key path -- and its output column order -- as it was.
    by = group_cols if len(group_cols) > 1 else group_cols[0]
    for group_val, sub in df.groupby(by):
        keys = group_val if isinstance(group_val, tuple) else (group_val,)
        group_key = dict(zip(group_cols, keys))
        for predictor in predictors:
            pair = sub[[score_col, predictor, N_USED_COL]].dropna()
            xs = pair[score_col].to_numpy()
            ys = pair[predictor].to_numpy()
            n_used = pair[N_USED_COL]
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
                    "%s vs %s: correlation undefined (%s, n=%d)",
                    ", ".join(f"{c}={v}" for c, v in group_key.items()),
                    predictor, note, n,
                )
            rows.append(
                {
                    **group_key,
                    "predictor": predictor,
                    "spearmanr": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                    "n_used_median": n_used.median(),
                    "n_used_min": n_used.min(),
                    "n_used_max": n_used.max(),
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
    # The hue legend title defaults to the raw column name ("k_senses"); give it
    # the same readable treatment as the table headers.
    if grid.legend is not None:
        grid.legend.set_title(human_col_name(hue_col))
    save_fig(grid.figure, figures_dir, name)
