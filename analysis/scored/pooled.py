"""Pooled all-PoS comparison of the shift-in-diversity scores.

The simulation runs as four independent datasets -- ``most_diverse_{noun,verb,adj,adv}``
-- and every stage from simulation through scoring to analysis is invoked once per
dataset. Part of speech is therefore a *directory partition*, not a column, and no
existing table answers either of the two questions this module exists for:

* **Does the ranking of the methods survive pooling the lemmata?** Per-PoS correlations
  rest on 30 (adverbs) to 100 (nouns) lemmata, and ``along_k`` is the thin scheme
  throughout. Pooling multiplies the pair count behind each rho by roughly four and
  tightens the bootstrap CIs correspondingly.
* **Does it hold across parts of speech?** The two-way PoS x scheme breakdown says
  whether an apparent PoS effect is really a scheme effect.

This module loads every PoS dataset, tags each row with its normalised PoS, and writes
the pooled rho-by-scheme table and the two-way PoS x scheme table with its faceted
figure.

The error-vs-n diagnostic is deliberately *not* pooled here. It asks whether a
method's error shrinks as the scored corpus grows, and pooling would mix noun pairs
(large ``n_used``) with adverb pairs (small ``n_used``), manufacturing a rho whenever
PoS happens to correlate with corpus size. Each dataset's own ``comparative`` run
answers that question over the pairs it can be asked of.

**Every table here groups by scheme.** That matters: :mod:`simulation.pairing`
deliberately emits the same ``(lemma_pos, source, target)`` triple under two scheme
tags -- a slope- or k-neighbour of the primary corpus appears in both ``primary`` and
``along_slope``/``along_k``, roughly 4.5% of unique pairs. Grouping by scheme keeps
those copies in separate cells, so nothing here double-counts them. A row pooling
*across* schemes would, and would need de-duplicating on that triple first; none is
produced.

Pooling across PoS needs no such care: ``lemma_pos`` carries the PoS suffix, so
``act_NOUN`` and ``act_VERB`` are distinct keys and the datasets have disjoint lemmata.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore

from analysis.io import human_col_name, save_fig, write_csv, write_table
from analysis.scored.methods import (
    GT_COLS,
    METHOD_ORDER,
    METHODS,
    CorpusIterator,
    dot_plot_by_group,
    load_all_methods,
    method_palette,
    score_col,
)
from analysis.scored.stats import (
    GT_SHIFT_COLS,
    N_USED_COL,
    POS_ORDER,
    add_pos_column,
    correlation_table,
)
from data_processing.simulation_loading import load_sim_corpora

logger = logging.getLogger("div")

# A cell computed on fewer than this many pairs is flagged, not dropped. Below roughly
# this many points the bootstrap CI spans most of [-1, 1], so the note tells the reader
# to read the CI rather than the point estimate. It also matches the smallest
# vocabulary in the study (adverbs, 30 lemmata), which is exactly the column a
# drop-thin-cells policy would silently remove from the PoS comparison.
SMALL_N_THRESHOLD = 30
SMALL_N_NOTE = "small n"

POS_COL = "pos"


def discover_pos_datasets(scores_root: Path, corpora_root: Path) -> list[str]:
    """Dataset directory names present under *both* roots and actually scored, sorted.

    A dataset counts when it is a directory under both roots *and* holds at least one
    method's pair-scores CSV. Requiring the scores side is what rules out the corpora
    root's non-dataset entries (``most_diverse_noun/stale/`` is a real example) and
    corpus dirs that were simulated but never scored -- either would otherwise join the
    pool as a silently empty dataset.

    Raises ``NotADirectoryError`` if either root is missing, so a mistyped path is
    reported as itself rather than as an empty result.
    """
    for label, root in (("scores root", scores_root), ("corpus root", corpora_root)):
        if not root.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {root}")

    names = []
    for scores_dir in sorted(p for p in scores_root.iterdir() if p.is_dir()):
        name = scores_dir.name
        if not (corpora_root / name).is_dir():
            continue
        if not any(
            (scores_dir / subdir / filename).exists()
            for subdir, filename, _ in METHODS.values()
        ):
            logger.warning(
                "%s has no method pair-scores under %s; skipping.", name, scores_dir
            )
            continue
        names.append(name)
    return names


def load_pooled(
    scores_root: Path,
    corpora_root: Path,
    datasets: list[str],
    iter_fn: CorpusIterator = load_sim_corpora,
) -> dict[str, pd.DataFrame]:
    """One frame per method, pooling every dataset in ``datasets`` and tagged by PoS.

    ``datasets`` are directory names present under both roots (``most_diverse_noun``
    and friends). Each is loaded with the per-dataset loader, given a ``pos`` column
    derived from its ``lemma_pos`` values and a ``dataset`` column recording where it
    came from, then concatenated. A dataset whose scores are absent is skipped with a
    warning rather than aborting the pool, matching
    :func:`~analysis.scored.methods.load_method`'s graceful degradation.

    PoS is derived per row rather than from the dataset directory name because the
    identifier is the authoritative source: the DWUG corpora mix ``nn`` and ``vb``
    lemmata inside one directory, where a directory-level tag would be wrong.
    """
    parts: dict[str, list[pd.DataFrame]] = {}
    for name in datasets:
        loaded = load_all_methods(
            scores_root / name, corpora_root / name, iter_fn
        )
        if not loaded:
            logger.warning("No method pair-scores for dataset %s; skipping.", name)
            continue
        for method, df in loaded.items():
            tagged = add_pos_column(df, POS_COL)
            tagged["dataset"] = name
            parts.setdefault(method, []).append(tagged)

    pooled = {
        method: pd.concat(frames, ignore_index=True)
        for method, frames in parts.items()
        if frames
    }

    for method in METHOD_ORDER:
        df = pooled.get(method)
        if df is None:
            continue
        logger.info(
            "pooled %s: %d pairs, %d lemmata, PoS %s",
            method,
            len(df),
            df["lemma_pos"].nunique(),
            sorted(df[POS_COL].unique()),
        )

    # A method scored on fewer PoS than its competitors makes the comparison rest on
    # different lemma sets, which the pooled rhos give no hint of. Warn rather than
    # fail: a partially-scored run should still be analysable, and pooled_coverage
    # spells the gap out.
    coverage = {m: frozenset(df[POS_COL].unique()) for m, df in pooled.items()}
    if len(set(coverage.values())) > 1:
        logger.warning(
            "Methods cover different PoS sets (%s); pooled rhos rest on different "
            "lemma sets and are not strictly comparable.",
            {m: sorted(p) for m, p in coverage.items()},
        )
    return pooled


def _group_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Lemma and PoS counts per group, for annotating a correlation table.

    Counted before the per-predictor dropna, so they describe the pairs present in the
    group rather than those with a defined ground truth for one measure; ``n`` in the
    correlation table is the post-dropna count and can be smaller.
    """
    grouped = df.groupby(group_cols)
    return pd.DataFrame(
        {
            "n_lemmata": grouped["lemma_pos"].nunique(),
            "n_pos": grouped[POS_COL].nunique(),
        }
    ).reset_index()


def add_small_n_note(
    df: pd.DataFrame, n_col: str = "n", threshold: int = SMALL_N_THRESHOLD
) -> pd.DataFrame:
    """Append a ``"small n"`` note to cells computed on fewer than ``threshold`` pairs.

    Appended to, not substituted for, any existing note: a cell can be both small and
    degenerate, and ``"n<3; small n"`` is more informative than either alone, while the
    ``n<3`` / ``constant predictor`` vocabulary established by
    :func:`~analysis.scored.stats.correlation_table` stays intact. The row keeps its
    rho and CI -- a wide CI is the honest signal, and dropping thin cells would quietly
    remove exactly the adverb column the PoS comparison exists to show.

    ``n_col`` names the count to test. The correlation tables call it ``n``; other
    tables in the scored analysis count pairs under different names, so the column is
    a parameter rather than hard-coded.
    """
    if df.empty:
        return df
    out = df.copy()
    out["note"] = [
        "; ".join(p for p in (note, SMALL_N_NOTE if n < threshold else "") if p)
        for note, n in zip(out["note"].fillna(""), out[n_col])
    ]
    return out


def pooled_correlation_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman rho per ``(method, scheme, predictor)``, pooled over every PoS.

    The same statistic as
    :func:`~analysis.scored.comparative.shift_correlation_table`, computed on the
    all-PoS frame, plus ``n_lemmata`` and ``n_pos``: a rho over four parts of speech
    and 265 lemmata means something different from one over a single vocabulary, and
    the table should say which it is.
    """
    parts = []
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        corr = correlation_table(df, score_col(method), GT_COLS, group_col="scheme")
        if corr.empty:
            # No groups at all: the method's pair scores were present but held no
            # rows. An empty frame carries none of the columns the merge joins on.
            logger.warning("%s contributed no correlation rows; skipping.", method)
            continue
        corr = corr.merge(_group_counts(df, ["scheme"]), on="scheme", how="left")
        corr.insert(0, "method", method)
        parts.append(corr)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # note last: it is the reader's footnote column, and add_small_n_note appends to
    # it. An all-empty frame carries no columns at all, so there is nothing to move.
    if "note" not in out.columns:
        return out
    return out[[c for c in out.columns if c != "note"] + ["note"]]


def pos_scheme_correlation_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman rho per ``(method, PoS, scheme, predictor)``.

    The two-way breakdown behind the faceted figure: it says whether the method ranking
    established on nouns holds for verbs, adjectives and adverbs, and whether any
    apparent PoS effect is really a scheme effect. Uses ``correlation_table``'s
    multi-key grouping rather than a nested loop, so the note conventions, pairwise
    dropna and n-summaries are identical to every other correlation table here.
    """
    parts = []
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        corr = correlation_table(
            df, score_col(method), GT_COLS, group_col=[POS_COL, "scheme"]
        )
        if corr.empty:
            logger.warning("%s contributed no correlation rows; skipping.", method)
            continue
        counts = _group_counts(df, [POS_COL, "scheme"]).drop(columns="n_pos")
        corr = corr.merge(counts, on=[POS_COL, "scheme"], how="left")
        corr.insert(0, "method", method)
        parts.append(corr)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "note" not in out.columns:
        return out
    return out[[c for c in out.columns if c != "note"] + ["note"]]


def coverage_table(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairs, lemmata and n_used per ``(dataset, pos, method)``.

    The table to read first when a pooled number looks wrong: it says immediately
    whether a PoS is under-represented or a method's scores were missing for one
    dataset.
    """
    rows = []
    for method in METHOD_ORDER:
        df = loaded.get(method)
        if df is None:
            continue
        for (dataset, pos), sub in df.groupby(["dataset", POS_COL]):
            rows.append(
                {
                    "dataset": dataset,
                    POS_COL: pos,
                    "method": method,
                    "n_pairs": len(sub),
                    "n_lemmata": sub["lemma_pos"].nunique(),
                    "n_used_median": sub[N_USED_COL].median(),
                    "n_used_min": sub[N_USED_COL].min(),
                    "n_used_max": sub[N_USED_COL].max(),
                }
            )
    return pd.DataFrame(rows)


def _scheme_order(schemes) -> list[str]:
    """Comparison schemes in a stable display order."""
    return sorted(set(schemes))


def _pos_values(present) -> list[str]:
    """PoS in :data:`~analysis.scored.stats.POS_ORDER`, unexpected tags appended.

    An ``UNKNOWN`` (or any future tag) keeps its own slot at the end rather than being
    dropped from the figure, matching ``pos_from_lemma``'s flag-don't-hide contract.
    """
    present = set(present)
    known = [p for p in POS_ORDER if p in present]
    return known + sorted(present - set(POS_ORDER))


def _plot_pooled_rho_by_scheme(
    corr_df: pd.DataFrame, gt_col: str, figures_dir: Path, name: str
) -> None:
    """Pooled SRC against comparison scheme, one colour per method.

    Deliberately the same shape as the per-PoS ``rho_by_scheme`` figure, so the pooled
    result and a single vocabulary's can be read side by side.
    """
    sub = corr_df[corr_df["predictor"] == gt_col]
    if sub.empty:
        return
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
    schemes = _scheme_order(sub["scheme"])

    fig, ax = plt.subplots()
    dot_plot_by_group(ax, sub, "scheme", methods, method_palette(), group_order=schemes)
    ax.set_xlabel("Comparison scheme")
    ax.set_ylabel("SRC (Spearman's rank correlation)")
    ax.legend(title="Method", fontsize="small")
    save_fig(fig, figures_dir, name)


def _plot_rho_pos_by_scheme(
    corr_df: pd.DataFrame, gt_col: str, figures_dir: Path, name: str
) -> None:
    """One panel per scheme; within a panel x = PoS, colour = method, CI error bars.

    Panels share a y-axis so a method's rho is comparable across schemes by eye -- the
    whole question the figure answers is whether the between-method ordering is stable
    across PoS *and* scheme, which a per-panel autoscale would obscure. The PoS
    vocabulary is pinned identically across panels, so a PoS missing from one scheme
    leaves a visible gap instead of shifting the others along.
    """
    sub = corr_df[corr_df["predictor"] == gt_col]
    if sub.empty:
        return
    schemes = _scheme_order(sub["scheme"])
    pos_values = _pos_values(sub[POS_COL])
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
    palette = method_palette()

    fig, axes = plt.subplots(
        1,
        len(schemes),
        figsize=(3.2 * len(schemes), 3.4),
        sharey=True,
        squeeze=False,
    )
    for ax, scheme in zip(axes[0], schemes):
        dot_plot_by_group(
            ax,
            sub[sub["scheme"] == scheme],
            POS_COL,
            methods,
            palette,
            group_order=pos_values,
        )
        ax.set_title(human_col_name(scheme))

    fig.supxlabel("Part of speech")
    fig.supylabel("SRC (Spearman's rank correlation)")
    # Lay out the panels first, then hand the legend its own reserved strip beneath
    # them. tight_layout knows nothing about a figure-level legend added afterwards,
    # so without reserving the strip the legend would sit on top of the x-label.
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Method",
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.0),
        fontsize="small",
    )
    save_fig(fig, figures_dir, name)


def analyse_comparative_pooled(
    scores_root: Path,
    corpora_root: Path,
    out_root: Path,
    datasets: list[str],
    iter_fn: CorpusIterator = load_sim_corpora,
) -> None:
    """Pooled all-PoS comparison: does the method ranking hold across parts of speech?

    The per-PoS ``comparative`` runs each answer the question within one vocabulary.
    This mode pools them, so the correlations rest on every lemma the simulation
    covers, and adds the PoS x scheme breakdown that says whether a per-PoS difference
    is a PoS effect or a scheme effect.
    """
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    loaded = load_pooled(scores_root, corpora_root, datasets, iter_fn)
    if not loaded:
        logger.warning(
            "No method pair-scores found under %s for datasets %s; nothing to analyse.",
            scores_root,
            datasets,
        )
        return

    corr_df = add_small_n_note(pooled_correlation_table(loaded), "n")
    if corr_df.empty:
        # Pair scores were found but held no rows, so every correlation table below
        # would be an empty frame without even the columns to slice on.
        logger.warning(
            "Pair scores under %s produced no correlation rows; nothing to analyse.",
            scores_root,
        )
        return
    write_table(
        corr_df, tables_dir, "pooled_shift_correlations", convert_col_names=True
    )

    two_way = add_small_n_note(pos_scheme_correlation_table(loaded), "n")
    # The full two-way table is wide for LaTeX (methods x PoS x schemes x measures), so
    # the machine-readable CSV carries all of it and each measure gets its own
    # paper-sized slice.
    write_csv(two_way, tables_dir, "pos_scheme_correlations")
    if not two_way.empty:
        for gt_col in GT_SHIFT_COLS.values():
            suffix = gt_col.removeprefix("gt_shift_")
            measure_slice = two_way[two_way["predictor"] == gt_col].drop(
                columns="predictor"
            )
            if measure_slice.empty:
                continue
            write_table(
                measure_slice,
                tables_dir,
                f"pos_scheme_correlations_{suffix}",
                convert_col_names=True,
            )

    coverage = coverage_table(loaded)
    if not coverage.empty:
        write_table(coverage, tables_dir, "pooled_coverage", convert_col_names=True)

    for gt_col in GT_SHIFT_COLS.values():
        suffix = gt_col.removeprefix("gt_shift_")
        _plot_pooled_rho_by_scheme(
            corr_df,
            gt_col,
            figures_dir / "rho_by_scheme",
            f"pooled_rho_by_scheme_{suffix}",
        )
        _plot_rho_pos_by_scheme(
            two_way,
            gt_col,
            figures_dir / "rho_by_pos",
            f"pos_scheme_rho_{suffix}",
        )

    pos_seen = sorted({p for df in loaded.values() for p in df[POS_COL].unique()})
    lemmata = len({lp for df in loaded.values() for lp in df["lemma_pos"].unique()})
    logger.info(
        "pooled: %d correlation rows and %d two-way rows across %d method(s), "
        "%d PoS (%s), %d lemmata",
        len(corr_df),
        len(two_way),
        len(loaded),
        len(pos_seen),
        ", ".join(pos_seen),
        lemmata,
    )
    if not two_way.empty:
        flagged = int(two_way["note"].str.contains(SMALL_N_NOTE).sum())
        logger.info(
            "pooled: %d/%d two-way cells below n=%d",
            flagged,
            len(two_way),
            SMALL_N_THRESHOLD,
        )


