"""Descriptive analysis of the raw simulated corpora.

For each simulated corpus (one ``(lemma, pos, k, offset)`` variant) we describe the
sense distribution two ways and compare them:

* **empirical** -- computed from the sampled CSV rows (what the corpus actually
  contains);
* **theoretical** -- the Zipfian design stored in the ``.meta.json`` sidecar
  (``sense_probs`` / ``entropy_bits``), i.e. what the sampler was aiming for.

The gap between the two is the sampling noise of a finite draw, summarised per corpus
by the Jensen-Shannon divergence between the empirical and theoretical sense
distributions.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore
from scipy.stats import entropy  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_csv, write_table
from data_processing.wic_conversion import Corpus, iter_corpora

logger = logging.getLogger("div")


def _corpus_row(corpus: Corpus) -> dict | None:
    """Build one per-corpus record, or ``None`` if the meta sidecar is missing."""
    if not corpus.meta_path.exists():
        logger.warning(
            "%s %s: no .meta.json, skipping", corpus.lemma_pos, corpus.csv_path.stem
        )
        return None

    meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(corpus.csv_path)

    # Empirical sense distribution from the sampled rows.
    emp_counts = df["sense"].value_counts()
    emp_probs = emp_counts / emp_counts.sum()
    emp_entropy = float(entropy(emp_probs.to_numpy(), base=2))

    # Theoretical distribution from the simulation design.
    theo_probs = meta["sense_probs"]  # dict: sense_id -> prob

    # Align both distributions on the union of sense ids (a designed sense may be
    # absent from a small sample). JS divergence is finite even with zero entries.
    senses = sorted(set(theo_probs) | set(emp_probs.index))
    emp_vec = np.array([emp_probs.get(s, 0.0) for s in senses])
    theo_vec = np.array([theo_probs.get(s, 0.0) for s in senses])
    js = float(jensenshannon(emp_vec, theo_vec, base=2))  # 0 == identical

    return {
        "lemma_pos": corpus.lemma_pos,
        "k": corpus.k,
        "offset": corpus.offset,
        "n_examples": int(len(df)),
        "n_senses_observed": int(emp_counts.size),
        "n_senses_available": int(meta["n_senses_available"]),
        "top_sense_share": float(emp_probs.max()),
        "rare_sense_share": float(emp_probs.min()),
        "entropy_empirical": emp_entropy,
        "entropy_theoretical": float(meta["entropy_bits"]),
        "js_divergence": js,
        "baseline_slope": float(meta["baseline_slope"]),
        "applied_slope": float(meta["applied_slope"]),
        "clamped": bool(meta["clamped"]),
    }


def _plot_entropy_vs_slope(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Empirical vs theoretical entropy against the applied Zipfian slope.

    One line per (kind, k) combination -- i.e. empirical and theoretical curves for
    each sense count k. The applied slope is lemma-specific, so there is no shared
    x-grid to connect; we aggregate each (k, offset) cell to its mean applied slope (x)
    and mean entropy (y) and draw the lines through those cell means.
    """

    long = per_corpus.melt(
        id_vars=["k", "offset", "applied_slope"],
        value_vars=["entropy_empirical", "entropy_theoretical"],
        var_name="kind",
        value_name="entropy_bits",
    )
    long["kind"] = long["kind"].str.replace("entropy_", "", regex=False)

    # Collapse each (k, kind, offset) cell to a single point: mean applied slope on x,
    # mean entropy on y. This gives every (kind, k) line a shared, ordered x-sequence.
    cells = long.groupby(["k", "kind", "offset"], as_index=False).agg(
        applied_slope=("applied_slope", "mean"),
        entropy_bits=("entropy_bits", "mean"),
    )

    grid = sns.relplot(
        data=cells,
        x="applied_slope",
        y="entropy_bits",
        hue="k",
        style="kind",
        kind="line",
        markers=True,
        sort=True,
    )
    grid.set_axis_labels("Applied Zipfian slope", "Sense entropy (bits)")
    save_fig(grid.figure, figures_dir, "entropy_vs_slope")


def _plot_n_examples(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Total examples contributed per sense count k (one bar per k)."""
    fig, ax = plt.subplots()
    sns.barplot(
        data=per_corpus,
        x="k",
        y="n_examples",
        estimator="sum",
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Number of senses (k)")
    ax.set_ylabel("Total examples")
    save_fig(fig, figures_dir, "n_examples_per_k")


# How many randomly-chosen example corpora the rank-frequency figure shows, and the
# seed for that choice (fixed so the figure is reproducible across runs).
_RANK_FREQ_N_EXAMPLES = 6
_RANK_FREQ_SEED = 0


def _draw_rank_frequency(ax, corpus: Corpus, row: pd.Series) -> None:
    """Draw one corpus's empirical-vs-theoretical rank-frequency curve onto ``ax``."""
    meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(corpus.csv_path)

    emp = (df["sense"].value_counts() / len(df)).sort_values(ascending=False)
    theo = pd.Series(meta["sense_probs"]).sort_values(ascending=False)

    ax.plot(range(1, len(emp) + 1), emp.to_numpy(), "o-", label="empirical")
    ax.plot(range(1, len(theo) + 1), theo.to_numpy(), "s--", label="theoretical")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"{row['lemma_pos']} k{row['k']} "
        f"slope {row['applied_slope']:.2f} (offset {row['offset']:+.2f})",
        fontsize="small",
    )


def _stratified_sample(per_corpus: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample ``n`` corpora spread roughly evenly across the sense-count ``k`` values.

    A plain random draw is dominated by the most common k, so we split the budget across
    the distinct k groups (largest groups absorb the remainder) and sample within each,
    seeded for reproducibility. Returns up to ``n`` rows, sorted by k for a tidy layout.
    """
    groups = dict(tuple(per_corpus.groupby("k")))
    n_groups = len(groups)
    base, extra = divmod(n, n_groups)
    # Give the leftover slots to the largest groups so quotas stay satisfiable.
    by_size = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
    quota = {k: base + (1 if i < extra else 0) for i, k in enumerate(by_size)}

    parts = [
        grp.sample(n=min(quota[k], len(grp)), random_state=_RANK_FREQ_SEED)
        for k, grp in groups.items()
    ]
    return pd.concat(parts).sort_values(["k", "lemma_pos", "offset"])


def _plot_rank_frequency(per_corpus: pd.DataFrame, data_dir: Path, figures_dir: Path) -> None:
    """Rank-frequency (Zipf) plots for several randomly-selected corpora: empirical vs theory.

    Draws up to :data:`_RANK_FREQ_N_EXAMPLES` corpora as small multiples so the figure
    is representative rather than reflecting a single hand-picked corpus. The sample is
    stratified across the sense-count k (see :func:`_stratified_sample`) and seeded
    (:data:`_RANK_FREQ_SEED`) for reproducibility.
    """
    n = min(_RANK_FREQ_N_EXAMPLES, len(per_corpus))
    sample = _stratified_sample(per_corpus, n)
    n = len(sample)  # stratified quotas may yield slightly fewer if a k group is small

    # Index corpora on disk by (lemma_pos, k, offset) so each sampled row finds its file
    # without re-walking the tree per example.
    by_key = {(c.lemma_pos, c.k, c.offset): c for c in iter_corpora(data_dir)}

    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, (_, row) in zip(flat, sample.iterrows()):
        corpus = by_key[(row["lemma_pos"], row["k"], row["offset"])]
        _draw_rank_frequency(ax, corpus, row)

    for ax in flat[n:]:  # hide any unused cells in the final row
        ax.set_visible(False)

    # Shared axis labels and a single legend keep the small multiples uncluttered.
    fig.supxlabel("Sense rank")
    fig.supylabel("Probability")
    fig.tight_layout()
    # Legend below the grid so it never collides with a panel title.
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.03))
    save_fig(fig, figures_dir, "rank_frequency_examples")


def analyse_raw_simulated(data_dir: Path, out_root: Path) -> None:
    """Run the raw-simulated-data analysis, writing tables and figures to ``out_root``."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    rows = [r for c in iter_corpora(data_dir) if (r := _corpus_row(c)) is not None]
    if not rows:
        logger.warning(
            "No simulated corpora with metadata found under %s; nothing to analyse. "
            "Run simulate_data.py first.",
            data_dir,
        )
        return

    per_corpus = pd.DataFrame(rows).sort_values(["lemma_pos", "k", "offset"])
    # The full per-corpus table is wide and long -- useful as data, but unwieldy as
    # Markdown/LaTeX -- so save it as CSV only.
    write_csv(per_corpus, tables_dir, "raw_per_corpus")

    # Per-(lemma, pos) sub-tables are small enough to render in all formats; drop the
    # now-redundant lemma_pos column and write each into its own sub-directory.
    per_lemma_dir = tables_dir / "per_lemma_pos"
    for lemma_pos, group in per_corpus.groupby("lemma_pos"):
        write_table(
            group.drop(columns="lemma_pos"),
            per_lemma_dir,
            str(lemma_pos),
        )

    # Group by the offset (the regular design grid); report the mean applied slope in
    # the cell too, since the applied slope is lemma-specific and so varies within a
    # cell. Both the offset and the (mean) actual slope are thus available in the table.
    summary = (
        per_corpus.groupby(["k", "offset"], as_index=False)
        .agg(
            n_corpora=("n_examples", "size"),
            mean_applied_slope=("applied_slope", "mean"),
            mean_n_examples=("n_examples", "mean"),
            mean_entropy_empirical=("entropy_empirical", "mean"),
            mean_entropy_theoretical=("entropy_theoretical", "mean"),
            mean_js_divergence=("js_divergence", "mean"),
        )
    )
    write_table(summary, tables_dir, "raw_summary")

    _plot_entropy_vs_slope(per_corpus, figures_dir)
    _plot_n_examples(per_corpus, figures_dir)
    _plot_rank_frequency(per_corpus, data_dir, figures_dir)

    logger.info(
        "raw_simulated: analysed %d corpora across %d (k, offset) cells",
        len(per_corpus),
        len(summary),
    )
