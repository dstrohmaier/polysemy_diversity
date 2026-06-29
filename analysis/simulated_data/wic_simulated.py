"""Descriptive analysis of the WiC-converted simulated corpora.

Each corpus has a ``.data`` sibling: a JSON array of sentence-pair examples produced
by ``convert_simulated_corpora``. Every pair carries a gold ``label`` -- ``1`` if the
two occurrences share a sense, ``0`` if they differ. Here we report, per corpus and in
aggregate, how the pairs split between same-sense and different-sense, plus the overall
data size.
"""

import json
import logging
from pathlib import Path

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_csv, write_table
from data_processing.wic_conversion import Corpus, iter_corpora

logger = logging.getLogger("div")


def _corpus_row(corpus: Corpus) -> dict | None:
    """Build one per-corpus record, or ``None`` if the ``.data`` file is missing."""
    if not corpus.data_path.exists():
        logger.warning(
            "%s %s: no .data file, skipping", corpus.lemma_pos, corpus.csv_path.stem
        )
        return None

    pairs = json.loads(corpus.data_path.read_text(encoding="utf-8"))
    n_pairs = len(pairs)
    n_same = sum(1 for p in pairs if p["label"] == 1)
    n_diff = n_pairs - n_same

    # The sense-distribution entropy (theoretical Zipfian design) and the Zipfian
    # slopes live in the corpus meta sidecar; pull them in so the same-sense rate can be
    # related to sense diversity, and so plots/tables can report the actual (applied)
    # slope alongside the nominal offset.
    entropy_bits = float("nan")
    baseline_slope = float("nan")
    applied_slope = float("nan")
    if corpus.meta_path.exists():
        meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
        entropy_bits = float(meta["entropy_bits"])
        baseline_slope = float(meta["baseline_slope"])
        applied_slope = float(meta["applied_slope"])

    return {
        "lemma_pos": corpus.lemma_pos,
        "k": corpus.k,
        "offset": corpus.offset,
        "baseline_slope": baseline_slope,
        "applied_slope": applied_slope,
        "n_pairs": n_pairs,
        "n_same": n_same,
        "n_diff": n_diff,
        "same_fraction": (n_same / n_pairs) if n_pairs else float("nan"),
        "entropy_bits": entropy_bits,
    }


def _plot_same_vs_diff_counts(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Same- vs different-sense pair counts against the applied slope, faceted by k.

    One point per corpus at its own ``applied_slope`` (baseline + offset). The applied
    slope is lemma-specific, so corpora do not share x-positions; this is a scatter
    rather than a grouped bar over the offset grid.
    """
    long = per_corpus.melt(
        id_vars=["k", "applied_slope"],
        value_vars=["n_same", "n_diff"],
        var_name="kind",
        value_name="count",
    )
    long["kind"] = long["kind"].str.replace("n_", "", regex=False)

    grid = sns.relplot(
        data=long,
        x="applied_slope",
        y="count",
        hue="kind",
        col="k",
        kind="scatter",
    )
    grid.set_axis_labels("Applied Zipfian slope", "Pairs")
    save_fig(grid.figure, figures_dir, "same_vs_diff_counts")


def _plot_same_fraction(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Same-sense fraction against the applied slope, hued by k (one point per corpus)."""
    grid = sns.relplot(
        data=per_corpus,
        x="applied_slope",
        y="same_fraction",
        hue="k",
        kind="scatter",
    )
    grid.set_axis_labels("Applied Zipfian slope", "Same-sense fraction")
    save_fig(grid.figure, figures_dir, "same_fraction_vs_slope")


def _plot_same_fraction_vs_entropy(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Same-sense fraction against sense-distribution entropy (one point per corpus).

    More diverse corpora (higher entropy) should pair fewer occurrences of the same
    sense, so same-sense fraction is expected to fall as entropy rises.
    """
    grid = sns.relplot(
        data=per_corpus,
        x="entropy_bits",
        y="same_fraction",
        hue="k",
        kind="scatter",
    )
    grid.set_axis_labels("Sense entropy (bits)", "Same-sense fraction")
    save_fig(grid.figure, figures_dir, "same_fraction_vs_entropy")


def analyse_wic_simulated(data_dir: Path, out_root: Path) -> None:
    """Run the WiC-simulated-data analysis, writing tables and figures to ``out_root``."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    rows = [r for c in iter_corpora(data_dir) if (r := _corpus_row(c)) is not None]
    if not rows:
        logger.warning(
            "No .data files found under %s; nothing to analyse. Run the WiC "
            "conversion (convert_simulated_corpora via simulate_data.py) first.",
            data_dir,
        )
        return

    per_corpus = pd.DataFrame(rows).sort_values(["lemma_pos", "k", "offset"])
    # The full per-corpus table is long -- useful as data, but unwieldy as
    # Markdown/LaTeX -- so save it as CSV only.
    write_csv(per_corpus, tables_dir, "wic_per_corpus")

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
            n_corpora=("n_pairs", "size"),
            mean_applied_slope=("applied_slope", "mean"),
            n_pairs=("n_pairs", "sum"),
            n_same=("n_same", "sum"),
            n_diff=("n_diff", "sum"),
            mean_same_fraction=("same_fraction", "mean"),
        )
    )
    write_table(summary, tables_dir, "wic_summary")

    _plot_same_vs_diff_counts(per_corpus, figures_dir)
    _plot_same_fraction(per_corpus, figures_dir)
    _plot_same_fraction_vs_entropy(per_corpus, figures_dir)

    logger.info(
        "wic_simulated: %d corpora, %d pairs total (%d same, %d diff)",
        len(per_corpus),
        int(per_corpus["n_pairs"].sum()),
        int(per_corpus["n_same"].sum()),
        int(per_corpus["n_diff"].sum()),
    )
