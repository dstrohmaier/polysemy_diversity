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
import seaborn as sns

from analysis.io import save_fig, write_table
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

    return {
        "lemma_pos": corpus.lemma_pos,
        "k": corpus.k,
        "offset": corpus.offset,
        "n_pairs": n_pairs,
        "n_same": n_same,
        "n_diff": n_diff,
        "same_fraction": (n_same / n_pairs) if n_pairs else float("nan"),
    }


def _plot_same_vs_diff_counts(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Grouped bar of same- vs different-sense pair counts by (k, offset)."""
    long = summary.melt(
        id_vars=["k", "offset"],
        value_vars=["n_same", "n_diff"],
        var_name="kind",
        value_name="count",
    )
    long["kind"] = long["kind"].str.replace("n_", "", regex=False)

    grid = sns.catplot(
        data=long,
        x="offset",
        y="count",
        hue="kind",
        col="k",
        kind="bar",
        errorbar=None,
    )
    grid.set_axis_labels("Zipfian slope offset", "Total pairs")
    save_fig(grid.figure, figures_dir, "same_vs_diff_counts")


def _plot_same_fraction(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Same-sense fraction across offset, hued by k."""
    grid = sns.relplot(
        data=per_corpus,
        x="offset",
        y="same_fraction",
        hue="k",
        kind="line",
        markers=True,
        errorbar="sd",
    )
    grid.set_axis_labels("Zipfian slope offset", "Same-sense fraction")
    save_fig(grid.figure, figures_dir, "same_fraction_vs_offset")


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
    write_table(per_corpus, tables_dir, "wic_per_corpus")

    summary = (
        per_corpus.groupby(["k", "offset"], as_index=False)
        .agg(
            n_corpora=("n_pairs", "size"),
            n_pairs=("n_pairs", "sum"),
            n_same=("n_same", "sum"),
            n_diff=("n_diff", "sum"),
            mean_same_fraction=("same_fraction", "mean"),
        )
    )
    write_table(summary, tables_dir, "wic_summary")

    _plot_same_vs_diff_counts(summary, figures_dir)
    _plot_same_fraction(per_corpus, figures_dir)

    logger.info(
        "wic_simulated: %d corpora, %d pairs total (%d same, %d diff)",
        len(per_corpus),
        int(per_corpus["n_pairs"].sum()),
        int(per_corpus["n_same"].sum()),
        int(per_corpus["n_diff"].sum()),
    )
