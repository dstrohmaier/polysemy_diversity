"""Analysis of vMF scoring output (``vmf_scores.csv``).

Relates the fitted vMF concentration (``vmf_kappa``) to each corpus's known
ground-truth properties: how strongly kappa tracks sense entropy and Zipfian slope
(conditional on the sense count k), and how kappa varies over the (slope, k) design
grid.
"""

import logging
from pathlib import Path

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from analysis.io import save_fig, write_csv, write_table
from analysis.scored.stats import (
    correlation_table,
    merge_entropy,
    score_grid,
    score_heatmap,
)

logger = logging.getLogger("div")

SCORE_COL = "vmf_kappa"
PREDICTORS = ["entropy_bits", "applied_slope"]


def _plot_kappa_vs_entropy(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """vMF kappa against sense entropy, one point per corpus, hued by k."""
    grid = sns.relplot(
        data=per_corpus,
        x="entropy_bits",
        y=SCORE_COL,
        hue="k_senses",
        kind="scatter",
    )
    grid.set_axis_labels("Sense entropy (bits)", "vMF concentration (kappa)")
    save_fig(grid.figure, figures_dir, "vmf_kappa_vs_entropy")


def analyse_vmf_scored(scores_dir: Path, sim_dir: Path, out_root: Path) -> None:
    """Analyse vMF scores in ``scores_dir`` against the corpora in ``sim_dir``."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"

    scores_path = scores_dir / "vmf" / "vmf_scores.csv"
    if not scores_path.exists():
        logger.warning(
            "No vmf_scores.csv at %s; nothing to analyse. Run score_data.py vmf first.",
            scores_path,
        )
        return

    per_corpus = merge_entropy(pd.read_csv(scores_path), sim_dir)
    per_corpus["lemma_pos"] = per_corpus["word"] + "_" + per_corpus["pos"]
    per_corpus = per_corpus.sort_values(["lemma_pos", "k_senses", "offset"])

    # Full per-corpus table is wide -- CSV only; small per-(lemma, pos) tables in all
    # formats (mirrors the descriptive modes).
    write_csv(per_corpus, tables_dir, "vmf_per_corpus")
    per_lemma_dir = tables_dir / "per_lemma_pos"
    for lemma_pos, group in per_corpus.groupby("lemma_pos"):
        write_table(group.drop(columns="lemma_pos"), per_lemma_dir, str(lemma_pos))

    corr = correlation_table(per_corpus, SCORE_COL, PREDICTORS)
    write_table(corr, tables_dir, "vmf_correlations", convert_col_names=True)

    grid = score_grid(per_corpus, SCORE_COL)
    write_table(grid, tables_dir, "vmf_kappa_grid", index=True)
    score_heatmap(grid, figures_dir, "vmf_kappa_heatmap", cbar_label="vMF kappa")

    _plot_kappa_vs_entropy(per_corpus, figures_dir)

    logger.info(
        "vmf_scored: analysed %d scored corpora across %d sense counts",
        len(per_corpus),
        per_corpus["k_senses"].nunique(),
    )
