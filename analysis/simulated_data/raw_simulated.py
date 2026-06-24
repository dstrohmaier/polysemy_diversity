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
import seaborn as sns

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


def _plot_entropy_vs_offset(per_corpus: pd.DataFrame, figures_dir: Path) -> None:
    """Empirical vs theoretical entropy across offset, hued by k (side by side)."""

    long = per_corpus.melt(
        id_vars=["k", "offset"],
        value_vars=["entropy_empirical", "entropy_theoretical"],
        var_name="kind",
        value_name="entropy_bits",
    )
    long["kind"] = long["kind"].str.replace("entropy_", "", regex=False)

    grid = sns.relplot(
        data=long,
        x="offset",
        y="entropy_bits",
        hue="k",
        style="kind",
        kind="line",
        markers=True,
        errorbar="sd",
    )
    grid.set_axis_labels("Zipfian slope offset", "Sense entropy (bits)")
    save_fig(grid.figure, figures_dir, "entropy_vs_offset")


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


def _plot_rank_frequency(per_corpus: pd.DataFrame, data_dir: Path, figures_dir: Path) -> None:
    """Rank-frequency (Zipf) plot for one representative corpus: empirical vs theory."""
    # Pick the corpus with the most senses observed as the representative example.
    top = per_corpus.sort_values("n_senses_observed", ascending=False).iloc[0]
    corpus = next(
        c
        for c in iter_corpora(data_dir)
        if c.lemma_pos == top["lemma_pos"]
        and c.k == top["k"]
        and c.offset == top["offset"]
    )
    meta = json.loads(corpus.meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(corpus.csv_path)

    emp = (df["sense"].value_counts() / len(df)).sort_values(ascending=False)
    theo = pd.Series(meta["sense_probs"]).sort_values(ascending=False)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(emp) + 1), emp.to_numpy(), "o-", label="empirical")
    ax.plot(range(1, len(theo) + 1), theo.to_numpy(), "s--", label="theoretical")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sense rank")
    ax.set_ylabel("Probability")
    ax.set_title(f"{top['lemma_pos']} k{top['k']} offset {top['offset']:+.2f}")
    ax.legend()
    save_fig(fig, figures_dir, "rank_frequency_example")


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

    summary = (
        per_corpus.groupby(["k", "offset"], as_index=False)
        .agg(
            n_corpora=("n_examples", "size"),
            mean_n_examples=("n_examples", "mean"),
            mean_entropy_empirical=("entropy_empirical", "mean"),
            mean_entropy_theoretical=("entropy_theoretical", "mean"),
            mean_js_divergence=("js_divergence", "mean"),
        )
    )
    write_table(summary, tables_dir, "raw_summary")

    _plot_entropy_vs_offset(per_corpus, figures_dir)
    _plot_n_examples(per_corpus, figures_dir)
    _plot_rank_frequency(per_corpus, data_dir, figures_dir)

    logger.info(
        "raw_simulated: analysed %d corpora across %d (k, offset) cells",
        len(per_corpus),
        len(summary),
    )
