import json
from typing import Any
from pathlib import Path

import click
import pandas as pd  # type: ignore
from nltk.metrics.distance import edit_distance  # type: ignore

from data_processing.constants import MIN_EXAMPLES, MIN_SENSES
from data_processing.lwsd_loading import wsd_generator

# The content-word universal POS labels used in the vocab files
# (e.g. source_data/vocabs/target_verbs.json). The WSD annotations already use
# these labels, so no POS mapping is needed.
CONTENT_POS = ["NOUN", "VERB", "ADJ", "ADV"]

# The WSD corpora to count senses from. "semcor_in_original_format" is excluded:
# it is a different format and would duplicate the senses already in "semcor".
WSD_SUBDIRS = ["semcor", "masc"]


def sense_instance_counts(wsd_dir: Path, min_examples: int = MIN_EXAMPLES) -> pd.DataFrame:
    """Count annotated instances per (lemma, pos, sense) across the WSD corpora.

    Reads the SEMCOR and MASC annotations under ``wsd_dir`` and returns a frame of
    ``lemma, pos, sense, n_instances`` -- one row per distinct sense, with how many
    times it is annotated. Only senses with at least ``min_examples`` instances are
    kept: a sense seen once or twice is too sparse to simulate from, so it should not
    contribute to a lemma's diversity ranking.

    This is the basis for both the most-diverse ranking (count distinct senses per
    lemma) and the statistics report (senses per lemma, instances per sense).
    """
    records: list[dict[str, Any]] = []
    for subdir in WSD_SUBDIRS:
        records.extend(wsd_generator(wsd_dir / subdir))
    df = pd.DataFrame(
        records, columns=["lemma", "pos", "sense", "sentence", "start", "end"]
    )

    # Single words only, content POS only.
    df = df[df["pos"].isin(CONTENT_POS) & ~df["lemma"].str.contains("_", na=False)]

    counts = (
        df.groupby(["lemma", "pos", "sense"]).size().reset_index(name="n_instances")
    )
    return counts[counts["n_instances"] >= min_examples]


def _report_near_duplicates(lemmas: list[str], pos: str, max_distance: int = 2) -> None:
    """Print selected-lemma pairs whose Levenshtein distance is < ``max_distance``.

    Flags near-duplicate lemmata within a POS (e.g. inflectional variants or typos
    that survived as separate annotation strings) so they can be reviewed.
    """
    for i, a in enumerate(lemmas):
        for b in lemmas[i + 1 :]:
            distance = edit_distance(a, b)
            if distance < max_distance:
                print(f"  [{pos}] {a!r} ~ {b!r} (levenshtein={distance})")


def _write_statistics(out_path: Path, pos: str, selected: pd.DataFrame) -> None:
    """Write a Markdown statistics report for one POS's selected vocabulary lemmata.

    ``selected`` is the per-(lemma, sense, n_instances) instance-count frame for a
    single POS, restricted to the lemmata written to that POS's vocab file. The report
    has two tables: senses-per-lemma (the diversity ranking) and instances-per-sense
    (the per-sense annotation counts behind it), listing only the selected lemmata.
    """
    # Senses per lemma (ranked, as in the vocab file), plus total instances.
    per_lemma = (
        selected.groupby("lemma")
        .agg(n_senses=("sense", "nunique"), n_instances=("n_instances", "sum"))
        .reset_index()
        .sort_values(["n_senses", "lemma"], ascending=[False, True])
    )

    # Instances per sense, lemmata in the same ranked order.
    per_sense = selected.sort_values(
        ["lemma", "n_instances"], ascending=[True, False]
    )[["lemma", "sense", "n_instances"]]

    sections = [
        f"# {pos} vocabulary statistics\n",
        "## Senses per lemma\n",
        per_lemma.to_markdown(index=False),
        "",
        "## Instances per sense\n",
        per_sense.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _overview_row(pos: str, selected: pd.DataFrame) -> dict:
    """Summary statistics for one POS's selected lemmata, for the overview table.

    ``selected`` is the per-(lemma, sense, n_instances) frame for the lemmata written
    to this POS's vocab file. For each quantity (senses per lemma, instances per
    lemma, instances per sense) the row reports the average, max, median and min.
    """
    per_lemma = selected.groupby("lemma").agg(
        n_senses=("sense", "nunique"), n_instances=("n_instances", "sum")
    )
    quantities = {
        "senses_per_lemma": per_lemma["n_senses"],
        "instances_per_lemma": per_lemma["n_instances"],
        "instances_per_sense": selected["n_instances"],
    }

    row = {"pos": pos, "n_lemmas": int(per_lemma.shape[0])}
    for name, series in quantities.items():
        row[f"avg_{name}"] = float(series.mean())
        row[f"max_{name}"] = float(series.max())
        row[f"median_{name}"] = float(series.median())
        row[f"min_{name}"] = float(series.min())
    return row


# Quantities summarised in the overview, each rendered as its own narrow table.
_OVERVIEW_QUANTITIES = {
    "senses_per_lemma": "Senses per lemma",
    "instances_per_lemma": "Instances per lemma",
    "instances_per_sense": "Instances per sense",
}


def _write_overview(out_path: Path, overview_rows: list[dict]) -> None:
    """Write the cross-POS overview as three narrow tables (one per quantity).

    Each table has one row per POS and the avg/max/median/min columns for that single
    quantity, which reads more easily than one very wide combined table.
    """
    overview = pd.DataFrame(overview_rows)
    sections = ["# Vocabulary statistics overview\n"]
    for name, title in _OVERVIEW_QUANTITIES.items():
        # Keep n_lemmas alongside the first table as context (it can be < n if a POS
        # has fewer than n qualifying lemmata).
        lead = ["pos", "n_lemmas"] if name == "senses_per_lemma" else ["pos"]
        cols = lead + [f"avg_{name}", f"max_{name}", f"median_{name}", f"min_{name}"]
        table = overview[cols].rename(
            columns={
                f"avg_{name}": "avg",
                f"max_{name}": "max",
                f"median_{name}": "median",
                f"min_{name}": "min",
            }
        )
        sections.append(f"## {title}\n")
        sections.append(table.to_markdown(index=False))
        sections.append("")
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def create_most_diverse(vocab_dir: Path, wsd_dir: Path, n: int = 100) -> None:
    """Write, for each content-word POS, the n lemmata with the most WSD senses.

    Produces one file per POS (e.g. ``most_diverse_verb.json``) so each vocabulary
    contains the most sense-diverse lemmata of a single POS, ranked by the number of
    distinct senses they are annotated with in SEMCOR + MASC. Also writes a per-POS
    ``statistics_<pos>.md`` report (senses per lemma, instances per sense), a
    cross-POS ``statistics_overview.md`` table, and prints, per POS, any pair of
    selected lemmata with a Levenshtein distance below 2.
    """
    vocab_dir.mkdir(parents=True, exist_ok=True)
    counts = sense_instance_counts(wsd_dir)

    overview_rows = []
    for pos in CONTENT_POS:
        pos_counts = counts[counts["pos"] == pos]
        # Rank lemmata by distinct-sense count (descending), breaking ties
        # alphabetically for reproducible output. Keep only lemmata with at least
        # MIN_SENSES senses -- fewer means no diversity to vary.
        ranked = (
            pos_counts.groupby("lemma")["sense"]
            .nunique()
            .reset_index(name="n_senses")
            .sort_values(["n_senses", "lemma"], ascending=[False, True])
        )
        ranked = ranked[ranked["n_senses"] >= MIN_SENSES]
        lemmas = ranked["lemma"].head(n).tolist()
        pairs = [[lemma, pos] for lemma in lemmas]

        out_path = vocab_dir / f"most_diverse_{pos.lower()}.json"
        # Match the compact one-pair-per-line layout of the existing vocab files.
        lines = ",\n".join(f"  {json.dumps(pair)}" for pair in pairs)
        out_path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")

        # Statistics for this POS, scoped to the lemmata just written.
        selected = pos_counts[pos_counts["lemma"].isin(lemmas)]
        _write_statistics(vocab_dir / f"statistics_{pos.lower()}.md", pos, selected)
        overview_rows.append(_overview_row(pos, selected))

        _report_near_duplicates(lemmas, pos)

    # Cross-POS overview: three narrow tables (senses/lemma, instances/lemma,
    # instances/sense), each with avg/max/median/min per POS.
    _write_overview(vocab_dir / "statistics_overview.md", overview_rows)


@click.command()
@click.argument("vocab_dir", type=Path)
@click.argument("vocab", type=str)
@click.argument("wsd_dir", type=Path)
@click.option(
    "-n", "n", type=int, default=100, help="Number of word-POS pairs to keep."
)
def main(vocab_dir: Path, vocab: str, wsd_dir: Path, n: int) -> None:

    match vocab:
        case "most_diverse":
            create_most_diverse(vocab_dir, wsd_dir, n=n)
        case _:
            raise ValueError(f"Unknown vocab: {vocab!r}")


if __name__ == "__main__":
    main()
