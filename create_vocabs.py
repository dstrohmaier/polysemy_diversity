import json
from typing import Any
from pathlib import Path

import click
import pandas as pd  # type: ignore
from nltk.metrics.distance import edit_distance  # type: ignore

from data_processing.loading_wsd import wsd_generator

# The content-word universal POS labels used in the vocab files
# (e.g. source_data/vocabs/target_verbs.json). The WSD annotations already use
# these labels, so no POS mapping is needed.
CONTENT_POS = ["NOUN", "VERB", "ADJ", "ADV"]

# The WSD corpora to count senses from. "semcor_in_original_format" is excluded:
# it is a different format and would duplicate the senses already in "semcor".
WSD_SUBDIRS = ["semcor", "masc"]


def count_senses(wsd_dir: Path, min_examples: int = 5) -> pd.DataFrame:
    """Count distinct annotated senses per (lemma, pos) across the WSD corpora.

    Reads the SEMCOR and MASC annotations under ``wsd_dir`` and returns a frame of
    ``lemma, pos, n_senses``, where ``n_senses`` is the number of distinct senses each
    single-word lemma is attested with. This replaces the previous WordNet synset count
    with the senses actually observed in the WSD data the simulations are built from.

    Only senses with at least ``min_examples`` annotated occurrences count toward
    ``n_senses``: a sense seen once or twice is too sparse to simulate from, so it
    should not inflate a lemma's diversity ranking.
    """
    records: list[dict[str, Any]] = []
    for subdir in WSD_SUBDIRS:
        records.extend(wsd_generator(wsd_dir / subdir))
    df = pd.DataFrame(
        records, columns=["lemma", "pos", "sense", "sentence", "start", "end"]
    )

    # Single words only, content POS only.
    df = df[df["pos"].isin(CONTENT_POS) & ~df["lemma"].str.contains("_", na=False)]

    # Keep only senses with enough instances, then count the surviving senses.
    instances = df.groupby(["lemma", "pos", "sense"])["sense"].transform("count")
    df = df[instances >= min_examples]

    counts = (
        df.groupby(["lemma", "pos"])["sense"].nunique().reset_index(name="n_senses")
    )
    return counts


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


def create_most_diverse(vocab_dir: Path, wsd_dir: Path, n: int = 100) -> None:
    """Write, for each content-word POS, the n lemmata with the most WSD senses.

    Produces one file per POS (e.g. ``most_diverse_verb.json``) so each vocabulary
    contains the most sense-diverse lemmata of a single POS, ranked by the number of
    distinct senses they are annotated with in SEMCOR + MASC. Also prints, per POS,
    any pair of selected lemmata with a Levenshtein distance below 2.
    """
    vocab_dir.mkdir(parents=True, exist_ok=True)
    counts = count_senses(wsd_dir)

    for pos in CONTENT_POS:
        pos_counts = counts[counts["pos"] == pos]
        # Sort by sense count (descending), breaking ties alphabetically for
        # reproducible output.
        ranked = pos_counts.sort_values(["n_senses", "lemma"], ascending=[False, True])
        lemmas = ranked["lemma"].head(n).tolist()
        pairs = [[lemma, pos] for lemma in lemmas]

        out_path = vocab_dir / f"most_diverse_{pos.lower()}.json"
        # Match the compact one-pair-per-line layout of the existing vocab files.
        lines = ",\n".join(f"  {json.dumps(pair)}" for pair in pairs)
        out_path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")

        _report_near_duplicates(lemmas, pos)


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
