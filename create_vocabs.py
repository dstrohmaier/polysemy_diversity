import json
from pathlib import Path

import click

from nltk.corpus import wordnet as wn

# WordNet part-of-speech tags mapped to the universal POS labels used in the
# vocab files (e.g. source_data/vocabs/target_verbs.json).
WORDNET_POS_TO_UNIVERSAL = {
    wn.NOUN: "NOUN",
    wn.VERB: "VERB",
    wn.ADJ: "ADJ",
    wn.ADV: "ADV",
}


def count_senses() -> dict[tuple[str, str], int]:
    """Count the number of WordNet synsets for every single-word lemma/POS pair."""
    counts: dict[tuple[str, str], int] = {}
    for wn_pos, universal_pos in WORDNET_POS_TO_UNIVERSAL.items():
        for lemma in wn.all_lemma_names(pos=wn_pos):
            # Skip multi-word expressions; we only want single words.
            if "_" in lemma:
                continue
            counts[(lemma, universal_pos)] = len(wn.synsets(lemma, pos=wn_pos))
    return counts


def create_most_diverse(vocab_dir: Path, n: int = 100) -> None:
    """Write the n lemma/POS pairs with the most distinct WordNet senses."""
    counts = count_senses()
    # Sort by sense count (descending), breaking ties alphabetically for
    # reproducible output.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    pairs = [[lemma, pos] for (lemma, pos), _ in ranked[:n]]

    vocab_dir.mkdir(parents=True, exist_ok=True)
    out_path = vocab_dir / "most_diverse.json"
    # Match the compact one-pair-per-line layout of the existing vocab files.
    lines = ",\n".join(f"  {json.dumps(pair)}" for pair in pairs)
    out_path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")


@click.command()
@click.argument("vocab_dir", type=Path)
@click.argument("vocab", type=str)
@click.option("-n", "n", type=int, default=100, help="Number of word-POS pairs to keep.")
def main(vocab_dir: Path, vocab: str, n: int) -> None:

    match vocab:
        case "most_diverse":
            create_most_diverse(vocab_dir, n=n)
        case _:
            raise ValueError(f"Unknown vocab: {vocab!r}")


if __name__ == "__main__":
    main()
