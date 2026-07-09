"""Synthesise a balanced WiC dataset from the FEWS WSD corpus.

FEWS (``source_data/fews/``) is a Wiktionary-based word-sense-disambiguation dataset:
each line is ``sentence \\t label`` where the target occurrence is wrapped in
``<WSD>…</WSD>`` and ``label`` is ``word.pos.sensenum`` (e.g. ``driving_force.noun.1``).

We turn it into WiC-style sentence pairs that share a target word:

* **positive** (``label = 1``, same sense): two occurrences of the *same* ``word.pos.sense``;
* **negative** (``label = 0``, different sense): two occurrences of the *same* ``word.pos``
  but a *different* sense.

The result is balanced to exactly 50% positive / 50% negative, and emits the same 8-field
schema as :func:`data_processing.loading_wic.get_wic_dsd`
(``lemma, sentence1, sentence2, label, start1, end1, start2, end2``) so it is a drop-in for
the WiC training pipeline. Note the WiC convention: ``label = 1`` means *same sense*.
"""

import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets  # type: ignore

# FEWS POS strings -> the NOUN/VERB/ADJ/ADV convention used across the simulated/WiC data.
_POS_MAP = {
    "noun": "NOUN",
    "verb": "VERB",
    "adjective": "ADJ",
    "adverb": "ADV",
}

_WSD_RE = re.compile(r"<WSD>(.*?)</WSD>", re.DOTALL)


def parse_fews_line(line: str) -> dict | None:
    """Parse one FEWS ``sentence \\t label`` line into an occurrence record.

    Returns ``{word, pos, sense_id, sentence, start, end}`` where ``sentence`` has the
    ``<WSD>`` markers stripped and ``sentence[start:end]`` is the target occurrence, or
    ``None`` for a malformed line (wrong field count, no ``<WSD>`` tag, or a label that is
    not ``word.pos.sensenum``).
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        return None
    raw_sentence, label = parts

    label_parts = label.split(".")
    if len(label_parts) < 3:
        return None
    word_token, pos_token = label_parts[0], label_parts[1]
    pos = _POS_MAP.get(pos_token)
    if pos is None:
        return None

    match = _WSD_RE.search(raw_sentence)
    if match is None:
        return None
    target = match.group(1)

    # Offset of the target in the tag-stripped sentence: strip tags from the prefix, then
    # the cleaned prefix length is the target's start. Only the first <WSD> matters.
    before_clean = _WSD_RE.sub(r"\1", raw_sentence[: match.start()])
    start = len(before_clean)
    end = start + len(target)
    sentence = _WSD_RE.sub(r"\1", raw_sentence)

    return {
        # Underscores in FEWS words join multi-word lemmas; use spaces as the surface lemma.
        "word": word_token.replace("_", " "),
        "pos": pos,
        "sense_id": label,
        "sentence": sentence,
        "start": start,
        "end": end,
    }


def load_fews_occurrences(txt_paths: list[Path]) -> list[dict]:
    """Parse every FEWS ``.txt`` file in ``txt_paths`` into occurrence records."""
    occurrences: list[dict] = []
    for path in txt_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = parse_fews_line(line)
                if record is not None:
                    occurrences.append(record)
    return occurrences


def _make_pair(word: str, pos: str, o0: dict, o1: dict, label: int) -> dict:
    return {
        "lemma": word,
        "sentence1": o0["sentence"],
        "sentence2": o1["sentence"],
        "label": label,
        "start1": o0["start"],
        "end1": o0["end"],
        "start2": o1["start"],
        "end2": o1["end"],
    }


def build_balanced_pairs(
    occurrences: list[dict],
    rng: random.Random,
    cap_per_word: int = 4,
) -> list[dict]:
    """Build WiC pairs from FEWS occurrences, balanced to 50% same / 50% different sense.

    Occurrences are grouped by ``word.pos``. For each word we sample up to ``cap_per_word``
    **positive** pairs (two occurrences of one sense) and up to ``cap_per_word``
    **negative** pairs (two occurrences of two different senses). The per-word cap stops
    high-frequency words from dominating. Finally the two classes are globally downsampled
    to the same count so the returned list is exactly 50/50 (its order is shuffled).
    """
    by_word: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for occ in occurrences:
        by_word[(occ["word"], occ["pos"])].append(occ)

    positives: list[dict] = []
    negatives: list[dict] = []

    for (word, pos), occs in by_word.items():
        by_sense: dict[str, list[dict]] = defaultdict(list)
        for occ in occs:
            by_sense[occ["sense_id"]].append(occ)

        # Positives: within-sense pairs from senses with >=2 occurrences.
        pos_candidates = [s for s, items in by_sense.items() if len(items) >= 2]
        word_positives: list[dict] = []
        for sense in pos_candidates:
            items = by_sense[sense][:]
            rng.shuffle(items)
            # Disjoint consecutive pairs so no occurrence is reused within a sense.
            for i in range(0, len(items) - 1, 2):
                word_positives.append(_make_pair(word, pos, items[i], items[i + 1], 1))
        rng.shuffle(word_positives)
        positives.extend(word_positives[:cap_per_word])

        # Negatives: cross-sense pairs; needs the word to have >=2 senses.
        senses = list(by_sense.keys())
        if len(senses) >= 2:
            word_negatives: list[dict] = []
            attempts = 0
            seen: set[tuple[int, int]] = set()
            max_attempts = cap_per_word * 8
            while len(word_negatives) < cap_per_word and attempts < max_attempts:
                attempts += 1
                s_a, s_b = rng.sample(senses, 2)
                o_a = rng.choice(by_sense[s_a])
                o_b = rng.choice(by_sense[s_b])
                key = (id(o_a), id(o_b))
                if key in seen:
                    continue
                seen.add(key)
                word_negatives.append(_make_pair(word, pos, o_a, o_b, 0))
            negatives.extend(word_negatives)

    # Global 50/50 balance: downsample the majority class.
    n = min(len(positives), len(negatives))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pairs = positives[:n] + negatives[:n]
    rng.shuffle(pairs)
    return pairs


def _split_pairs_by_lemma(
    pairs: list[dict], rng: random.Random, val_fraction: float
) -> tuple[list[dict], list[dict]]:
    """Partition ``pairs`` into (train, validation), disjoint by lemma.

    Splitting by lemma (not by row) keeps a word entirely on one side, so the
    validation set measures generalisation to unseen words rather than memorised ones.
    """
    lemmas = sorted({p["lemma"] for p in pairs})
    rng.shuffle(lemmas)
    n_val = max(1, int(len(lemmas) * val_fraction)) if lemmas else 0
    val_lemmas = set(lemmas[:n_val])
    train_pairs = [p for p in pairs if p["lemma"] not in val_lemmas]
    val_pairs = [p for p in pairs if p["lemma"] in val_lemmas]
    return train_pairs, val_pairs


def get_fews_wic_dsd(
    dataset_dir: Path,
    use_test: bool = False,
    seed: int = 1848,
    cap_per_word: int = 4,
    val_fraction: float = 0.1,
) -> DatasetDict:
    """Build a balanced synthetic-WiC ``DatasetDict`` from the FEWS corpus.

    Pairs are synthesised from ``train/train.txt`` (the FEWS dev/test splits are too
    low-shot to yield same-sense pairs, so they are not used) and partitioned into
    ``train``/``validation`` **disjoint by lemma**. Mirrors
    :func:`data_processing.loading_wic.get_wic_dsd`: with ``use_test=True`` the held-out
    validation pairs are folded back into ``train`` (so the final model trains on
    everything) and ``validation`` reuses them for the reported metric. Returns the
    8-field WiC schema, balanced 50/50 within each partition's construction.
    """
    rng = random.Random(seed)

    train_occ = load_fews_occurrences([dataset_dir / "train" / "train.txt"])
    pairs = build_balanced_pairs(train_occ, rng, cap_per_word)
    train_pairs, val_pairs = _split_pairs_by_lemma(pairs, rng, val_fraction)

    train_ds = Dataset.from_list(train_pairs)
    val_ds = Dataset.from_list(val_pairs)

    if use_test:
        train_ds = concatenate_datasets([train_ds, val_ds])

    return DatasetDict({"train": train_ds, "validation": val_ds})
