import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Any, Iterator

import pandas as pd  # type: ignore

logger = logging.getLogger("div")

# Matches a variant stem such as "k3_offset_m0.20" or "k5_offset_p0.00", produced by
# simulate_zipfian_corpora as f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}".
_VARIANT_RE = re.compile(r"^k(?P<k>\d+)_offset_(?P<sign>[mp])(?P<mag>[\d.]+)$")


@dataclass(frozen=True)
class Corpus:
    """One simulated corpus: a single (lemma, pos, k, offset) variant on disk.

    ``data_path`` is the WiC ``.data`` sibling; it only exists once this corpus has
    been through ``convert_simulated_corpora``.
    """

    lemma_pos: str  # the parent directory name, e.g. "<lemma>_<pos>"
    k: int
    offset: float
    csv_path: Path
    meta_path: Path
    data_path: Path


def parse_variant(stem: str) -> tuple[int, float]:
    """Parse a variant stem like ``k3_offset_m0.20`` into ``(k, offset)``."""
    match = _VARIANT_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognised variant stem: {stem!r}")
    offset = float(match.group("mag"))
    if match.group("sign") == "m":
        offset = -offset
    return int(match.group("k")), offset


def iter_corpora(sim_dir: Path) -> Iterator[Corpus]:
    """Yield one :class:`Corpus` per simulated CSV under ``sim_dir``.

    Globs ``<lemma>_<pos>/k*_offset_*.csv`` (the layout from
    ``simulate_zipfian_corpora``) and derives the sibling ``.meta.json`` / ``.data``
    paths. The trailing suffix is swapped explicitly rather than via
    ``Path.with_suffix`` because the ``.`` in the offset magnitude (e.g.
    ``k3_offset_p0.00``) confuses pathlib's suffix handling.
    """
    for csv_path in sorted(sim_dir.glob("*/k*_offset_*.csv")):
        base = csv_path.name[: -len(".csv")]
        k, offset = parse_variant(base)
        yield Corpus(
            lemma_pos=csv_path.parent.name,
            k=k,
            offset=offset,
            csv_path=csv_path,
            meta_path=csv_path.parent / (base + ".meta.json"),
            data_path=csv_path.parent / (base + ".data"),
        )


def generate_comparison_pairs(
    df: pd.DataFrame, seed: int = 1848
) -> Generator[dict[str, Any], None, None]:
    """Pair occurrences of each lemma into WiC-style sentence-pair examples.

    Operates on the simulated-corpus schema (lemma/pos/sense/sentence/start/end).
    Within each lemma the rows are shuffled and paired two-at-a-time; the gold
    ``label`` is whether the two occurrences share a sense.
    """
    for lemma, sub_df in df.groupby("lemma"):
        shuffled_df = sub_df.sample(frac=1, random_state=seed).reset_index()
        pairs = [group for _, group in shuffled_df.groupby(shuffled_df.index // 2)]

        for pair_df in pairs:
            if len(pair_df) < 2:  # can happen for odd numbers
                continue

            r0, r1 = pair_df.iloc[0], pair_df.iloc[1]
            assert (
                r0.pos == r1.pos
            ), f"paired rows disagree on PoS: {r0.pos!r} vs {r1.pos!r}"

            # No writing_id in the simulated data: synthesize a per-corpus-unique
            # id from the original (pre-shuffle) row indices plus char offsets.
            data_id = (
                f"{r0['index']}_{r0.start}_{r0.end}"
                f"__{r1['index']}_{r1.start}_{r1.end}"
            )

            yield {
                "id": data_id,
                "lemma": lemma,
                "pos": r0.pos,
                "sentence1": r0.sentence,
                "sentence2": r1.sentence,
                "label": int(r0.sense == r1.sense),
                "start1": int(r0.start),
                "end1": int(r0.end),
                "start2": int(r1.start),
                "end2": int(r1.end),
            }


def convert_simulated_corpora(
    sim_dir: Path, output_dir: Path, seed: int = 1848
) -> None:
    """Convert every simulated corpus under ``sim_dir`` to WiC ``.data`` files.

    Walks ``sim_dir/<lemma>_<pos>/k*_offset_*.csv`` (the layout produced by
    ``simulate_zipfian_corpora``) and writes a JSON array of sentence pairs to
    ``output_dir/<lemma>_<pos>/k*_offset_*.data`` for each, ready for
    ``apply_wic.py`` to consume.
    """
    for corpus in iter_corpora(sim_dir):
        if not corpus.meta_path.exists():
            continue  # skip stray CSVs without sidecar metadata

        df = pd.read_csv(corpus.csv_path)
        pairs = list(generate_comparison_pairs(df, seed=seed))

        out_path = output_dir / corpus.lemma_pos / (corpus.csv_path.stem + ".data")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        logger.info(
            "%s %s: %d pairs", corpus.lemma_pos, corpus.csv_path.stem, len(pairs)
        )
