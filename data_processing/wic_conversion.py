import json
import logging
from pathlib import Path
from typing import Generator, Any

import pandas as pd  # type: ignore

from data_processing.simulation_loading import iter_corpora

logger = logging.getLogger("div")


def generate_comparison_pairs(
    df: pd.DataFrame, seed: int = 1848
) -> Generator[dict[str, Any], None, None]:
    """Pair occurrences of each lemma into WiC-style sentence-pair examples.

    Operates on the simulated-corpus schema (lemma/pos/sense/sentence/start/end).
    Within each lemma the rows are shuffled and arranged in a cycle: each row ``i`` is
    paired with its successor ``(i + 1) mod N``. This yields exactly ``N`` pairs (one
    per sentence) with every sentence appearing in exactly two pairs -- once as the
    left element and once as the right -- so the pair count matches the number of
    sentences. The gold ``label`` is whether the two occurrences share a sense.
    """
    for lemma, sub_df in df.groupby("lemma"):
        shuffled_df = sub_df.sample(frac=1, random_state=seed).reset_index()
        n = len(shuffled_df)
        if n < 2:  # a lone sentence cannot form a (non-self) pair
            continue

        for i in range(n):
            r0 = shuffled_df.iloc[i]
            r1 = shuffled_df.iloc[(i + 1) % n]
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
