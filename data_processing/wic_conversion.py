import json
from pathlib import Path
from typing import Generator, Any

import pandas as pd  # type: ignore


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
    for csv_path in sorted(sim_dir.glob("*/k*_offset_*.csv")):
        # Not csv_path.with_suffix(...): the "0.00" in the variant name confuses
        # pathlib's suffix handling. Swap the trailing ".csv" explicitly.
        meta_path = csv_path.parent / (csv_path.name[: -len(".csv")] + ".meta.json")
        if not meta_path.exists():
            continue  # skip stray CSVs without sidecar metadata

        df = pd.read_csv(csv_path)
        pairs = list(generate_comparison_pairs(df, seed=seed))

        out_path = output_dir / csv_path.parent.name / (csv_path.stem + ".data")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
        print(f"  {csv_path.parent.name} {csv_path.stem}: {len(pairs)} pairs")
