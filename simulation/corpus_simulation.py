"""Methods to create an artifical dataset with varying senses."""

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from simulation.zipfian import estimate_slopes_for_words, zipfian_probs_for_senses


def uniform_sense_probs(wsd_df: pd.DataFrame, lemma: str, pos: str) -> dict[str, float]:
    """Return a uniform probability distribution over all senses of (lemma, pos)."""
    senses = (
        wsd_df.loc[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos), "sense"]
        .unique()
        .tolist()
    )
    if not senses:
        raise ValueError(f"No examples found for lemma={lemma!r}, pos={pos!r}")
    p = 1.0 / len(senses)
    return {s: p for s in senses}


def zipfian_sense_probs(
    wsd_df: pd.DataFrame, lemma: str, pos: str, slope: float
) -> dict[str, float]:
    """Return a Zipfian probability distribution over all senses of (lemma, pos).

    Senses are ranked by descending corpus frequency; rank 1 gets the highest probability.
    """
    subset = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]
    if subset.empty:
        raise ValueError(f"No examples found for lemma={lemma!r}, pos={pos!r}")

    senses = subset["sense"].value_counts().index.tolist()

    Z = sum((i + 1) ** (-slope) for i in range(len(senses)))
    return {s: (i + 1) ** (-slope) / Z for i, s in enumerate(senses)}


def simulate_polysemy(
    wsd_df: pd.DataFrame,
    lemma: str,
    pos: str,
    sense_probs: dict[str, float],
    n_draws: int = 2000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Draw n_draws examples from (lemma, pos) using sense_probs.

    Each draw independently samples a sense then a random example from that sense.
    Returns a DataFrame with columns: lemma, pos, sense, sentence, start, end.
    """
    if rng is None:
        rng = np.random.default_rng()

    subset = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]

    missing = set(sense_probs) - set(subset["sense"].unique())
    if missing:
        raise ValueError(f"sense_probs contains senses not found in wsd_df: {missing}")
    if not np.isclose(sum(sense_probs.values()), 1.0):
        raise ValueError(
            f"sense_probs values must sum to 1.0 (got {sum(sense_probs.values()):.6f})"
        )

    senses = list(sense_probs.keys())
    probs = np.array([sense_probs[s] for s in senses], dtype=float)
    sense_groups = {s: subset[subset["sense"] == s] for s in senses}

    # Pre-shuffle each sense's examples; cycle through the shuffled order before repeating.
    sense_queues: dict[str, list[int]] = {}
    for s, group in sense_groups.items():
        sense_queues[s] = rng.permutation(len(group)).tolist()

    def next_row(sense: str) -> pd.Series:
        queue = sense_queues[sense]
        if not queue:
            warnings.warn(
                f"Exhausted examples for sense {sense!r}; repeating from the beginning."
            )
            queue.extend(rng.permutation(len(sense_groups[sense])).tolist())
        return sense_groups[sense].iloc[queue.pop()]

    records = []
    for sense in rng.choice(senses, size=n_draws, p=probs):
        row = next_row(sense)
        records.append(
            {
                "lemma": row["lemma"],
                "pos": row["pos"],
                "sense": row["sense"],
                "sentence": row["sentence"],
                "start": row["start"],
                "end": row["end"],
            }
        )

    return pd.DataFrame(
        records, columns=["lemma", "pos", "sense", "sentence", "start", "end"]
    )


def _offset_grid(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive grid of offsets from lo to hi, rounded to avoid float drift."""
    return [float(x) for x in np.round(np.arange(lo, hi + step / 2, step), 4)]


SLOPE_FLOOR = 0.05  # keep the distribution well-defined when baseline+offset is small
MAX_DRAW_ATTEMPTS = 50  # redraws allowed to realise all k senses in a finite sample


@dataclass(frozen=True)
class SimConfig:
    """Run-wide simulation settings shared across all target words."""

    k_senses: tuple[int, ...]
    offsets: list[float]
    n_draws: int
    seed: int


def simulate_word_corpus(
    sub_df: pd.DataFrame,
    baseline: float,
    config: SimConfig,
    output_dir: Path,
) -> None:
    lemma = sub_df["lemma"].iloc[0]
    pos = sub_df["pos"].iloc[0]
    senses = sub_df["sense"].value_counts().index.tolist()

    # Deterministic per-(lemma, pos) id (unlike built-in hash(), which is salted
    # per run). Keyed on both fields so the same lemma under different POS gets
    # independent streams.
    word_id = int(hashlib.sha256(f"{lemma}\t{pos}".encode()).hexdigest(), 16) % (2**32)

    # Start from a clean directory so files from a previous run (e.g. with a
    # different k/offset grid) cannot linger alongside this run's output. Rather
    # than deleting, archive any existing output under stale/<run_stamp>/.
    word_dir = output_dir / f"{lemma}_{pos}"
    if word_dir.exists():
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = output_dir / "stale" / run_stamp / f"{lemma}_{pos}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(word_dir), str(archive))
    word_dir.mkdir(parents=True, exist_ok=True)

    for k in config.k_senses:
        for i, offset in enumerate(config.offsets):
            if len(senses) < k:
                continue

            top_k = senses[:k]
            applied = max(baseline + offset, SLOPE_FLOOR)

            sense_probs = zipfian_probs_for_senses(top_k, applied)
            # A finite multinomial draw can miss a rare sense, leaving fewer
            # than k senses realised. Reject and redraw until all k appear so
            # every corpus has *exactly* k senses per word.
            corpus = None
            # List seed -> SeedSequence: independent, collision-free streams per
            # (word, k, offset), fully determined by config.seed.
            rng = np.random.default_rng([config.seed, word_id, k, i])
            for _ in range(MAX_DRAW_ATTEMPTS):
                candidate = simulate_polysemy(
                    sub_df, lemma, pos, sense_probs, n_draws=config.n_draws, rng=rng
                )
                if candidate["sense"].nunique() == k:
                    corpus = candidate
                    break
            if corpus is None:
                warnings.warn(
                    f"{lemma!r}: could not realise all {k} senses in "
                    f"{config.n_draws} draws at slope {applied:.3f} after "
                    f"{MAX_DRAW_ATTEMPTS} attempts; word omitted for "
                    f"k={k}, offset={offset}."
                )
                continue

            variant = f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}"
            corpus.to_csv(word_dir / f"{variant}.csv", index=False)
            variant_meta = {
                "lemma": lemma,
                "pos": pos,
                "baseline_slope": baseline,
                "applied_slope": applied,
                "clamped": applied != baseline + offset,
                "k_senses": k,
                "n_senses_available": len(senses),
            }
            meta_path = word_dir / f"{variant}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(variant_meta, f, indent=2)


def simulate_zipfian_corpora(
    wsd_df: pd.DataFrame,
    target_pairs: list[tuple[str, str]],
    output_dir: Path,
    k_senses: tuple[int, ...],
    offset_min: float,
    offset_max: float,
    offset_step: float,
    n_draws: int,
    seed: int,
) -> None:
    # Baseline slopes are estimated once on each verb's full sense inventory;
    # they depend on neither k nor the offset.
    fits = estimate_slopes_for_words(wsd_df, target_pairs)

    fittable = fits[fits["status"] == "ok"]

    if fittable.empty:
        raise ValueError("No target word had a fittable baseline slope.")

    offsets = _offset_grid(offset_min, offset_max, offset_step)
    config = SimConfig(k_senses=k_senses, offsets=offsets, n_draws=n_draws, seed=seed)

    for _, fit in fittable.iterrows():
        lemma, pos = fit["lemma"], fit["pos"]
        sub_df = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]
        # Baseline fit on the full sense inventory.
        baseline = float(fit["slope"])
        simulate_word_corpus(sub_df, baseline, config, output_dir)
