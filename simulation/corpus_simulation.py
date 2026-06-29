"""Methods to create an artifical dataset with varying senses."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from itertools import product
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import entropy  # type: ignore

from simulation.zipfian import estimate_slopes_for_words, zipfian_probs_for_senses

logger = logging.getLogger("div")


def _zipfian_unique_counts(
    probs: np.ndarray, available: np.ndarray, n_draws: int
) -> np.ndarray:
    """Per-sense sentence counts that stay Zipfian without ever repeating a sentence.

    Duplicated sentences are uninformative, so each sentence may be used at most
    once. A sense with only ``available_s`` distinct sentences can therefore supply
    at most that many examples, which caps how large a corpus can stay proportional
    to ``probs``. We take the largest total ``N <= n_draws`` for which the Zipfian
    target ``round(probs_s * N)`` fits within every sense's supply -- set by the
    binding sense ``min_s floor(available_s / probs_s)`` -- then allocate
    ``round(probs_s * N)`` per sense, repairing rounding so the counts sum to ``N``,
    stay within ``available``, and keep every sense at >= 1 (so all k senses survive).
    The result is smaller than ``n_draws`` when some sense is sentence-poor, but its
    sense proportions track the Zipfian design as closely as integer counts allow.
    """
    n_max = int(np.floor(available / probs).min())
    target_n = min(n_draws, n_max)

    counts = np.round(probs * target_n).astype(int)
    counts = np.clip(counts, 1, available.astype(int))

    # Repair the total toward target_n: rounding and the >=1 floor can push the sum
    # off. Remove from / add to the sense with the most slack, never crossing the
    # [1, available] bounds, so the realised distribution stays as Zipfian as possible.
    while counts.sum() > target_n and (counts > 1).any():
        counts[np.argmax(np.where(counts > 1, counts - 1, -1))] -= 1
    while counts.sum() < target_n:
        slack = available.astype(int) - counts
        if slack.max() <= 0:
            break
        counts[np.argmax(slack)] += 1

    return counts


def distinct_sentence_pool(sub_df: pd.DataFrame, min_examples: int) -> pd.DataFrame:
    """Reduce a (lemma, pos) subset to its sampleable pool of distinct sentences.

    A sentence can appear several times in the WSD data: both as exact repeats within
    a sense and, occasionally, under *different* senses. A sentence tagged with
    conflicting senses is ambiguous, so drop all its copies; then collapse the
    remaining exact repeats so every surviving sentence is unique and maps to one
    sense. Finally drop senses left with fewer than ``min_examples`` distinct
    sentences -- too sparse to sample a corpus from.

    The result is the pool used both to rank/select senses and to draw from, so the
    minimum-examples floor is enforced on *distinct sentences* (not raw occurrences).
    """
    sense_per_sentence = sub_df.groupby("sentence")["sense"].transform("nunique")
    pool = sub_df[sense_per_sentence == 1].drop_duplicates("sentence")

    sentences_per_sense = pool.groupby("sense")["sentence"].transform("count")
    return pool[sentences_per_sense >= min_examples]


def simulate_polysemy(
    pool: pd.DataFrame,
    sense_probs: dict[str, float],
    n_draws: int = 2000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Draw up to n_draws *distinct* example sentences from ``pool`` using sense_probs.

    ``pool`` must already be the distinct-sentence pool for one (lemma, pos) -- see
    :func:`distinct_sentence_pool`. Sentences are sampled without replacement per
    sense, so the corpus never repeats a sentence. Because a sense supplies at most
    its number of distinct sentences, the corpus shrinks below n_draws when a sense is
    sentence-poor; the realised sense proportions still track the Zipfian sense_probs
    as closely as integer counts allow (see :func:`_zipfian_unique_counts`).

    Returns a DataFrame with columns: lemma, pos, sense, sentence, start, end.
    """
    if rng is None:
        rng = np.random.default_rng()

    missing = set(sense_probs) - set(pool["sense"].unique())
    if missing:
        raise ValueError(f"sense_probs contains senses not found in pool: {missing}")
    if not np.isclose(sum(sense_probs.values()), 1.0):
        raise ValueError(
            f"sense_probs values must sum to 1.0 (got {sum(sense_probs.values()):.6f})"
        )

    senses = list(sense_probs.keys())
    probs = np.array([sense_probs[s] for s in senses], dtype=float)
    sense_groups = {s: pool[pool["sense"] == s] for s in senses}
    available = np.array([len(sense_groups[s]) for s in senses], dtype=int)

    counts = _zipfian_unique_counts(probs, available, n_draws)

    parts = [
        sense_groups[s].sample(n=int(c), replace=False, random_state=rng)
        for s, c in zip(senses, counts)
        if c > 0
    ]
    corpus = pd.concat(parts, ignore_index=True)
    return corpus[["lemma", "pos", "sense", "sentence", "start", "end"]]


def _offset_grid(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive grid of offsets from lo to hi, rounded to avoid float drift."""
    return [float(x) for x in np.round(np.arange(lo, hi + step / 2, step), 4)]


SLOPE_FLOOR = 0.05  # keep the distribution well-defined when baseline+offset is small
# Skip variants too small (after dedup) to score robustly.
MIN_CORPUS_SENTENCES = 30


@dataclass(frozen=True)
class SimConfig:
    """Run-wide simulation settings shared across all target words."""

    k_senses: tuple[int, ...]
    offsets: list[float]
    n_draws: int
    seed: int
    min_examples: int  # min distinct sentences (post-dedup) for a sense to be usable


def _word_id(lemma: str, pos: str) -> int:
    """Deterministic per-(lemma, pos) id for collision-free RNG seeding.

    Unlike built-in hash() (salted per run), this is stable across runs. Keyed
    on both fields so the same lemma under different POS gets independent streams.
    """
    return int(hashlib.sha256(f"{lemma}\t{pos}".encode()).hexdigest(), 16) % (2**32)


def _prepare_word_dir(output_dir: Path, lemma: str, pos: str) -> Path:
    """Return a clean output dir for (lemma, pos), archiving any prior run.

    Files from a previous run (e.g. with a different k/offset grid) must not
    linger alongside this run's output. Rather than deleting, archive existing
    output under stale/<run_stamp>/.
    """
    word_dir = output_dir / f"{lemma}_{pos}"
    if word_dir.exists():
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = output_dir / "stale" / run_stamp / f"{lemma}_{pos}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(word_dir), str(archive))
    word_dir.mkdir(parents=True, exist_ok=True)
    return word_dir


def _draw_corpus_with_all_senses(
    pool: pd.DataFrame,
    lemma: str,
    pos: str,
    sense_probs: dict[str, float],
    k: int,
    n_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw a unique-sentence corpus in which all k senses are realised.

    ``simulate_polysemy`` allocates at least one distinct sentence to every sense
    (each sense in ``pool`` has >= min_examples distinct sentences by construction),
    so a single draw realises all k senses -- no rejection loop is needed. We still
    assert the contract so a future change that breaks it fails loudly rather than
    silently dropping a sense.

    Raises
    ------
    RuntimeError
        If the drawn corpus does not realise all k senses.
    """
    corpus = simulate_polysemy(pool, sense_probs, n_draws=n_draws, rng=rng)
    realised = corpus["sense"].nunique()
    if realised != k:
        raise RuntimeError(
            f"{lemma!r} ({pos}): drew {realised} senses but expected {k}; the "
            f"per-sense sentence supply is too small to realise all senses."
        )
    return corpus


def _write_variant(
    word_dir: Path, variant: str, corpus: pd.DataFrame, meta: dict
) -> None:
    """Write the corpus CSV and its sidecar meta JSON for one variant."""
    corpus.to_csv(word_dir / f"{variant}.csv", index=False)
    with open(word_dir / f"{variant}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def simulate_word_corpus(
    sub_df: pd.DataFrame,
    baseline: float,
    config: SimConfig,
    output_dir: Path,
) -> None:
    lemma = sub_df["lemma"].iloc[0]
    pos = sub_df["pos"].iloc[0]

    # Reduce to the sampleable pool (distinct, unambiguous sentences; senses with
    # >= min_examples of them) and rank senses by frequency *within that pool*, so
    # sense selection, sense_probs, and sampling are all consistent.
    pool = distinct_sentence_pool(sub_df, config.min_examples)
    senses = pool["sense"].value_counts().index.tolist()

    word_id = _word_id(lemma, pos)
    word_dir = _prepare_word_dir(output_dir, lemma, pos)

    # Map each offset to its grid index so the RNG seed is a clean integer,
    # independent of the float grid's resolution (int(offset * 100) could
    # truncate-collide when offsets carry sub-0.01 precision).
    offset_index = {offset: j for j, offset in enumerate(config.offsets)}

    for k, offset in product(config.k_senses, config.offsets):
        if len(senses) < k:
            continue
        top_k = senses[:k]

        applied = max(baseline + offset, SLOPE_FLOOR)
        sense_probs = zipfian_probs_for_senses(top_k, applied)

        # List seed -> SeedSequence: independent, collision-free streams per
        # (word, k, offset), fully determined by config.seed.
        rng = np.random.default_rng([config.seed, word_id, k, offset_index[offset]])
        corpus = _draw_corpus_with_all_senses(
            pool, lemma, pos, sense_probs, k, config.n_draws, rng
        )

        variant = f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}"
        if len(corpus) < MIN_CORPUS_SENTENCES:
            # The unique-sentence supply was too thin for this (k, offset) to yield an
            # informative corpus; skip it rather than write a near-empty variant.
            logger.info(
                "%s_%s %s: only %d unique sentences (< %d), skipping",
                lemma,
                pos,
                variant,
                len(corpus),
                MIN_CORPUS_SENTENCES,
            )
            continue
        _write_variant(
            word_dir,
            variant,
            corpus,
            {
                "lemma": lemma,
                "pos": pos,
                "baseline_slope": baseline,
                "applied_slope": applied,
                "clamped": applied != baseline + offset,
                "k_senses": k,
                "n_senses_available": len(senses),
                "sense_probs": sense_probs,
                "entropy_bits": float(entropy(list(sense_probs.values()), base=2)),
            },
        )


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
    min_examples: int,
) -> None:
    # Baseline slopes are estimated once on each verb's full sense inventory, from the
    # raw occurrence counts -- they depend on neither k nor the offset, and are not
    # affected by the post-dedup min_examples filter applied during sampling.
    fits = estimate_slopes_for_words(wsd_df, target_pairs)

    fittable = fits[fits["status"] == "ok"]

    if fittable.empty:
        raise ValueError("No target word had a fittable baseline slope.")

    offsets = _offset_grid(offset_min, offset_max, offset_step)
    config = SimConfig(
        k_senses=k_senses,
        offsets=offsets,
        n_draws=n_draws,
        seed=seed,
        min_examples=min_examples,
    )

    for _, fit in fittable.iterrows():
        lemma, pos = fit["lemma"], fit["pos"]
        sub_df = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]
        # Baseline fit on the full sense inventory.
        baseline = float(fit["slope"])
        simulate_word_corpus(sub_df, baseline, config, output_dir)
