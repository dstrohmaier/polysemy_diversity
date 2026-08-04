"""Materialise DWUG EN into the on-disk corpus layout the scorers already consume.

DWUG ships one ``uses.csv`` per lemma (all decades together) plus an optimal
clustering in ``clusters/opt/``. The diachronic evaluation needs, per lemma, two
corpora -- grouping 1 (1810-1860, the *source*) and grouping 2 (1960-2010, the
target) -- written in the simulated-corpus schema
``lemma,pos,sense,sentence,start,end``. Writing that schema means
:func:`~data_processing.wic_conversion.generate_comparison_pairs` and every
``score_pair_*`` function work verbatim on DWUG, and the WiC ``.data`` labels come
out as a free accuracy diagnostic against the DWUG clustering.

Ground truth
------------
Each corpus's ``.meta.json`` stores ``sense_probs``, the empirical cluster
distribution of that grouping. The analysis layer reads it under the same key as the
simulation's design distribution, so the existing ground-truth machinery applies
unchanged. Note the asymmetry this hides: for the simulation ``sense_probs`` is the
*design* distribution (independent of the sample), whereas here it is the *observed*
one, and so -- richness especially -- is subject to sampling bias at ~100 usages per
grouping.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from data_processing.dwug_loading import SOURCE_STEM, TARGET_STEM
from data_processing.wic_conversion import generate_comparison_pairs
from simulation.diversity import STANDARD_ORDERS, diversity_shift, hill_diversity

logger = logging.getLogger("div")

# DWUG marks usages its optimal clustering left unassigned (noise) with -1. They carry
# no sense label, so they are dropped from both the ground truth and the data the
# scorers see. Dropping them exactly reproduces DWUG's own
# stats/opt/stats_groupings.csv cluster_freq_dist1/2 columns.
NOISE_CLUSTER = -1

# Both scorers need at least two usages: loo_centroid_distance asserts n >= 2, and
# generate_comparison_pairs emits nothing for n < 2.
MIN_GROUPING_USAGES = 2

# The two decade groupings of the readme's "Second Evaluation".
DECADE_LABELS = {1: "1810-1860", 2: "1960-2010"}
GROUPING_STEMS = {1: SOURCE_STEM, 2: TARGET_STEM}

# Columns of the simulated-corpus schema, in order.
CORPUS_COLUMNS = ["lemma", "pos", "sense", "sentence", "start", "end"]


def _target_span(indexes_target_token: str) -> tuple[int, int]:
    """Parse DWUG's ``"27:36"`` char-offset span into ``(start, end)``.

    The offsets index into ``context`` (not ``context_tokenized``), and ``context`` is
    what we write as ``sentence``, so they transfer unchanged.
    """
    start, end = str(indexes_target_token).split(":")
    return int(start), int(end)


def dwug_lemma_frame(
    uses: pd.DataFrame, clusters: pd.DataFrame, lemma_pos: str
) -> pd.DataFrame:
    """Join uses to their cluster and reshape into the simulated-corpus schema.

    Returns the :data:`CORPUS_COLUMNS` plus the two DWUG keys ``grouping`` and
    ``identifier``, which the caller uses to split by decade and to trace provenance;
    both are dropped before the corpus CSV is written.

    Two column choices are deliberate and easy to get wrong:

    * ``lemma`` is set to ``lemma_pos`` (e.g. ``"afternoon_nn"``).
      ``generate_comparison_pairs`` groups by ``lemma``, so one constant value per file
      yields exactly one group.
    * ``pos`` is the **coarse suffix** of ``lemma_pos`` (``"nn"`` / ``"vb"``), *not*
      DWUG's own ``pos`` column. That column holds fine-grained CLAWS tags which vary
      *within* every one of the 46 lemmata (``pin_vb`` alone has 16 distinct values),
      and ``generate_comparison_pairs`` asserts that paired rows agree on ``pos`` --
      so passing DWUG's column through would fire that assert on essentially every
      lemma.

    Noise rows (``cluster == -1``) are dropped: DWUG's clustering assigned them no
    sense.
    """
    merged = uses.merge(clusters, on="identifier", how="inner")
    # A DWUG release whose clustering does not cover every usage would silently shrink
    # the ground truth; currently every identifier matches, so require that.
    assert len(merged) == len(uses), (
        f"{lemma_pos}: {len(uses) - len(merged)} of {len(uses)} usages have no cluster "
        f"in clusters/opt/; the ground truth would silently omit them"
    )

    merged = merged[merged["cluster"] != NOISE_CLUSTER]
    spans = merged["indexes_target_token"].map(_target_span)

    return pd.DataFrame(
        {
            "lemma": lemma_pos,
            "pos": lemma_pos.rsplit("_", 1)[1],
            "sense": merged["cluster"].astype(str),
            "sentence": merged["context"],
            "start": [s for s, _ in spans],
            "end": [e for _, e in spans],
            "grouping": merged["grouping"].astype(int),
            "identifier": merged["identifier"],
        }
    ).reset_index(drop=True)


def cluster_probs(frame: pd.DataFrame, grouping: int) -> dict[str, float]:
    """Sense (= cluster) probability distribution of one grouping.

    The empirical relative frequency of each cluster among that grouping's noise-free
    usages -- the diachronic study's analogue of the simulation's design
    ``sense_probs``, and what the ground-truth Hill diversities are computed from.

    Deliberately taken over the *full* grouping, before any downsampling.
    Downsampling exists only to stop a corpus-size difference contaminating the
    *scores* (readme "Source and Target Corpus"); the ground truth is a property of
    the data itself, mirroring the simulation where it is the design distribution
    rather than the realised sample.
    """
    counts = frame.loc[frame["grouping"] == grouping, "sense"].value_counts()
    total = float(counts.sum())
    return {str(sense): float(n) / total for sense, n in counts.items()}


def _entropy_bits(probs: dict[str, float]) -> float:
    """Shannon entropy in bits, matching the simulation sidecar's ``entropy_bits``."""
    values = np.asarray(list(probs.values()), dtype=float)
    values = values[values > 0]
    return float(-np.sum(values * np.log2(values)))


def _grouping_meta(
    frame: pd.DataFrame, lemma_pos: str, grouping: int, n_raw: int
) -> dict:
    """The ``.meta.json`` sidecar for one grouping.

    ``sense_probs`` is load-bearing: the analysis layer's ground-truth lookup reads it
    under exactly this key for both evaluations. The rest is diagnostic.

    The simulation-only keys (``baseline_slope``, ``applied_slope``, ``clamped``) are
    deliberately absent -- they are meaningless for DWUG, and omitting them makes an
    accidental run of a simulation-only analysis mode fail loudly rather than report
    nonsense.
    """
    sub = frame[frame["grouping"] == grouping]
    probs = cluster_probs(frame, grouping)
    return {
        "lemma": lemma_pos,
        "pos": lemma_pos.rsplit("_", 1)[1],
        "dataset": "dwug_en",
        "grouping": grouping,
        "decades": DECADE_LABELS[grouping],
        "role": "source" if grouping == 1 else "target",
        "sense_probs": probs,
        "k_senses": len(probs),
        "n_usages": int(len(sub)),
        "n_usages_raw": int(n_raw),
        "n_noise_dropped": int(n_raw - len(sub)),
        "hill_diversity": {str(q): hill_diversity(probs, q) for q in STANDARD_ORDERS},
        "entropy_bits": _entropy_bits(probs),
        # sum_i p_i^2 -- the Simpson concentration the WiC method estimates as
        # p(same). Directly comparable to the measured p_same, so it doubles as a
        # per-corpus calibration check for the WiC scorer.
        "p_same_empirical": float(sum(p * p for p in probs.values())),
    }


def write_dwug_lemma(
    frame: pd.DataFrame,
    lemma_pos: str,
    output_dir: Path,
    raw_sizes: dict[int, int],
    seed: int = 1848,
) -> bool:
    """Write one lemma's two grouping corpora (CSV + ``.meta.json`` + ``.data``).

    ``raw_sizes`` maps grouping -> usage count *before* the noise drop, for the
    sidecar's audit trail. Returns ``True`` if both groupings were written, ``False``
    if the lemma was skipped.

    A grouping with fewer than :data:`MIN_GROUPING_USAGES` usages after the noise drop
    cannot be scored, so the **whole lemma** is skipped: a pair needs both sides, and
    skipping early keeps the preparation summary honest rather than letting the lemma
    vanish silently at pair enumeration.
    """
    sizes = {g: int((frame["grouping"] == g).sum()) for g in GROUPING_STEMS}
    too_small = {g: n for g, n in sizes.items() if n < MIN_GROUPING_USAGES}
    if too_small:
        logger.warning(
            "%s: grouping(s) %s have < %d usages after dropping noise; skipping lemma",
            lemma_pos, too_small, MIN_GROUPING_USAGES,
        )
        return False

    word_dir = output_dir / lemma_pos
    word_dir.mkdir(parents=True, exist_ok=True)

    for grouping, stem in GROUPING_STEMS.items():
        sub = frame[frame["grouping"] == grouping]
        csv_path = word_dir / f"{stem}.csv"
        sub[CORPUS_COLUMNS].to_csv(csv_path, index=False)

        meta = _grouping_meta(frame, lemma_pos, grouping, raw_sizes[grouping])
        (word_dir / f"{stem}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        # Read the CSV back rather than pairing `sub` directly: generate_comparison_pairs
        # builds each pair id from the incoming frame's reset index, so a filtered slice
        # would embed pre-filter DWUG row numbers. Reading back gives a clean 0..n-1
        # index (as the simulation path produces) and guarantees the .data pairs come
        # from exactly the bytes the vMF and cosine scorers read.
        pairs = list(generate_comparison_pairs(pd.read_csv(csv_path), seed=seed))
        (word_dir / f"{stem}.data").write_text(
            json.dumps(pairs, indent=2), encoding="utf-8"
        )

        logger.info(
            "%s %s: %d usages, %d senses, %d WiC pairs",
            lemma_pos, stem, len(sub), meta["k_senses"], len(pairs),
        )

    return True


def prepare_dwug_corpora(
    dwug_root: Path, output_dir: Path, seed: int = 1848
) -> pd.DataFrame:
    """Convert every DWUG EN lemma into the on-disk corpus layout.

    ``dwug_root`` is the unpacked dataset (containing ``data/`` and ``clusters/opt/``).
    Writes ``output_dir/<lemma>_<pos>/{g1,g2}.{csv,meta.json,data}`` and returns a
    one-row-per-lemma summary (sizes, noise counts, senses per grouping, and the
    ground-truth shifts) for the caller to write out as a preparation report.
    """
    data_dir = dwug_root / "data"
    clusters_dir = dwug_root / "clusters" / "opt"

    rows = []
    for lemma_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        lemma_pos = lemma_dir.name
        # uses.csv embeds unescaped double quotes in `context`, which the default
        # dialect mis-parses; QUOTE_NONE is safe as no context holds a tab or newline.
        uses = pd.read_csv(lemma_dir / "uses.csv", sep="\t", quoting=csv.QUOTE_NONE)
        clusters = pd.read_csv(clusters_dir / f"{lemma_pos}.csv", sep="\t")
        raw_sizes = {
            g: int((uses["grouping"].astype(int) == g).sum()) for g in GROUPING_STEMS
        }

        frame = dwug_lemma_frame(uses, clusters, lemma_pos)
        written = write_dwug_lemma(frame, lemma_pos, output_dir, raw_sizes, seed=seed)

        n_g1 = int((frame["grouping"] == 1).sum())
        n_g2 = int((frame["grouping"] == 2).sum())
        row = {
            "lemma_pos": lemma_pos,
            "n_raw_g1": raw_sizes[1],
            "n_raw_g2": raw_sizes[2],
            "n_noise": int(len(uses) - len(frame)),
            "n_g1": n_g1,
            "n_g2": n_g2,
            "written": written,
        }
        if written:
            probs_s = cluster_probs(frame, 1)
            probs_t = cluster_probs(frame, 2)
            row["k_g1"] = len(probs_s)
            row["k_g2"] = len(probs_t)
            row["n_equalised"] = min(n_g1, n_g2)
            for q in STANDARD_ORDERS:
                row[f"gt_shift_q{q}"] = diversity_shift(probs_s, probs_t, q)
        rows.append(row)

    summary = pd.DataFrame(rows)
    logger.info(
        "prepared %d/%d DWUG lemmata (%d noise usages dropped)",
        int(summary["written"].sum()), len(summary), int(summary["n_noise"].sum()),
    )
    return summary
