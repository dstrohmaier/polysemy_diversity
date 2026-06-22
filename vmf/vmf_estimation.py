"""Estimate von Mises-Fisher concentration for simulated word-sense corpora.

The von Mises-Fisher (vMF) distribution is the analogue of a Gaussian for points
that live on the surface of a unit hypersphere -- i.e. for directions rather than
positions. It is parameterised by a mean direction ``mu`` (a unit vector pointing
to the centre of the cloud) and a concentration ``kappa >= 0``. Large ``kappa``
means the directions are tightly clustered around ``mu``; ``kappa = 0`` means they
are spread uniformly over the sphere.

We treat each word occurrence's contextual embedding as a direction (only its
orientation matters, not its magnitude) and fit a single vMF per corpus. The
resulting ``kappa`` is our scalar measure of sense spread: a polysemous or
sense-diverse word produces context vectors pointing in many directions and
therefore a low ``kappa``, whereas a word used in one consistent sense yields a
high ``kappa``. ``kappa`` is recovered from the resultant length ``r`` (the norm
of the mean vector) via the standard approximation in ``_estimate_kappa``.

This module walks the simulated corpora produced by ``simulate_zipfian_corpora``
(one corpus per (lemma, pos) x (k_senses, offset) variant), fits one ``kappa`` per
corpus, and writes the collected scores to a single ``vmf_scores.csv``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from data_processing.vector_extraction import ExtractionConfig, WordVectorExtractor


def estimate_vmf_parameters(vectors):
    if len(vectors) == 0:
        raise ValueError("Empty vectors")
    mean_vector = np.mean(vectors, axis=0)
    r = np.linalg.norm(mean_vector)
    if r == 0:
        return None, 0
    mu = mean_vector / r
    d = vectors.shape[1]
    kappa = _estimate_kappa(r, d)
    return mu, kappa


def _estimate_kappa(r, d):
    if r >= 1.0:
        return float("inf")
    if r <= 0:
        return 0
    if d <= 2:
        return 2 * r / (1 - r**2)
    return (r * (d - r**2)) / (1 - r**2)


def score_corpus(df: pd.DataFrame, extractor: WordVectorExtractor, meta: dict):
    """Fit one vMF concentration (kappa) for a target corpus.

    All rows of ``df`` share one (lemma, pos) and represent a single
    (k, offset) variant; their contextual vectors are pooled into one fit.
    Returns a result dict, or None if fewer than two vectors were extracted.
    """
    word = df["lemma"].iloc[0]
    target_pos = df["pos"].iloc[0]
    contexts = df.to_dict("records")

    vectors = extractor.get_word_vectors(contexts, word, target_pos)
    if len(vectors) < 2:
        return None

    _, kappa = estimate_vmf_parameters(vectors)
    kappa = kappa if kappa is not None else 0
    return {
        "word": word,
        "pos": target_pos,
        "k_senses": meta["k_senses"],
        "baseline_slope": meta["baseline_slope"],
        "applied_slope": meta["applied_slope"],
        # Effective offset; equals the nominal offset unless the slope was
        # clamped to the floor (meta["clamped"]).
        "offset": meta["applied_slope"] - meta["baseline_slope"],
        "clamped": meta["clamped"],
        "vmf_kappa": kappa,
        "vector_count": len(vectors),
    }


def get_corpora_vmf(
    sim_dir: Path,
    output_dir: Path,
    hf_model_name: str = "answerdotai/ModernBERT-base",
) -> pd.DataFrame:
    """Compute a vMF kappa for every simulated corpus under ``sim_dir``.

    Walks ``sim_dir/<lemma>_<pos>/k*_offset_*.csv`` (the layout produced by
    ``simulate_zipfian_corpora``), fits one kappa per corpus, and writes a
    single combined ``vmf_scores.csv`` to ``output_dir``.
    """
    extractor = WordVectorExtractor.from_config(
        ExtractionConfig(hf_model_name=hf_model_name)
    )

    rows = []
    for csv_path in sorted(sim_dir.glob("*/k*_offset_*.csv")):
        # Not csv_path.with_suffix(...): the "0.00" in the variant name confuses
        # pathlib's suffix handling. Swap the trailing ".csv" explicitly.
        meta_path = csv_path.parent / (csv_path.name[: -len(".csv")] + ".meta.json")
        if not meta_path.exists():
            continue  # skip stray CSVs without sidecar metadata

        df = pd.read_csv(csv_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        record = score_corpus(df, extractor, meta)
        if record is None:
            continue

        rows.append(record)
        print(
            f"  {csv_path.parent.name} {csv_path.stem} "
            f"vMF κ: {record['vmf_kappa']:.4f} (n={record['vector_count']})"
        )

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "vmf_scores.csv", index=False)
    return result
