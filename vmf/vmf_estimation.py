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

For the shift evaluation this module fits ``kappa`` on each corpus of a (source,
target) pair (down-sampled to equal size) and scores the pair by the log-ratio
``log(kappa_S / kappa_T)`` -- positive when the target is more diverse. Pairs are
enumerated across the simulated corpora produced by ``simulate_zipfian_corpora``;
``get_corpora_vmf_pairs`` writes the collected scores to ``vmf_pair_scores.csv``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from data_processing.vector_extraction import ExtractionConfig, WordVectorExtractor
from simulation.pairing import (
    CorpusPair,
    PairBuilder,
    equalise_indices,
    build_simulated_pairs,
)

logger = logging.getLogger("div")


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


def _corpus_vectors(csv_path: Path, extractor: WordVectorExtractor) -> np.ndarray:
    """Extract the target word's (L2-normalised) contextual vectors for one corpus.

    The simulated corpus carries each occurrence's gold span, so occurrences are
    located by span rather than re-found with spaCy (which would drop occurrences
    whose lemma/POS disagree with the annotation).
    """
    contexts = pd.read_csv(csv_path).to_dict("records")
    return extractor.get_word_vectors_from_spans(contexts)


def score_pair_vmf(
    pair: CorpusPair, extractor: WordVectorExtractor, seed: int = 0
) -> dict:
    """vMF shift score ``log(kappa_S / kappa_T)`` for one (source, target) pair.

    Down-samples both corpora's vectors to equal n before fitting so the
    resultant-length floor (which grows as n shrinks) cancels in the ratio rather
    than contaminating it. Anomalies raise rather than skip: with the >= 30 sentence
    floor a corpus can hardly drop below 2 vectors, and a zero/infinite kappa needs
    perfectly cancelling or perfectly aligned directions -- neither should occur for
    real embeddings, so if they do it is worth a loud failure.
    """
    vecs_s = _corpus_vectors(pair.source.csv_path, extractor)
    vecs_t = _corpus_vectors(pair.target.csv_path, extractor)
    assert len(vecs_s) >= 2 and len(vecs_t) >= 2, (
        f"{pair.lemma_pos}: < 2 vectors for {pair.source.csv_path.stem} or "
        f"{pair.target.csv_path.stem}; a kept corpus should yield >= 2"
    )

    idx_s, idx_t = equalise_indices(len(vecs_s), len(vecs_t), seed=seed)
    vecs_s, vecs_t = vecs_s[idx_s], vecs_t[idx_t]

    _, kappa_s = estimate_vmf_parameters(vecs_s)
    _, kappa_t = estimate_vmf_parameters(vecs_t)
    # A zero or infinite kappa (r == 0 or r >= 1) makes log(kappa_S / kappa_T)
    # undefined; these are not expected for real embeddings, so fail loudly.
    assert kappa_s and kappa_t and not (np.isinf(kappa_s) or np.isinf(kappa_t)), (
        f"{pair.lemma_pos}: degenerate kappa (S={kappa_s}, T={kappa_t}) for "
        f"{pair.source.csv_path.stem}->{pair.target.csv_path.stem}"
    )

    return {
        "lemma_pos": pair.lemma_pos,
        "scheme": pair.scheme,
        "source_variant": pair.source.csv_path.stem,
        "target_variant": pair.target.csv_path.stem,
        "vmf_log_ratio": float(np.log(kappa_s / kappa_t)),
        "n_used": len(vecs_s),
    }


def get_corpora_vmf_pairs(
    sim_dir: Path,
    output_dir: Path,
    hf_model_name: str = "answerdotai/ModernBERT-large",
    seed: int = 0,
    build_corpus_pairs: PairBuilder = build_simulated_pairs,
) -> pd.DataFrame:
    """Compute the vMF shift score for every corpus pair under ``sim_dir``.

    ``build_corpus_pairs`` decides which (source, target) pairs ``sim_dir``
    yields: the default covers the simulation's three comparison schemes, while
    :func:`~simulation.pairing.build_dwug_pairs` gives the diachronic evaluation's single
    pair per lemma. Writes one combined ``vmf_pair_scores.csv`` to ``output_dir``.
    """
    extractor = WordVectorExtractor.from_config(
        ExtractionConfig(hf_model_name=hf_model_name)
    )
    pairs = build_corpus_pairs(sim_dir)
    assert pairs, (
        f"no corpus pairs found under {sim_dir}; is the directory layout the one the "
        f"chosen enumerator expects (see score_data.py --dataset)?"
    )

    rows = []
    for pair in pairs:
        record = score_pair_vmf(pair, extractor, seed=seed)
        rows.append(record)
        logger.info(
            "%s [%s] %s->%s vMF log-ratio: %.4f (n=%d)",
            pair.lemma_pos, record["scheme"], record["source_variant"],
            record["target_variant"], record["vmf_log_ratio"], record["n_used"],
        )

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "vmf_pair_scores.csv", index=False)
    return result
