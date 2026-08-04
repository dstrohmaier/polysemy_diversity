"""Cosine-similarity baseline for the shift-in-diversity evaluation.

A deliberately simple counterpart to the vMF and WiC methods: measure a corpus's
usage diversity from the geometry of its (L2-normalised) contextual vectors alone,
then report the shift as ``log(D_cos_T / D_cos_S)`` between source and target
(target in the numerator, as cosine diversity is a direct -- not inverse -- measure).

Per-corpus diversity is the leave-one-out mean distance to centroid
(:func:`loo_centroid_distance`): each vector's cosine distance to the mean direction
of all the *other* vectors, averaged. Leave-one-out avoids comparing a vector to a
centroid it helped form, which would otherwise shrink apparent diversity (worst for
small corpora).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from data_processing.vector_extraction import ExtractionConfig, WordVectorExtractor
from data_processing.simulation_loading import iter_corpora
from simulation.pairing import CorpusPair, enumerate_pairs, equalise_indices

logger = logging.getLogger("div")


def loo_centroid_distance(vectors: np.ndarray) -> float:
    """Diversity = ``mean_i (1 - cos(x_i, centroid of the *other* vectors))``.

    Leave-one-out (LOO): each vector is compared to the mean direction of every
    vector *except itself*, removing the self-inclusion bias of comparing a vector to
    a centroid it helped form (which shrinks apparent diversity, worst for small n).

    Vectors are unit-norm (the extractor L2-normalises), so with ``S = sum_j x_j`` the
    LOO centroid for ``i`` is ``(S - x_i) / (n - 1)``; cosine drops the positive
    scalar, leaving ``cos_i = (x_i . S - 1) / ||S - x_i||`` with
    ``||S - x_i||^2 = ||S||^2 - 2 (x_i . S) + 1``. That is an O(n*d) closed form -- one
    matmul ``X @ S`` plus ``||S||^2`` -- with no O(n^2) pairwise matrix. Requires n >= 2.
    """
    n = len(vectors)
    assert n >= 2, "loo_centroid_distance needs at least 2 vectors"
    total = vectors.sum(axis=0)
    dots = vectors @ total  # x_i . S, shape (n,)
    sq_norms = float(total @ total) - 2.0 * dots + 1.0  # ||S - x_i||^2
    # ||S - x_i|| == 0 means the other vectors cancel exactly onto -x_i: unreachable
    # for real embeddings and would yield a nan cosine that poisons the mean.
    assert np.all(sq_norms > 0), "degenerate LOO centroid (others cancel a vector)"
    cos = (dots - 1.0) / np.sqrt(sq_norms)
    return float((1.0 - cos).mean())


def _corpus_vectors(csv_path: Path, extractor: WordVectorExtractor) -> np.ndarray:
    """Extract the target word's (L2-normalised) contextual vectors for one corpus.

    Occurrences are located by their stored gold span, matching the vMF scorer's
    :func:`vmf.vmf_estimation._corpus_vectors` so the two methods see the same set.
    """
    contexts = pd.read_csv(csv_path).to_dict("records")
    return extractor.get_word_vectors_from_spans(contexts)


def score_pair_cosine(
    pair: CorpusPair, extractor: WordVectorExtractor, seed: int = 0
) -> dict:
    """Cosine baseline shift ``log(D_cos_T / D_cos_S)`` for one (source, target) pair.

    Cosine diversity is a *direct* diversity measure (larger = more diverse), unlike
    the vMF concentration kappa which is inverse; the target therefore goes in the
    numerator so that, as with every method, positive => target more diverse. Vectors
    are extracted, down-sampled to equal n, then measured by
    :func:`loo_centroid_distance` (which depends on n, so it runs *after* equalising).
    Anomalies raise rather than skip: with the >= 30 sentence floor a corpus can hardly
    drop below 2 vectors, and a non-positive diversity (which would make the log
    undefined) needs perfectly identical usages.
    """
    vecs_s = _corpus_vectors(pair.source.csv_path, extractor)
    vecs_t = _corpus_vectors(pair.target.csv_path, extractor)
    assert len(vecs_s) >= 2 and len(vecs_t) >= 2, (
        f"{pair.lemma_pos}: < 2 vectors for {pair.source.csv_path.stem} or "
        f"{pair.target.csv_path.stem}; a kept corpus should yield >= 2"
    )

    idx_s, idx_t = equalise_indices(len(vecs_s), len(vecs_t), seed=seed)
    vecs_s, vecs_t = vecs_s[idx_s], vecs_t[idx_t]
    div_s = loo_centroid_distance(vecs_s)
    div_t = loo_centroid_distance(vecs_t)
    assert div_s > 0 and div_t > 0, (
        f"{pair.lemma_pos}: non-positive cosine diversity (S={div_s}, T={div_t}) for "
        f"{pair.source.csv_path.stem}->{pair.target.csv_path.stem}"
    )

    return {
        "lemma_pos": pair.lemma_pos,
        "scheme": pair.scheme,
        "source_variant": pair.source.csv_path.stem,
        "target_variant": pair.target.csv_path.stem,
        "cosine_log_ratio": float(np.log(div_t / div_s)),
        "n_used": len(vecs_s),
    }


def get_corpora_cosine_pairs(
    sim_dir: Path,
    output_dir: Path,
    hf_model_name: str = "answerdotai/ModernBERT-large",
    seed: int = 0,
) -> pd.DataFrame:
    """Compute the cosine baseline shift score for every corpus pair under ``sim_dir``.

    Writes one combined ``cosine_pair_scores.csv`` to ``output_dir``.
    """
    extractor = WordVectorExtractor.from_config(
        ExtractionConfig(hf_model_name=hf_model_name)
    )
    pairs = enumerate_pairs(list(iter_corpora(sim_dir)))

    rows = []
    for pair in pairs:
        record = score_pair_cosine(pair, extractor, seed=seed)
        rows.append(record)
        logger.info(
            "%s [%s] %s->%s cosine log-ratio: %.4f (n=%d)",
            pair.lemma_pos, record["scheme"], record["source_variant"],
            record["target_variant"], record["cosine_log_ratio"], record["n_used"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "cosine_pair_scores.csv", index=False)
    return result
