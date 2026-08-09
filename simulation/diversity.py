"""Hill-number diversity measures over a sense distribution.

The simulation stores each corpus's design sense distribution as ``sense_probs``
in its ``.meta.json`` sidecar (see :func:`simulation.corpus_simulation.simulate_word_corpus`).
The ground truth for the shift evaluation is the *diversity shift*
``log(qD(T) / qD(S))`` between a source (S) and target (T) corpus, for the three
standard Hill orders q in {0, 1, 2} (richness, Shannon, Simpson diversity), plus the
*evenness shift* ``log(E(T) / E(S))`` for the evenness ratio ``E = 1D / 0D``.

The simulation varies its corpora along two dimensions -- richness (the number of
senses k) and evenness (the Zipfian slope) -- while every method returns a single
score, so the evaluation needs a ground-truth target for each dimension separately:
q=0 isolates richness, and E isolates evenness.

These are pure functions of ``sense_probs`` -- nothing is read from the corpus or
persisted; the analysis layer computes them on the fly from the existing sidecars.
"""

import numpy as np

# The three Hill orders reported throughout the shift evaluation: richness (0),
# Shannon diversity (1), Simpson diversity (2). See the readme's diversity table.
STANDARD_ORDERS: tuple[int, ...] = (0, 1, 2)

# Name of the evenness measure wherever ground-truth measures are keyed by order
# alongside the integer q's (sidecar dicts, shift-column suffixes). Spelled out rather
# than given a pseudo-q because E is a *ratio of two* orders, not an order itself.
EVENNESS_KEY = "evenness"


def hill_diversity(sense_probs: dict[str, float] | list[float], q: int) -> float:
    """The Hill number ``{}^{q}D`` of a sense distribution.

    ``{}^{q}D = (sum_i p_i^q)^{1/(1-q)}`` in general, with the standard closed forms
    at the three orders used here:

    * ``q = 0`` -- richness, the number of senses with non-zero probability.
    * ``q = 1`` -- ``exp(-sum_i p_i ln p_i)`` (the limit; Shannon diversity).
    * ``q = 2`` -- ``1 / sum_i p_i^2`` (inverse Simpson concentration).

    ``sense_probs`` may be the ``{sense: prob}`` mapping stored in ``.meta.json`` or
    a bare sequence of probabilities; only the values are used. Zero-probability
    senses are dropped so they neither inflate richness nor break the ``q = 1`` log.
    """
    probs = np.asarray(
        list(sense_probs.values()) if isinstance(sense_probs, dict) else sense_probs,
        dtype=float,
    )
    probs = probs[probs > 0]
    if probs.size == 0:
        raise ValueError("sense_probs has no positive-probability sense")

    if q == 0:
        return float(probs.size)
    if q == 1:
        return float(np.exp(-np.sum(probs * np.log(probs))))
    if q == 2:
        return float(1.0 / np.sum(probs**2))
    return float(np.sum(probs**q) ** (1.0 / (1.0 - q)))


def diversity_shift(
    source_probs: dict[str, float] | list[float],
    target_probs: dict[str, float] | list[float],
    q: int,
) -> float:
    """Ground-truth diversity shift ``log(qD(target) / qD(source))`` at order ``q``.

    Positive when the target corpus is more diverse than the source, matching the
    orientation of every method's log-ratio score (see the readme's WiC/vMF sign
    discussion). Identical distributions give exactly ``0``.
    """
    return float(
        np.log(hill_diversity(target_probs, q) / hill_diversity(source_probs, q))
    )


def evenness_shift(
    source_probs: dict[str, float] | list[float],
    target_probs: dict[str, float] | list[float],
) -> float:
    """Ground-truth evenness shift ``log(E(target) / E(source))`` for ``E = 1D / 0D``.

    Since E is that ratio, this is just the q=1 shift minus the q=0 one: the Shannon
    shift with the pure-richness part divided out. Oriented like
    :func:`diversity_shift` -- positive when the target's senses are more evenly
    spread, zero for identical distributions.
    """
    return diversity_shift(source_probs, target_probs, 1) - diversity_shift(
        source_probs, target_probs, 0
    )
