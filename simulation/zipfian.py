"""Estimate the Zipfian slope of the sense-frequency distribution.

We fit a one-parameter Zipf law
    p(rank) ∝ rank**(-slope)
to observed sense counts (senses ranked by descending frequency), in two scopes:

* :func:`estimate_word_slope` -- one word's counts, used descriptively.
* :func:`estimate_pooled_slope` -- **one slope for the whole vocabulary**, which is
  what the simulation applies. See that function for why the simulation uses a single
  pooled baseline rather than a per-word one.

With only a handful of senses per word the per-word maximum-likelihood slope is
unbiased but high-variance (SE ≈ 0.1 for ~200 observations over a few senses),
so every estimate is reported together with its standard error and the
sense/observation counts it was derived from. Pooling across the vocabulary is what
buys the precision the per-word fits lack.
"""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore


def zipfian_probs_for_senses(senses: list[str], slope: float) -> dict[str, float]:
    """Zipfian probabilities over an explicit, frequency-ranked list of senses.

    ``senses[0]`` is treated as rank 1 (highest probability). Takes the sense
    list directly, so it works on a truncated top-k subset of a verb's inventory.
    """
    weights = np.array([(i + 1) ** (-slope) for i in range(len(senses))])
    weights /= weights.sum()
    return {s: float(w) for s, w in zip(senses, weights)}


@dataclass(frozen=True)
class SlopeFit:
    lemma: str
    pos: str
    slope: float  # NaN when not fittable
    se: float  # standard error of the slope, NaN when not fittable
    n_senses: int
    n_obs: int
    status: str  # "ok" | "too_few_senses" | "no_variation" | "no_convergence"


def estimate_word_slope(
    counts: np.ndarray,
    *,
    init: float = 1.2,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[float, float, str]:
    """MLE of the Zipf slope from one word's sense counts.

    Parameters
    ----------
    counts : array of sense occurrence counts (order does not matter; the
        function ranks senses by descending count internally).

    Returns
    -------
    (slope, se, status). ``slope`` and ``se`` are NaN unless ``status == "ok"``.
    The standard error is ``1/sqrt(H)`` where ``H`` is the Hessian of the
    negative log-likelihood (the observed Fisher information).
    """
    counts_np = np.sort(np.asarray(counts, dtype=float))[::-1]  # rank 1 == most frequent
    n = len(counts_np)

    if n < 3:
        return float("nan"), float("nan"), "too_few_senses"
    if np.all(counts_np == counts_np[0]):
        # all senses equally frequent: likelihood is flat, slope undefined
        return float("nan"), float("nan"), "no_variation"

    counts_j = jnp.asarray(counts_np, dtype=jnp.float64)
    ranks = jnp.arange(1, n + 1, dtype=jnp.float64)

    @jax.jit
    def negloglik(slope: jnp.ndarray) -> jnp.ndarray:
        """Zipfian negative log-likelihood.

        log p(k) = -slope*log(k) - log(H(n, slope)),  H = sum_k k**(-slope).
        H is computed via a stable log-sum-exp rather than in linear space.
        """
        log_unnorm = -slope * jnp.log(ranks)
        log_Z = jax.nn.logsumexp(log_unnorm)
        log_probs = log_unnorm - log_Z
        return -jnp.dot(counts_j, log_probs)

    grad_fn = jax.jit(jax.grad(negloglik))
    hessian_fn = jax.jit(jax.hessian(negloglik))

    slope = init
    for _ in range(max_iter):
        g = float(grad_fn(slope))
        h = float(hessian_fn(slope))
        if abs(h) < 1e-15:
            return float("nan"), float("nan"), "no_convergence"
        step = g / h  # Newton step
        slope -= step
        if abs(step) < tol:
            h = float(hessian_fn(slope))
            return slope, h ** -0.5, "ok"

    return float("nan"), float("nan"), "no_convergence"


@dataclass(frozen=True)
class PooledSlopeFit:
    """The single vocabulary-wide slope the simulation applies to every word."""

    slope: float
    se: float
    n_words: int  # words contributing to the likelihood
    n_obs: int  # sense occurrences summed over those words
    status: str  # "ok" | "no_fittable_words" | "no_convergence"


def estimate_pooled_slope(
    counts_per_word: list[np.ndarray],
    *,
    init: float = 1.2,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> PooledSlopeFit:
    """MLE of one Zipf slope shared by every word in ``counts_per_word``.

    Why pooled rather than per-word
    -------------------------------
    The simulation offsets each corpus's slope from a baseline, and the analysis grids
    the results by that offset. A per-word baseline makes the offset axis incoherent:
    offset ``-0.3`` is a different applied slope for every lemma, so a grid column
    pools genuinely different distributions and comparisons across lemmata and PoS
    become uninterpretable. One vocabulary-wide baseline makes an offset mean the same
    applied slope everywhere, which is the whole point of the design grid.

    It is also the better-identified estimate: with 3-5 senses per word the per-word
    MLE has SE ≈ 0.1, whereas pooling hundreds of words drives that down by an order of
    magnitude. The trade is realism -- the applied slope no longer tracks any
    individual lemma's empirical sense distribution.

    The likelihood
    --------------
    Ranks are commensurate across words (rank 1 is "most frequent sense" for every
    lemma), but the *normaliser* is not: a 3-sense word and a 5-sense word have
    different partition functions ``H(n, slope) = sum_{r=1..n} r**(-slope)``. So the
    pooled negative log-likelihood sums each word's term with its **own** ``n``:

        -sum_w sum_r c_{w,r} * [ -slope*log(r) - log H(n_w, slope) ]

    Concatenating all counts into one long vector and fitting that would instead treat
    word 2's rank-1 sense as competing with word 1's -- a different, wrong model.

    Words with fewer than 3 senses or with no count variation carry no information
    about the slope and are dropped, matching :func:`estimate_word_slope`'s statuses.
    Because high-count words contribute proportionally more to the likelihood, the
    pooled estimate is dominated by well-attested lemmata rather than by the noisy
    low-frequency tail.
    """
    usable = [
        np.sort(np.asarray(c, dtype=float))[::-1]
        for c in counts_per_word
        if len(c) >= 3 and not np.all(np.asarray(c) == np.asarray(c)[0])
    ]
    if not usable:
        return PooledSlopeFit(
            float("nan"), float("nan"), 0, 0, "no_fittable_words"
        )

    n_obs = int(sum(c.sum() for c in usable))

    # Group words by sense count so each distinct n contributes one logsumexp over its
    # rank vector, rather than one per word. The likelihood only sees a word through
    # (its n, its per-rank counts), so summing the counts within an n-group first is
    # exact, not an approximation -- and it collapses hundreds of words onto a handful
    # of distinct n values.
    by_n: dict[int, np.ndarray] = {}
    for c in usable:
        n = len(c)
        by_n[n] = by_n.get(n, np.zeros(n)) + c

    sizes = sorted(by_n)
    counts_j = [jnp.asarray(by_n[n], dtype=jnp.float64) for n in sizes]
    ranks_j = [jnp.arange(1, n + 1, dtype=jnp.float64) for n in sizes]

    @jax.jit
    def negloglik(slope: jnp.ndarray) -> jnp.ndarray:
        total = 0.0
        for counts_n, ranks_n in zip(counts_j, ranks_j):
            log_unnorm = -slope * jnp.log(ranks_n)
            log_probs = log_unnorm - jax.nn.logsumexp(log_unnorm)
            total = total - jnp.dot(counts_n, log_probs)
        return total

    grad_fn = jax.jit(jax.grad(negloglik))
    hessian_fn = jax.jit(jax.hessian(negloglik))

    slope = init
    for _ in range(max_iter):
        g = float(grad_fn(slope))
        h = float(hessian_fn(slope))
        if abs(h) < 1e-15:
            return PooledSlopeFit(
                float("nan"), float("nan"), len(usable), n_obs, "no_convergence"
            )
        step = g / h
        slope -= step
        if abs(step) < tol:
            h = float(hessian_fn(slope))
            return PooledSlopeFit(
                slope, h**-0.5, len(usable), n_obs, "ok"
            )

    return PooledSlopeFit(
        float("nan"), float("nan"), len(usable), n_obs, "no_convergence"
    )


def sense_counts_for_words(
    wsd_df: pd.DataFrame,
    targets: list[tuple[str, str]],
) -> list[np.ndarray]:
    """Per-word sense-occurrence count vectors for ``targets``, for pooled fitting.

    Counts come from the full sense inventory and the raw occurrences, matching
    :func:`estimate_slopes_for_words`: the baseline describes the word's real sense
    distribution and must not depend on the simulation's k or on the post-dedup
    ``min_examples`` filter applied later during sampling.
    """
    out = []
    for lemma, pos in targets:
        subset = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]
        counts = subset["sense"].value_counts().to_numpy()
        if counts.sum() > 0:
            out.append(counts)
    return out


def estimate_slopes_for_words(
    wsd_df: pd.DataFrame,
    targets: list[tuple[str, str]],
) -> pd.DataFrame:
    """Fit a Zipf sense-slope for each (lemma, pos) in ``targets``.

    Returns a DataFrame with one row per target:
    [lemma, pos, slope, se, n_senses, n_obs, status].
    """
    fits: list[SlopeFit] = []
    for lemma, pos in targets:
        subset = wsd_df[(wsd_df["lemma"] == lemma) & (wsd_df["pos"] == pos)]
        sense_counts = subset["sense"].value_counts().to_numpy()
        n_obs = int(sense_counts.sum())

        if n_obs == 0:
            fits.append(
                SlopeFit(lemma, pos, float("nan"), float("nan"), 0, 0, "too_few_senses")
            )
            continue

        slope, se, status = estimate_word_slope(sense_counts)
        fits.append(
            SlopeFit(lemma, pos, slope, se, len(sense_counts), n_obs, status)
        )

    return pd.DataFrame(fits)
