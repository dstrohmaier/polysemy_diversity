"""Estimate the Zipfian slope of the sense-frequency distribution per word.

For each target (lemma, pos) we fit a one-parameter Zipf law
    p(rank) ∝ rank**(-slope)
to the observed sense counts (senses ranked by descending frequency).

With only a handful of senses per word the maximum-likelihood slope is
unbiased but high-variance (SE ≈ 0.1 for ~200 observations over a few senses),
so every estimate is reported together with its standard error and the
sense/observation counts it was derived from.
"""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore


def zipfian_probs_for_senses(senses: list[str], slope: float) -> dict[str, float]:
    """Zipfian probabilities over an explicit, frequency-ranked list of senses.

    ``senses[0]`` is treated as rank 1 (highest probability). Unlike
    ``corpus_simulation.zipfian_sense_probs`` this takes the sense list directly,
    so it works on a truncated top-k subset of a verb's inventory.
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
