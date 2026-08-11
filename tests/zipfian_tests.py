import unittest

import numpy as np

from simulation.zipfian import estimate_pooled_slope, estimate_word_slope


def _sample_finite_zipf_counts(
    true_slope: float, n_senses: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw per-sense counts from the finite, rank-truncated Zipf the estimator fits.

    p(rank k) ∝ k**(-true_slope) for k = 1..n_senses, normalised over those ranks
    only -- the same generative model as estimate_word_slope, so a recovered slope
    should match true_slope. (np.random.zipf samples the *infinite* zeta
    distribution, a different model, and would not round-trip here.)
    """
    ranks = np.arange(1, n_senses + 1)
    probs = ranks ** (-true_slope)
    probs /= probs.sum()
    draws = rng.choice(ranks, size=size, p=probs)
    _, counts = np.unique(draws, return_counts=True)
    return counts


class ZipfianTestCase(unittest.TestCase):
    def test_recovers_slope_from_finite_zipf(self):
        rng = np.random.default_rng(298)
        true_slope = 1.2
        n_senses = 5  # the handful-of-senses regime this module targets

        # Large sample so the MLE variance is small relative to our tolerance.
        counts = _sample_finite_zipf_counts(true_slope, n_senses, 200_000, rng)

        estimate, se, status = estimate_word_slope(counts)
        self.assertLess(se, 0.05)
        self.assertEqual(status, "ok")
        # Tolerance comfortably exceeds the sampling SE (~0.004 at this n).
        self.assertAlmostEqual(estimate, true_slope, delta=0.05)

    def test_recovers_across_slope_range(self):
        # Recovery should hold across the shallow/steep range of plausible slopes.
        for true_slope in (0.6, 1.2, 1.8):
            with self.subTest(true_slope=true_slope):
                rng = np.random.default_rng(7)
                counts = _sample_finite_zipf_counts(true_slope, 5, 200_000, rng)
                estimate, _, status = estimate_word_slope(counts)
                self.assertEqual(status, "ok")
                self.assertAlmostEqual(estimate, true_slope, delta=0.05)

    def test_too_few_senses_status(self):
        # n < 3 senses: the slope is not estimated.
        slope, se, status = estimate_word_slope(np.array([10, 4]))
        self.assertEqual(status, "too_few_senses")
        self.assertTrue(np.isnan(slope))
        self.assertTrue(np.isnan(se))

    def test_no_variation_status(self):
        # All senses equally frequent: the likelihood is flat, slope undefined.
        slope, se, status = estimate_word_slope(np.array([5, 5, 5, 5]))
        self.assertEqual(status, "no_variation")
        self.assertTrue(np.isnan(slope))
        self.assertTrue(np.isnan(se))

    def test_order_invariance(self):
        # The function ranks senses by descending count internally, so input
        # order must not change the result.
        counts = np.array([100, 40, 25, 10, 5])
        a = estimate_word_slope(counts)
        b = estimate_word_slope(counts[::-1])
        self.assertEqual(a[2], "ok")
        self.assertAlmostEqual(a[0], b[0])


class PooledZipfianTestCase(unittest.TestCase):
    """The vocabulary-wide fit the simulation's baseline slope comes from."""

    def test_recovers_shared_slope_from_many_small_words(self):
        # The regime that motivates pooling: many words, each with too few
        # observations to fit on its own, all drawn from one shared slope.
        rng = np.random.default_rng(11)
        true_slope = 1.1
        counts = [
            _sample_finite_zipf_counts(true_slope, n_senses, 200, rng)
            for n_senses in (3, 4, 5) * 40
        ]

        fit = estimate_pooled_slope(counts)
        self.assertEqual(fit.status, "ok")
        self.assertEqual(fit.n_words, len(counts))
        self.assertAlmostEqual(fit.slope, true_slope, delta=0.05)
        # Pooling is what buys the precision a single small word cannot reach.
        self.assertLess(fit.se, 0.05)

    def test_varying_sense_counts_use_their_own_normaliser(self):
        # Words with different sense counts have different partition functions. If
        # they were pooled under a single shared normaliser the estimate would be
        # biased; drawing 3- and 5-sense words from one slope and recovering it
        # confirms each word's own n is used.
        rng = np.random.default_rng(5)
        true_slope = 0.8
        counts = [_sample_finite_zipf_counts(true_slope, 3, 400, rng) for _ in range(60)]
        counts += [_sample_finite_zipf_counts(true_slope, 5, 400, rng) for _ in range(60)]

        fit = estimate_pooled_slope(counts)
        self.assertEqual(fit.status, "ok")
        self.assertAlmostEqual(fit.slope, true_slope, delta=0.05)

    def test_skips_uninformative_words(self):
        # Words with <3 senses or no count variation carry no slope information and
        # are dropped, mirroring estimate_word_slope's statuses.
        rng = np.random.default_rng(3)
        informative = [
            _sample_finite_zipf_counts(1.2, 4, 300, rng) for _ in range(30)
        ]
        fit = estimate_pooled_slope(
            informative + [np.array([7, 2]), np.array([5, 5, 5, 5])]
        )
        self.assertEqual(fit.status, "ok")
        self.assertEqual(fit.n_words, len(informative))

    def test_no_fittable_words_status(self):
        fit = estimate_pooled_slope([np.array([7, 2]), np.array([4, 4, 4])])
        self.assertEqual(fit.status, "no_fittable_words")
        self.assertTrue(np.isnan(fit.slope))
        self.assertEqual(fit.n_words, 0)

    def test_dominated_by_high_count_words(self):
        # High-count words contribute proportionally more to the likelihood, so the
        # pooled estimate should sit near the well-attested word's slope rather than
        # midway between it and a noisy low-frequency one.
        rng = np.random.default_rng(19)
        heavy = [_sample_finite_zipf_counts(1.5, 5, 20_000, rng)]
        light = [_sample_finite_zipf_counts(0.5, 5, 20, rng) for _ in range(5)]

        fit = estimate_pooled_slope(heavy + light)
        self.assertEqual(fit.status, "ok")
        self.assertGreater(fit.slope, 1.2)


if __name__ == "__main__":
    unittest.main()
