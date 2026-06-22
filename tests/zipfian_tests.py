import unittest

import numpy as np

from simulation.zipfian import estimate_word_slope


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


if __name__ == "__main__":
    unittest.main()
