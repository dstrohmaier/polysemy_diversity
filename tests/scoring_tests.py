import unittest
from unittest import mock

import numpy as np

from vmf.vmf_estimation import _estimate_kappa, estimate_vmf_parameters
from wic.wic_estimation import score_corpus_wic


class VMFScoringTestCase(unittest.TestCase):
    def test_empty_vectors_raises(self):
        with self.assertRaises(ValueError):
            estimate_vmf_parameters(np.empty((0, 3)))

    def test_zero_resultant_returns_none_and_zero(self):
        # Antipodal points cancel exactly: the mean vector is the origin, so no
        # mean direction is defined and concentration is zero.
        vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        mu, kappa = estimate_vmf_parameters(vectors)
        self.assertIsNone(mu)
        self.assertEqual(kappa, 0)

    def test_identical_vectors_give_infinite_kappa(self):
        # All directions coincide -> resultant length r == 1 -> kappa -> inf.
        vectors = np.tile([0.0, 1.0, 0.0], (5, 1))
        mu, kappa = estimate_vmf_parameters(vectors)
        np.testing.assert_allclose(mu, [0.0, 1.0, 0.0])
        self.assertEqual(kappa, float("inf"))

    def test_mu_is_unit_length(self):
        rng = np.random.default_rng(0)
        # A cloud loosely centred on +x so the resultant is non-zero.
        vectors = np.array([1.0, 0.0, 0.0]) + 0.1 * rng.standard_normal((50, 3))
        mu, _ = estimate_vmf_parameters(vectors)
        self.assertAlmostEqual(float(np.linalg.norm(mu)), 1.0, places=10)

    def test_tighter_cluster_gives_higher_kappa(self):
        # The core property the module relies on: directions pointing many ways
        # (sense-diverse) yield low kappa; one consistent direction yields high
        # kappa. A tight cloud must score strictly higher than a spread one.
        rng = np.random.default_rng(1)
        centre = np.array([1.0, 0.0, 0.0])
        tight = centre + 0.05 * rng.standard_normal((200, 3))
        spread = centre + 0.8 * rng.standard_normal((200, 3))

        _, kappa_tight = estimate_vmf_parameters(tight)
        _, kappa_spread = estimate_vmf_parameters(spread)
        self.assertGreater(kappa_tight, kappa_spread)


class EstimateKappaTestCase(unittest.TestCase):
    def test_r_at_or_above_one_is_infinite(self):
        self.assertEqual(_estimate_kappa(1.0, 5), float("inf"))
        self.assertEqual(_estimate_kappa(1.5, 5), float("inf"))

    def test_nonpositive_r_is_zero(self):
        self.assertEqual(_estimate_kappa(0.0, 5), 0)
        self.assertEqual(_estimate_kappa(-0.1, 5), 0)

    def test_low_dimension_branch(self):
        # d <= 2 uses 2r / (1 - r^2).
        r, d = 0.5, 2
        expected = 2 * r / (1 - r**2)
        self.assertAlmostEqual(_estimate_kappa(r, d), expected)

    def test_high_dimension_branch(self):
        # d > 2 uses r * (d - r^2) / (1 - r^2).
        r, d = 0.5, 10
        expected = (r * (d - r**2)) / (1 - r**2)
        self.assertAlmostEqual(_estimate_kappa(r, d), expected)

    def test_kappa_increases_with_resultant_length(self):
        d = 1024  # ModernBERT-large hidden size; the regime this module runs in.
        kappas = [_estimate_kappa(r, d) for r in (0.1, 0.3, 0.6, 0.9)]
        self.assertEqual(kappas, sorted(kappas))


def _entry(id_, label, lemma="run", pos="VERB"):
    """A minimal WiC entry; sentence fields are placeholders since the model
    call is mocked out, but score_corpus_wic reads lemma/pos/id/label."""
    return {
        "id": id_,
        "lemma": lemma,
        "pos": pos,
        "sentence1": "a",
        "sentence2": "b",
        "label": label,
    }


_META = {
    "k_senses": 3,
    "baseline_slope": 1.2,
    "applied_slope": 1.0,  # offset = applied - baseline = -0.2
    "clamped": False,
}


class WiCScoringTestCase(unittest.TestCase):
    """Tests for score_corpus_wic with the model call (_predict_logits) mocked.

    Only the scoring math -- softmax -> p_diff (class 0), argmax -> preds,
    accuracy, row assembly -- is under test; the transformers model is replaced
    by fixed logits so the tests are deterministic and need no model download.
    """

    def _score(self, entries, logits):
        with mock.patch(
            "wic.wic_estimation._predict_logits", return_value=np.asarray(logits)
        ):
            # model/tokenizer are unused once _predict_logits is patched.
            return score_corpus_wic(entries, model=None, tokenizer=None, meta=_META)

    def test_p_diff_is_class_zero_softmax(self):
        # Logits favouring class 0 -> high p_diff; favouring class 1 -> low.
        entries = [_entry("p0", label=0), _entry("p1", label=1)]
        logits = [[2.0, 0.0], [0.0, 2.0]]
        summary, pairs = self._score(entries, logits)

        sm = np.exp(2.0) / (np.exp(2.0) + np.exp(0.0))  # softmax of the larger logit
        self.assertAlmostEqual(pairs[0]["p_diff"], sm)
        self.assertAlmostEqual(pairs[1]["p_diff"], 1 - sm)
        self.assertAlmostEqual(summary["wic_p_diff_mean"], 0.5)

    def test_predictions_and_accuracy(self):
        # preds = argmax(logits). Rows: 0 correct, 1 correct, 1 wrong -> acc 2/3.
        entries = [
            _entry("a", label=0),  # logits pick 0 -> correct
            _entry("b", label=1),  # logits pick 1 -> correct
            _entry("c", label=1),  # logits pick 0 -> wrong
        ]
        logits = [[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]]
        summary, pairs = self._score(entries, logits)

        self.assertEqual([p["pred"] for p in pairs], [0, 1, 0])
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["pair_count"], 3)

    def test_summary_carries_meta_and_offset(self):
        entries = [_entry("a", label=0)]
        summary, _ = self._score(entries, [[1.0, 0.0]])

        self.assertEqual(summary["word"], "run")
        self.assertEqual(summary["pos"], "VERB")
        self.assertEqual(summary["k_senses"], 3)
        self.assertEqual(summary["baseline_slope"], 1.2)
        self.assertEqual(summary["applied_slope"], 1.0)
        self.assertAlmostEqual(summary["offset"], -0.2)
        self.assertFalse(summary["clamped"])

    def test_pair_rows_preserve_ids_and_labels(self):
        entries = [_entry("x", label=0), _entry("y", label=1)]
        _, pairs = self._score(entries, [[1.0, 0.0], [0.0, 1.0]])

        self.assertEqual([p["id"] for p in pairs], ["x", "y"])
        self.assertEqual([p["label"] for p in pairs], [0, 1])
        self.assertTrue(all(isinstance(p["label"], int) for p in pairs))
        # Pair rows carry the Zipfian slopes (the analysis plots performance against
        # the applied slope, so each pair must know its corpus's actual slope).
        self.assertTrue(all(p["baseline_slope"] == 1.2 for p in pairs))
        self.assertTrue(all(p["applied_slope"] == 1.0 for p in pairs))

    def test_empty_entries_rejected(self):
        with self.assertRaises(AssertionError):
            score_corpus_wic([], model=None, tokenizer=None, meta=_META)


if __name__ == "__main__":
    unittest.main()
