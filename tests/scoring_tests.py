import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import pandas as pd

from analysis.scored.stats import correlation_table
from cosine.cosine_estimation import loo_centroid_distance
from simulation.diversity import diversity_shift, hill_diversity
from simulation.pairing import build_simulated_pairs, equalise_indices
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
            # trainer/tokenizer are unused once _predict_logits is patched.
            return score_corpus_wic(entries, trainer=None, tokenizer=None, meta=_META)

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
            score_corpus_wic([], trainer=None, tokenizer=None, meta=_META)


class HillDiversityTestCase(unittest.TestCase):
    def test_q0_is_richness(self):
        # q=0 counts the senses with non-zero probability, ignoring their weights.
        self.assertEqual(hill_diversity({"a": 0.9, "b": 0.05, "c": 0.05}, 0), 3.0)

    def test_q0_drops_zero_probability_senses(self):
        self.assertEqual(hill_diversity({"a": 0.5, "b": 0.5, "c": 0.0}, 0), 2.0)

    def test_q2_is_inverse_simpson(self):
        probs = {"a": 0.6, "b": 0.3, "c": 0.1}
        expected = 1.0 / sum(p * p for p in probs.values())
        self.assertAlmostEqual(hill_diversity(probs, 2), expected)

    def test_q1_uniform_equals_support(self):
        # Shannon diversity of a uniform distribution over k senses is exactly k.
        self.assertAlmostEqual(hill_diversity([0.25] * 4, 1), 4.0)

    def test_diversity_more_even_is_more_diverse(self):
        skewed = {"a": 0.8, "b": 0.1, "c": 0.1}
        even = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        for q in (1, 2):
            self.assertLess(hill_diversity(skewed, q), hill_diversity(even, q))

    def test_shift_identical_is_zero(self):
        probs = {"a": 0.7, "b": 0.3}
        for q in (0, 1, 2):
            self.assertEqual(diversity_shift(probs, probs, q), 0.0)

    def test_shift_positive_when_target_more_diverse(self):
        source = {"a": 0.8, "b": 0.2}
        target = {"a": 0.5, "b": 0.5}
        self.assertGreater(diversity_shift(source, target, 2), 0.0)


def _write_corpora(root: Path, variants):
    """Materialise ``(lemma_pos, k, offset)`` variants as the on-disk layout.

    ``build_simulated_pairs`` reads the corpus dir itself, so the pairing tests set up
    real (empty) CSVs whose stems carry the k/offset the logic parses.
    """
    for lemma_pos, k, offset in variants:
        stem = f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}"
        word_dir = root / lemma_pos
        word_dir.mkdir(parents=True, exist_ok=True)
        (word_dir / f"{stem}.csv").write_text("lemma\n", encoding="utf-8")
    return root


class PairingTestCase(unittest.TestCase):
    def setUp(self):
        # One lemma, k in {3, 4}, offset in {-0.1, 0.0, 0.1}.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = _write_corpora(
            Path(self._tmp.name),
            [("run_VERB", k, o) for k in (3, 4) for o in (-0.1, 0.0, 0.1)],
        )
        self.n_corpora = 6
        self.pairs = build_simulated_pairs(root)

    def test_primary_source_is_low_diversity_anchor(self):
        # Every "primary" pair sources from the lowest-k, steepest-slope corpus.
        primary = [p for p in self.pairs if p.scheme == "primary"]
        self.assertTrue(
            all(p.source.csv_path.stem == "k3_offset_m0.10" for p in primary)
        )
        self.assertEqual(len(primary), self.n_corpora - 1)

    def test_along_slope_source_is_steeper(self):
        # Same k, so the source (lower diversity) has the smaller offset.
        for p in (p for p in self.pairs if p.scheme == "along_slope"):
            self.assertEqual(p.source.k, p.target.k)
            self.assertLess(p.source.offset, p.target.offset)

    def test_along_k_source_has_lower_k(self):
        for p in (p for p in self.pairs if p.scheme == "along_k"):
            self.assertEqual(p.source.offset, p.target.offset)
            self.assertLess(p.source.k, p.target.k)

    def test_single_corpus_lemma_yields_no_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_corpora(Path(tmp), [("lone_NOUN", 3, 0.0)])
            self.assertEqual(build_simulated_pairs(root), [])

    def test_duplicate_variant_skipped_not_crashed(self):
        # Two stems parsing to the same (k, offset) must not abort the run; the
        # degenerate neighbour pair is skipped, other pairs still produced. The
        # duplicate needs distinct *filenames* that parse alike, which the trailing
        # zero of "p0.000" supplies -- two files cannot share one stem on disk.
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_corpora(Path(tmp), [("dup_VERB", 4, 0.0)])
            word_dir = root / "dup_VERB"
            for stem in ("k3_offset_p0.00", "k3_offset_p0.000"):
                (word_dir / f"{stem}.csv").write_text("lemma\n", encoding="utf-8")

            pairs = build_simulated_pairs(root)  # must not raise
        # The k3->k4 comparison still exists; no pair has identical endpoints.
        self.assertTrue(any(p.source.k != p.target.k for p in pairs))
        for p in pairs:
            self.assertFalse(p.source.k == p.target.k and p.source.offset == p.target.offset)


class CorrelationTableTestCase(unittest.TestCase):
    """The degenerate-predictor guard: a constant predictor (e.g. the q=0 richness
    shift within a same-k scheme) must be flagged, not silently NaN'd via scipy."""

    def _df(self, scores, predictor_vals, group="along_slope"):
        return pd.DataFrame(
            {"scheme": [group] * len(scores), "score": scores, "gt": predictor_vals}
        )

    def test_constant_predictor_flagged(self):
        # gt is identically 0 (the q0 same-k case): rho undefined, note explains why.
        df = self._df([0.1, 0.4, 0.9], [0.0, 0.0, 0.0])
        out = correlation_table(df, "score", ["gt"], group_col="scheme")
        row = out.iloc[0]
        self.assertTrue(np.isnan(row["spearmanr"]))
        self.assertEqual(row["note"], "constant predictor")
        self.assertEqual(row["n"], 3)

    def test_small_sample_flagged_distinctly(self):
        # Fewer than three points: a different, non-degenerate reason.
        df = self._df([0.1, 0.4], [0.2, 0.5])
        out = correlation_table(df, "score", ["gt"], group_col="scheme")
        self.assertEqual(out.iloc[0]["note"], "n<3")

    def test_varying_predictor_correlates(self):
        # A clean monotone case still produces a finite rho and empty note.
        df = self._df([0.1, 0.4, 0.9], [0.2, 0.5, 0.8])
        out = correlation_table(df, "score", ["gt"], group_col="scheme")
        row = out.iloc[0]
        self.assertAlmostEqual(row["spearmanr"], 1.0)
        self.assertEqual(row["note"], "")


def _naive_loo_centroid_distance(vectors: np.ndarray) -> float:
    """O(n^2) reference: recompute each leave-one-out centroid explicitly.

    Deliberately literal (loop + delete row + mean + normalise) so it validates the
    vectorised ``S - x_i`` identity in loo_centroid_distance rather than mirroring it.
    """
    n = len(vectors)
    terms = []
    for i in range(n):
        others = np.delete(vectors, i, axis=0)
        centroid = others.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        terms.append(1.0 - float(vectors[i] @ centroid))
    return float(np.mean(terms))


def _unit_rows(arr) -> np.ndarray:
    """L2-normalise each row (the extractor's precondition for the functional)."""
    arr = np.asarray(arr, dtype=float)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class CosineDiversityTestCase(unittest.TestCase):
    def test_identical_vectors_give_zero(self):
        # Every LOO centroid points the same way as the held-out vector -> cos 1 -> 0.
        vecs = np.tile([1.0, 0.0, 0.0], (5, 1))
        self.assertAlmostEqual(loo_centroid_distance(vecs), 0.0)

    def test_spread_exceeds_tight_cluster(self):
        # The property the baseline relies on: more-dispersed usages score higher.
        rng = np.random.default_rng(0)
        centre = np.array([1.0, 0.0, 0.0])
        tight = _unit_rows(centre + 0.05 * rng.standard_normal((50, 3)))
        spread = _unit_rows(centre + 0.8 * rng.standard_normal((50, 3)))
        self.assertGreater(
            loo_centroid_distance(spread), loo_centroid_distance(tight)
        )

    def test_matches_naive_reference(self):
        # The key correctness guard: the O(n*d) closed form equals the explicit
        # leave-one-out computation on an asymmetric, non-trivial cloud.
        rng = np.random.default_rng(1)
        vecs = _unit_rows(rng.standard_normal((40, 8)))
        self.assertAlmostEqual(
            loo_centroid_distance(vecs), _naive_loo_centroid_distance(vecs), places=10
        )

    def test_n_equals_two_is_pair_distance(self):
        # With n == 2 the "others" centroid is just the single other vector, so each
        # term is 1 - cos(x_0, x_1); the mean equals that one distance.
        vecs = _unit_rows([[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> cos 0 -> dist 1
        self.assertAlmostEqual(loo_centroid_distance(vecs), 1.0)

    def test_returns_plain_float(self):
        vecs = _unit_rows(np.random.default_rng(2).standard_normal((6, 4)))
        self.assertIsInstance(loo_centroid_distance(vecs), float)

    def test_too_few_vectors_asserts(self):
        with self.assertRaises(AssertionError):
            loo_centroid_distance(_unit_rows([[1.0, 0.0, 0.0]]))


class EqualiseIndicesTestCase(unittest.TestCase):
    def test_both_indices_trim_to_smaller(self):
        idx_a, idx_b = equalise_indices(10, 3, seed=1)
        self.assertEqual(len(idx_a), 3)
        self.assertEqual(len(idx_b), 3)

    def test_smaller_side_is_full_identity(self):
        # The shorter side keeps every row, in order (arange), so the caller's
        # unchanged data passes through untouched.
        idx_a, idx_b = equalise_indices(10, 3, seed=1)
        np.testing.assert_array_equal(idx_b, np.arange(3))
        # The trimmed side's indices are a sorted subset of its range.
        self.assertTrue(set(idx_a.tolist()) <= set(range(10)))
        self.assertEqual(list(idx_a), sorted(idx_a))

    def test_indices_apply_to_array_and_list_alike(self):
        # The point of the index-based API: one index set, applied to either an
        # array of vectors (vMF) or a list of pair dicts (WiC).
        vecs = np.arange(20).reshape(10, 2)
        dicts = [{"i": i} for i in range(10)]
        idx_a, _ = equalise_indices(10, 4, seed=2)
        kept_vecs = vecs[idx_a]
        kept_dicts = [dicts[i] for i in idx_a]
        self.assertEqual(len(kept_vecs), 4)
        self.assertEqual([d["i"] for d in kept_dicts], idx_a.tolist())

    def test_equal_length_is_identity_both_sides(self):
        idx_a, idx_b = equalise_indices(3, 3, seed=1)
        np.testing.assert_array_equal(idx_a, np.arange(3))
        np.testing.assert_array_equal(idx_b, np.arange(3))

    def test_deterministic_under_seed(self):
        first, _ = equalise_indices(10, 3, seed=42)
        second, _ = equalise_indices(10, 3, seed=42)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
