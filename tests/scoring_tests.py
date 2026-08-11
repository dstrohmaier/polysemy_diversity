import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pandas as pd

from analysis.scored.comparative import n_sensitivity_table
from analysis.scored.pooled import (
    SMALL_N_NOTE,
    SMALL_N_THRESHOLD,
    add_small_n_note,
    discover_pos_datasets,
    pooled_correlation_table,
    pos_scheme_correlation_table,
)
from analysis.scored.stats import (
    GT_SHIFT_COLS,
    UNKNOWN_POS,
    add_pos_column,
    correlation_table,
    pos_from_lemma,
)
from cosine.cosine_estimation import loo_centroid_distance, score_pair_cosine
from simulation.diversity import diversity_shift, evenness_shift, hill_diversity
from simulation.pairing import (
    SLOPE_STRIDES,
    CorpusPair,
    _offset_index,
    build_simulated_pairs,
    equalise_indices,
)
from vmf.vmf_estimation import (
    _estimate_kappa,
    estimate_vmf_parameters,
    score_pair_vmf,
)
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


class EvennessShiftTestCase(unittest.TestCase):
    def test_shift_identical_is_zero(self):
        probs = {"a": 0.7, "b": 0.3}
        self.assertEqual(evenness_shift(probs, probs), 0.0)

    def test_shift_is_q1_minus_q0(self):
        # E = 1D/0D, so log E(T)/E(S) is the Shannon shift less the richness shift.
        source = {"a": 0.8, "b": 0.15, "c": 0.05}
        target = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        self.assertAlmostEqual(
            evenness_shift(source, target),
            diversity_shift(source, target, 1) - diversity_shift(source, target, 0),
        )

    def test_shift_positive_when_target_more_even(self):
        # Same richness, so the shift isolates the change in evenness alone.
        source = {"a": 0.8, "b": 0.1, "c": 0.1}
        target = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        self.assertGreater(evenness_shift(source, target), 0.0)

    def test_shift_ignores_pure_richness_change(self):
        # Doubling a uniform distribution's support leaves E = 1 on both sides: a
        # change in richness with the evenness held fixed must not move the metric.
        self.assertAlmostEqual(evenness_shift([0.25] * 4, [0.125] * 8), 0.0)

    def test_uniform_target_is_maximally_even(self):
        # E <= 1 with equality only for a uniform distribution, so a uniform target
        # cannot be less even than any source over the same support.
        source = {"a": 0.6, "b": 0.25, "c": 0.15}
        target = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        self.assertAlmostEqual(evenness_shift(target, target), 0.0)
        self.assertGreater(evenness_shift(source, target), 0.0)
        self.assertLess(evenness_shift(target, source), 0.0)


def _span(pair) -> float:
    """Offset distance a pair covers, rounded past the grid's float drift."""
    return round(pair.target.offset - pair.source.offset, 2)


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

    def test_along_slope_emits_every_stride(self):
        """Each stride in SLOPE_STRIDES contributes its own set of pairs.

        A one-step slope move shifts diversity several times less than a one-step k
        move, so the wider stride exists to put the two families on a comparable
        footing. Needs more offsets than the shared fixture has, hence its own corpora.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # The full ladder, so every stride has room to partition it.
            root = _write_corpora(
                Path(tmp),
                [("run_VERB", 3, round(-0.5 + 0.1 * i, 2)) for i in range(11)],
            )
            slope = [
                p for p in build_simulated_pairs(root) if p.scheme == "along_slope"
            ]
            distances = sorted(
                {round(p.target.offset - p.source.offset, 2) for p in slope}
            )
            self.assertEqual(
                distances, [round(0.1 * s, 2) for s in sorted(SLOPE_STRIDES)]
            )
            # Eleven offsets: ten neighbour pairs, and three 3-step pairs partitioning
            # the ladder from its low end (-0.5 -> -0.2 -> +0.1 -> +0.4).
            self.assertEqual(sum(1 for p in slope if _span(p) == 0.1), 10)
            self.assertEqual(
                sorted(
                    (round(p.source.offset, 2), round(p.target.offset, 2))
                    for p in slope
                    if _span(p) == 0.3
                ),
                [(-0.5, -0.2), (-0.2, 0.1), (0.1, 0.4)],
            )
            # Orientation must hold at every stride, not just the neighbour one.
            for p in slope:
                self.assertEqual(p.source.k, p.target.k)
                self.assertLess(p.source.offset, p.target.offset)

    def test_wide_stride_partition_is_anchored_not_positional(self):
        """A lemma missing variants still cuts the ladder at the same offsets.

        The simulation drops variants a lemma has too few senses or sentences for, so
        lemmata reach the pairing stage with different subsets of the offset grid.
        Partitioning by position in each lemma's own list would slide the cut points
        onto whatever offsets that lemma happens to have, scattering the wide
        comparisons across the axis instead of pinning them to shared boundaries.
        """
        def wide_pairs(offsets, lemma):
            """Pairs whose endpoints both sit on a partition boundary.

            A gappy lemma can make two *neighbours* span 0.3 as well, so the span
            alone does not identify a wide pair; the boundary test does.
            """
            root = _write_corpora(
                Path(tempfile.mkdtemp(dir=self._tmp.name)),
                [(lemma, 3, o) for o in offsets],
            )
            stride = max(SLOPE_STRIDES)
            return sorted(
                (round(p.source.offset, 2), round(p.target.offset, 2))
                for p in build_simulated_pairs(root)
                if p.scheme == "along_slope"
                and _span(p) == round(0.1 * stride, 2)
                and _offset_index(p.source) % stride == 0
                and _offset_index(p.target) % stride == 0
            )

        # Variants missing *inside* a segment cost nothing: the cuts stay on the
        # shared boundaries. A positional walk would have paired -0.5 with +0.0 here,
        # that being this lemma's own third element.
        self.assertEqual(
            wide_pairs([-0.5, -0.3, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4], "interior_VERB"),
            [(-0.5, -0.2), (-0.2, 0.1), (0.1, 0.4)],
        )
        # A missing *boundary* (-0.2) drops only the two segments that needed it,
        # rather than sliding the remaining cuts onto other offsets.
        self.assertEqual(
            wide_pairs([-0.5, -0.4, -0.3, 0.0, 0.1, 0.2, 0.3, 0.4], "boundary_VERB"),
            [(0.1, 0.4)],
        )

    def test_along_slope_strides_emit_distinct_pairs(self):
        """No pair is emitted twice, which would double-count it in the analysis."""
        slope = [
            (p.lemma_pos, p.source.csv_path.stem, p.target.csv_path.stem)
            for p in self.pairs
            if p.scheme == "along_slope"
        ]
        self.assertEqual(len(slope), len(set(slope)))

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

    def _df(self, scores, predictor_vals, group="along_slope", n_used=None, pos=None):
        # ``pos`` is added only when a test asks for it, so the single-key tests
        # exercise exactly the frame shape the per-PoS analysis passes.
        frame = pd.DataFrame(
            {
                "scheme": group if isinstance(group, list) else [group] * len(scores),
                "score": scores,
                "gt": predictor_vals,
                "n_used": n_used if n_used is not None else [100] * len(scores),
            }
        )
        if pos is not None:
            frame["pos"] = pos if isinstance(pos, list) else [pos] * len(scores)
        return frame

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

    def test_n_used_summarised_alongside_rho(self):
        # A rho is only interpretable next to the corpus size it was computed at.
        df = self._df([0.1, 0.4, 0.9], [0.2, 0.5, 0.8], n_used=[40, 100, 220])
        row = correlation_table(df, "score", ["gt"], group_col="scheme").iloc[0]
        self.assertEqual(row["n_used_median"], 100.0)
        self.assertEqual(row["n_used_min"], 40.0)
        self.assertEqual(row["n_used_max"], 220.0)

    def test_n_used_summary_covers_only_correlated_rows(self):
        # The dropna decides which rows reach spearmanr; the n summary must describe
        # that same subset, not the pre-drop frame.
        df = self._df([0.1, 0.4, 0.9, 0.95], [0.2, 0.5, 0.8, np.nan],
                      n_used=[40, 100, 220, 9999])
        row = correlation_table(df, "score", ["gt"], group_col="scheme").iloc[0]
        self.assertEqual(row["n"], 3)
        self.assertEqual(row["n_used_max"], 220.0)

    def test_missing_n_used_column_rejected(self):
        df = self._df([0.1, 0.4, 0.9], [0.2, 0.5, 0.8]).drop(columns="n_used")
        with self.assertRaises(AssertionError):
            correlation_table(df, "score", ["gt"], group_col="scheme")

    def test_single_element_list_group_col_matches_string(self):
        # pandas yields a scalar key for a string groupby but a 1-tuple for a
        # one-element list; both must produce the same table, or the pooled mode's
        # multi-key path would silently differ from the per-PoS one.
        df = self._df([0.1, 0.4, 0.9], [0.2, 0.5, 0.8])
        as_string = correlation_table(df, "score", ["gt"], group_col="scheme")
        as_list = correlation_table(df, "score", ["gt"], group_col=["scheme"])
        pd.testing.assert_frame_equal(as_string, as_list)

    def test_two_key_group_col_yields_one_column_per_key(self):
        df = self._df(
            [0.1, 0.4, 0.9, 0.2, 0.5, 0.8],
            [0.2, 0.5, 0.8, 0.1, 0.6, 0.7],
            group=["primary"] * 3 + ["along_k"] * 3,
            pos=["NOUN"] * 3 + ["VERB"] * 3,
        )
        out = correlation_table(df, "score", ["gt"], group_col=["pos", "scheme"])
        self.assertIn("pos", out.columns)
        self.assertIn("scheme", out.columns)
        # The failure mode if the group tuple is used as a dict key verbatim.
        self.assertNotIn("['pos', 'scheme']", out.columns)
        self.assertEqual(len(out), 2)

    def test_two_key_group_values_are_unpacked_not_tupled(self):
        df = self._df(
            [0.1, 0.4, 0.9],
            [0.2, 0.5, 0.8],
            group="primary",
            pos="NOUN",
        )
        out = correlation_table(df, "score", ["gt"], group_col=["pos", "scheme"])
        self.assertEqual(out.iloc[0]["pos"], "NOUN")
        self.assertEqual(out.iloc[0]["scheme"], "primary")

    def test_missing_group_column_rejected(self):
        # A clear assertion, not a bare KeyError out of pandas.
        df = self._df([0.1, 0.4, 0.9], [0.2, 0.5, 0.8])
        with self.assertRaises(AssertionError):
            correlation_table(df, "score", ["gt"], group_col=["pos"])


class NSensitivityTestCase(unittest.TestCase):
    """Error-vs-n diagnostic: separates a weak method from one run below the sample
    size it needs (readme's vMF/Nagata regime caveat)."""

    def _loaded(self, scores, gts, ns, schemes=None, pos=None):
        # Every ground-truth shift column gets the same values: this diagnostic is
        # about error-vs-n, not about telling the measures apart. Built from
        # GT_SHIFT_COLS so a newly added measure cannot leave the fixture short a
        # column the table under test requires. ``pos`` is omitted unless a test
        # asks for it, keeping the single-key fixtures at the per-PoS frame shape.
        frame = pd.DataFrame(
            {
                "scheme": schemes if schemes is not None else ["primary"] * len(scores),
                "vmf_log_ratio": scores,
                **{col: gts for col in GT_SHIFT_COLS.values()},
                "n_used": ns,
            }
        )
        if pos is not None:
            frame["pos"] = pos
        return {"vMF": frame}

    def test_error_shrinking_with_n_gives_negative_rho(self):
        # Error falls as n grows: the signature of a sample-size-driven deficit.
        loaded = self._loaded([1.0, 0.7, 0.55, 0.5], [0.5] * 4, [25, 50, 100, 400])
        out = n_sensitivity_table(loaded)
        row = out[out["predictor"] == "gt_shift_q2"].iloc[0]
        self.assertLess(row["rho_err_vs_n"], 0)
        self.assertEqual(row["n_used_min"], 25.0)
        self.assertEqual(row["n_used_max"], 400.0)

    def test_constant_n_flagged_not_dropped(self):
        # The simulation can hold corpus size fixed within a scheme; that is a
        # distinct, reportable reason for a missing rho.
        loaded = self._loaded([1.0, 0.7, 0.55], [0.5] * 3, [100, 100, 100])
        out = n_sensitivity_table(loaded)
        self.assertTrue((out["note"] == "constant n_used").all())
        self.assertTrue(out["rho_err_vs_n"].isna().all())

    def test_unscored_method_skipped(self):
        # analyse_comparative drops methods whose CSV is absent; the table follows.
        self.assertTrue(n_sensitivity_table({"vMF": None}).empty)

    def test_group_col_list_matches_string_default(self):
        loaded = self._loaded([1.0, 0.7, 0.55, 0.5], [0.5] * 4, [25, 50, 100, 400])
        pd.testing.assert_frame_equal(
            n_sensitivity_table(loaded),
            n_sensitivity_table(loaded, group_col=["scheme"]),
        )

    def test_two_key_grouping_splits_by_pos(self):
        loaded = self._loaded(
            [1.0, 0.7, 0.55, 0.9, 0.6, 0.5],
            [0.5] * 6,
            [25, 50, 100, 25, 50, 100],
            schemes=["primary"] * 6,
            pos=["NOUN"] * 3 + ["VERB"] * 3,
        )
        out = n_sensitivity_table(loaded, group_col=["pos", "scheme"])
        q2 = out[out["predictor"] == "gt_shift_q2"]
        self.assertEqual(sorted(q2["pos"]), ["NOUN", "VERB"])


class PosDerivationTestCase(unittest.TestCase):
    """The two evaluations tag PoS differently ("act_NOUN" vs "graft_nn"), but the
    pooled analysis groups on one vocabulary, so the mapping must be total and stable."""

    def test_simulation_tags_pass_through_uppercase(self):
        self.assertEqual(pos_from_lemma("act_NOUN"), "NOUN")
        self.assertEqual(pos_from_lemma("lose_VERB"), "VERB")
        self.assertEqual(pos_from_lemma("quick_ADJ"), "ADJ")
        self.assertEqual(pos_from_lemma("quickly_ADV"), "ADV")

    def test_dwug_tags_map_to_the_simulation_vocabulary(self):
        # Asserted equal to the simulation's result, so the two datasets provably
        # land in one bucket rather than merely both being uppercase.
        self.assertEqual(pos_from_lemma("graft_nn"), pos_from_lemma("act_NOUN"))
        self.assertEqual(pos_from_lemma("bar_vb"), pos_from_lemma("lose_VERB"))

    def test_multiword_lemma_splits_from_the_right(self):
        # Guards the rsplit-vs-split choice: a lemma may contain an underscore.
        self.assertEqual(pos_from_lemma("take_off_VERB"), "VERB")

    def test_unrecognised_tag_is_flagged_not_dropped(self):
        # A pooled run over four datasets must not abort on one odd directory name.
        self.assertEqual(pos_from_lemma("foo_XYZ"), UNKNOWN_POS)
        self.assertEqual(pos_from_lemma("nolemma"), UNKNOWN_POS)

    def test_add_pos_column_leaves_caller_frame_untouched(self):
        df = pd.DataFrame({"lemma_pos": ["act_NOUN", "graft_nn"]})
        out = add_pos_column(df)
        self.assertNotIn("pos", df.columns)
        self.assertEqual(list(out["pos"]), ["NOUN", "NOUN"])


class SmallNNoteTestCase(unittest.TestCase):
    """A thin cell is reported with its rho and CI plus a flag, not dropped: the
    adverb vocabulary is 30 lemmata, and dropping thin cells would remove exactly the
    column the PoS comparison exists to show."""

    def _table(self, ns, notes=None, n_col="n"):
        return pd.DataFrame(
            {
                n_col: ns,
                "spearmanr": [0.5] * len(ns),
                "note": notes if notes is not None else [""] * len(ns),
            }
        )

    def test_small_cell_flagged_and_kept(self):
        out = add_small_n_note(self._table([5, 500]))
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["note"], SMALL_N_NOTE)
        self.assertEqual(out.iloc[1]["note"], "")
        self.assertTrue(np.isfinite(out.iloc[0]["spearmanr"]))

    def test_note_is_appended_not_replaced(self):
        out = add_small_n_note(self._table([2], notes=["n<3"]))
        self.assertEqual(out.iloc[0]["note"], f"n<3; {SMALL_N_NOTE}")

    def test_cell_at_threshold_not_flagged(self):
        # The comparison is "<", not "<=".
        out = add_small_n_note(self._table([SMALL_N_THRESHOLD]))
        self.assertEqual(out.iloc[0]["note"], "")

    def test_n_pairs_column_supported(self):
        # n_sensitivity_table names its count n_pairs, not n.
        out = add_small_n_note(self._table([5], n_col="n_pairs"), n_col="n_pairs")
        self.assertEqual(out.iloc[0]["note"], SMALL_N_NOTE)


class DiscoverPosDatasetsTestCase(unittest.TestCase):
    """Discovery must require the *scores* side: the corpora root holds non-dataset
    entries (a real ``most_diverse_noun/stale/`` exists) and dirs simulated but not
    yet scored, either of which would join the pool as a silently empty dataset."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.scores = self.root / "scores"
        self.corpora = self.root / "corpora"
        self.scores.mkdir()
        self.corpora.mkdir()

    def _make(self, name, with_scores=True, with_corpora=True):
        if with_corpora:
            (self.corpora / name).mkdir(parents=True, exist_ok=True)
        if with_scores:
            method_dir = self.scores / name / "cosine"
            method_dir.mkdir(parents=True, exist_ok=True)
            (method_dir / "cosine_pair_scores.csv").write_text("lemma_pos\n")
        else:
            (self.scores / name).mkdir(parents=True, exist_ok=True)

    def test_dataset_present_in_both_roots_is_found(self):
        self._make("most_diverse_noun")
        self.assertEqual(
            discover_pos_datasets(self.scores, self.corpora), ["most_diverse_noun"]
        )

    def test_corpus_dir_without_scores_is_skipped(self):
        # The "stale/" and "simulated but never scored" case.
        self._make("most_diverse_noun")
        self._make("stale", with_scores=False)
        self.assertEqual(
            discover_pos_datasets(self.scores, self.corpora), ["most_diverse_noun"]
        )

    def test_scores_dir_without_corpora_is_skipped(self):
        self._make("most_diverse_noun")
        self._make("orphan_scores", with_corpora=False)
        self.assertEqual(
            discover_pos_datasets(self.scores, self.corpora), ["most_diverse_noun"]
        )

    def test_result_is_sorted(self):
        # Determinism of the pooled frame's row order.
        for name in ["most_diverse_verb", "most_diverse_adj", "most_diverse_noun"]:
            self._make(name)
        self.assertEqual(
            discover_pos_datasets(self.scores, self.corpora),
            ["most_diverse_adj", "most_diverse_noun", "most_diverse_verb"],
        )


class PosSchemeTableTestCase(unittest.TestCase):
    """The two-way breakdown: whether the method ranking established on nouns holds
    for the other parts of speech, and whether a PoS effect is really a scheme one."""

    def _loaded(self):
        rng = np.random.default_rng(0)
        rows = []
        for pos, lemma_tag in [("NOUN", "NOUN"), ("VERB", "VERB")]:
            for scheme in ["primary", "along_k"]:
                for i in range(6):
                    rows.append(
                        {
                            "lemma_pos": f"w{i}_{lemma_tag}",
                            "pos": pos,
                            "scheme": scheme,
                            "source_variant": "a",
                            "target_variant": "b",
                            "cosine_log_ratio": float(rng.normal()),
                            "n_used": 100,
                            **{c: float(rng.normal()) for c in GT_SHIFT_COLS.values()},
                        }
                    )
        return {"cosine": pd.DataFrame(rows)}

    def test_schema_has_pos_and_scheme_columns(self):
        out = pos_scheme_correlation_table(self._loaded())
        for col in ("method", "pos", "scheme", "predictor", "spearmanr", "n_lemmata"):
            self.assertIn(col, out.columns)
        # note is the reader's footnote column and belongs last.
        self.assertEqual(out.columns[-1], "note")

    def test_row_count_is_methods_times_pos_times_schemes_times_predictors(self):
        # The combinatorial guard: catches a silently dropped group.
        out = pos_scheme_correlation_table(self._loaded())
        self.assertEqual(len(out), 1 * 2 * 2 * len(GT_SHIFT_COLS))

    def test_each_pos_scheme_cell_present(self):
        out = pos_scheme_correlation_table(self._loaded())
        cells = set(zip(out["pos"], out["scheme"]))
        self.assertEqual(
            cells,
            {("NOUN", "primary"), ("NOUN", "along_k"),
             ("VERB", "primary"), ("VERB", "along_k")},
        )

    def test_rowless_frame_yields_empty_table_not_keyerror(self):
        # discover_pos_datasets admits a dataset on the strength of the pair-scores
        # CSV existing, so a header-only file reaches here. correlation_table then
        # returns a frame with no columns at all, which the group-count merge and
        # the note reorder would both index into.
        empty = self._loaded()["cosine"].iloc[0:0]
        for table in (pos_scheme_correlation_table, pooled_correlation_table):
            with self.subTest(table=table.__name__):
                self.assertTrue(table({"cosine": empty}).empty)


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


class StubCache:
    """Serves fixed vectors per corpus path, standing in for CorpusVectorCache."""

    def __init__(self, by_path):
        self.by_path = by_path

    def vectors(self, csv_path):
        return self.by_path[Path(csv_path).name]


def _pair(source_name: str, target_name: str) -> CorpusPair:
    """A pair whose handles carry only the csv_path the scorers read."""
    def handle(name):
        return SimpleNamespace(
            lemma_pos="run_VERB",
            csv_path=Path("run_VERB") / name,
            meta_path=Path("x"),
            data_path=Path("y"),
        )

    return CorpusPair("run_VERB", "primary", handle(source_name), handle(target_name))


class ScorePairTestCase(unittest.TestCase):
    """The scorers read their vectors through the cache, and the log-ratio is the
    documented direction: positive when the target is more diverse."""

    def setUp(self):
        rng = np.random.default_rng(7)
        centre = np.array([1.0, 0.0, 0.0])
        # Source tight (low diversity), target spread (high diversity), equal n so
        # equalise_indices keeps everything and the expected value is computable.
        self.tight = _unit_rows(centre + 0.05 * rng.standard_normal((20, 3)))
        self.spread = _unit_rows(centre + 0.9 * rng.standard_normal((20, 3)))
        self.cache = StubCache({"s.csv": self.tight, "t.csv": self.spread})

    def test_vmf_log_ratio_matches_kappa_ratio(self):
        record = score_pair_vmf(_pair("s.csv", "t.csv"), self.cache, seed=0)
        expected = np.log(
            estimate_vmf_parameters(self.tight)[1]
            / estimate_vmf_parameters(self.spread)[1]
        )
        self.assertAlmostEqual(record["vmf_log_ratio"], float(expected), places=10)
        self.assertEqual(record["n_used"], 20)
        # kappa falls as diversity rises, so a more diverse target gives a positive
        # log-ratio -- the sign convention the readme and the analysis depend on.
        self.assertGreater(record["vmf_log_ratio"], 0.0)

    def test_cosine_log_ratio_matches_diversity_ratio(self):
        record = score_pair_cosine(_pair("s.csv", "t.csv"), self.cache, seed=0)
        expected = np.log(
            loo_centroid_distance(self.spread) / loo_centroid_distance(self.tight)
        )
        self.assertAlmostEqual(record["cosine_log_ratio"], float(expected), places=10)
        self.assertEqual(record["n_used"], 20)
        self.assertGreater(record["cosine_log_ratio"], 0.0)

    def test_unequal_corpora_are_downsampled_to_the_smaller(self):
        cache = StubCache({"s.csv": self.tight[:12], "t.csv": self.spread})
        for scorer in (score_pair_vmf, score_pair_cosine):
            record = scorer(_pair("s.csv", "t.csv"), cache, seed=0)
            self.assertEqual(record["n_used"], 12, scorer.__name__)

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
