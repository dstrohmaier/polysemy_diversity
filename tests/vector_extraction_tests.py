"""Tests for the batched contextual-vector extractor.

The extractor feeds the vMF and cosine scorers, whose ``equalise_indices`` step
down-samples *positionally* from a seeded RNG. Row order and row count are therefore
part of the contract, not incidental: a batching change that reorders or drops
differently would silently score a different subset of occurrences. Most of what is
pinned here is that contract, plus the padding behaviour batching introduces.

The real ModernBERT tokenizer is used (a fast tokenizer, required for the
``return_offsets_mapping`` the span alignment relies on) with a shrunken *config*, so
no large weights download and everything runs on CPU -- the same trick as
``wic_model_tests``.
"""

import unittest

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data_processing.vector_extraction import WordVectorExtractor

_BASE = "answerdotai/ModernBERT-large"


def _tiny_extractor(tokenizer):
    """An extractor over a 2-layer, 64-dim ModernBERT: real code path, tiny weights."""
    cfg = AutoConfig.from_pretrained(_BASE)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    # ModernBERT validates layer_types against num_hidden_layers.
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]
    torch.manual_seed(0)
    return WordVectorExtractor(AutoModel.from_config(cfg), tokenizer, device="cpu")


def _ctx(sentence, target):
    """A context dict with the gold span of ``target`` inside ``sentence``."""
    start = sentence.index(target)
    return {"sentence": sentence, "start": start, "end": start + len(target)}


# Deliberately varied lengths: with equal-length inputs padding is never exercised, so
# a pad-contamination bug would pass every batching test.
VARIED = [
    _ctx("The bank approved my loan.", "bank"),
    _ctx(
        "I sat on the bank of the river and watched the slow brown water carry "
        "leaves and broken branches past the old stone bridge downstream.",
        "bank",
    ),
    _ctx("A bank holiday.", "bank"),
    _ctx(
        "Every bank in the district reported record profits this quarter, which "
        "the regulator described as unusual but not in itself alarming.",
        "bank",
    ),
    _ctx("She will bank the cheque tomorrow morning before work.", "bank"),
    _ctx("Snow drifted against the bank.", "bank"),
]


class BatchedExtractionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(_BASE)
        cls.extractor = _tiny_extractor(cls.tokenizer)
        cls.hidden = cls.extractor.model.config.hidden_size

    def test_batched_matches_unbatched(self):
        """Batching must not change the vectors, only how many forwards produce them."""
        one = self.extractor.get_word_vectors_from_spans(VARIED, batch_size=1)
        many = self.extractor.get_word_vectors_from_spans(VARIED, batch_size=8)
        self.assertEqual(one.shape, many.shape)
        np.testing.assert_allclose(one, many, atol=1e-5)

    def test_padding_does_not_affect_short_row(self):
        """A short sentence batched beside a long one must match its solo extraction.

        This is the direct test for pad contamination: if a padded position leaked
        into the pooled mean, the short row would shift while still looking healthy
        (finite and unit-norm).
        """
        short, long_ = VARIED[2], VARIED[1]
        solo = self.extractor.get_word_vectors_from_spans([short], batch_size=1)
        together = self.extractor.get_word_vectors_from_spans(
            [short, long_], batch_size=2
        )
        np.testing.assert_allclose(solo[0], together[0], atol=1e-5)

    def test_order_preserved(self):
        """Row i must be the embedding of context i -- the equalise_indices contract."""
        batched = self.extractor.get_word_vectors_from_spans(VARIED, batch_size=4)
        for i, ctx in enumerate(VARIED):
            solo = self.extractor.get_word_vectors_from_spans([ctx], batch_size=1)
            np.testing.assert_allclose(
                batched[i], solo[0], atol=1e-5, err_msg=f"row {i} is not context {i}"
            )

    def test_batch_size_does_not_change_count(self):
        """Non-divisor batch sizes must not lose a tail chunk."""
        counts = {
            bs: len(self.extractor.get_word_vectors_from_spans(VARIED, batch_size=bs))
            for bs in (1, 3, 7, 32)
        }
        self.assertEqual(set(counts.values()), {len(VARIED)}, counts)

    def test_contexts_without_a_usable_span_are_dropped_in_order(self):
        """Dropped contexts must not disturb the surviving rows' order."""
        bad_span = {"sentence": "A bank holiday.", "start": 5, "end": 5}  # end <= start
        missing = {"sentence": "Another bank sentence."}  # no start/end at all
        contexts = [VARIED[0], bad_span, VARIED[2], missing, VARIED[5]]

        got = self.extractor.get_word_vectors_from_spans(contexts, batch_size=2)
        expected = self.extractor.get_word_vectors_from_spans(
            [VARIED[0], VARIED[2], VARIED[5]], batch_size=2
        )
        self.assertEqual(len(got), 3)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_empty_returns_empty_with_hidden_size(self):
        """Downstream reads ``.shape[1]``, so the empty case must stay 2-D."""
        got = self.extractor.get_word_vectors_from_spans([])
        self.assertEqual(got.shape, (0, self.hidden))

    def test_rows_are_unit_norm(self):
        """loo_centroid_distance's closed form is only valid for unit-norm rows."""
        got = self.extractor.get_word_vectors_from_spans(VARIED, batch_size=4)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)

    def test_multiple_target_layers(self):
        """Layer averaging must work per batch row.

        No production path passes a non-default ``target_layers``, and the unbatched
        code indexed batch row 0 while stacking layers -- so this generalisation is
        only exercised here.
        """
        got = self.extractor.get_word_vectors_from_spans(
            VARIED, target_layers=(-1, -2), batch_size=4
        )
        self.assertEqual(got.shape, (len(VARIED), self.hidden))
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)

        one = self.extractor.get_word_vectors_from_spans(
            VARIED, target_layers=(-1, -2), batch_size=1
        )
        np.testing.assert_allclose(one, got, atol=1e-5)

    def test_multi_subword_target_is_pooled(self):
        """A target spanning several subwords still yields one unit-norm row."""
        ctx = _ctx("The photosynthesis experiment failed.", "photosynthesis")
        got = self.extractor.get_word_vectors_from_spans([ctx])
        self.assertEqual(got.shape, (1, self.hidden))
        self.assertAlmostEqual(float(np.linalg.norm(got[0])), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
