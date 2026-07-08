import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from wic.preprocessing import preprocess_wic_targets
from wic.target_vector_model import (
    HEAD_PARAM_NAMES,
    TargetVectorConfig,
    TargetVectorForWiC,
    WiCTargetDataCollator,
)

# The real project tokenizer (ModernBERT is a fast tokenizer, required for the
# return_offsets_mapping the masking relies on). For the model tests we shrink the
# ModernBERT *config* to a couple of tiny layers so no large weights are downloaded and
# everything runs on CPU, while keeping model_type == the real pipeline's.
_BASE = "answerdotai/ModernBERT-large"


def _tiny_encoder_config():
    cfg = AutoConfig.from_pretrained(_BASE)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    # ModernBERT validates that layer_types matches num_hidden_layers, so shrink it too.
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]
    return cfg


def _tiny_model():
    config = TargetVectorConfig(base_model_name=_BASE, classifier_dropout=0.0)
    encoder = AutoModel.from_config(_tiny_encoder_config())
    return TargetVectorForWiC(config, encoder=encoder)


class TargetMaskTestCase(unittest.TestCase):
    """The span -> subword mask is the model-independent, load-bearing piece."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(_BASE)

    def _decode(self, input_ids, mask):
        selected = [i for i, m in zip(input_ids, mask) if m == 1]
        return self.tokenizer.decode(selected).strip()

    def test_masks_select_the_target_words(self):
        # "bank" occurs at chars 9-13 in sentence1 and 4-8 in sentence2.
        s1 = "I sat on bank of the river."
        s2 = "The bank approved my loan."
        self.assertEqual(s1[9:13], "bank")
        self.assertEqual(s2[4:8], "bank")

        out = preprocess_wic_targets(
            {
                "lemma": ["bank"],
                "sentence1": [s1],
                "sentence2": [s2],
                "start1": [9],
                "end1": [13],
                "start2": [4],
                "end2": [8],
            },
            self.tokenizer,
        )
        input_ids = out["input_ids"][0]
        mask1 = out["target_mask_1"][0]
        mask2 = out["target_mask_2"][0]

        self.assertEqual(len(mask1), len(input_ids))
        self.assertEqual(len(mask2), len(input_ids))
        # Each mask must decode back to exactly the target word.
        self.assertEqual(self._decode(input_ids, mask1), "bank")
        self.assertEqual(self._decode(input_ids, mask2), "bank")
        # The two masks must select disjoint tokens (different occurrences).
        self.assertFalse(any(a and b for a, b in zip(mask1, mask2)))

    def test_prefix_shift_handles_lemma_prefix(self):
        # A real lemma length must still land on the sentence1 target, because the
        # "{lemma}: " prefix shift is derived from len(lemma).
        s1 = "They play chess."
        s2 = "A short play tonight."
        out = preprocess_wic_targets(
            {
                "lemma": ["play"],
                "sentence1": [s1],
                "sentence2": [s2],
                "start1": [s1.index("play")],
                "end1": [s1.index("play") + 4],
                "start2": [s2.index("play")],
                "end2": [s2.index("play") + 4],
            },
            self.tokenizer,
        )
        input_ids = out["input_ids"][0]
        self.assertEqual(self._decode(input_ids, out["target_mask_1"][0]), "play")
        self.assertEqual(self._decode(input_ids, out["target_mask_2"][0]), "play")


class TargetVectorModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(_BASE)

    def setUp(self):
        self.model = _tiny_model()
        self.hidden = self.model.encoder.config.hidden_size

    def _batch(self):
        examples = {
            "lemma": ["bank", "play"],
            "sentence1": ["I sat on bank side.", "They play chess."],
            "sentence2": ["The bank is closed.", "A short play tonight."],
            "start1": [9, 5],
            "end1": [13, 9],
            "start2": [4, 8],
            "end2": [8, 12],
            "label": [0, 1],
        }
        tok = preprocess_wic_targets(examples, self.tokenizer)
        tok["labels"] = examples["label"]
        features = [{k: tok[k][i] for k in tok} for i in range(len(examples["lemma"]))]
        return WiCTargetDataCollator(self.tokenizer)(features)

    def test_forward_shapes_and_loss(self):
        out = self.model(**self._batch())
        self.assertEqual(out.logits.shape, (2, 2))
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.ndim, 0)

    def test_feature_dimension_is_4h(self):
        # The classifier's first Linear must consume a 4*hidden feature.
        first_linear = self.model.classifier[1]
        self.assertEqual(first_linear.in_features, 4 * self.hidden)

    def test_empty_mask_pools_to_zero(self):
        h = torch.randn(1, 5, self.hidden)
        zero_mask = torch.zeros(1, 5, dtype=torch.long)
        pooled = TargetVectorForWiC._pool_target(h, zero_mask)
        torch.testing.assert_close(pooled, torch.zeros(1, self.hidden))

    def test_head_param_names_present(self):
        state = dict(self.model.named_parameters())
        for name in HEAD_PARAM_NAMES:
            self.assertIn(name, state)

    def test_save_reload_roundtrip(self):
        self.model.eval()
        batch = self._batch()
        with torch.no_grad():
            before = self.model(**batch).logits

        with tempfile.TemporaryDirectory() as d:
            self.model.save_pretrained(d)
            # Saved checkpoint must carry the head weights assert_trained_head checks.
            self.assertTrue(
                (Path(d) / "model.safetensors").exists()
                or (Path(d) / "pytorch_model.bin").exists()
            )
            reloaded = TargetVectorForWiC.from_pretrained(d)

        reloaded.eval()
        with torch.no_grad():
            after = reloaded(**batch).logits
        torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
