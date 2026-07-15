"""Target-vector WiC model: classify sense match from `[u; v; |u−v|]`.

The model the training and scoring pipelines use. Rather than mean-pooling the whole
sequence, it locates the target word in each sentence, extracts its contextual vector
— `u` (sentence1) and `v` (sentence2) — and feeds the InferSent-style interaction
feature ``[u; v; |u−v|]`` (dimension ``3 * hidden_size``) into a small MLP classifier.

The two target vectors are pooled from a single joined encoding of
``"{lemma}: {sentence1}" [SEP] sentence2`` using the ``target_mask_1``/``target_mask_2``
multi-hot masks produced by :func:`wic.preprocessing.preprocess_wic_targets`.

Being a :class:`~transformers.PreTrainedModel` subclass, it round-trips through
``save_pretrained``/``from_pretrained``, which the ``wic+fews`` transfer stage relies
on (a saved ``final/`` dir is reloaded as the stage-2 base).
"""

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput

# Head parameter names as they appear in the saved state dict (the two Linear layers of
# the Sequential head at indices 1 and 4). wic_estimation.py's assert_trained_head checks
# for these to detect an un-fine-tuned checkpoint.
HEAD_PARAM_NAMES = ("classifier.1.weight", "classifier.4.weight")


class TargetVectorConfig(PretrainedConfig):
    model_type = "target_vector_wic"

    def __init__(
        self,
        base_model_name: str = "answerdotai/ModernBERT-large",
        classifier_dropout: float = 0.1,
        num_labels: int = 2,
        encoder_config: dict | None = None,
        **kwargs,
    ):
        self.base_model_name = base_model_name
        self.classifier_dropout = classifier_dropout
        # Serialised architecture of the wrapped encoder, so a saved checkpoint rebuilds
        # the exact encoder (its hidden size etc.) on reload rather than re-deriving it
        # from base_model_name. Populated by from_base; consumed in __init__ on reload.
        self.encoder_config = encoder_config
        super().__init__(num_labels=num_labels, **kwargs)


class TargetVectorForWiC(PreTrainedModel):
    """WiC classifier over the target-vector interaction feature ``[u; v; |u−v|]``."""

    config_class = TargetVectorConfig

    def __init__(self, config: TargetVectorConfig, encoder=None):
        super().__init__(config)
        # On a fresh build ``encoder`` carries the pretrained weights (see from_base);
        # on ``from_pretrained`` reload it is None and we rebuild an empty encoder from
        # the serialised encoder_config (so its architecture matches the saved weights),
        # which the saved state dict then fills.
        if encoder is None:
            if config.encoder_config is not None:
                enc_dict = dict(config.encoder_config)
                encoder_config = AutoConfig.for_model(
                    enc_dict.pop("model_type"), **enc_dict
                )
            else:
                encoder_config = AutoConfig.from_pretrained(config.base_model_name)
            encoder = AutoModel.from_config(encoder_config)
        self.encoder = encoder
        # Persist the encoder's architecture so reloads reconstruct it faithfully.
        config.encoder_config = self.encoder.config.to_dict()

        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden, config.num_labels),
        )
        # post_init sets up backend state (tied-weight keys etc.) and initialises the
        # head. It skips modules already flagged initialised, so a pretrained/loaded
        # encoder is left untouched; on from_pretrained the saved state dict overwrites
        # the head afterwards.
        self.post_init()

    @classmethod
    def from_base(cls, base_model_name: str, **config_kwargs) -> "TargetVectorForWiC":
        """Build from a base HF encoder checkpoint, loading its pretrained weights."""
        config = TargetVectorConfig(base_model_name=base_model_name, **config_kwargs)
        encoder = AutoModel.from_pretrained(base_model_name)
        return cls(config, encoder=encoder)

    @staticmethod
    def _pool_target(hidden_states, target_mask):
        """Masked mean of ``hidden_states`` over ``target_mask`` (B, L) → (B, H).

        A row whose mask is all-zero (target truncated away) pools to the zero vector
        rather than dividing by zero.
        """
        mask = target_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / counts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        target_mask_1=None,
        target_mask_2=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = outputs.last_hidden_state  # (B, L, H)

        u = self._pool_target(hidden_states, target_mask_1)
        v = self._pool_target(hidden_states, target_mask_2)
        feat = torch.cat([u, v, (u - v).abs()], dim=-1)  # (B, 3H)
        logits = self.classifier(feat)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class WiCTargetDataCollator:
    """Dynamically pad ``input_ids``/``attention_mask`` and both target masks.

    ``DataCollatorWithPadding`` pads the tokenizer fields but leaves the extra
    ``target_mask_1``/``target_mask_2`` columns as ragged lists. We pad those masks with
    zeros to the batch's padded length (a padded position is never a target).
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        mask_1 = [f.pop("target_mask_1") for f in features]
        mask_2 = [f.pop("target_mask_2") for f in features]
        batch = self._base(features)
        padded_len = batch["input_ids"].shape[1]

        def pad(masks):
            return torch.tensor(
                [m + [0] * (padded_len - len(m)) for m in masks], dtype=torch.long
            )

        batch["target_mask_1"] = pad(mask_1)
        batch["target_mask_2"] = pad(mask_2)
        return batch


def load_wic_model(path_or_name: str) -> TargetVectorForWiC:
    """Load a trained target-vector model from a saved dir, or build one from a base name.

    A saved ``final/`` directory (produced by ``save_pretrained``) reloads with its
    trained encoder+head; a bare HF model name builds a fresh model with pretrained
    encoder weights and a randomly-initialised head.
    """
    from pathlib import Path

    if Path(path_or_name).is_dir():
        return TargetVectorForWiC.from_pretrained(path_or_name)
    return TargetVectorForWiC.from_base(path_or_name)
