"""Shared WiC tokenization used by both training and scoring."""

import logging

logger = logging.getLogger("div")


def preprocess_wic(examples, tokenizer):
    """Tokenize WiC sentence pairs as "{lemma}: {sentence1}" + sentence2.

    The "word: sentence1" prefix guides the model's attention onto the target
    word's context. Returns the tokenizer output (no labels); callers add a
    "labels" column as needed. Dynamic padding is left to the DataCollator.
    """
    first_sentences = [
        f"{w}: {s}" for w, s in zip(examples["lemma"], examples["sentence1"])
    ]
    return tokenizer(
        first_sentences,
        examples["sentence2"],
        truncation=True,
        max_length=256,
        padding=False,
    )


# Prefix prepended to sentence1 by both preprocessors: "{lemma}: ". The target span
# recorded against the raw sentence1 must be shifted right by len(lemma) + len(": ").
_PREFIX_SUFFIX = ": "


def _target_mask(offsets, sequence_ids, segment, span_start, span_end):
    """Multi-hot mask over subwords overlapping ``[span_start, span_end)`` in ``segment``.

    ``segment`` is the tokenizer sequence id (0 = sentence1/prefix, 1 = sentence2).
    Uses the same half-open overlap rule as ``WordVectorExtractor._embed_span``:
    a subword is selected when its char span overlaps the target span. Special tokens
    (``sequence_id is None`` or zero-width offset) are never selected. Returns a list of
    0/1 ints the length of the token sequence; all-zero if the target was truncated away.
    """
    mask = []
    for (tok_start, tok_end), seq_id in zip(offsets, sequence_ids):
        hit = (
            seq_id == segment
            and tok_end > tok_start  # skip special tokens (span (0, 0))
            and not (tok_end <= span_start or tok_start >= span_end)
        )
        mask.append(1 if hit else 0)
    return mask


def preprocess_wic_targets(examples, tokenizer):
    """Tokenize the joined WiC pair and emit target-word subword masks for u and v.

    Produces the same ``"{lemma}: {sentence1}" [SEP] sentence2`` encoding as
    :func:`preprocess_wic`, plus ``target_mask_1``/``target_mask_2``: multi-hot masks
    selecting the subwords of the target occurrence in sentence1 (segment 0) and
    sentence2 (segment 1) respectively. The target-vector model masked-mean-pools these
    into u and v. Requires ``start1/end1/start2/end2`` (character spans against the raw
    sentences) on each example. An all-zero mask means the target was pushed past the
    ``max_length`` truncation; the example is kept and a warning is logged.
    """
    first_sentences = [
        f"{w}: {s}" for w, s in zip(examples["lemma"], examples["sentence1"])
    ]
    tokenized = tokenizer(
        first_sentences,
        examples["sentence2"],
        truncation=True,
        max_length=256,
        padding=False,
        return_offsets_mapping=True,
    )

    offset_batch = tokenized.pop("offset_mapping")
    masks_1 = []
    masks_2 = []
    n_truncated = 0
    for i, lemma in enumerate(examples["lemma"]):
        prefix_shift = len(lemma) + len(_PREFIX_SUFFIX)
        seq_ids = tokenized.sequence_ids(i)
        offsets = offset_batch[i]

        # sentence1 lives in segment 0 behind the "{lemma}: " prefix, so its raw span
        # is shifted right by the prefix length. sentence2 (segment 1) is unshifted.
        mask1 = _target_mask(
            offsets,
            seq_ids,
            segment=0,
            span_start=examples["start1"][i] + prefix_shift,
            span_end=examples["end1"][i] + prefix_shift,
        )
        mask2 = _target_mask(
            offsets,
            seq_ids,
            segment=1,
            span_start=examples["start2"][i],
            span_end=examples["end2"][i],
        )
        if not any(mask1) or not any(mask2):
            n_truncated += 1
        masks_1.append(mask1)
        masks_2.append(mask2)

    if n_truncated:
        logger.warning(
            "preprocess_wic_targets: %d/%d examples had a target truncated away "
            "(empty mask); their u/v fall back to a zero vector.",
            n_truncated,
            len(examples["lemma"]),
        )

    tokenized["target_mask_1"] = masks_1
    tokenized["target_mask_2"] = masks_2
    return tokenized
