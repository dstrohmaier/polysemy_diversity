"""Shared WiC tokenization used by both training and scoring."""

import logging

logger = logging.getLogger("div")


def preprocess_wic(examples, tokenizer):
    """Tokenize WiC sentence pairs as "{lemma}: sentence1" + sentence2.

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


# Marker tokens wrapped around each target-word occurrence, e.g. "... [unused0] bank
# [unused1] ...". Both are real single-token entries in ModernBERT's vocabulary that are
# never used by the pretrained model, so they act as untrained boundary markers rather
# than colliding with any trained embedding.
_MARK_START = "[unused0]"
_MARK_END = "[unused1]"


def _wrap_target(sentence, span_start, span_end):
    """Insert marker tokens around ``sentence[span_start:span_end]``.

    A space is kept on both sides of the wrapped word (before it and before the closing
    marker) so it still tokenizes with its normal leading-space subword, matching how it
    would tokenize mid-sentence without the markers. Without that space the word loses
    its leading-space marker and falls back to an out-of-distribution subword the model
    rarely sees mid-sentence.
    """
    return (
        f"{sentence[:span_start]}{_MARK_START} "
        f"{sentence[span_start:span_end]} {_MARK_END}"
        f"{sentence[span_end:]}"
    )


def _target_mask_from_markers(input_ids, sequence_ids, segment, mark_start_id, mark_end_id):
    """Multi-hot mask over the subwords strictly between the markers in ``segment``.

    ``segment`` is the tokenizer sequence id (0 = sentence1/prefix, 1 = sentence2).
    Returns an all-zero mask if either marker was truncated away, rather than leaving an
    unterminated span that would otherwise bleed to the end of the sequence.
    """
    mask = [0] * len(input_ids)
    in_span = False
    saw_end = False
    for i, (tok_id, seq_id) in enumerate(zip(input_ids, sequence_ids)):
        if seq_id != segment:
            continue
        if tok_id == mark_start_id:
            in_span = True
            continue
        if tok_id == mark_end_id:
            in_span = False
            saw_end = True
            continue
        if in_span:
            mask[i] = 1
    if not saw_end:
        return [0] * len(input_ids)
    return mask


def preprocess_wic_targets(examples, tokenizer):
    """Tokenize the joined WiC pair and emit target-word subword masks for u and v.

    Wraps each target occurrence in ``[unused0] ... [unused1]`` marker tokens before
    tokenizing, producing ``"{lemma}: {sentence1 with markers}" [SEP] {sentence2 with
    markers}``, then emits ``target_mask_1``/``target_mask_2``: multi-hot masks selecting
    the subwords strictly between the markers in sentence1 (segment 0) and sentence2
    (segment 1) respectively. The target-vector model masked-mean-pools these into u and
    v. Requires ``start1/end1/start2/end2`` (character spans against the raw sentences)
    on each example. A mask is all-zero if either of its markers was pushed past the
    ``max_length`` truncation; the example is kept and a warning is logged.
    """
    mark_start_id, mark_end_id = tokenizer.convert_tokens_to_ids(
        [_MARK_START, _MARK_END]
    )

    first_sentences = [
        f"{w}: {_wrap_target(s, st, en)}"
        for w, s, st, en in zip(
            examples["lemma"],
            examples["sentence1"],
            examples["start1"],
            examples["end1"],
        )
    ]
    second_sentences = [
        _wrap_target(s, st, en)
        for s, st, en in zip(
            examples["sentence2"], examples["start2"], examples["end2"]
        )
    ]

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=256,
        padding=False,
    )

    masks_1 = []
    masks_2 = []
    n_truncated = 0
    for i in range(len(examples["lemma"])):
        input_ids = tokenized["input_ids"][i]
        seq_ids = tokenized.sequence_ids(i)

        mask1 = _target_mask_from_markers(
            input_ids, seq_ids, segment=0,
            mark_start_id=mark_start_id, mark_end_id=mark_end_id,
        )
        mask2 = _target_mask_from_markers(
            input_ids, seq_ids, segment=1,
            mark_start_id=mark_start_id, mark_end_id=mark_end_id,
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
