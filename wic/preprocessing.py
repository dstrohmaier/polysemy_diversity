"""Shared WiC tokenization used by both training and scoring."""


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
