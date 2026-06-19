"""Score simulated corpora with a trained WiC sequence-classification model.

For each simulated corpus the model judges every sentence pair and we report the
probability that the two occurrences *differ* in sense (class 0; class 1 = same
sense, matching the WiC training label convention in
``data_processing/loading_wic.py``).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import torch
from datasets import Dataset  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from wic.preprocessing import preprocess_wic


def _predict_logits(entries: list[dict], model, tokenizer) -> np.ndarray:
    """Return the model's raw logits (n_pairs, 2) for a corpus's sentence pairs."""
    dataset = Dataset.from_list(
        [
            {
                "lemma": e["lemma"],
                "sentence1": e["sentence1"],
                "sentence2": e["sentence2"],
            }
            for e in entries
        ]
    )
    tokenized = dataset.map(
        lambda x: preprocess_wic(x, tokenizer),
        batched=True,
        remove_columns=["lemma", "sentence1", "sentence2"],
    )
    # Trainer needs a labels column to not crash; add a placeholder.
    tokenized = tokenized.map(
        lambda x: {"labels": [0] * len(x["input_ids"])}, batched=True
    )

    training_args = TrainingArguments(
        output_dir="_tmp_wic_predict",
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, processing_class=tokenizer)
    return trainer.predict(tokenized).predictions  # type: ignore


def score_corpus_wic(
    entries: list[dict], model, tokenizer, meta: dict
) -> tuple[dict, list[dict]]:
    """Score one corpus's sentence pairs with the WiC model.

    Returns ``(summary_row, pair_rows)``. ``p_diff`` is the softmax probability
    of class 0 (different sense). Requires a non-empty ``entries``.
    """
    assert entries

    word = entries[0]["lemma"]
    pos = entries[0]["pos"]
    offset = meta["applied_slope"] - meta["baseline_slope"]

    logits = _predict_logits(entries, model, tokenizer)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    p_diff = probs[:, 0]
    preds = np.argmax(logits, axis=1)
    labels = np.array([e["label"] for e in entries])

    summary = {
        "word": word,
        "pos": pos,
        "k_senses": meta["k_senses"],
        "baseline_slope": meta["baseline_slope"],
        "applied_slope": meta["applied_slope"],
        "offset": offset,
        "clamped": meta["clamped"],
        "wic_p_diff_mean": float(p_diff.mean()),
        "wic_p_diff_std": float(p_diff.std()),
        "accuracy": float((preds == labels).mean()),
        "pair_count": len(entries),
    }

    pair_rows = [
        {
            "id": e["id"],
            "word": word,
            "pos": pos,
            "k_senses": meta["k_senses"],
            "offset": offset,
            "p_diff": float(pd_),
            "label": int(e["label"]),
            "pred": int(pred),
        }
        for e, pd_, pred in zip(entries, p_diff, preds)
    ]
    return summary, pair_rows


def get_corpora_wic_score(
    sim_dir: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    base_model: str = "answerdotai/ModernBERT-large",
    models_root: Path = Path("output/models"),
) -> pd.DataFrame:
    """Score every simulated corpus under ``sim_dir`` with a trained WiC model.

    Walks ``sim_dir/<lemma>_<pos>/k*_offset_*.csv`` (for the ``.meta.json`` sidecar)
    and reads the sibling ``.data`` file produced by the conversion step. Writes a
    per-corpus ``wic_scores.csv`` and a per-pair ``wic_pair_scores.csv`` to
    ``output_dir`` and returns the per-corpus summary DataFrame.
    """

    if model_dir is None:
        model_dir = (
            models_root
            / base_model.replace("/", "--")
            / "wic+tempowic"
            / "tempowic"
            / "final"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    summary_rows = []
    pair_rows = []
    for csv_path in sorted(sim_dir.glob("*/k*_offset_*.csv")):
        # Not csv_path.with_suffix(...): the "0.00" in the variant name confuses
        # pathlib's suffix handling. Swap the trailing ".csv" explicitly.
        stem = csv_path.name[: -len(".csv")]
        meta_path = csv_path.parent / (stem + ".meta.json")
        data_path = csv_path.parent / (stem + ".data")
        if not meta_path.exists():
            warnings.warn(f"Missing meta_path: {meta_path}")
            continue  # skip stray CSVs without sidecar metadata
        if not data_path.exists():
            warnings.warn(
                f"  {csv_path.parent.name} {stem}: missing .data (run conversion); skipped"
            )
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entries = json.loads(data_path.read_text(encoding="utf-8"))

        summary, corpus_pairs = score_corpus_wic(entries, model, tokenizer, meta)

        summary_rows.append(summary)
        pair_rows.extend(corpus_pairs)
        print(
            f"  {csv_path.parent.name} {stem} "
            f"P(diff) mean: {summary['wic_p_diff_mean']:.4f} "
            f"acc: {summary['accuracy']:.4f} (n={summary['pair_count']})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "wic_scores.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "wic_pair_scores.csv", index=False)
    return summary_df
