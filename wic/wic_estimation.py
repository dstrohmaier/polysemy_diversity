"""Score simulated corpus *pairs* with a trained WiC sequence-classification model.

Each corpus's intra-corpus sentence pairs are judged by the model; class 1 is the
same-sense probability (matching the WiC training label convention in
``data_processing/loading_wic.py``). A (source, target) pair is then scored by the
log-ratio ``log(p_same_S / p_same_T)`` of the mean same-sense probabilities, positive
when the target is more diverse. ``score_corpus_wic`` remains as the per-corpus
building block (also used by the tests).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import torch
from datasets import Dataset  # type: ignore
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from simulation.pairing import (
    CorpusPair,
    PairBuilder,
    equalise_indices,
    build_simulated_pairs,
)
from wic.preprocessing import preprocess_wic_targets
from wic.target_vector_model import (
    HEAD_PARAM_NAMES,
    WiCTargetDataCollator,
    load_wic_model,
)

logger = logging.getLogger("div")


def assert_trained_head(model_dir: Path) -> None:
    """Fail loudly if the checkpoint at ``model_dir`` has no trained classifier head.

    ``AutoModelForSequenceClassification.from_pretrained`` only *warns* when a head is
    missing from the checkpoint and then initialises it randomly, so a stale or
    base-model directory scores silently with a nonsense head. We inspect the saved
    weights directly (rather than the loaded module, whose head is always populated)
    and require the classifier parameters to be present.
    """
    try:
        from safetensors import safe_open  # type: ignore

        weight_files = sorted(model_dir.glob("*.safetensors"))
        present: set[str] = set()
        for wf in weight_files:
            with safe_open(wf, framework="pt") as f:  # type: ignore
                present.update(f.keys())
        if not weight_files:  # fall back to the PyTorch .bin format
            bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
            for bf in bin_files:
                present.update(
                    torch.load(bf, map_location="cpu", weights_only=True).keys()
                )
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Could not read model weights under {model_dir} to verify the WiC "
            f"classifier head: {exc}"
        ) from exc

    missing = [name for name in HEAD_PARAM_NAMES if name not in present]
    assert not missing, (
        f"WiC checkpoint {model_dir} is missing classifier-head weights {missing}; "
        f"the model was never fine-tuned for WiC (from_pretrained would randomly "
        f"initialise the head, producing a flat, high P(diff)). Train the model "
        f"(e.g. `just train-wic-fews`) or point --wic-model-dir at a trained "
        f"checkpoint's `final` directory."
    )


def _build_predict_trainer(model, tokenizer) -> Trainer:
    """Build the one `Trainer` reused across all corpora for prediction.

    A fresh `Trainer` re-wraps `model` via its `Accelerator` (adding hooks around the
    existing forward) without undoing the previous wrapping. Building it once per
    `get_corpora_wic_pairs` run rather than once per corpus keeps the wrapper depth
    constant instead of growing with every corpus and eventually hitting a
    RecursionError deep inside the encoder's forward.
    """
    training_args = TrainingArguments(
        output_dir="_tmp_wic_predict",
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=WiCTargetDataCollator(tokenizer),
    )


def _predict_logits(entries: list[dict], trainer: Trainer, tokenizer) -> np.ndarray:
    """Return the model's raw logits (n_pairs, 2) for a corpus's sentence pairs."""
    dataset = Dataset.from_list(
        [
            {
                "lemma": e["lemma"],
                "sentence1": e["sentence1"],
                "sentence2": e["sentence2"],
                # Target spans, needed by the target-vector model to locate u and v.
                "start1": e["start1"],
                "end1": e["end1"],
                "start2": e["start2"],
                "end2": e["end2"],
            }
            for e in entries
        ]
    )
    tokenized = dataset.map(
        lambda x: preprocess_wic_targets(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    # Trainer needs a labels column to not crash; add a placeholder.
    tokenized = tokenized.map(
        lambda x: {"labels": [0] * len(x["input_ids"])}, batched=True
    )

    return trainer.predict(tokenized).predictions  # type: ignore


def score_corpus_wic(
    entries: list[dict], trainer: Trainer, tokenizer, meta: dict
) -> tuple[dict, list[dict]]:
    """Score one corpus's sentence pairs with the WiC model.

    Returns ``(summary_row, pair_rows)``. ``p_diff`` is the softmax probability
    of class 0 (different sense). Requires a non-empty ``entries``.
    """
    assert entries

    word = entries[0]["lemma"]
    pos = entries[0]["pos"]
    offset = meta["applied_slope"] - meta["baseline_slope"]

    logits = _predict_logits(entries, trainer, tokenizer)
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
            "baseline_slope": meta["baseline_slope"],
            "applied_slope": meta["applied_slope"],
            "offset": offset,
            "p_diff": float(pd_),
            "label": int(e["label"]),
            "pred": int(pred),
        }
        for e, pd_, pred in zip(entries, p_diff, preds)
    ]
    return summary, pair_rows


def _corpus_p_same(entries: list[dict], trainer: Trainer, tokenizer) -> float:
    """Mean P(same sense) over a corpus's intra-corpus WiC pairs.

    ``p_same = 1 - p_diff`` where ``p_diff`` is the class-0 softmax probability (see
    :func:`score_corpus_wic`). Under a perfect model this approximates Simpson
    concentration ``sum_i p_i^2`` -- the readme's basis for the WiC shift score.
    """
    logits = _predict_logits(entries, trainer, tokenizer)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    p_same = probs[:, 1]
    return float(p_same.mean())


def score_pair_wic(
    pair: CorpusPair, trainer: Trainer, tokenizer, seed: int = 0
) -> dict:
    """WiC shift score ``log(p_same_S / p_same_T)`` for one (source, target) pair.

    Down-samples the larger pair set to the smaller's count (so the mean is over
    equal n) and returns the log-ratio; positive => target more diverse. Anomalies
    raise rather than skip -- they signal a broken pipeline, not a data condition.
    """
    if not (pair.source.data_path.exists() and pair.target.data_path.exists()):
        raise FileNotFoundError(
            f"{pair.lemma_pos}: missing .data for {pair.source.csv_path.stem} or "
            f"{pair.target.csv_path.stem}; run convert_simulated_corpora first"
        )

    entries_s = json.loads(pair.source.data_path.read_text(encoding="utf-8"))
    entries_t = json.loads(pair.target.data_path.read_text(encoding="utf-8"))
    assert entries_s and entries_t, (
        f"{pair.lemma_pos}: empty .data; simulation should never write < 2 pairs"
    )
    idx_s, idx_t = equalise_indices(len(entries_s), len(entries_t), seed=seed)
    entries_s = [entries_s[i] for i in idx_s]
    entries_t = [entries_t[i] for i in idx_t]

    p_same_s = _corpus_p_same(entries_s, trainer, tokenizer)
    p_same_t = _corpus_p_same(entries_t, trainer, tokenizer)

    return {
        "lemma_pos": pair.lemma_pos,
        "scheme": pair.scheme,
        "source_variant": pair.source.csv_path.stem,
        "target_variant": pair.target.csv_path.stem,
        "wic_log_ratio": float(np.log(p_same_s / p_same_t)),
        "n_used": len(entries_s),
    }


def get_corpora_wic_pairs(
    sim_dir: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    base_model: str = "answerdotai/ModernBERT-large",
    models_root: Path = Path("output/models"),
    seed: int = 0,
    build_corpus_pairs: PairBuilder = build_simulated_pairs,
) -> pd.DataFrame:
    """Compute the WiC shift score for every corpus pair under ``sim_dir``.

    ``build_corpus_pairs`` decides which (source, target) pairs ``sim_dir``
    yields: the default covers the simulation's three comparison schemes, while
    :func:`~simulation.pairing.build_dwug_pairs` gives the diachronic evaluation's single
    pair per lemma. Writes one combined ``wic_pair_scores.csv`` to ``output_dir``.
    """
    if model_dir is None:
        model_dir = (
            models_root / base_model.replace("/", "--") / "wic+fews" / "fews" / "final"
        )
    assert_trained_head(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_wic_model(str(model_dir))
    trainer = _build_predict_trainer(model, tokenizer)

    pairs = build_corpus_pairs(sim_dir)
    assert pairs, (
        f"no corpus pairs found under {sim_dir}; is the directory layout the one the "
        f"chosen enumerator expects (see score_data.py --dataset)?"
    )
    rows = []
    for pair in pairs:
        record = score_pair_wic(pair, trainer, tokenizer, seed=seed)
        rows.append(record)
        logger.info(
            "%s [%s] %s->%s WiC log-ratio: %.4f (n=%d)",
            pair.lemma_pos, record["scheme"], record["source_variant"],
            record["target_variant"], record["wic_log_ratio"], record["n_used"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "wic_pair_scores.csv", index=False)
    return result
