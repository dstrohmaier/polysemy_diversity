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
from collections import OrderedDict
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

# Sentence pairs per forward pass. Raise it to trade GPU memory for throughput: the
# WiC corpora are short-sentence pairs, so a larger batch mostly buys better occupancy
# rather than hitting a memory wall. Exposed via score_data.py --batch-size.
DEFAULT_EVAL_BATCH_SIZE = 128

# Corpora held in the logit cache. Pairs arrive grouped by lemma and no lemma has more
# than ~33 corpora on the default grid, so this holds a lemma's whole working set.
DEFAULT_CACHE_CAPACITY = 64


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


def _build_predict_trainer(
    model, tokenizer, batch_size: int = DEFAULT_EVAL_BATCH_SIZE
) -> Trainer:
    """Build the one `Trainer` reused across all corpora for prediction.

    A fresh `Trainer` re-wraps `model` via its `Accelerator` (adding hooks around the
    existing forward) without undoing the previous wrapping. Building it once per
    `get_corpora_wic_pairs` run rather than once per corpus keeps the wrapper depth
    constant instead of growing with every corpus and eventually hitting a
    RecursionError deep inside the encoder's forward.
    """
    training_args = TrainingArguments(
        output_dir="_tmp_wic_predict",
        per_device_eval_batch_size=batch_size,
        fp16=torch.cuda.is_available(),
        # Overlap host-side tokenised-batch collation with GPU compute. Inference is
        # short per batch, so a single-threaded loader leaves the GPU waiting between
        # batches; pinned memory makes the H2D copy async on top.
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
    )
    return Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=WiCTargetDataCollator(tokenizer),
    )


class WiCLogitCache:
    """Corpus ``.data`` path -> per-entry logits, memoised across the pairs sharing it.

    A corpus belongs to several pairs (the primary anchor to one per sibling variant),
    and the model's logits for a sentence pair depend only on that pair, so re-running
    the encoder for every comparison is redundant work -- ~5x on the default grid, and
    ~34x for a lemma's anchor.

    Why the cache is keyed per *entry* rather than per corpus: ``score_pair_wic``
    trims both sides to equal n with :func:`~simulation.pairing.equalise_indices`,
    whose kept subset depends on the *partner's* length. The same corpus therefore
    arrives with a different subset in each of its pairs, so a per-corpus cache of the
    mean would be wrong. Logits are per-entry and partner-independent, so caching those
    and averaging the requested subset afterwards is exact -- it returns bit-identical
    values to the uncached path.

    The whole corpus is embedded on first touch (its entries are needed in full sooner
    or later), so every later pair is a pure hit whatever subset it asks for.
    """

    def __init__(self, trainer: Trainer, tokenizer, capacity: int = 64):
        assert capacity > 0, "capacity must be positive"
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.capacity = capacity
        self._entries: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def logits_for(
        self, data_path: Path, entries: list[dict], indices: np.ndarray
    ) -> np.ndarray:
        """Logits for ``entries[indices]``, embedding the full corpus on first touch.

        ``entries`` must be the corpus's complete, in-order entry list -- the same one
        ``indices`` came from -- so a cached row lines up with the entry at that
        position on every later call.
        """
        key = Path(data_path).resolve()
        cached = self._entries.get(key)
        if cached is None:
            self.misses += 1
            cached = _predict_logits(entries, self.trainer, self.tokenizer)
            self._entries[key] = cached
            if len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        else:
            self.hits += 1
            self._entries.move_to_end(key)
        return cached[indices]

    def log_summary(self) -> None:
        """Log the hit rate, so a run shows whether the cache actually paid off."""
        total = self.hits + self.misses
        if not total:
            return
        logger.info(
            "WiC logit cache: %d corpus evaluations for %d requests (%.1f%% hits, "
            "capacity %d)",
            self.misses,
            total,
            100.0 * self.hits / total,
            self.capacity,
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


def _mean_p_same(logits: np.ndarray) -> float:
    """Mean P(same sense) over a corpus's intra-corpus WiC pairs.

    ``p_same = 1 - p_diff`` where ``p_diff`` is the class-0 softmax probability (see
    :func:`score_corpus_wic`). Under a perfect model this approximates Simpson
    concentration ``sum_i p_i^2`` -- the readme's basis for the WiC shift score.
    """
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return float(probs[:, 1].mean())


def _corpus_p_same(entries: list[dict], trainer: Trainer, tokenizer) -> float:
    """Mean P(same sense) for ``entries``, embedding them directly (no cache)."""
    return _mean_p_same(_predict_logits(entries, trainer, tokenizer))


def score_pair_wic(
    pair: CorpusPair, cache: WiCLogitCache, seed: int = 0
) -> dict:
    """WiC shift score ``log(p_same_S / p_same_T)`` for one (source, target) pair.

    Down-samples the larger pair set to the smaller's count (so the mean is over
    equal n) and returns the log-ratio; positive => target more diverse. Anomalies
    raise rather than skip -- they signal a broken pipeline, not a data condition.

    The equal-n trim is applied to the *cached logits* rather than to the entries fed
    to the model, which is what lets a corpus be embedded once and reused across its
    pairs even though each pair trims it differently (see :class:`WiCLogitCache`).
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

    p_same_s = _mean_p_same(
        cache.logits_for(pair.source.data_path, entries_s, idx_s)
    )
    p_same_t = _mean_p_same(
        cache.logits_for(pair.target.data_path, entries_t, idx_t)
    )

    return {
        "lemma_pos": pair.lemma_pos,
        "scheme": pair.scheme,
        "source_variant": pair.source.csv_path.stem,
        "target_variant": pair.target.csv_path.stem,
        "wic_log_ratio": float(np.log(p_same_s / p_same_t)),
        "n_used": len(idx_s),
    }


def get_corpora_wic_pairs(
    sim_dir: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    base_model: str = "answerdotai/ModernBERT-large",
    models_root: Path = Path("output/models"),
    seed: int = 0,
    build_corpus_pairs: PairBuilder = build_simulated_pairs,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    cache_capacity: int = DEFAULT_CACHE_CAPACITY,
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
    trainer = _build_predict_trainer(model, tokenizer, batch_size=batch_size)

    pairs = build_corpus_pairs(sim_dir)
    assert pairs, (
        f"no corpus pairs found under {sim_dir}; is the directory layout the one the "
        f"chosen enumerator expects (see score_data.py --dataset)?"
    )
    # Pairs arrive grouped by lemma, so a capacity covering the largest lemma's corpus
    # count keeps that lemma's whole working set resident (the same reasoning as
    # data_processing.vector_cache.DEFAULT_CAPACITY).
    cache = WiCLogitCache(trainer, tokenizer, capacity=cache_capacity)
    rows = []
    for pair in pairs:
        record = score_pair_wic(pair, cache, seed=seed)
        rows.append(record)
        logger.info(
            "%s [%s] %s->%s WiC log-ratio: %.4f (n=%d)",
            pair.lemma_pos, record["scheme"], record["source_variant"],
            record["target_variant"], record["wic_log_ratio"], record["n_used"],
        )

    cache.log_summary()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "wic_pair_scores.csv", index=False)
    return result
