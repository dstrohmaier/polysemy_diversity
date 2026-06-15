import json
from pathlib import Path

import click
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def load_efcamdat(data_path: Path) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    records = [
        {
            "id": e["id"],
            "lemma": e["lemma"],
            "sentence1": e["sentence1"],
            "sentence2": e["sentence2"],
        }
        for e in entries
    ]
    return Dataset.from_list(records)


def preprocess(examples, tokenizer):
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


def run_inference(model_dir: Path, dataset: Dataset, tokenizer) -> np.ndarray:
    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=["lemma", "sentence1", "sentence2"],
    )
    # Trainer needs a dummy label column to not crash; we add a placeholder
    tokenized = tokenized.map(lambda x: {"labels": [0] * len(x["input_ids"])}, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Minimal TrainingArguments just for prediction
    training_args = TrainingArguments(
        output_dir=str(model_dir / "_tmp_predict"),
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, processing_class=tokenizer)

    output = trainer.predict(tokenized)
    return np.argmax(output.predictions, axis=1)


@click.command()
@click.argument("model_dir", type=Path)
@click.argument("efcamdat_dir", type=Path)
@click.argument("output_dir", type=Path)
def main(model_dir: Path, efcamdat_dir: Path, output_dir: Path) -> None:
    """Run WiC inference on all efcamdat .data files using MODEL_DIR."""
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA-capable GPU found. Aborting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    for data_path in sorted(efcamdat_dir.glob("*.data")):
        print(f"Processing {data_path.name} ...")
        dataset = load_efcamdat(data_path)
        ids = dataset["id"]

        preds = run_inference(model_dir, dataset, tokenizer)

        out_path = output_dir / (data_path.stem + ".predictions.json")
        results = [
            {"id": id_, "prediction": int(pred)}
            for id_, pred in zip(ids, preds)
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} predictions to {out_path.name}")


if __name__ == "__main__":
    main()
