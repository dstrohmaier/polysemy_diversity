import json
from typing import Any
from pathlib import Path

import numpy as np
import random
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate  # type: ignore

from data_processing.loading_wic import get_wic_dsd
from data_processing.loading_fews import get_fews_wic_dsd
from utilities.reproducibility import make_reproducible
from wic.preprocessing import preprocess_wic_targets
from wic.target_vector_model import WiCTargetDataCollator, load_wic_model


def preprocess_wic_with_labels(examples, tokenizer):
    # Tokenize with target-word masks, then map the SuperGLUE labels (0 or 1) to the
    # standard "labels" key expected by Trainer.
    tokenized = preprocess_wic_targets(examples, tokenizer)
    tokenized["labels"] = examples["label"]
    return tokenized


METRIC_F1 = evaluate.load("f1")
METRIC_ACCURACY = evaluate.load("accuracy")

# Search space for randomised hyperparameter search
HP_SEARCH_SPACE: dict[str, list[float] | list[int]] = {
    "learning_rate": [8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 5e-5],
    "per_device_train_batch_size": [4, 8, 16, 32],
    "num_train_epochs": [1, 2, 3, 4, 5],
}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        **METRIC_F1.compute(predictions=preds, references=labels, average="binary"),
        "f1_macro": METRIC_F1.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"],
        **METRIC_ACCURACY.compute(predictions=preds, references=labels),
    }


def train_model(
    model,
    tokenized_datasets,
    tokenizer,
    output_dir: Path,
    hparams: dict[str, Any],
    save: bool = True,
):
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=hparams["learning_rate"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        per_device_eval_batch_size=32,
        num_train_epochs=hparams["num_train_epochs"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=WiCTargetDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if save:
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(output_dir / "final")

    last_eval = next(
        (e for e in reversed(trainer.state.log_history) if "eval_f1" in e), None
    )
    assert (
        last_eval is not None
    ), "training produced no eval_f1. Check eval_strategy/epochs"
    return last_eval["eval_f1"]


def sample_hparams(rng: random.Random) -> dict:
    return {key: rng.choice(vals) for key, vals in HP_SEARCH_SPACE.items()}


def hyperparameter_search(
    model_name: str,
    tokenized_datasets,
    tokenizer,
    output_dir: Path,
    n_trials: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Run randomised hyperparameter search, keep the best trial's model."""
    best_f1 = -1.0
    best_hparams = {}
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for trial in range(n_trials):
        hparams = sample_hparams(rng)
        trial_dir = output_dir / f"trial_{trial}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[HPSearch] Trial {trial + 1}/{n_trials}: {hparams}")
        with open(trial_dir / "hparams.json", "w") as out_f:
            json.dump(hparams, out_f)

        model = load_wic_model(model_name)
        f1 = train_model(
            model, tokenized_datasets, tokenizer, trial_dir, hparams, save=False
        )
        results.append({"trial": trial, "hparams": hparams, "f1": f1})


        print(f"[HPSearch] Trial {trial + 1} F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_hparams = hparams

    print(f"\n[HPSearch] Best F1: {best_f1:.4f} | Best hparams: {best_hparams}")

    # Save search summary
    with open(output_dir / "hp_search_results.json", "w") as f:
        json.dump(
            {"best_f1": best_f1, "best_hparams": best_hparams, "all_trials": results},
            f,
            indent=2,
        )

    return best_hparams


def train_final_model(
    model_name: str, tokenized_datasets, tokenizer, output_dir: Path, hparams: dict
):
    """Retrain on combined train+dev with the best hparams; overwrites output_dir/final."""
    model = load_wic_model(model_name)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=hparams["learning_rate"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        per_device_eval_batch_size=32,
        num_train_epochs=hparams["num_train_epochs"],
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        fp16=True,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        processing_class=tokenizer,
        data_collator=WiCTargetDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")


def evaluate_final_model(output_dir: Path, tokenized_test_dataset, tokenizer):
    """Evaluate the saved final model on the test split and write test_results.json."""
    model = load_wic_model(str(output_dir / "final"))
    eval_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=32,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=WiCTargetDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # print(f"[Eval] Test results: {metrics}")


def tokenize(dsd, tokenizer):
    return dsd.map(
        lambda x: preprocess_wic_with_labels(x, tokenizer),
        batched=True,
        remove_columns=dsd["train"].column_names,
    )


def run_pipeline(
    model_name: str,
    source_dir: Path,
    output_dir: Path,
    dataset: str,
    n_trials: int,
    seed: int,
) -> None:

    make_reproducible(seed, deterministic=False)
    rng = random.Random(seed)

    base_output = output_dir / model_name.replace("/", "--")
    wic_dir = source_dir / "base_dataset"
    fews_dir = source_dir / "fews"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    match dataset:
        case "wic":
            wic_search_data = tokenize(get_wic_dsd(wic_dir), tokenizer)
            wic_final_data = tokenize(get_wic_dsd(wic_dir, use_test=True), tokenizer)
            best_hparams = hyperparameter_search(
                model_name,
                wic_search_data,
                tokenizer,
                base_output / dataset / "hp_search",
                n_trials,
                rng,
            )
            print("[Final] Retraining WiC on train+dev with best hparams")
            train_final_model(
                model_name,
                wic_final_data,
                tokenizer,
                base_output / dataset / "hp_search",
                best_hparams,
            )
            evaluate_final_model(
                base_output / dataset, wic_final_data["validation"], tokenizer
            )

        case "fews":
            fews_search_data = tokenize(get_fews_wic_dsd(fews_dir, seed=seed), tokenizer)
            fews_final_data = tokenize(
                get_fews_wic_dsd(fews_dir, use_test=True, seed=seed), tokenizer
            )
            best_hparams = hyperparameter_search(
                model_name,
                fews_search_data,
                tokenizer,
                base_output / dataset / "hp_search",
                n_trials,
                rng,
            )
            print("[Final] Retraining FEWS on train+dev with best hparams")
            train_final_model(
                model_name,
                fews_final_data,
                tokenizer,
                base_output / dataset / "hp_search",
                best_hparams,
            )
            evaluate_final_model(
                base_output / dataset, fews_final_data["validation"], tokenizer
            )

        case "wic+fews":
            wic_output = base_output / "wic+fews" / "wic"
            fews_output = base_output / "wic+fews" / "fews"
            wic_search_data = tokenize(get_wic_dsd(wic_dir), tokenizer)
            wic_final_data = tokenize(get_wic_dsd(wic_dir, use_test=True), tokenizer)
            fews_search_data = tokenize(get_fews_wic_dsd(fews_dir, seed=seed), tokenizer)
            fews_final_data = tokenize(
                get_fews_wic_dsd(fews_dir, use_test=True, seed=seed), tokenizer
            )

            print("[HPSearch] Stage 1: WiC hyperparameter search")
            wic_best_hparams = hyperparameter_search(
                model_name, wic_search_data, tokenizer, wic_output, n_trials, rng
            )
            print("[Final] Retraining WiC on train+dev with best hparams")
            train_final_model(
                model_name, wic_final_data, tokenizer, wic_output, wic_best_hparams
            )
            evaluate_final_model(wic_output, wic_final_data["validation"], tokenizer)

            print(
                "[HPSearch] Stage 2: FEWS hyperparameter search (starting from best WiC model)"
            )
            fews_best_hparams = hyperparameter_search(
                str(wic_output / "final"),
                fews_search_data,
                tokenizer,
                fews_output,
                n_trials,
                rng,
            )
            print("[Final] Retraining FEWS on train+dev with best hparams")
            train_final_model(
                str(wic_output / "final"),
                fews_final_data,
                tokenizer,
                fews_output,
                fews_best_hparams,
            )
            evaluate_final_model(
                fews_output, fews_final_data["validation"], tokenizer
            )
