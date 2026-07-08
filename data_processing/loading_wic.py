import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets  # type: ignore


def load_split_tempowic(data_jl_path: Path, labels_tsv_path: Path) -> Dataset:
    """
    Parses a TempoWiC JSONL data file and its paired TSV labels file.
    Labels are 0/1 integers keyed by example id.
    """
    with open(labels_tsv_path, "r", encoding="utf-8") as f:
        labels = {row[0]: int(row[1]) for line in f if (row := line.strip().split("\t"))}

    records = []
    with open(data_jl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["id"] not in labels:
                continue
            records.append(
                {
                    "lemma": entry["word"],
                    "sentence1": entry["tweet1"]["text"],
                    "sentence2": entry["tweet2"]["text"],
                    "label": labels[entry["id"]],
                    # Character spans of the target occurrence in each tweet, used by
                    # the target-vector WiC model to locate u and v.
                    "start1": int(entry["tweet1"]["text_start"]),
                    "end1": int(entry["tweet1"]["text_end"]),
                    "start2": int(entry["tweet2"]["text_start"]),
                    "end2": int(entry["tweet2"]["text_end"]),
                }
            )
    return Dataset.from_list(records)


def get_tempowic_dsd(dataset_dir: Path, use_test: bool = False) -> DatasetDict:
    train_ds = load_split_tempowic(
        dataset_dir / "train.data.jl",
        dataset_dir / "train.labels.tsv",
    )
    if use_test:
        val_ds = load_split_tempowic(
            dataset_dir / "test-codalab-10k.data.jl",
            dataset_dir / "test.gold.tsv",
        )
        train_ds = concatenate_datasets([train_ds, load_split_tempowic(
            dataset_dir / "validation.data.jl",
            dataset_dir / "validation.labels.tsv",
        )])
    else:
        val_ds = load_split_tempowic(
            dataset_dir / "validation.data.jl",
            dataset_dir / "validation.labels.tsv",
        )
    return DatasetDict({"train": train_ds, "validation": val_ds})


def load_split_json_wic(data_json_path: Path, gold_json_path: Path) -> Dataset:
    """
    Parses separate data and gold JSON/JSONL files.
    Maps 'T'/'F' labels to 1/0 integers.
    """
    # 1. Parse the text data
    with open(data_json_path, "r", encoding="utf-8") as f:
        data_entries = json.load(f)
    # 2. Parse the gold label data
    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_entries = json.load(f)

    # Ensure parallel alignment between data entries and gold labels
    assert len(data_entries) == len(
        gold_entries
    ), f"Mismatch: {len(data_entries)} data rows vs {len(gold_entries)} gold rows."

    # 3. Zip and structural normalization
    processed_records = []
    for data, gold in zip(data_entries, gold_entries):
        raw_label = gold["tag"] 
        binary_label = 1 if str(raw_label).strip().upper() == "T" else 0

        processed_records.append(
            {
                "lemma": data["lemma"],
                "sentence1": data["sentence1"],
                "sentence2": data["sentence2"],
                "label": binary_label,
                # WiC records the target's character span in each sentence as strings;
                # keep them (as ints) for the target-vector model to locate u and v.
                "start1": int(data["start1"]),
                "end1": int(data["end1"]),
                "start2": int(data["start2"]),
                "end2": int(data["end2"]),
            }
        )

    # Convert directly into a Hugging Face Dataset
    return Dataset.from_list(processed_records)


# 4. Construct DatasetDict for the Trainer
def get_wic_dsd(dataset_dir: Path, use_test: bool = False) -> DatasetDict:
    train_ds = load_split_json_wic(
        dataset_dir / "train" / "training.en-en.data",
        dataset_dir / "train" / "training.en-en.gold",
    )
    if use_test:
        val_ds = load_split_json_wic(
            dataset_dir / "test" / "test.en-en.data",
            dataset_dir / "test" / "test.en-en.gold",
        )
        train_ds = concatenate_datasets([train_ds, load_split_json_wic(
            dataset_dir / "dev" / "dev.en-en.data",
            dataset_dir / "dev" / "dev.en-en.gold",
        )])
    else:
        val_ds = load_split_json_wic(
            dataset_dir / "dev" / "dev.en-en.data",
            dataset_dir / "dev" / "dev.en-en.gold",
        )
    return DatasetDict({"train": train_ds, "validation": val_ds})
