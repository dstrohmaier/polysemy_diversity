from pathlib import Path

import click
import torch

from wic.modelling import run_pipeline


@click.command()
@click.argument("model_name", type=str)
@click.argument("source_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option(
    "--dataset",
    type=click.Choice(["wic", "tempowic", "wic+tempowic"]),
    default="wic",
    show_default=True,
)
@click.option(
    "--n-trials",
    type=click.IntRange(min=2),
    default=30,
    show_default=True,
    help="Number of random hyperparameter trials.",
)
@click.option(
    "--seed",
    type=int,
    default=1848,
    show_default=True,
    help="Random seed for hyperparameter sampling.",
)
def run_training(
    model_name: str,
    source_dir: Path,
    output_dir: Path,
    dataset: str,
    n_trials: int,
    seed: int,
):
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA-capable GPU found. Aborting.")

    run_pipeline(model_name, source_dir, output_dir, dataset, n_trials, seed)


if __name__ == "__main__":
    run_training()
