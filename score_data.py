"""Score simulated corpora, either with a von Mises-Fisher concentration (kappa)
or with a trained WiC sequence-classification model.

For every simulated corpus under SIM_DIR (the ``<lemma>_<pos>/k*_offset_*.csv``
layout produced by simulate_data.py):

* ``vmf``  -- the target word's contextual embeddings are extracted with a
  transformer model and a single vMF kappa is fitted. Writes ``vmf_scores.csv``.
* ``wic``  -- a trained WiC model judges every sentence pair (read from the sibling
  ``.data`` files) and we record the probability that the two occurrences differ in
  sense. Writes a per-corpus ``wic_scores.csv`` and a per-pair ``wic_pair_scores.csv``.

All outputs are written to OUTPUT_DIR.
"""

from pathlib import Path

import click
import torch

from vmf.vmf_estimation import get_corpora_vmf
from wic.wic_estimation import get_corpora_wic_score


@click.command()
@click.argument("scoring", type=click.Choice(["vmf", "wic"]))
@click.argument("sim_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option(
    "--hf-model-name",
    type=str,
    default="answerdotai/ModernBERT-base",
    show_default=True,
    help="[vmf] Hugging Face model used to extract contextual embeddings.",
)
@click.option(
    "--wic-model-dir",
    type=Path,
    default=None,
    help="[wic] Trained WiC model dir. Defaults to the wic+tempowic final model "
    "under output/models for --base-model.",
)
@click.option(
    "--base-model",
    type=str,
    default="answerdotai/ModernBERT-large",
    show_default=True,
    help="[wic] Base model name used to locate the default trained WiC model dir.",
)
def score(
    scoring: str,
    sim_dir: Path,
    output_dir: Path,
    hf_model_name: str,
    wic_model_dir: Path | None,
    base_model: str,
) -> None:
    """Score every simulated corpus under SIM_DIR using SCORING (vmf or wic)."""

    match scoring:
        case "vmf":
            get_corpora_vmf(sim_dir, output_dir / "vmf", hf_model_name=hf_model_name)
        case "wic":
            if not torch.cuda.is_available():
                raise SystemExit("No CUDA-capable GPU found. Aborting.")
            get_corpora_wic_score(
                sim_dir,
                output_dir / "wic",
                model_dir=wic_model_dir,
                base_model=base_model,
            )


if __name__ == "__main__":
    score()
