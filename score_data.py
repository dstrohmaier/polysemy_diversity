"""Score simulated corpus *pairs* with a shift-in-diversity measure.

Every method scores a (source, target) corpus pair of the same lemma and reports a
log-ratio that is positive when the target is more diverse (see the readme). Pairs
are enumerated per lemma across three comparison schemes (see
:mod:`simulation.pairing`). For every SIM_DIR (the ``<lemma>_<pos>/k*_offset_*.csv``
layout produced by simulate_data.py):

* ``vmf``    -- fit a vMF concentration kappa on each corpus's (equal-n) contextual
  embeddings; score ``log(kappa_S / kappa_T)``. Writes ``vmf_pair_scores.csv``.
* ``wic``    -- a trained WiC model judges each corpus's intra-corpus sentence pairs;
  score ``log(p_same_S / p_same_T)``. Writes ``wic_pair_scores.csv``.
* ``cosine`` -- baseline: a cosine-geometry diversity per corpus; score
  ``log(D_cos_T / D_cos_S)``. Writes ``cosine_pair_scores.csv``.

``--dataset`` selects the corpus layout under SIM_DIR, and with it how pairs are
enumerated: the simulated ``k*_offset_*`` grid (three schemes per lemma), or the
DWUG decade groupings prepared by ``prepare_dwug.py`` (one ``g1``->``g2`` pair per
lemma, the readme's second evaluation).

All outputs are written to OUTPUT_DIR.
"""

from pathlib import Path

import click
import torch

from cosine.cosine_estimation import get_corpora_cosine_pairs
from simulation.pairing import build_dwug_pairs, build_simulated_pairs
from utilities.logging_utils import start_logging
from utilities.reproducibility import make_reproducible
from vmf.vmf_estimation import get_corpora_vmf_pairs
from wic.wic_estimation import get_corpora_wic_pairs

# Corpus layout under SIM_DIR -> the enumerator that turns it into scoring pairs.
_PAIR_ENUMERATORS = {"simulated": build_simulated_pairs, "dwug": build_dwug_pairs}


@click.command()
@click.argument("scoring", type=click.Choice(["vmf", "wic", "cosine"]))
@click.argument("sim_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.option(
    "--hf-model-name",
    type=str,
    default="answerdotai/ModernBERT-large",
    show_default=True,
    help="[vmf] Hugging Face model used to extract contextual embeddings.",
)
@click.option(
    "--wic-model-dir",
    type=Path,
    default=None,
    help="[wic] Trained WiC model dir. Defaults to the wic+fews final model "
    "under output/models for --base-model.",
)
@click.option(
    "--base-model",
    type=str,
    default="answerdotai/ModernBERT-large",
    show_default=True,
    help="[wic] Base model name used to locate the default trained WiC model dir.",
)
@click.option(
    "--dataset",
    type=click.Choice(sorted(_PAIR_ENUMERATORS)),
    default="simulated",
    show_default=True,
    help="Corpus layout under SIM_DIR: the simulated k/offset grid, or DWUG decade "
    "groupings (one g1->g2 pair per lemma).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for the equal-n downsampling of each corpus pair (and global RNG "
    "state). The default matches the scorers' own default, so it reproduces runs "
    "made before this option existed.",
)
def score(
    scoring: str,
    sim_dir: Path,
    output_dir: Path,
    hf_model_name: str,
    wic_model_dir: Path | None,
    base_model: str,
    dataset: str,
    seed: int,
) -> None:
    """Score every corpus pair under SIM_DIR using SCORING (vmf, wic, or cosine)."""

    start_logging(output_dir / "logs", file_name=f"score_{scoring}.log")
    # Scoring is pure inference, so the determinism flags cost no measurable
    # throughput here (unlike training, which opts out to keep cudnn.benchmark).
    make_reproducible(seed, deterministic=True)
    build_corpus_pairs = _PAIR_ENUMERATORS[dataset]

    match scoring:
        case "vmf":
            get_corpora_vmf_pairs(
                sim_dir,
                output_dir / "vmf",
                hf_model_name=hf_model_name,
                seed=seed,
                build_corpus_pairs=build_corpus_pairs,
            )
        case "cosine":
            get_corpora_cosine_pairs(
                sim_dir,
                output_dir / "cosine",
                hf_model_name=hf_model_name,
                seed=seed,
                build_corpus_pairs=build_corpus_pairs,
            )
        case "wic":
            if not torch.cuda.is_available():
                raise SystemExit("No CUDA-capable GPU found. Aborting.")
            get_corpora_wic_pairs(
                sim_dir,
                output_dir / "wic",
                model_dir=wic_model_dir,
                base_model=base_model,
                seed=seed,
                build_corpus_pairs=build_corpus_pairs,
            )


if __name__ == "__main__":
    score()
