"""Simulate corpora whose verb sense distributions vary in skew and sense count.

The simulations vary along two dimensions:

* ``k_senses`` -- each verb is restricted to exactly ``k`` senses (its ``k`` most
  frequent real senses). Verbs with fewer than ``k`` annotated senses are skipped
  for that ``k``.
* ``offset``   -- the Zipfian sense-frequency slope is shifted up or down relative
  to the verb's baseline slope (estimated once from its full real sense inventory).

For every (k, offset) pair we generate one corpus, so the output is a grid of
corpora spanning both dimensions.
"""

import json
from pathlib import Path

import click

from data_processing.loading_wsd import load_wsd
from data_processing.wic_conversion import convert_simulated_corpora
from simulation.corpus_simulation import simulate_zipfian_corpora


def _load_targets(path: Path) -> list[tuple[str, str]]:
    """Read a JSON list of [lemma, pos] pairs into a list of tuples."""
    pairs = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(pair) for pair in pairs]


@click.command()
@click.argument("wsd_dir", type=Path)
@click.argument("targets_fp", type=Path)
@click.argument("output_dir", type=Path)
@click.option(
    "--k-senses",
    type=int,
    multiple=True,
    default=(3, 4, 5),
    show_default=True,
    help="Exact sense count(s) per verb. Repeat to sweep several k values.",
)
@click.option("--offset-min", type=float, default=-0.5, show_default=True)
@click.option("--offset-max", type=float, default=0.5, show_default=True)
@click.option("--offset-step", type=float, default=0.1, show_default=True)
@click.option(
    "--n-draws",
    type=int,
    default=200,
    show_default=True,
    help="Occurrences simulated per verb per corpus.",
)
@click.option(
    "--min-examples",
    type=int,
    default=5,
    show_default=True,
    help="Passed to load_wsd; min examples per (lemma, pos, sense).",
)
@click.option("--seed", type=int, default=42, show_default=True)
def simulate(
    wsd_dir: Path,
    targets_fp: Path,
    output_dir: Path,
    k_senses: tuple[int, ...],
    offset_min: float,
    offset_max: float,
    offset_step: float,
    n_draws: int,
    min_examples: int,
    seed: int,
) -> None:
    """Generate corpora varying in sense count (k) and Zipfian slope from WSD_DIR."""
    wsd_df = load_wsd([wsd_dir], min_examples=min_examples)
    target_pairs = _load_targets(targets_fp)

    simulate_zipfian_corpora(
        wsd_df,
        target_pairs,
        output_dir,
        k_senses,
        offset_min,
        offset_max,
        offset_step,
        n_draws,
        seed,
    )

    # Convert the freshly generated corpora to WiC-format .data files alongside
    # the CSVs, ready for apply_wic.py.
    convert_simulated_corpora(output_dir, output_dir, seed=seed)


if __name__ == "__main__":
    simulate()
