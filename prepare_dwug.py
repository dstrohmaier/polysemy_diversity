"""Materialise DWUG EN into the corpus layout used by the shift scorers.

The diachronic evaluation (readme "Second Evaluation") compares each lemma's
1810-1860 usages against its 1960-2010 ones. DWUG stores both decades in a single
``uses.csv`` per lemma, so this step splits them into two corpora written in the same
on-disk shape the simulation produces -- ``<lemma>_<pos>/{g1,g2}.csv`` plus
``.meta.json`` and ``.data`` siblings. Everything downstream (vMF, WiC and cosine
scoring, and the comparative analysis) then runs on DWUG unchanged, selected with
``--dataset dwug``.

Writes a ``tables/dwug_preparation.csv`` report: per-lemma usage counts before and
after the noise drop, senses per grouping, the equalised corpus size, and the
ground-truth diversity shifts.
"""

from pathlib import Path

import click

from analysis.io import write_csv
from data_processing.dwug_conversion import prepare_dwug_corpora
from utilities.logging_utils import start_logging


@click.command()
@click.argument("dwug_root", type=Path)
@click.argument("output_dir", type=Path)
@click.option(
    "--seed",
    type=int,
    default=1848,
    show_default=True,
    help="Seed for the WiC pair shuffling used to build the .data files.",
)
def prepare(dwug_root: Path, output_dir: Path, seed: int) -> None:
    """Convert DWUG_ROOT into per-lemma grouping corpora under OUTPUT_DIR."""
    start_logging(output_dir / "logs", file_name="prepare_dwug.log")

    summary = prepare_dwug_corpora(dwug_root, output_dir, seed=seed)
    write_csv(summary, output_dir / "tables", "dwug_preparation")


if __name__ == "__main__":
    prepare()
