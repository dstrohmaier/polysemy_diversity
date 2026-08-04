"""Analysis of the simulated data and its scoring.

ANALYSIS_TYPE selects what to analyse:

* ``raw_simulated``  -- the raw simulated corpora (sense distribution, entropy, corpus
  size), comparing the empirical sample against the theoretical Zipfian design.
* ``wic_simulated``  -- the WiC-converted ``.data`` pairs (same- vs different-sense
  pair counts and overall data size).
* ``comparative``    -- the shift-in-diversity comparison: Spearman rho of each
  method's per-pair log-ratio against the ground-truth diversity shifts
  ``log(qD(T)/qD(S))`` for q in {0, 1, 2}, grouped by comparison scheme, as a table
  and per-scheme dot-plot figures.

The descriptive modes take ``DATA_DIR OUTPUT_DIR`` (DATA_DIR = a simulated_data
dataset dir). The ``comparative`` mode takes ``SCORES_DIR OUTPUT_DIR SIM_DIR``:
SCORES_DIR is the dataset's scoring output (``output/scores/<dataset>``) and SIM_DIR
is its corpus dir (for the ``.meta.json`` sense-probability sidecars). Pass
``--dataset dwug`` there to analyse the diachronic evaluation, whose corpus dir is
the one written by ``prepare_dwug.py``.

Each mode writes CSV/Markdown/LaTeX tables and PDF figures under
``OUTPUT_DIR/<analysis_type>/``.
"""

# Select the non-interactive Agg backend before any module imports pyplot, so figures
# render headless (to PDF) without a display.
import matplotlib

matplotlib.use("Agg")
import seaborn as sns  # type: ignore
sns.set_style("darkgrid")

from pathlib import Path  # noqa: E402

import click  # noqa: E402

from analysis.scored.comparative import analyse_comparative  # noqa: E402
from analysis.simulated_data.raw_simulated import analyse_raw_simulated  # noqa: E402
from analysis.simulated_data.wic_simulated import analyse_wic_simulated  # noqa: E402
from data_processing.dwug_loading import iter_dwug_corpora  # noqa: E402
from data_processing.simulation_loading import iter_corpora  # noqa: E402
from utilities.logging_utils import start_logging  # noqa: E402

# Corpus layout under SIM_DIR -> how to walk it for the .meta.json sidecars.
_CORPUS_ITERATORS = {"simulated": iter_corpora, "dwug": iter_dwug_corpora}


@click.command()
@click.argument(
    "analysis_type",
    type=click.Choice(["raw_simulated", "wic_simulated", "comparative"]),
)
@click.argument("data_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.argument("sim_dir", type=Path, required=False)
@click.option(
    "--dataset",
    type=click.Choice(sorted(_CORPUS_ITERATORS)),
    default="simulated",
    show_default=True,
    help="[comparative] Layout of SIM_DIR, selecting how corpus .meta.json sidecars "
    "are enumerated for the ground truth.",
)
def main(
    analysis_type: str,
    data_dir: Path,
    output_dir: Path,
    sim_dir: Path | None,
    dataset: str,
) -> None:
    """Run ANALYSIS_TYPE, writing tables and figures under OUTPUT_DIR/<analysis_type>/.

    For the descriptive modes, DATA_DIR is a simulated_data dataset dir. For
    ``comparative``, DATA_DIR is the dataset's scoring output dir and SIM_DIR
    (required) is its corpus dir.
    """
    out_root = output_dir / analysis_type
    out_root.mkdir(parents=True, exist_ok=True)
    start_logging(out_root / "logs", file_name=f"analysis_{analysis_type}.log")

    match analysis_type:
        case "raw_simulated":
            analyse_raw_simulated(data_dir, out_root)
        case "wic_simulated":
            analyse_wic_simulated(data_dir, out_root)
        case "comparative":
            if sim_dir is None:
                raise click.UsageError(
                    "comparative requires SIM_DIR (the corpus dir holding the "
                    ".meta.json sidecars)."
                )
            analyse_comparative(
                data_dir, sim_dir, out_root, iter_fn=_CORPUS_ITERATORS[dataset]
            )


if __name__ == "__main__":
    main()
