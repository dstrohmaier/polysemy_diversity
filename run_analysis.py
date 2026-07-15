"""Analysis of the simulated data and its scoring.

ANALYSIS_TYPE selects what to analyse:

* ``raw_simulated``  -- the raw simulated corpora (sense distribution, entropy, corpus
  size), comparing the empirical sample against the theoretical Zipfian design.
* ``wic_simulated``  -- the WiC-converted ``.data`` pairs (same- vs different-sense
  pair counts and overall data size).
* ``vmf_scored``     -- vMF scoring output: kappa vs entropy/slope (conditional on k),
  a (slope, k) score grid table, and per-corpus dot plots against both properties.
* ``wic_scored``     -- WiC scoring output: P(diff) vs entropy/slope (grid table +
  per-corpus dot plots), a calibration plot with a pooled OLS fit, and accuracy + F1
  of the WiC model with bootstrap CIs against slope and entropy.

The descriptive modes take ``DATA_DIR OUTPUT_DIR`` (DATA_DIR = a simulated_data
dataset dir). The scored modes take ``SCORES_DIR OUTPUT_DIR SIM_DIR``: SCORES_DIR is
the dataset's scoring output (``output/scores/<dataset>``) and SIM_DIR is its
simulated_data dir (for the corpus ``.meta.json`` entropy sidecars).

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

from analysis.scored.vmf_scored import analyse_vmf_scored  # noqa: E402
from analysis.scored.wic_scored import analyse_wic_scored  # noqa: E402
from analysis.simulated_data.raw_simulated import analyse_raw_simulated  # noqa: E402
from analysis.simulated_data.wic_simulated import analyse_wic_simulated  # noqa: E402
from utilities.logging_utils import start_logging  # noqa: E402


@click.command()
@click.argument(
    "analysis_type",
    type=click.Choice(
        ["raw_simulated", "wic_simulated", "vmf_scored", "wic_scored"]
    ),
)
@click.argument("data_dir", type=Path)
@click.argument("output_dir", type=Path)
@click.argument("sim_dir", type=Path, required=False)
def main(
    analysis_type: str,
    data_dir: Path,
    output_dir: Path,
    sim_dir: Path | None,
) -> None:
    """Run ANALYSIS_TYPE, writing tables and figures under OUTPUT_DIR/<analysis_type>/.

    For descriptive modes, DATA_DIR is a simulated_data dataset dir. For scored modes,
    DATA_DIR is the dataset's scoring output dir and SIM_DIR (required) is its
    simulated_data dir.
    """
    out_root = output_dir / analysis_type
    out_root.mkdir(parents=True, exist_ok=True)
    start_logging(out_root / "logs", file_name=f"analysis_{analysis_type}.log")

    match analysis_type:
        case "raw_simulated":
            analyse_raw_simulated(data_dir, out_root)
        case "wic_simulated":
            analyse_wic_simulated(data_dir, out_root)
        case "vmf_scored" | "wic_scored":
            if sim_dir is None:
                raise click.UsageError(
                    f"{analysis_type} requires SIM_DIR (the simulated_data dir "
                    "holding corpus .meta.json sidecars)."
                )
            analyse = (
                analyse_vmf_scored
                if analysis_type == "vmf_scored"
                else analyse_wic_scored
            )
            analyse(data_dir, sim_dir, out_root)


if __name__ == "__main__":
    main()
