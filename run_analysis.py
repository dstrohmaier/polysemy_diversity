"""Descriptive analysis of the simulated data.

ANALYSIS_TYPE selects what to analyse:

* ``raw_simulated``  -- the raw simulated corpora (sense distribution, entropy, corpus
  size), comparing the empirical sample against the theoretical Zipfian design.
* ``wic_simulated``  -- the WiC-converted ``.data`` pairs (same- vs different-sense
  pair counts and overall data size).

Each mode writes CSV/Markdown/LaTeX tables and PDF figures under
``OUTPUT_DIR/<analysis_type>/``. More analysis modes are expected to be added here.
"""

# Select the non-interactive Agg backend before any module imports pyplot, so figures
# render headless (to PDF) without a display.
import matplotlib

matplotlib.use("Agg")
import seaborn as sns  # type: ignore
sns.set_style("darkgrid")

from pathlib import Path  # noqa: E402

import click  # noqa: E402

from analysis.simulated_data.raw_simulated import analyse_raw_simulated  # noqa: E402
from analysis.simulated_data.wic_simulated import analyse_wic_simulated  # noqa: E402
from utilities.logging_utils import start_logging  # noqa: E402


@click.command()
@click.argument(
    "analysis_type", type=click.Choice(["raw_simulated", "wic_simulated"])
)
@click.argument("data_dir", type=Path)
@click.argument("output_dir", type=Path)
def main(analysis_type: str, data_dir: Path, output_dir: Path) -> None:
    """Run ANALYSIS_TYPE over the simulated dataset at DATA_DIR, writing to OUTPUT_DIR."""
    out_root = output_dir / analysis_type
    out_root.mkdir(parents=True, exist_ok=True)
    start_logging(out_root / "logs", file_name=f"analysis_{analysis_type}.log")

    match analysis_type:
        case "raw_simulated":
            analyse_raw_simulated(data_dir, out_root)
        case "wic_simulated":
            analyse_wic_simulated(data_dir, out_root)


if __name__ == "__main__":
    main()
