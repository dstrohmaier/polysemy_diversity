"""Mode-agnostic output helpers shared by all analysis modes.

Analysis modes produce (a) tables and (b) figures (written as PDF). Tables come in
two flavours: :func:`write_table` emits three formats -- CSV for downstream loading,
Markdown for quick human reading in the repo, and LaTeX for direct inclusion in the
paper -- while :func:`write_csv` emits CSV only, for tables too wide or long to be
useful as Markdown/LaTeX. These helpers keep that output contract in one place so all
modes stay consistent.
"""

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
from matplotlib.figure import Figure

from utilities.latex_utils import df_to_latex, format_float

logger = logging.getLogger("div")


def write_table(
    df: pd.DataFrame,
    tables_dir: Path,
    name: str,
    cols_to_formatter: dict[str, Callable[[Any], str]] | None = None,
    index: bool = False,
) -> None:
    """Write ``df`` to ``tables_dir`` as ``name``.{csv,md,tex}.

    ``cols_to_formatter`` maps a column (or list of columns) to a LaTeX cell
    formatter from :mod:`utilities.latex_utils`. When omitted, every float column
    is rendered with :data:`~utilities.latex_utils.format_float`.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    tex_path = tables_dir / f"{name}.tex"

    df.to_csv(csv_path, index=index)
    md_path.write_text(df.to_markdown(index=index), encoding="utf-8")

    if cols_to_formatter is None:
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        cols_to_formatter = {c: format_float for c in float_cols}
    tex_path.write_text(
        df_to_latex(df, cols_to_formatter, index=index), encoding="utf-8"
    )

    logger.info("Wrote table %s (.csv/.md/.tex, %d rows)", name, len(df))


def write_csv(df: pd.DataFrame, tables_dir: Path, name: str, index: bool = False) -> None:
    """Write ``df`` to ``tables_dir`` as ``name.csv`` only.

    For tables that are too large/wide to be useful as Markdown or LaTeX (e.g.
    per-corpus dumps), where only the machine-readable CSV is wanted.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{name}.csv", index=index)
    logger.info("Wrote table %s (.csv only, %d rows)", name, len(df))


def save_fig(fig: Figure, figures_dir: Path, name: str) -> None:
    """Save ``fig`` as ``figures_dir/name.pdf`` and close it."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{name}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s.pdf", name)
