"""Mode-agnostic output helpers shared by all analysis modes.

Analysis modes produce (a) tables and (b) figures (written as PDF). Tables come in
two flavours: :func:`write_table` emits three formats -- CSV for downstream loading,
Markdown for quick human reading in the repo, and LaTeX for direct inclusion in the
paper -- while :func:`write_csv` emits CSV only, for tables too wide or long to be
useful as Markdown/LaTeX. These helpers keep that output contract in one place so all
modes stay consistent.

This is the entry point of a three-module output layer, ordered by how much they
know about the output format: :mod:`analysis.naming` maps raw column names to
readable labels and knows about no format at all (figures use it too);
:mod:`analysis.latex_utils` adds the ``.tex``-specific escaping and Styler
rendering; and this module drives both. :func:`~analysis.naming.human_col_name` is
re-exported here so callers labelling a figure need only one import.
"""

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
from matplotlib.figure import Figure

from analysis.latex_utils import df_to_latex, format_float
from analysis.naming import LABEL_VALUE_COLS, human_col_name

# Re-exported so the output layer presents one import surface: callers that write
# tables or label figures reach for analysis.io and need not know whether a helper
# is format-agnostic (analysis.naming) or LaTeX-specific (analysis.latex_utils).
__all__ = ["human_col_name", "save_fig", "write_csv", "write_table"]

logger = logging.getLogger("div")


def _drop_empty_note(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` without a ``note`` column that holds no notes.

    Notes are written as ``""`` when a row is unexceptional, so a table where no
    row needed flagging carries a column of empty strings. Returns ``df``
    unchanged (not a copy) when there is a ``note`` column with content, or none
    at all; never mutates the caller's frame.
    """
    if "note" not in df.columns:
        return df
    notes = df["note"]
    if notes.isna().all() or (notes.fillna("").astype(str).str.strip() == "").all():
        return df.drop(columns="note")
    return df


def _humanise_label_values(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite identifier-valued cells in :data:`~analysis.naming.LABEL_VALUE_COLS`.

    Only touches strings that still look like raw schema identifiers -- no
    whitespace, and not already containing an acronym's casing -- so a value that
    is already prose is left alone. Copies before writing; never mutates the
    caller's frame.
    """
    cols = [c for c in LABEL_VALUE_COLS if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for col in cols:
        out[col] = out[col].map(
            lambda v: human_col_name(v) if isinstance(v, str) and " " not in v else v
        )
    return out


def write_table(
    df: pd.DataFrame,
    tables_dir: Path,
    name: str,
    cols_to_formatter: dict[str, Callable[[Any], str]] | None = None,
    index: bool = False,
    convert_col_names: bool = False,
) -> None:
    """Write ``df`` to ``tables_dir`` as ``name``.{csv,md,tex}.

    ``cols_to_formatter`` maps a column (or list of columns) to a LaTeX cell
    formatter from :mod:`analysis.latex_utils`. When omitted, every float column
    is rendered with :data:`~analysis.latex_utils.format_float`. Set
    ``convert_col_names`` to render the LaTeX headers via
    :func:`~analysis.latex_utils.col_formatter` (e.g. ``spearmanr`` -> ``SRC``,
    ``F1`` -> ``F``\\ :sub:`1`); this replaces the default header escaping.

    The CSV is the machine-readable output: it keeps the raw column names and the
    full schema, so downstream loaders see a stable table. The Markdown and LaTeX
    are the human-readable outputs, and get three readability passes: headers go
    through :func:`~analysis.naming.human_col_name` (``gt_shift_q0`` -> ``GT shift
    (q=0)``); identifier-valued cells in
    :data:`~analysis.naming.LABEL_VALUE_COLS` get the same rewriting
    (``same_lemma`` -> ``Same lemma``); and a ``note`` column carrying no notes
    (every cell empty or NaN) is dropped, since it only exists to flag
    exceptional rows.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    tex_path = tables_dir / f"{name}.tex"

    df.to_csv(csv_path, index=index)

    # Presentation-only frame: the CSV above already has the raw schema.
    shown = _humanise_label_values(_drop_empty_note(df))
    md_path.write_text(
        shown.rename(columns=human_col_name).to_markdown(index=index),
        encoding="utf-8",
    )

    if cols_to_formatter is None:
        float_cols = [c for c in shown.columns if pd.api.types.is_float_dtype(shown[c])]
        cols_to_formatter = {c: format_float for c in float_cols}
    else:
        # Formatters are keyed by raw column name; drop any naming a column that
        # is not in the shown frame (e.g. ``note``) so Styler.format is not handed
        # a missing subset.
        cols_to_formatter = {
            c: f for c, f in cols_to_formatter.items()
            if not isinstance(c, str) or c in shown.columns
        }
    tex_path.write_text(
        df_to_latex(
            shown,
            cols_to_formatter,
            index=index,
            # df_to_latex forbids escaping and converting headers at once.
            escape_col_names=not convert_col_names,
            convert_col_names=convert_col_names,
        ),
        encoding="utf-8",
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
