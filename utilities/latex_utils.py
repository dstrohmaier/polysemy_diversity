from typing import Any, Callable, Optional

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

# Rendered in place of a missing (NaN) cell. The analysis pivots are sparse -- most
# (corpus, matrix) cells are empty -- so every numeric formatter routes NaN here
# rather than letting "nan" leak into a table. An en-dash reads better in LaTeX
# than a hyphen.
MISSING = "--"


def make_float_formatter(
    decimals: int = 2,
    scale: float = 1.0,
    percent_sign: bool = False,
    missing: str = MISSING,
) -> Callable[[float], str]:
    """Build a cell formatter for floats with uniform NaN handling.

    One factory replaces the previous family of near-identical percentage
    formatters. `decimals` fixes the precision, `scale` multiplies the value
    (100 to render a fraction on a 0-100 scale), and `percent_sign` appends an
    escaped percent sign. NaN always renders as `missing`, so a column never
    silently mixes a number with the literal "nan".
    """

    def fmt(content: float) -> str:
        if content is None or (isinstance(content, float) and np.isnan(content)):
            return missing
        suffix = "\\%" if percent_sign else ""
        return "{:.{}f}{}".format(content * scale, decimals, suffix)

    return fmt


# Common instances, built from the factory so the NaN handling stays in one place.
# A "percentage" scales a fraction by 100 and shows the sign; "percentage points"
# scales by 100 but shows a bare number (a difference of percentages).
format_float = make_float_formatter(decimals=2)
format_percentage = make_float_formatter(decimals=1, scale=100, percent_sign=True)
format_percentage_2f = make_float_formatter(decimals=2, scale=100, percent_sign=True)
format_percentage_points = make_float_formatter(decimals=1, scale=100)
format_percentage_points_2f = make_float_formatter(decimals=2, scale=100)


def format_bool(content: bool) -> str:
    if content:
        return "\\checkmark"
    else:
        return ""


def col_formatter(col_name: str) -> str:
    """Render a column name as its LaTeX header.

    Covers this project's evaluation metrics -- accuracy, the Spearman rank
    correlation of the similarity task, and the OCS/PCS relation scores -- the
    matrix sub-types used as headers in the metric tables, plus a few generic
    statistics. Unknown names (e.g. corpus columns) pass through unchanged.
    """
    match col_name:
        case "accuracy":
            return "Acc."
        case "spearmanr" | "spearman_r":
            return "SRC"
        case "ocs":
            return "OCS"
        case "pcs":
            return "PCS"
        case "coverage":
            return "Cov."
        # Matrix sub-types (metric-table column headers). sg_neg is rewritten so
        # its underscore is not left raw when column escaping is bypassed.
        case "sg_neg":
            return "sg-neg"
        case "count" | "log" | "ppmi" | "svd" | "baseline":
            return col_name
        case "F1":
            return "F\\textsubscript{1}"
        case "F2":
            return "F\\textsubscript{2}"
        case "R2":
            return "R\\textsuperscript{2}"
        case _:
            return col_name


def df_to_latex(
    df: pd.DataFrame,
    cols_to_formatter: dict[str, Any],
    index: bool = True,
    escape_col_names: bool = True,
    convert_col_names: bool = False,
    highlight_max: bool = False,
    highlight_max_axis: int | str = "index",
    highlight_max_subset: Optional[list[str]] = None,
    highlight_min_subset: Optional[list[str]] = None,
) -> str:
    df = df.copy()

    s = df.style

    for cols, formatter in cols_to_formatter.items():
        s = s.format(subset=cols, formatter=formatter)  # type: ignore

    assert not (
        convert_col_names and escape_col_names
    ), "convert_col_names and escape_col_name cannot both be applied"

    if escape_col_names:
        s.format_index(escape="latex", axis=1)

    if convert_col_names:
        s.format_index(col_formatter, axis=1)  # type: ignore

    s.format_index(escape="latex", axis=0)

    if not index:
        s = s.hide()

    if highlight_max:
        s = s.highlight_max(axis=highlight_max_axis, props="font-weight: bold")

    if highlight_max_subset is not None:
        s = s.highlight_max(
            axis=0, props="font-weight: bold;", subset=highlight_max_subset
        )
    if highlight_min_subset is not None:
        s = s.highlight_min(
            axis=0, props="font-weight: bold;", subset=highlight_min_subset
        )

    latex = s.to_latex(
        multirow_align="c",
        clines="skip-last;data",
        hrules=True,
        convert_css=True,
    )

    latex = latex.replace("cline", "cmidrule(lr)")

    return latex
