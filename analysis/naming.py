"""Readable labels for the analysis schema's raw column names.

The analysis frames use snake_case identifiers (``gt_shift_q0``, ``lemma_pos``)
so they stay convenient to index, but every human-facing surface -- Markdown and
LaTeX table headers, figure axis labels, legend titles -- wants prose. This module
holds that one mapping, so those surfaces cannot drift apart.

Output-format-agnostic by design: :func:`human_col_name` returns plain text, and
:mod:`analysis.latex_utils` layers the LaTeX-only escaping on top. It is the
lowest layer of the output stack and imports nothing from :mod:`analysis`, which
is what lets both :mod:`analysis.io` and :mod:`analysis.latex_utils` use it.
"""

import re

# Acronyms that must render with fixed casing wherever they appear in a column
# name, mapped from the lower-case token used in the raw schema. Applied token-wise
# by :func:`human_col_name`, so ``gt_shift_q0`` and ``rho_err_vs_n`` pick these up
# without needing an entry of their own. Not all are all-caps: "PoS" is Part of
# Speech and "vMF" is the von Mises-Fisher distribution.
ACRONYMS = {
    "gt": "GT",
    "src": "SRC",
    "ocs": "OCS",
    "pcs": "PCS",
    "wic": "WiC",
    "js": "JS",
    "ci": "CI",
    "vmf": "vMF",
    "pos": "PoS",
    "ppmi": "PPMI",
    "svd": "SVD",
    "sg": "SG",
}

# Whole-name overrides, applied before any token-wise rewriting. These are names
# whose readable form is not a mechanical transform of the raw name -- an acronym
# for a multi-word statistic, or a phrase that reads better than its schema name.
COL_NAMES = {
    "spearmanr": "SRC",
    "spearman_r": "SRC",
    "rho": "SRC",
    "rho_err_vs_n": "SRC (error vs n)",
    "accuracy": "Acc.",
    "coverage": "Cov.",
    "ci_low": "CI low",
    "ci_high": "CI high",
    "n_used": "n used",
    "n_used_min": "n used (min)",
    "n_used_max": "n used (max)",
    "n_used_median": "n used (median)",
    "lemma_pos": "Lemma PoS",
    "sg_neg": "SG-neg",
    "js_divergence": "JS divergence",
    "entropy_bits": "Entropy (bits)",
    "p_diff_theoretical": "P(diff), theoretical",
    # Evenness is the one Hill-family target without a q-order; name its
    # definition the way MEASURE_LABELS does in analysis.scored.stats.
    "gt_shift_evenness": "GT shift (E=1D/0D)",
    "same_fraction": "Same-sense fraction",
    "diff_fraction": "Diff-sense fraction",
    "k_senses": "Senses (k)",
    "n_senses_available": "Senses available",
    "n_senses_observed": "Senses observed",
    "n_lemmata": "Lemmata (n)",
    "n_pos": "PoS (n)",
}

# Tokens that keep their exact casing rather than being title-cased or upper-cased:
# single-letter maths variables and small connecting words.
LOWER_TOKENS = {"n", "k", "q", "p", "e", "vs", "per", "of", "used"}

# Columns whose cells are schema identifiers rather than free text or data values,
# so they get the same readable rewriting as the headers ("same_lemma" -> "Same
# lemma"). ``note`` is deliberately absent: its cells are prose ("n<3"). A ``pos``
# cell is already an all-caps tag ("NOUN"), which human_col_name passes through
# unchanged; it is listed so an unexpected lower-case tag still renders readably.
LABEL_VALUE_COLS = ("scheme", "predictor", "method", "lemma_pos", "pos", "dataset")


def human_col_name(col_name: str) -> str:
    """Render a raw column name as a readable label (plain text, no LaTeX).

    Three rules, in order: whole-name overrides from :data:`COL_NAMES`; a
    ``q<digit>`` suffix becomes a parenthesised ``(q=<digit>)``; then the
    remaining underscore-separated tokens are joined with spaces, with known
    acronyms recased via :data:`ACRONYMS`. Used for Markdown headers and figure
    labels, and as the base for the LaTeX headers built by
    :func:`~analysis.latex_utils.col_formatter`. Unknown names degrade gracefully
    -- an unrecognised token keeps its own casing, so a corpus column passes
    through readably.
    """
    if not isinstance(col_name, str):
        return col_name
    if col_name in COL_NAMES:
        return COL_NAMES[col_name]

    tokens = col_name.split("_")
    # A trailing Hill-order marker ("q0", "q1") reads better as "(q=0)".
    suffix = ""
    if len(tokens) > 1 and re.fullmatch(r"q\d+", tokens[-1]):
        suffix = f" (q={tokens[-1][1:]})"
        tokens = tokens[:-1]

    out = []
    for i, token in enumerate(tokens):
        if token.lower() in ACRONYMS:
            out.append(ACRONYMS[token.lower()])
        elif token.lower() in LOWER_TOKENS:
            out.append(token.lower())
        elif token.isupper():
            out.append(token)
        elif i == 0:
            out.append(token[:1].upper() + token[1:])
        else:
            out.append(token)
    return " ".join(out) + suffix
