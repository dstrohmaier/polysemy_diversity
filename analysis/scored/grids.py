"""The scored methods laid back onto the simulation's (k, offset) design grid.

The rest of the scored analysis reduces each method to one scalar per (method, scheme,
measure) -- a Spearman rho with a CI, drawn as a dot plot. That says whether a method
ranks the pairs correctly but collapses the design the corpora were simulated over, so
it cannot say *where* in that design a method tracks the ground truth and where it
fails. This module puts the per-pair results back onto the (k, offset) grid.

Two geometries, from the two families of comparison scheme (see
:mod:`simulation.pairing`):

``primary``
    Every variant against the lemma's low-diversity anchor, so each pair has a
    well-defined target ``(k, offset)`` -- a heatmap cell.
``along_k`` / ``along_slope``
    One-step comparisons between neighbouring grid nodes -- an arrow from the source
    node to the target node. Both schemes share one panel: ``along_k`` steps k at a
    fixed offset (vertical) and ``along_slope`` steps offset at a fixed k
    (horizontal).

Three figures per geometry, each **panelled along the axis its content actually varies
over**. That is the organising principle here and it is worth stating plainly, because
the obvious alternative is wrong: a method's log-ratio is one number per pair and
carries *no* measure dimension, so panelling the score figure by measure would repeat
identical numbers across every row and invite a reader to see structure that is not
there. Hence:

===================  =============  ==========================================
Figure               Panels         Varies over
===================  =============  ==========================================
score                1x3 method     the method
ground truth         1x4 measure    the measure
signed error         3x4 both       both -- error is against a specific GT column
===================  =============  ==========================================

The score and ground-truth figures share one colour scale and are meant to be read
side by side; the signed-error figure has its own symmetric scale. All three are
diverging and centred at 0, which is meaningful for each: no shift, and no error.

Import position: this module sits after :mod:`analysis.scored.methods` and before the
two mode modules, keeping the one-way chain ``naming -> io -> stats -> methods ->
grids -> {comparative, pooled}``.
"""

import logging
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from matplotlib.colors import Normalize  # type: ignore

from analysis.io import save_fig, write_table
from analysis.scored.methods import METHOD_ORDER, score_col
from analysis.scored.stats import GT_SHIFT_COLS, MEASURE_LABELS
from data_processing.simulation_loading import parse_variant

logger = logging.getLogger("div")

__all__ = [
    "add_grid_coords",
    "primary_cell_table",
    "step_arrow_table",
    "symmetric_limits",
    "write_grid_figures",
]

# Diverging and centred at 0, which is meaningful for both statistics drawn here (no
# shift, and no correlation). Pinned as a constant so all six figures agree.
GRID_CMAP = "RdBu_r"

# The comparison schemes each geometry draws (see simulation.pairing).
PRIMARY_SCHEME = "primary"
STEP_SCHEMES = ("along_k", "along_slope")

# Grid keys per geometry. A primary cell is identified by its target node alone (the
# source is the lemma's anchor); a step arrow needs both endpoints plus the scheme.
_PRIMARY_KEYS = ["target_k", "target_offset"]
_STEP_KEYS = ["scheme", "source_k", "source_offset", "target_k", "target_offset"]

# Offsets round-trip through a 2-dp string in the variant stem ("k3_offset_m0.50"), so
# group and pivot on a rounded value: raw float equality would fragment a grid column.
_OFFSET_DP = 2

# How much each arrow is pulled back from its endpoints, in grid-cell units, so it
# reads as a move *between* two nodes rather than through them. Both families share
# one gap: the axes are locked square, so trimming them differently would make an
# identical one-step move look shorter in one direction.
_ARROW_GAP = 0.16


def add_grid_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``(k, offset)`` parsed from the variant stems.

    Adds ``source_k``, ``source_offset``, ``target_k``, ``target_offset``, reusing
    :func:`~data_processing.simulation_loading.parse_variant` rather than re-deriving
    the stem format.

    Stems that do not parse yield NaN instead of raising. That is the DWUG case --
    its variants are ``g1``/``g2`` and it has no (k, offset) design -- so a DWUG
    invocation drops out cleanly at the caller's emptiness check rather than aborting
    the analysis.
    """
    out = df.copy()
    for side in ("source", "target"):
        ks: list[float] = []
        offsets: list[float] = []
        for stem in out[f"{side}_variant"]:
            try:
                k, offset = parse_variant(str(stem))
            except ValueError:
                ks.append(np.nan)
                offsets.append(np.nan)
            else:
                ks.append(float(k))
                offsets.append(round(offset, _OFFSET_DP))
        out[f"{side}_k"] = ks
        out[f"{side}_offset"] = offsets
    return out


def _grid_table(
    df: pd.DataFrame, keys: list[str], value_col: str, gt_col: str | None
) -> pd.DataFrame:
    """One row per grid group: both statistics plus the counts behind them.

    Columns are ``*keys``, ``mean_score``, ``signed_error``, ``n_lemmata``,
    ``n_pairs`` and ``note``. ``mean_score`` is the mean of ``value_col`` over the
    group's lemmata; ``signed_error`` is the mean of ``value_col - gt_col``, positive
    where the method overstates the shift.

    **Why signed error rather than a rank correlation.** The simulation applies one
    vocabulary-wide baseline slope, so a pair's ground truth is a pure function of its
    ``(k, offset)`` endpoints -- exactly what the grid keys pin. Every lemma in a cell
    therefore has the *same* ground truth, and a within-cell correlation has no
    variation to work with: it is undefined in every cell of both geometries. An error
    against that known constant is the statistic the geometry actually supports.
    Ranking evidence still lives in the pooled/comparative dot plots, which correlate
    across cells rather than within one.

    Both quantities are log-ratios in the same units, so their difference is directly
    interpretable and shares the score figure's diverging scale centred at 0.

    ``note`` distinguishes an empty cell (``n=0``) from a grid node missing from the
    table entirely, which is absent data.

    **The q=0 (richness) column is a special case**: ``gt_shift_q0`` is
    ``log(k_T / k_S)``, which is exactly 0 wherever a group fixes both k values -- every
    ``along_slope`` arrow, and every primary cell. There the signed error collapses to
    ``mean_score`` itself. The column stays readable but is measuring raw magnitude
    against a zero target, not error against a varying one.

    ``gt_col=None`` means "means only": the ground-truth figure plots a ``gt_shift_*``
    column as its own ``mean_score``, and differencing that column against itself would
    be a trivial 0 everywhere. Those rows carry a NaN ``signed_error`` instead.
    """
    rows = []
    cols = [value_col] if gt_col is None else [value_col, gt_col]
    for group_val, sub in df.groupby(keys, dropna=True):
        keyvals = group_val if isinstance(group_val, tuple) else (group_val,)
        pair = sub[cols].dropna()
        xs = pair[value_col].to_numpy(dtype=float)
        if gt_col is None:
            err, note = float("nan"), ""
        elif len(xs) == 0:
            err, note = float("nan"), "n=0"
        else:
            # Differenced per lemma, then averaged. Equivalent to mean(score) - GT
            # while the cell's GT is constant, but stays correct if it ever is not.
            errs = xs - pair[gt_col].to_numpy(dtype=float)
            err, note = float(np.mean(errs)), ""
        rows.append(
            {
                **dict(zip(keys, keyvals)),
                "mean_score": float(np.mean(xs)) if len(xs) else float("nan"),
                "signed_error": err,
                "n_lemmata": sub["lemma_pos"].nunique(),
                "n_pairs": len(pair),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def primary_cell_table(
    df: pd.DataFrame, value_col: str, gt_col: str | None
) -> pd.DataFrame:
    """Per-``(k, offset)`` cell statistics over the ``primary`` scheme's pairs."""
    sub = df[df["scheme"] == PRIMARY_SCHEME]
    if sub.empty:
        return pd.DataFrame()
    return _grid_table(sub, _PRIMARY_KEYS, value_col, gt_col)


def step_arrow_table(
    df: pd.DataFrame, value_col: str, gt_col: str | None
) -> pd.DataFrame:
    """Per-arrow statistics over the one-step ``along_k`` / ``along_slope`` pairs."""
    sub = df[df["scheme"].isin(STEP_SCHEMES)]
    if sub.empty:
        return pd.DataFrame()
    return _grid_table(sub, _STEP_KEYS, value_col, gt_col)


def symmetric_limits(values: Iterable[float]) -> float:
    """``M`` such that ``[-M, +M]`` covers every finite value, for a centred colour map.

    Callers pool everything that must share a scale into one call -- for the score and
    ground-truth figures that is every method cell mean *and* every GT cell mean, so
    the two figures can be read side by side. Degenerate input (all NaN, or all zero)
    falls back to ``1.0`` rather than producing a zero-width normalisation.
    """
    finite = [abs(float(v)) for v in values if np.isfinite(v)]
    m = max(finite) if finite else 0.0
    return m if m > 0 else 1.0


def _axis_orders(cells: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Sorted k and offset vocabularies, pinned across every panel of a figure.

    Computed once over the union of all panels so a node present for one method but
    absent for another still occupies its slot, instead of shifting the grid between
    panels -- the same discipline as ``_scheme_order`` / ``_pos_values`` in
    :mod:`analysis.scored.pooled`.
    """
    ks = sorted({k for k in cells["target_k"].dropna()})
    offsets = sorted({o for o in cells["target_offset"].dropna()})
    if "source_k" in cells.columns:
        ks = sorted(set(ks) | {k for k in cells["source_k"].dropna()})
        offsets = sorted(set(offsets) | {o for o in cells["source_offset"].dropna()})
    return ks, offsets


def draw_grid_heatmap(
    ax,
    cells: pd.DataFrame,
    value_col: str,
    k_order: list[float],
    offset_order: list[float],
    norm: Normalize,
    cmap,
) -> None:
    """Fill one panel with a ``k`` x ``offset`` heatmap of ``value_col``.

    Cells absent from ``cells`` stay NaN and render as the colormap's "bad" colour
    rather than as its zero colour, so missing data cannot be misread as a value of 0.
    """
    matrix = np.full((len(k_order), len(offset_order)), np.nan)
    for _, row in cells.iterrows():
        if pd.isna(row["target_k"]) or pd.isna(row["target_offset"]):
            continue
        i = k_order.index(row["target_k"])
        j = offset_order.index(row["target_offset"])
        matrix[i, j] = row[value_col]
    ax.pcolormesh(
        np.arange(len(offset_order) + 1),
        np.arange(len(k_order) + 1),
        np.ma.masked_invalid(matrix),
        norm=norm,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.3,
    )
    _label_axes(ax, k_order, offset_order, centred=True)


def draw_step_arrows(
    ax,
    arrows: pd.DataFrame,
    value_col: str,
    k_order: list[float],
    offset_order: list[float],
    norm: Normalize,
    cmap,
) -> None:
    """Fill one panel with one coloured arrow per step comparison.

    ``along_k`` rows step k at a fixed offset and so draw vertically; ``along_slope``
    rows step offset at a fixed k and draw horizontally. Both families share the panel
    because they live on the same grid, and their orientation alone identifies which
    scheme an arrow belongs to.

    Both families are one-step moves between neighbouring nodes, so every arrow sits
    on the node line and no lane separation is needed.

    Arrows are pulled back from their endpoints in *data* units, so the gap is the
    same fraction of a cell whatever the span. Trimming by a fixed number of points
    instead (matplotlib's ``shrinkA``/``shrinkB``) eats a much larger share of a
    one-cell arrow than of a three-cell one, which makes single steps look stubby
    beside the vertical arrows they should match.

    A near-zero value maps to a near-white fill, so every arrow also carries a thin
    outline; without it the low-magnitude slope arrows vanish against the panel even
    though they are the denser family. Faint node markers keep the grid legible where
    arrows are absent -- and absent they will be, ``along_k`` being the thin scheme
    throughout.

    See :func:`_grid_table` for why the signed-error figure's q=0 panel reduces to the
    raw score.
    """
    xs, ys = np.meshgrid(range(len(offset_order)), range(len(k_order)))
    ax.scatter(xs, ys, s=5, color="0.75", zorder=1)
    cmap_obj = plt.get_cmap(cmap)
    for _, row in arrows.iterrows():
        if any(pd.isna(row[c]) for c in _STEP_KEYS[1:]) or pd.isna(row[value_col]):
            continue
        x0 = offset_order.index(row["source_offset"])
        y0 = k_order.index(row["source_k"])
        x1 = offset_order.index(row["target_offset"])
        y1 = k_order.index(row["target_k"])
        # Trim in data units rather than points (shrinkA/shrinkB): a fixed number of
        # points removes a far larger fraction of a short arrow than of a long one.
        if y0 == y1:
            direction = 1 if x1 > x0 else -1
            x0 += direction * _ARROW_GAP
            x1 -= direction * _ARROW_GAP
        else:
            direction = 1 if y1 > y0 else -1
            y0 += direction * _ARROW_GAP
            y1 -= direction * _ARROW_GAP
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            zorder=2,
            arrowprops=dict(
                arrowstyle="simple,head_width=0.62,head_length=0.5,tail_width=0.34",
                facecolor=cmap_obj(norm(row[value_col])),
                edgecolor="0.4",
                linewidth=0.4,
            ),
        )
    _label_axes(ax, k_order, offset_order, centred=False)


def _label_axes(
    ax, k_order: list[float], offset_order: list[float], centred: bool
) -> None:
    """Tick the shared (k, offset) axes and lock the cells square.

    ``centred`` shifts the ticks by half a cell for the heatmap, whose pcolormesh
    quadrilaterals span integer edges, versus the arrow panel whose nodes sit *on* the
    integers.

    Both axes are in units of one grid step, so ``aspect="equal"`` makes every cell
    square. That is not cosmetic on the arrow panels: with unequal aspect a vertical
    (k) arrow is drawn longer than a horizontal (slope) one spanning the same number
    of steps, which reads as a difference in magnitude when it is only a difference in
    the axes' scaling.
    """
    shift = 0.5 if centred else 0.0
    ax.set_xticks([i + shift for i in range(len(offset_order))])
    ax.set_xticklabels([f"{o:+.1f}" for o in offset_order], fontsize="x-small")
    ax.set_yticks([i + shift for i in range(len(k_order))])
    ax.set_yticklabels([f"{k:.0f}" for k in k_order], fontsize="x-small")
    if not centred:
        ax.set_xlim(-0.5, len(offset_order) - 0.5)
        ax.set_ylim(-0.5, len(k_order) - 0.5)
    ax.set_aspect("equal", adjustable="box")


DrawFn = Callable[..., None]


def _panel_figure(
    panels: list[tuple[str, pd.DataFrame]],
    draw_fn: DrawFn,
    value_col: str,
    k_order: list[float],
    offset_order: list[float],
    norm: Normalize,
    nrows: int,
    ncols: int,
    cbar_label: str,
    row_labels: list[str] | None = None,
) -> plt.Figure:
    """Lay ``panels`` out as an ``nrows`` x ``ncols`` grid sharing one colour scale.

    One routine serves all three figure shapes (1x3 method, 1x4 measure, 3x4 both);
    they differ only in their panel vocabulary and titles.

    Seaborn's ``FacetGrid`` was the obvious fit but maps a *long dataframe* through a
    plotting function and offers no control over a shared colour normalisation or
    figure-level colourbar placement, neither of which these figures work without. The
    repo's own precedent for a panel grid is ``plt.subplots(..., squeeze=False)`` plus
    ``axes.ravel()`` (see :mod:`analysis.scored.pooled` and
    :mod:`analysis.simulated_data.raw_simulated`); ``squeeze=False`` also keeps the
    axes array 2-D so the single-row shapes need no special case.
    """
    # Square cells mean a panel's proportions follow the grid's: 11 offsets x 3 k is
    # wide and short. Size the panels from that ratio (plus a fixed allowance for the
    # tick labels and titles) so the axes fill the figure instead of floating in
    # whitespace with the aspect lock doing the shrinking.
    cell = 0.55
    panel_w = cell * len(offset_order) + 0.55
    panel_h = cell * len(k_order) + 0.75
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_w * ncols + 1.2, panel_h * nrows + 0.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax, (title, cells) in zip(flat, panels):
        if cells is not None and not cells.empty:
            draw_fn(ax, cells, value_col, k_order, offset_order, norm, GRID_CMAP)
        else:
            _label_axes(ax, k_order, offset_order, centred=draw_fn is draw_grid_heatmap)
        ax.set_title(title, fontsize="small")
    for ax in flat[len(panels) :]:
        ax.set_visible(False)

    # Row identity on the leftmost panel of each row, so the 3xN grid's two axes stay
    # distinguishable (columns are titled, rows are not). Single-row figures are
    # already fully identified by their panel titles, and labelling that one row would
    # collide with the figure-level "Sense count k" ylabel.
    # A figure-level supylabel is centred on the figure and would print straight
    # through the middle row's own ylabel, so when rows are labelled the k axis is
    # named on each row label instead of once for the figure.
    labelled_rows = bool(row_labels) and nrows > 1
    if labelled_rows:
        for r, label in enumerate(row_labels):
            axes[r, 0].set_ylabel(f"{label}\nSense count k", fontsize="small")

    fig.supxlabel("Zipfian slope offset", fontsize="small")
    if not labelled_rows:
        fig.supylabel("Sense count k", fontsize="small")
    # tight_layout before the figure-level colourbar: the reverse order lets the
    # layout pass reclaim the space the colourbar was given (same constraint the
    # pooled module documents for its figure-level legend).
    fig.tight_layout()
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=GRID_CMAP)
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize="small")
    cbar.ax.tick_params(labelsize="x-small")
    return fig


def _geometry_figures(
    loaded: dict[str, pd.DataFrame],
    table_fn: Callable[[pd.DataFrame, str, str | None], pd.DataFrame],
    draw_fn: DrawFn,
    figures_dir: Path,
    stem: str,
) -> pd.DataFrame:
    """Emit one geometry's three figures and return its long-format companion table.

    ``table_fn`` selects the geometry (primary cells or step arrows) and ``draw_fn``
    the matching drawing primitive; everything else -- which panels exist, which
    colour scale they share -- follows from the figure shapes described in the module
    docstring.
    """
    methods = [m for m in METHOD_ORDER if m in loaded]
    if not methods:
        return pd.DataFrame()

    # Per (method, measure) cell tables. mean_score does not depend on the measure --
    # it is the same column in all four -- so the score figure reads the first measure
    # arbitrarily while the correlation figure uses all four.
    cells: dict[tuple[str, object], pd.DataFrame] = {}
    for method in methods:
        for measure, gt_col in GT_SHIFT_COLS.items():
            cells[(method, measure)] = table_fn(loaded[method], score_col(method), gt_col)

    # Ground truth is a property of the pair, not of the method, so take it from any
    # one method's frame: the pair rows are identical across methods.
    ref = loaded[methods[0]]
    gt_cells = {
        measure: table_fn(ref, gt_col, None)
        for measure, gt_col in GT_SHIFT_COLS.items()
    }

    # Axis vocabularies come from the union of every panel, so a node present for one
    # method but not another still holds its slot instead of shifting the grid.
    non_empty = [c for c in list(cells.values()) + list(gt_cells.values()) if not c.empty]
    if not non_empty:
        return pd.DataFrame()
    k_order, offset_order = _axis_orders(pd.concat(non_empty, ignore_index=True))

    first_measure = next(iter(GT_SHIFT_COLS))
    score_panels = [(m, cells[(m, first_measure)]) for m in methods]
    gt_panels = [
        (MEASURE_LABELS[measure], gt_cells[measure]) for measure in GT_SHIFT_COLS
    ]

    # One scale across both figures -- every method cell mean and every GT cell mean --
    # so a method panel and a GT panel can be compared directly.
    shared = symmetric_limits(
        pd.concat([df for _, df in score_panels + gt_panels if not df.empty])[
            "mean_score"
        ]
    )
    score_norm = Normalize(vmin=-shared, vmax=shared)

    # Stacked one method per row, for the same reason the GT figure is 2x2: square
    # cells make each panel wide and short, so panels read better above one another
    # than side by side.
    save_fig(
        _panel_figure(
            score_panels, draw_fn, "mean_score", k_order, offset_order, score_norm,
            len(score_panels), 1, "Mean log-ratio",
        ),
        figures_dir,
        stem,
    )
    # 2x2 rather than 1x4: a square-celled panel of 11 offsets by 3 k values is wide
    # and short, so four of them in a row would make an unreadably long figure.
    save_fig(
        _panel_figure(
            gt_panels, draw_fn, "mean_score", k_order, offset_order, score_norm,
            2, 2, "Mean GT shift",
        ),
        figures_dir,
        f"{stem}_gt",
    )

    # Signed error: 3x4, method x measure. Its own symmetric scale rather than the
    # score figure's -- an error can exceed the scores it is built from, and pinning it
    # to a shared scale would flatten the differences this figure exists to show.
    error_panels = [
        (MEASURE_LABELS[measure], cells[(method, measure)])
        for method in methods
        for measure in GT_SHIFT_COLS
    ]
    error_limit = symmetric_limits(
        pd.concat([df for _, df in error_panels if not df.empty])["signed_error"]
        if any(not df.empty for _, df in error_panels)
        else []
    )
    save_fig(
        _panel_figure(
            error_panels, draw_fn, "signed_error", k_order, offset_order,
            Normalize(vmin=-error_limit, vmax=error_limit),
            len(methods), len(GT_SHIFT_COLS),
            "Mean signed error (score - GT)", row_labels=methods,
        ),
        figures_dir,
        stem.replace("shift_", "error_"),
    )

    parts = []
    for (method, measure), df in cells.items():
        if df.empty:
            continue
        out = df.copy()
        out.insert(0, "method", method)
        out.insert(1, "measure", MEASURE_LABELS[measure])
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_grid_figures(
    loaded: dict[str, pd.DataFrame],
    figures_dir: Path,
    tables_dir: Path,
    prefix: str = "",
) -> None:
    """Emit both geometries' figures and companion tables for ``loaded``.

    The single entry point both comparative modes call. ``prefix`` distinguishes the
    pooled mode's output stems, matching the existing pooled naming convention.

    Silently skips a dataset with no (k, offset) design -- the DWUG evaluation, whose
    variants are ``g1``/``g2`` -- after logging why, since these figures are specific
    to the simulated grid.
    """
    with_coords = {m: add_grid_coords(df) for m, df in loaded.items()}
    if all(df["target_k"].isna().all() for df in with_coords.values()):
        logger.info(
            "No (k, offset) grid coordinates in these pair scores "
            "(non-simulated layout); skipping the grid figures."
        )
        return

    for table_fn, draw_fn, subdir, stem, table_name in (
        (primary_cell_table, draw_grid_heatmap, "grid_primary",
         "shift_grid_primary", "grid_primary_cells"),
        (step_arrow_table, draw_step_arrows, "grid_steps",
         "shift_grid_steps", "grid_step_arrows"),
    ):
        table = _geometry_figures(
            with_coords, table_fn, draw_fn, figures_dir / subdir, f"{prefix}{stem}"
        )
        if table.empty:
            logger.warning("No cells for the %s geometry; no table written.", subdir)
            continue
        # note last, matching the convention the other scored tables follow.
        cols = [c for c in table.columns if c != "note"] + ["note"]
        write_table(
            table[cols], tables_dir, f"{prefix}{table_name}", convert_col_names=True
        )
        logger.info("%s: %d cells across %d method(s)", subdir, len(table),
                    table["method"].nunique())
