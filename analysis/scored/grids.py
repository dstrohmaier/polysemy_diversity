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
correlation          3x4 both       both -- rho is against a specific GT column
===================  =============  ==========================================

The score and ground-truth figures share one colour scale and are meant to be read
side by side; the correlation figure stands alone on a fixed ``[-1, 1]``.

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
from analysis.scored.stats import GT_SHIFT_COLS, MEASURE_LABELS, rho_or_note
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

# How far off the node line each horizontal arrow lane sits, in grid-cell units. The
# two along_slope strides would otherwise be drawn on top of each other.
_LANE_OFFSET = 0.26

# How much each arrow is pulled back from its endpoints, in grid-cell units, so it
# reads as a move *between* two nodes rather than through them. Multi-step arrows are
# trimmed harder: they pass the column where the vertical arrows sit, and the wider
# gap keeps the two families from touching.
_ARROW_GAP = 0.16
_WIDE_ARROW_GAP = 0.30


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

    Columns are ``*keys``, ``mean_score``, ``spearmanr``, ``n_lemmata``, ``n_pairs``
    and ``note``. ``mean_score`` is the mean of ``value_col`` over the group's lemmata;
    ``spearmanr`` is that column against ``gt_col``, delegated to
    :func:`~analysis.scored.stats.rho_or_note` so this path and
    :func:`~analysis.scored.stats.correlation_table` agree on which cells are
    undefined.

    The plain rho is used here, **not** ``spearman_with_ci``: a CI cannot be drawn in a
    heatmap cell or on an arrow, and bootstrapping every cell would cost ~1000
    resamples x ~400 cells per geometry for nothing. The companion tables carry
    ``n_lemmata`` so a reader can still judge how much weight a cell bears. If CIs are
    ever wanted here that trade-off has to be revisited, not silently reversed.

    ``note`` distinguishes the two reasons a cell can be blank in the correlation
    figures: a non-empty note with ``n_pairs > 0`` is a degenerate statistic, whereas
    a grid node missing from the table entirely is absent data.

    **The q=0 correlation is largely undefined on this grid**, so the richness column
    of the rho figures is blank or nearly so. That is geometry, not a data shortage:
    ``gt_shift_q0`` is ``log(k_T / k_S)``, a function of the two k values alone, and
    the grid keys pin those values. A step arrow fixes both endpoints, so its q=0
    predictor is *always* constant and the arrow rho is undefined without exception.
    A primary cell fixes only the target k; its rho becomes defined exactly when the
    cell pools lemmata whose anchors sit at different k, which happens in the pooled
    mode but rarely within one PoS. The mean-score and ground-truth figures are
    unaffected -- only the correlation ones lose that column.

    ``gt_col=None`` means "means only": the ground-truth figure plots a ``gt_shift_*``
    column as its own ``mean_score``, and correlating that column with itself would be
    a trivial 1.0 everywhere. Those rows carry a NaN ``spearmanr`` rather than a
    meaningless one.
    """
    rows = []
    cols = [value_col] if gt_col is None else [value_col, gt_col]
    for group_val, sub in df.groupby(keys, dropna=True):
        keyvals = group_val if isinstance(group_val, tuple) else (group_val,)
        pair = sub[cols].dropna()
        xs = pair[value_col].to_numpy(dtype=float)
        if gt_col is None:
            rho, note = float("nan"), ""
        else:
            rho, _, _, _, note = rho_or_note(
                xs, pair[gt_col].to_numpy(dtype=float), with_ci=False
            )
        rows.append(
            {
                **dict(zip(keys, keyvals)),
                "mean_score": float(np.mean(xs)) if len(xs) else float("nan"),
                "spearmanr": rho,
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

    ``along_slope`` is emitted at two strides (see
    :data:`~simulation.pairing.SLOPE_STRIDES`): neighbour moves and wider ones that
    reach the same magnitude as a single k step. The two strides would overlap on the
    node line, so they are drawn on separate lanes -- one-step above the row, longer
    strides below -- with the node markers on the line between them. Vertical arrows
    stay on the line, since the k axis has a single stride. The wider strides are
    emitted end to end rather than sliding, so they partition their row instead of
    stacking on top of each other.

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

    See :func:`_grid_table` for why the correlation figure's q=0 panel has no arrows
    at all.
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
        span = max(abs(x1 - x0), abs(y1 - y0))
        if y0 == y1:
            # Horizontal arrows share their row with the vertical ones, so they sit
            # off the node line: neighbours above it, multi-step below. The wider
            # arrows also start and end a little inside their endpoints, clearing the
            # column each vertical arrow occupies.
            lane = _LANE_OFFSET if span <= 1 else -_LANE_OFFSET
            y0 += lane
            y1 += lane
        # Trim in data units rather than points (shrinkA/shrinkB): a fixed number of
        # points removes a far larger fraction of a one-cell arrow than of a
        # three-cell one, which is what made the single-step arrows look stubby
        # beside the vertical ones. A proportional gap keeps every arrow reading as
        # the number of steps it actually spans.
        gap = _ARROW_GAP if span <= 1 else _WIDE_ARROW_GAP
        if x1 != x0:
            direction = 1 if x1 > x0 else -1
            x0 += direction * gap
            x1 -= direction * gap
        if y1 != y0:
            direction = 1 if y1 > y0 else -1
            y0 += direction * gap
            y1 -= direction * gap
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
    if row_labels and nrows > 1:
        for r, label in enumerate(row_labels):
            axes[r, 0].set_ylabel(label, fontsize="small")

    fig.supxlabel("Zipfian slope offset", fontsize="small")
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

    # Correlation: 3x4, method x measure, on the fixed [-1, 1] rho scale.
    rho_panels = [
        (MEASURE_LABELS[measure], cells[(method, measure)])
        for method in methods
        for measure in GT_SHIFT_COLS
    ]
    save_fig(
        _panel_figure(
            rho_panels, draw_fn, "spearmanr", k_order, offset_order,
            Normalize(vmin=-1, vmax=1), len(methods), len(GT_SHIFT_COLS),
            "SRC (Spearman's rank correlation)", row_labels=methods,
        ),
        figures_dir,
        stem.replace("shift_", "rho_"),
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
