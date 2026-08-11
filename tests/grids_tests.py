"""The (k, offset) design-grid figures and their per-cell statistic.

Two things are pinned here, both of which had silently broken before:

* the per-cell statistic is a **signed error** against ground truth, not a rank
  correlation. With one vocabulary-wide baseline slope the ground truth is a pure
  function of a pair's ``(k, offset)`` endpoints -- exactly what the grid keys pin --
  so every lemma in a cell shares one ground-truth value and a within-cell rho has no
  variation to work with. The rho grids came out blank in every cell of both
  geometries; the error is well defined there.
* both arrow families are drawn at the same length for a one-step move. The axes are
  locked square, so trimming vertical arrows harder than horizontal ones made an
  identical step read as a smaller one.
"""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from analysis.scored.grids import (
    add_grid_coords,
    draw_step_arrows,
    primary_cell_table,
    step_arrow_table,
)


def _pairs(n_lemmata: int = 4) -> pd.DataFrame:
    """Primary + step pairs over a small grid, with a constant-per-cell ground truth.

    The ground truth depends only on the pair's endpoints -- the real generative
    situation under a shared baseline -- so every lemma in a cell carries the same
    value.
    """
    rows = []
    for lemma in range(n_lemmata):
        for k in (3, 4):
            for offset in (-0.3, 0.0, 0.3):
                if (k, offset) == (3, 0.3):
                    continue  # the anchor
                rows.append(
                    {
                        "lemma_pos": f"w{lemma}_ADJ",
                        "scheme": "primary",
                        "source_variant": "k3_offset_p0.30",
                        "target_variant": (
                            f"k{k}_offset_{'m' if offset < 0 else 'p'}"
                            f"{abs(offset):.2f}"
                        ),
                        # GT is a function of the endpoints alone.
                        "gt": float(k) + offset,
                        "score": float(k) + offset + 0.5,  # a constant +0.5 bias
                    }
                )
    return add_grid_coords(pd.DataFrame(rows))


class SignedErrorTestCase(unittest.TestCase):
    def test_error_is_defined_when_gt_is_constant_in_a_cell(self):
        """The regression: a within-cell rho is undefined here, an error is not."""
        table = primary_cell_table(_pairs(), "score", "gt")
        self.assertFalse(table.empty)
        self.assertTrue(
            table["signed_error"].notna().all(),
            "signed error must be defined in every cell",
        )

    def test_error_recovers_a_known_bias(self):
        # Scores are ground truth plus exactly 0.5 everywhere.
        table = primary_cell_table(_pairs(), "score", "gt")
        np.testing.assert_allclose(table["signed_error"], 0.5)

    def test_error_sign_marks_over_and_under_statement(self):
        df = _pairs()
        df["score"] = df["gt"] - 0.25  # understate the shift everywhere
        table = primary_cell_table(df, "score", "gt")
        self.assertTrue((table["signed_error"] < 0).all())

    def test_error_is_nan_without_a_gt_column(self):
        # The ground-truth figure plots a GT column as its own mean_score; differencing
        # it against itself would be a trivial zero.
        table = primary_cell_table(_pairs(), "gt", None)
        self.assertTrue(table["signed_error"].isna().all())

    def test_step_arrows_also_carry_a_defined_error(self):
        rows = []
        for lemma in range(4):
            for k in (3, 4):
                rows.append(
                    {
                        "lemma_pos": f"w{lemma}_ADJ",
                        "scheme": "along_slope",
                        "source_variant": f"k{k}_offset_p0.30",
                        "target_variant": f"k{k}_offset_p0.00",
                        "gt": 0.2,
                        "score": 0.35,
                    }
                )
        table = step_arrow_table(add_grid_coords(pd.DataFrame(rows)), "score", "gt")
        self.assertFalse(table.empty)
        np.testing.assert_allclose(table["signed_error"], 0.15)


class ArrowGeometryTestCase(unittest.TestCase):
    def _drawn_lengths(self, arrows: pd.DataFrame) -> list[float]:
        fig, ax = plt.subplots()
        try:
            draw_step_arrows(
                ax, arrows, "signed_error", [3.0, 4.0], [0.0, 0.3],
                Normalize(-1, 1), "RdBu_r",
            )
            lengths = []
            for ann in ax.texts:
                (x0, y0), (x1, y1) = ann.xyann, ann.xy
                lengths.append(float(np.hypot(x1 - x0, y1 - y0)))
            return lengths
        finally:
            plt.close(fig)

    def test_vertical_and_horizontal_steps_draw_the_same_length(self):
        """A one-step move must look the same size whichever axis it runs along."""
        arrows = pd.DataFrame(
            [
                dict(scheme="along_slope", source_k=3.0, source_offset=0.3,
                     target_k=3.0, target_offset=0.0, signed_error=0.4),
                dict(scheme="along_k", source_k=3.0, source_offset=0.0,
                     target_k=4.0, target_offset=0.0, signed_error=0.8),
            ]
        )
        horizontal, vertical = self._drawn_lengths(arrows)
        self.assertAlmostEqual(horizontal, vertical, places=9)

    def test_arrows_are_trimmed_clear_of_their_nodes(self):
        # Still shorter than the full step, so an arrow reads as a move *between*
        # nodes rather than through them.
        arrows = pd.DataFrame(
            [
                dict(scheme="along_k", source_k=3.0, source_offset=0.0,
                     target_k=4.0, target_offset=0.0, signed_error=0.8),
            ]
        )
        (length,) = self._drawn_lengths(arrows)
        self.assertLess(length, 1.0)
        self.assertGreater(length, 0.5)

    def test_rows_without_a_value_are_skipped(self):
        arrows = pd.DataFrame(
            [
                dict(scheme="along_k", source_k=3.0, source_offset=0.0,
                     target_k=4.0, target_offset=0.0, signed_error=float("nan")),
            ]
        )
        self.assertEqual(self._drawn_lengths(arrows), [])


if __name__ == "__main__":
    unittest.main()
