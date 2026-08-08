"""Loaders over the on-disk DWUG corpus layout.

``prepare_dwug_corpora`` materialises DWUG EN into the same directory shape the
simulation uses -- ``<lemma>_<pos>/{g1,g2}.csv`` with sibling ``.meta.json`` and
``.data`` files -- so the existing scorers consume the diachronic data unchanged.
This module is the DWUG counterpart of :mod:`data_processing.simulation_loading`:
it enumerates those corpora and parses a grouping stem.

DWUG corpora are kept a *separate* type from the simulated :class:`Corpus` rather
than a variant of it. A DWUG corpus is identified by its decade grouping, and has no design
distribution at all. 
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Grouping stems written by the converter. Source is the *older* corpus here (readme
# "Second Evaluation"), unlike the simulation where source is the least diverse one.
SOURCE_STEM = "g1"  # 1810-1860
TARGET_STEM = "g2"  # 1960-2010

_GROUPING_RE = re.compile(r"^g(?P<grouping>[12])$")


@dataclass(frozen=True)
class DwugCorpus:
    """One DWUG decade grouping of one lemma, on disk.

    ``data_path`` is the WiC ``.data`` sibling; unlike the simulated corpora it is
    written by the same pass that writes the CSV, so it always exists.
    """

    lemma_pos: str  # the parent directory name, e.g. "afternoon_nn"
    grouping: int  # 1 (1810-1860, source) or 2 (1960-2010, target)
    csv_path: Path
    meta_path: Path
    data_path: Path


def parse_grouping(stem: str) -> int:
    """Parse a DWUG corpus stem like ``g1`` into its grouping number."""
    match = _GROUPING_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognised DWUG grouping stem: {stem!r}")
    return int(match.group("grouping"))


def load_dwug_corpora(dwug_dir: Path) -> list[DwugCorpus]:
    """Return one :class:`DwugCorpus` per grouping CSV under ``dwug_dir``, sorted by path.

    Globs ``<lemma>_<pos>/g[12].csv`` (the layout from ``prepare_dwug_corpora``) and
    derives the sibling ``.meta.json`` / ``.data`` paths. ``Path.with_suffix`` is safe
    here -- unlike in ``load_corpora``, where the ``.`` inside an offset magnitude
    (``k3_offset_p0.00``) confuses pathlib -- because a grouping stem has no dot.
    """
    return [
        DwugCorpus(
            lemma_pos=csv_path.parent.name,
            grouping=parse_grouping(csv_path.stem),
            csv_path=csv_path,
            meta_path=csv_path.with_suffix(".meta.json"),
            data_path=csv_path.with_suffix(".data"),
        )
        for csv_path in sorted(dwug_dir.glob("*/g[12].csv"))
    ]
