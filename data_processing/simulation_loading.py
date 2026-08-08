"""Shared loaders over the on-disk simulated-corpus layout.

The simulation writes one corpus per ``<lemma>_<pos>/k*_offset_*.csv`` variant, each
with sibling ``.meta.json`` and (after WiC conversion) ``.data`` files. Several
unrelated stages -- WiC conversion, vMF/WiC/cosine scoring, and the analysis modes --
need to enumerate these corpora and parse a variant stem, so those pieces live here
rather than in any one consumer.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Matches a variant stem such as "k3_offset_m0.20" or "k5_offset_p0.00", produced by
# simulate_zipfian_corpora as f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}".
_VARIANT_RE = re.compile(r"^k(?P<k>\d+)_offset_(?P<sign>[mp])(?P<mag>[\d.]+)$")


@dataclass(frozen=True)
class Corpus:
    """One simulated corpus: a single (lemma, pos, k, offset) variant on disk.

    ``data_path`` is the WiC ``.data`` sibling; it only exists once this corpus has
    been through ``convert_simulated_corpora``.
    """

    lemma_pos: str  # the parent directory name, e.g. "<lemma>_<pos>"
    k: int
    offset: float
    csv_path: Path
    meta_path: Path
    data_path: Path


def parse_variant(stem: str) -> tuple[int, float]:
    """Parse a variant stem like ``k3_offset_m0.20`` into ``(k, offset)``."""
    match = _VARIANT_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognised variant stem: {stem!r}")
    offset = float(match.group("mag"))
    if match.group("sign") == "m":
        offset = -offset
    return int(match.group("k")), offset


def load_sim_corpora(sim_dir: Path) -> list[Corpus]:
    """Return one :class:`Corpus` per simulated CSV under ``sim_dir``, sorted by path.

    Globs ``<lemma>_<pos>/k*_offset_*.csv`` (the layout from
    ``simulate_zipfian_corpora``) and derives the sibling ``.meta.json`` / ``.data``
    paths. The trailing suffix is swapped explicitly rather than via
    ``Path.with_suffix`` because the ``.`` in the offset magnitude (e.g.
    ``k3_offset_p0.00``) confuses pathlib's suffix handling.
    """
    corpora = []
    for csv_path in sorted(sim_dir.glob("*/k*_offset_*.csv")):
        base = csv_path.name[: -len(".csv")]
        k, offset = parse_variant(base)
        corpora.append(
            Corpus(
                lemma_pos=csv_path.parent.name,
                k=k,
                offset=offset,
                csv_path=csv_path,
                meta_path=csv_path.parent / (base + ".meta.json"),
                data_path=csv_path.parent / (base + ".data"),
            )
        )
    return corpora
