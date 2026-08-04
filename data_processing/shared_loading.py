"""The corpus surface shared by every on-disk corpus layout.

Both evaluations put their corpora on disk as a CSV of usages with sibling
``.meta.json`` and ``.data`` files, differing only in how a lemma's corpora are
named and enumerated: the simulation by design axes (see
:mod:`data_processing.simulation_loading`), DWUG by decade grouping (see
:mod:`data_processing.dwug_loading`). This module holds the protocol that surface
satisfies, so neither loader has to depend on the other.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class CorpusHandle(Protocol):
    """The on-disk surface every scorer needs from a corpus.

    :class:`~data_processing.simulation_loading.Corpus` and
    :class:`~data_processing.dwug_loading.DwugCorpus` both satisfy this structurally,
    so :class:`~simulation.pairing.CorpusPair` can hold either kind and the
    ``score_pair_*`` functions work for both. Those functions read nothing beyond
    these four members -- notably not ``k``/``offset``, which are simulation-only.

    The members are declared as read-only ``@property`` rather than bare annotations
    on purpose: a protocol with plain attribute members is invariant and requires the
    attribute to be *settable*, which neither frozen dataclass provides. Simplifying
    these to ``lemma_pos: str`` would stop both corpus types matching.
    """

    @property
    def lemma_pos(self) -> str: ...

    @property
    def csv_path(self) -> Path: ...

    @property
    def meta_path(self) -> Path: ...

    @property
    def data_path(self) -> Path: ...
