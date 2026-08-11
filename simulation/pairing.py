"""Pair simulated corpora into (source, target) comparisons for shift scoring.

The shift evaluation scores a *pair* of corpora: every method
reports ``log(score_S / score_T)`` for a source (S) and target (T) corpus of the
same lemma. This module turns the flat set of per-variant corpora produced by
``simulate_zipfian_corpora`` into the (S, T) pairs the scorers consume.

Orientation contract
--------------------
S is usually the **lower-expected-diversity** member of a pair, so every method's
``log(·_S / ·_T)`` increases with the target's diversity (see the readme's sign
discussion for vMF and WiC). Expected diversity rises as the Zipfian slope gets
*flatter* (smaller applied slope -> more even senses) and as k grows, so the
low-diversity anchor is the **steepest** slope and **lowest** k.

However, when evaluating on DWUG, S is the **older** corpus. No assumption is made
about diversity.

Comparison schemes (readme "Source and Target Corpus")
-----------------------------------------------------
* ``primary``     -- every other corpus of the lemma against the single primary
  source (steepest slope, lowest k).
* ``along_k``     -- corpora that share a slope but differ in k; S = lower k.
* ``along_slope`` -- corpora that share k but differ in slope; S = steeper slope.
  Emitted at each stride in :data:`SLOPE_STRIDES`: neighbours, and a wider move that
  puts a slope comparison on the same magnitude footing as a k comparison.
* ``diachronic``  -- the DWUG evaluation's single comparison per lemma: the
  1810-1860 grouping against the 1960-2010 one.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from data_processing.dwug_loading import DwugCorpus, load_dwug_corpora
from data_processing.shared_loading import CorpusHandle
from data_processing.simulation_loading import Corpus, load_sim_corpora

logger = logging.getLogger("div")

# The DWUG evaluation has one comparison per lemma, so every pair carries this single
# scheme tag. The comparative analysis groups correlations by scheme, which then
# yields one row per (method, ground-truth order) across all lemmata.
DIACHRONIC_SCHEME = "diachronic"

# How many offset steps an ``along_slope`` comparison spans. One step is the natural
# neighbour comparison; the wider stride exists because the two axes are not on
# comparable footings -- a single offset step moves ground-truth diversity roughly 2-6x
# less than a single k step (Shannon 6x, Simpson 3x, evenness 2x on the simulated
# grid), so one-step slope comparisons cluster near zero and are hard to read next to
# the k comparisons drawn beside them. A 3-step slope move lands in the same magnitude
# range as a 1-step k move, which is what makes the two families comparable on one
# colour scale in the grid figures.
#
# Both strides are emitted under the same ``along_slope`` scheme: they are the same
# kind of comparison, differing only in how far they reach, and the analysis
# distinguishes them by the offset distance between the pair's endpoints.
SLOPE_STRIDES = (1, 3)

# Spacing and low end of the simulation's offset ladder, matching ``simulate_data.py``'s
# ``--offset-step`` / ``--offset-min`` defaults. Pairing reads finished corpora off disk
# and never sees that config, so the values are restated here; they only number the
# rungs a strided walk partitions on, and a run with a different grid still pairs
# correctly as long as these match it.
#
# The partition counts rungs from OFFSET_MIN rather than from zero, so the first wide
# arrow starts at the steepest slope and the row is covered from its edge inward.
OFFSET_STEP = 0.1
OFFSET_MIN = -0.5


@dataclass(frozen=True)
class CorpusPair:
    """A (source, target) corpus comparison of one lemma.

    For the simulation schemes ``source`` is the lower-expected-diversity corpus
    (steeper slope / lower k) and ``target`` the higher; for ``diachronic`` it is
    simply the older corpus. ``scheme`` records which comparison produced the pair so
    the analysis can group correlations by comparison type.

    The members are typed as :class:`CorpusHandle` -- the structural surface the
    scorers actually use -- so a pair can hold either a simulated ``Corpus`` or a
    ``DwugCorpus``.
    """

    lemma_pos: str
    scheme: str  # "primary" | "along_k" | "along_slope" | "diachronic"
    source: CorpusHandle
    target: CorpusHandle


def _more_diverse(a: Corpus, b: Corpus) -> Corpus:
    """Return whichever corpus has the higher *expected* diversity.

    Larger offset (flatter slope, i.e. applied slope shifted down) and larger k both
    raise expected diversity. Corpora paired here always differ on exactly one axis,
    so comparing that axis alone is unambiguous; the assert guards misuse.
    """
    assert (a.k == b.k) != (a.offset == b.offset), (
        "pairing helper expects corpora differing on exactly one axis"
    )
    if a.k != b.k:
        return a if a.k > b.k else b
    # Same k: the more diverse corpus is the one with the larger offset (flatter
    # slope). offset = applied_slope - baseline_slope, so larger offset => flatter.
    return a if a.offset > b.offset else b


def _order(a: Corpus, b: Corpus) -> tuple[Corpus, Corpus]:
    """Return ``(source, target)`` with source = lower expected diversity."""
    target = _more_diverse(a, b)
    source = b if target is a else a
    return source, target


def build_simulated_pairs(sim_dir: Path) -> list[CorpusPair]:
    """Build all (source, target) pairs for the corpora under ``sim_dir``.

    Groups by lemma (the ``lemma_pos`` directory), then emits the three comparison
    schemes. A pair is included once per scheme it belongs to (a slope-neighbour of
    the primary appears in both ``primary`` and ``along_slope``); the scheme tag
    distinguishes them so the analysis groups them separately rather than
    de-duplicating.
    """
    corpora = load_sim_corpora(sim_dir)
    by_lemma: dict[str, list[Corpus]] = defaultdict(list)
    for c in corpora:
        by_lemma[c.lemma_pos].append(c)

    pairs: list[CorpusPair] = []
    for lemma_pos, group in sorted(by_lemma.items()):
        if len(group) < 2:
            logger.info("%s: <2 corpora, no pairs", lemma_pos)
            continue

        # The primary is the lemma's low-diversity anchor: lowest k, then steepest
        # slope (smallest offset). Every other corpus is weakly more diverse on both
        # axes, so the primary is always the source.
        primary = min(group, key=lambda c: (c.k, c.offset))
        for c in group:
            if c is primary:
                continue
            if c.k == primary.k and c.offset == primary.offset:
                # A duplicate of the anchor variant: same (k, offset), so no
                # diversity shift to measure. Skip rather than emit a 0-shift pair.
                logger.warning(
                    "%s: duplicate of primary variant (k=%d, offset=%.4f) in %s; skipping",
                    lemma_pos, c.k, c.offset, c.csv_path.name,
                )
                continue
            pairs.append(CorpusPair(lemma_pos, "primary", primary, c))

        # along_k: fix offset, vary k.
        by_offset: dict[float, list[Corpus]] = defaultdict(list)
        for c in group:
            by_offset[c.offset].append(c)
        for cs in by_offset.values():
            for source, target in _adjacent_pairs(cs, key=lambda c: c.k):
                pairs.append(CorpusPair(lemma_pos, "along_k", source, target))

        # along_slope: fix k, vary offset -- at each stride in SLOPE_STRIDES.
        by_k: dict[int, list[Corpus]] = defaultdict(list)
        for c in group:
            by_k[c.k].append(c)
        for cs in by_k.values():
            for stride in SLOPE_STRIDES:
                for source, target in _adjacent_pairs(
                    cs,
                    key=lambda c: c.offset,
                    stride=stride,
                    anchor_index=_offset_index,
                ):
                    pairs.append(CorpusPair(lemma_pos, "along_slope", source, target))

    logger.info("built %d corpus pairs across %d lemmata", len(pairs), len(by_lemma))
    return pairs


def _offset_index(corpus: Corpus) -> int:
    """Position of a corpus's offset on the simulation's offset ladder.

    The grid is a regular step apart (``_offset_grid`` in
    :mod:`simulation.corpus_simulation`), so the index follows from the value and
    needs no reference to the run's config. Counting from :data:`OFFSET_MIN` puts rung
    0 at the steepest slope; rounding absorbs float drift, keeping a value like
    ``-0.30000000000000004`` on its intended rung.

    This is the anchor a strided ``along_slope`` walk partitions on, so that every
    lemma cuts the offset axis at the same offsets regardless of which variants it
    happens to have on disk.
    """
    return round((corpus.offset - OFFSET_MIN) / OFFSET_STEP)


def _adjacent_pairs(corpora, key, stride: int = 1, anchor_index=None):
    """Yield ``(source, target)`` for each pair ``stride`` apart along a sorted axis.

    Sorting by ``key`` (k or offset) and pairing along that order keeps the comparison
    to a clean move on one dimension; ``_order`` then assigns source/target by
    expected diversity. Strided rather than all-vs-all so the shift stays a known
    number of steps on the dimension under study.

    ``stride=1`` is the neighbour case and slides one position at a time, so every
    adjacent pair is covered. A wider stride instead steps **end to end**: the next
    pair starts where the previous one finished (0->3, 3->6, ...), rather than sliding
    (0->3, 1->4, ...). Consecutive multi-step comparisons would otherwise re-use the
    same corpora and, drawn as arrows on the design grid, would lie on top of one
    another. A partitioning of the axis says as much with a fraction of the pairs --
    and every pair here costs a scoring run.

    The partition is anchored to ``anchor_index`` -- a position *in the full axis*,
    not in this lemma's own list -- so that every lemma cuts the axis at the same
    places. Anchoring to list position instead would shift a lemma's cut points
    whenever it is missing a variant (the simulation drops variants a lemma has too
    few senses or sentences for), scattering the wide arrows across every offset and
    filling the row rather than partitioning it.

    A wider stride is what the slope axis needs: one offset step moves diversity
    several times less than one k step (see :data:`SLOPE_STRIDES`), so single-step
    slope comparisons sit near zero and are hard to read against the k comparisons
    they are drawn beside.

    Two corpora with the same (k, offset) -- an accidental duplicate variant in a
    lemma dir -- would be indistinguishable on both axes; such a pair is skipped with
    a warning rather than passed to ``_order`` (which cannot orient it).
    """
    ordered = sorted(corpora, key=key)
    if stride > 1 and anchor_index is not None:
        # Keep only the corpora sitting on a partition boundary, then walk those.
        # A boundary this lemma lacks simply drops the pair rather than sliding the
        # cut onto a neighbouring offset.
        ordered = [c for c in ordered if anchor_index(c) % stride == 0]
        for a, b in zip(ordered, ordered[1:]):
            if anchor_index(b) - anchor_index(a) != stride:
                continue
            if a.k == b.k and a.offset == b.offset:
                logger.warning(
                    "%s: duplicate variant (k=%d, offset=%.4f) for %s and %s; skipping pair",
                    a.lemma_pos, a.k, a.offset, a.csv_path.name, b.csv_path.name,
                )
                continue
            yield _order(a, b)
        return
    for i in range(0, len(ordered) - stride, stride):
        a, b = ordered[i], ordered[i + stride]
        if a.k == b.k and a.offset == b.offset:
            logger.warning(
                "%s: duplicate variant (k=%d, offset=%.4f) for %s and %s; skipping pair",
                a.lemma_pos, a.k, a.offset, a.csv_path.name, b.csv_path.name,
            )
            continue
        yield _order(a, b)


def _keep_indices(length: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of ``n`` items to keep from a length-``length`` sequence.

    Returns all indices when ``length <= n`` (nothing to drop); otherwise a
    without-replacement sample, sorted so the kept items stay in their original order.
    """
    if length <= n:
        return np.arange(length)
    return np.sort(rng.choice(length, size=n, replace=False))


def equalise_indices(len_a: int, len_b: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Row indices that trim the longer of two sequences to the shorter's length.

    Returns ``(idx_a, idx_b)`` -- the indices to keep from each side so both end up
    at ``min(len_a, len_b)`` rows. The caller applies them to whatever it holds
    (a vector array for vMF, a list of pair dicts for WiC).

    The vMF resultant-length floor and the WiC pair count both depend on n. A
    log-ratio between two differently-sized corpora conflates a size difference with
    a diversity difference; trimming to equal n removes that confound (readme
    "Source and Target Corpus").
    """
    n = min(len_a, len_b)
    rng = np.random.default_rng(seed)
    return _keep_indices(len_a, n, rng), _keep_indices(len_b, n, rng)


def build_dwug_pairs(dwug_dir: Path) -> list[CorpusPair]:
    """Build the one (source, target) pair per lemma for the diachronic evaluation.

    Source is grouping 1 (1810-1860), target is grouping 2 (1960-2010): here the source
    is the *older* corpus rather than the less diverse one (readme "Second
    Evaluation").

    A lemma missing either grouping is skipped with a warning rather than aborting the
    run.
    """
    corpora = load_dwug_corpora(dwug_dir)
    by_lemma: dict[str, dict[int, DwugCorpus]] = defaultdict(dict)
    for c in corpora:
        by_lemma[c.lemma_pos][c.grouping] = c

    pairs: list[CorpusPair] = []
    for lemma_pos, groupings in sorted(by_lemma.items()):
        source, target = groupings.get(1), groupings.get(2)
        if source is None or target is None:
            logger.warning(
                "%s: missing grouping(s) %s; skipping",
                lemma_pos,
                sorted({1, 2} - set(groupings)),
            )
            continue
        pairs.append(CorpusPair(lemma_pos, DIACHRONIC_SCHEME, source, target))

    logger.info("built %d diachronic pairs across %d lemmata", len(pairs), len(by_lemma))
    return pairs


# How a scoring driver turns a dataset directory into the pairs to score. The drivers
# take one of these rather than sniffing the directory layout: a half-populated or
# misnamed directory would make sniffing silently pick the wrong branch, whereas an
# explicit builder puts the choice at the call site (score_data.py's --dataset).
PairBuilder = Callable[[Path], list[CorpusPair]]
