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

Since ``applied_slope = baseline_slope + offset``, the steepest slope is the
**largest** offset: the offset axis runs from most diverse (most negative) to least
diverse (most positive), opposite to the k axis. Comparisons therefore run from
larger to smaller offset, and from smaller to larger k.

However, when evaluating on DWUG, S is the **older** corpus. No assumption is made
about diversity.

Comparison schemes (readme "Source and Target Corpus")
-----------------------------------------------------
* ``primary``     -- every other corpus of the lemma against the single primary
  source (steepest slope, lowest k).
* ``along_k``     -- corpora that share a slope but differ in k; S = lower k.
* ``along_slope`` -- neighbouring corpora that share k but differ in slope;
  S = steeper slope.
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

    Larger k raises expected diversity; larger *offset* lowers it. The two axes
    therefore run in opposite directions, which is the easy thing to get wrong here:
    ``applied_slope = baseline_slope + offset``, so a larger offset is a **steeper**
    Zipf law, concentrating mass on the top senses and making the corpus *less*
    diverse. The flattest, most diverse slope sits at the most negative offset.

    Corpora paired here always differ on exactly one axis, so comparing that axis
    alone is unambiguous; the assert guards misuse.
    """
    assert (a.k == b.k) != (a.offset == b.offset), (
        "pairing helper expects corpora differing on exactly one axis"
    )
    if a.k != b.k:
        return a if a.k > b.k else b
    # Same k: the more diverse corpus is the one with the smaller offset (flatter,
    # more even slope).
    return a if a.offset < b.offset else b


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
        # slope. Steepest is the *largest* offset (applied = baseline + offset), hence
        # the negation -- the two axes run in opposite directions, see _more_diverse.
        # Every other corpus is weakly more diverse on both axes, so the primary is
        # always the source.
        primary = min(group, key=lambda c: (c.k, -c.offset))
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

        # along_slope: fix k, vary offset.
        by_k: dict[int, list[Corpus]] = defaultdict(list)
        for c in group:
            by_k[c.k].append(c)
        for cs in by_k.values():
            for source, target in _adjacent_pairs(cs, key=lambda c: c.offset):
                pairs.append(CorpusPair(lemma_pos, "along_slope", source, target))

    logger.info("built %d corpus pairs across %d lemmata", len(pairs), len(by_lemma))
    return pairs


def _adjacent_pairs(corpora, key):
    """Yield ``(source, target)`` for each neighbouring pair along a sorted axis.

    Sorting by ``key`` (k or offset) and pairing along that order keeps the comparison
    to a clean one-step move on a single dimension; ``_order`` then assigns
    source/target by expected diversity. Neighbours rather than all-vs-all so the
    shift stays one known step on the dimension under study -- and every pair here
    costs a scoring run.

    Note this walks each lemma's *own* sorted list, so a lemma missing a variant (the
    simulation drops those it has too few senses or sentences for) pairs across the
    gap rather than dropping the comparison. The step is then wider than one rung for
    that lemma; the analysis reads the endpoints off the pair, so such a step lands on
    the grid as the longer move it is.

    Two corpora with the same (k, offset) -- an accidental duplicate variant in a
    lemma dir -- would be indistinguishable on both axes; such a pair is skipped with
    a warning rather than passed to ``_order`` (which cannot orient it).
    """
    ordered = sorted(corpora, key=key)
    for a, b in zip(ordered, ordered[1:]):
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
