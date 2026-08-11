"""Orientation of the simulated (source, target) pairs.

Every method reports ``log(score_S / score_T)`` and the ground truth is
``log(qD(T) / qD(S))``, so both only line up if the source really is the
lower-diversity corpus. The offset axis is the easy one to get backwards --
``applied_slope = baseline + offset``, so a *larger* offset is a steeper, less
diverse distribution, running opposite to k. These tests pin that direction against
the diversity measures themselves rather than against a restatement of the
convention.
"""

import unittest
from pathlib import Path

from simulation.diversity import diversity_shift, hill_diversity
from simulation.pairing import _adjacent_pairs, _order
from simulation.zipfian import zipfian_probs_for_senses
from data_processing.simulation_loading import Corpus

BASELINE = 1.2


def _corpus(k: int, offset: float, lemma_pos: str = "test_VERB") -> Corpus:
    """A Corpus stub carrying only the fields pairing looks at."""
    stem = f"k{k}_offset_{'m' if offset < 0 else 'p'}{abs(offset):.2f}"
    return Corpus(
        lemma_pos=lemma_pos,
        k=k,
        offset=round(offset, 4),
        csv_path=Path(f"{lemma_pos}/{stem}.csv"),
        meta_path=Path(f"{lemma_pos}/{stem}.meta.json"),
        data_path=Path(f"{lemma_pos}/{stem}.data"),
    )


def _probs(corpus: Corpus) -> dict[str, float]:
    """The design sense distribution a corpus would be simulated from."""
    senses = [f"s{i}" for i in range(corpus.k)]
    return zipfian_probs_for_senses(senses, BASELINE + corpus.offset)


class OffsetDirectionTestCase(unittest.TestCase):
    def test_larger_offset_is_less_diverse(self):
        # The premise the whole orientation rests on. Guards against anyone
        # "fixing" the sign back the other way.
        flat = _probs(_corpus(5, -0.5))
        steep = _probs(_corpus(5, 0.5))
        for q in (1, 2):
            self.assertGreater(hill_diversity(flat, q), hill_diversity(steep, q))


class OrderTestCase(unittest.TestCase):
    def test_source_is_less_diverse_along_slope(self):
        # _order must put the steeper (larger-offset) corpus first.
        a, b = _corpus(4, -0.3), _corpus(4, 0.2)
        source, target = _order(a, b)
        self.assertEqual(source.offset, 0.2)
        self.assertEqual(target.offset, -0.3)
        # And that ordering must make the ground-truth shift positive.
        for q in (1, 2):
            self.assertGreater(diversity_shift(_probs(source), _probs(target), q), 0.0)

    def test_source_is_less_diverse_along_k(self):
        a, b = _corpus(3, 0.0), _corpus(5, 0.0)
        source, target = _order(a, b)
        self.assertEqual(source.k, 3)
        self.assertEqual(target.k, 5)
        self.assertGreater(diversity_shift(_probs(source), _probs(target), 0), 0.0)

    def test_order_is_symmetric_in_its_arguments(self):
        a, b = _corpus(4, -0.4), _corpus(4, 0.1)
        self.assertEqual(_order(a, b), _order(b, a))


class AdjacentPairsTestCase(unittest.TestCase):
    def _offsets(self):
        return [round(-0.5 + 0.1 * i, 4) for i in range(11)]

    def test_neighbour_pairs_all_run_toward_diversity(self):
        corpora = [_corpus(4, o) for o in self._offsets()]
        pairs = list(_adjacent_pairs(corpora, key=lambda c: c.offset))
        self.assertEqual(len(pairs), 10)
        for source, target in pairs:
            self.assertGreater(source.offset, target.offset)
            self.assertGreater(
                diversity_shift(_probs(source), _probs(target), 2), 0.0
            )

    def test_every_pair_is_a_single_step(self):
        corpora = [_corpus(4, o) for o in self._offsets()]
        pairs = list(_adjacent_pairs(corpora, key=lambda c: c.offset))
        for source, target in pairs:
            self.assertAlmostEqual(source.offset - target.offset, 0.1)

    def test_pairs_tile_the_row(self):
        # The walk covers the row without gaps or overlaps: consecutive pairs share
        # an endpoint. The walk runs up the offset axis while each pair is oriented
        # back down it (steep -> flat), so the shared node is the earlier pair's
        # *source* and the later pair's target.
        corpora = [_corpus(4, o) for o in self._offsets()]
        pairs = list(_adjacent_pairs(corpora, key=lambda c: c.offset))
        for (prev_source, _), (_, next_target) in zip(pairs, pairs[1:]):
            self.assertEqual(prev_source.offset, next_target.offset)


if __name__ == "__main__":
    unittest.main()
