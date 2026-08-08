"""Tests for the per-corpus vector cache.

The extractor is faked throughout: what matters here is how often it is called and
that the array handed back is the one the scorers can consume, neither of which needs
a transformer.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_processing.vector_cache import CorpusVectorCache
from simulation.pairing import equalise_indices


class FakeExtractor:
    """Counts extractions and returns a distinct array per corpus."""

    def __init__(self, n_rows: int = 4, hidden: int = 8):
        self.calls: list[int] = []
        self.n_rows = n_rows
        self.hidden = hidden

    def get_word_vectors_from_spans(self, contexts, **kwargs):
        self.calls.append(len(contexts))
        # Value keyed on the corpus content so a wrong cache hit is detectable.
        seed = abs(hash(contexts[0]["sentence"])) % (2**31)
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(self.n_rows, self.hidden)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _write_corpus(directory: Path, name: str, sentence: str) -> Path:
    """A minimal corpus CSV with the columns the extractor reads."""
    csv_path = directory / f"{name}.csv"
    rows = ["lemma,pos,sense,sentence,start,end"]
    rows += [f"bank,NOUN,s1,{sentence} {i},0,4" for i in range(4)]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


class CorpusVectorCacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.extractor = FakeExtractor()

    def test_second_request_is_served_from_cache(self):
        """The repeat that motivates the cache: same corpus, one extraction."""
        path = _write_corpus(self.dir, "a", "the bank")
        cache = CorpusVectorCache(self.extractor)

        first = cache.vectors(path)
        second = cache.vectors(path)

        self.assertEqual(len(self.extractor.calls), 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_distinct_corpora_are_extracted_separately(self):
        a = _write_corpus(self.dir, "a", "the bank")
        b = _write_corpus(self.dir, "b", "a river")
        cache = CorpusVectorCache(self.extractor)

        va, vb = cache.vectors(a), cache.vectors(b)

        self.assertEqual(len(self.extractor.calls), 2)
        self.assertFalse(np.allclose(va, vb), "distinct corpora share vectors")

    def test_equivalent_paths_hit_the_same_entry(self):
        """Keys are resolved, so ``dir/a.csv`` and ``dir/./a.csv`` are one corpus."""
        path = _write_corpus(self.dir, "a", "the bank")
        cache = CorpusVectorCache(self.extractor)

        cache.vectors(path)
        cache.vectors(path.parent / "." / path.name)

        self.assertEqual(len(self.extractor.calls), 1)

    def test_lru_evicts_beyond_capacity(self):
        paths = [_write_corpus(self.dir, n, f"sentence {n}") for n in "abc"]
        cache = CorpusVectorCache(self.extractor, capacity=2)

        cache.vectors(paths[0])
        cache.vectors(paths[1])
        cache.vectors(paths[2])  # evicts paths[0], the least recently used
        self.assertEqual(len(self.extractor.calls), 3)

        cache.vectors(paths[0])  # so this must recompute
        self.assertEqual(len(self.extractor.calls), 4)

        cache.vectors(paths[2])  # still resident
        self.assertEqual(len(self.extractor.calls), 4)

    def test_recent_use_postpones_eviction(self):
        """A re-read refreshes an entry, which is what keeps a lemma's anchor warm."""
        paths = [_write_corpus(self.dir, n, f"sentence {n}") for n in "abc"]
        cache = CorpusVectorCache(self.extractor, capacity=2)

        cache.vectors(paths[0])
        cache.vectors(paths[1])
        cache.vectors(paths[0])  # refresh a, so b is now least recently used
        cache.vectors(paths[2])  # evicts b, not a
        self.assertEqual(len(self.extractor.calls), 3)

        cache.vectors(paths[0])
        self.assertEqual(len(self.extractor.calls), 3, "a should still be cached")

    def test_cached_array_satisfies_the_scorer_contract(self):
        """float32, 2-D, and fancy-indexable by equalise_indices' output."""
        path = _write_corpus(self.dir, "a", "the bank")
        cache = CorpusVectorCache(self.extractor)

        vectors = cache.vectors(path)
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(vectors.ndim, 2)

        idx_a, _ = equalise_indices(len(vectors), 2, seed=0)
        self.assertEqual(len(vectors[idx_a]), 2)

    def test_zero_capacity_is_rejected(self):
        with self.assertRaises(AssertionError):
            CorpusVectorCache(self.extractor, capacity=0)


if __name__ == "__main__":
    unittest.main()
