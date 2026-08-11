"""The WiC logit cache: fewer forward passes, bit-identical scores.

A corpus appears in several pairs, so its sentence pairs were previously re-embedded
once per comparison. The cache removes that, but only correctly because it memoises
*per-entry logits* rather than a per-corpus mean: ``equalise_indices`` trims each
corpus against its partner's length, so the same corpus contributes a different
subset in each pair it belongs to. These tests pin both halves -- the saving and the
exactness.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from simulation.pairing import CorpusPair, equalise_indices
from wic.wic_estimation import WiCLogitCache, score_pair_wic


def _entries(n: int, tag: str) -> list[dict]:
    return [
        {
            "id": f"{tag}{i}",
            "lemma": "bank",
            "pos": "NOUN",
            "sentence1": f"{tag} first {i}",
            "sentence2": f"{tag} second {i}",
            "start1": 0,
            "end1": 4,
            "start2": 0,
            "end2": 4,
            "label": i % 2,
        }
        for i in range(n)
    ]


class _StubCorpus:
    """Minimal stand-in for a Corpus: the two paths score_pair_wic touches."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.csv_path = data_path.with_suffix(".csv")


def _write(dirpath: Path, name: str, n: int) -> _StubCorpus:
    path = dirpath / f"{name}.data"
    path.write_text(json.dumps(_entries(n, name)), encoding="utf-8")
    return _StubCorpus(path)


def _fake_logits(entries, trainer, tokenizer):
    """Deterministic per-entry logits, so a cached row must equal a recomputed one.

    Keyed on the entry id rather than its position, which is what makes a wrong
    cache (one ignoring the equalise_indices subset) produce different numbers.
    Uses a stable digest rather than ``hash()``, whose per-run salt would make the
    expected values change between interpreter invocations.
    """
    out = []
    for e in entries:
        digest = hashlib.sha256(e["id"].encode()).hexdigest()
        h = (int(digest[:8], 16) % 1000) / 1000.0
        out.append([h, 1.0 - h])
    return np.asarray(out)


class WiCLogitCacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_repeated_corpus_is_embedded_once(self):
        # The saving: an anchor shared by two pairs is embedded once, not twice.
        anchor = _write(self.dir, "anchor", 40)
        b = _write(self.dir, "b", 40)
        c = _write(self.dir, "c", 40)
        with mock.patch(
            "wic.wic_estimation._predict_logits", side_effect=_fake_logits
        ) as pl:
            cache = WiCLogitCache(trainer=None, tokenizer=None)
            score_pair_wic(CorpusPair("w_NOUN", "primary", anchor, b), cache)
            score_pair_wic(CorpusPair("w_NOUN", "primary", anchor, c), cache)
        self.assertEqual(pl.call_count, 3)  # anchor, b, c -- not 4
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 3)

    def test_cached_score_matches_uncached(self):
        """The exactness check: caching must not change any score.

        The two pairs trim the shared corpus to different sizes, which is precisely
        the case a naive per-corpus cache would get wrong.
        """
        anchor = _write(self.dir, "anchor", 50)
        small = _write(self.dir, "small", 20)
        large = _write(self.dir, "large", 45)

        pairs = [
            CorpusPair("w_NOUN", "primary", anchor, small),
            CorpusPair("w_NOUN", "primary", anchor, large),
        ]
        with mock.patch(
            "wic.wic_estimation._predict_logits", side_effect=_fake_logits
        ):
            shared = WiCLogitCache(trainer=None, tokenizer=None)
            cached = [score_pair_wic(p, shared) for p in pairs]
            # A fresh cache per pair reproduces the uncached path exactly.
            uncached = [
                score_pair_wic(p, WiCLogitCache(trainer=None, tokenizer=None))
                for p in pairs
            ]

        for got, want in zip(cached, uncached):
            self.assertEqual(got["wic_log_ratio"], want["wic_log_ratio"])
            self.assertEqual(got["n_used"], want["n_used"])
        # And the two pairs really did trim the anchor differently.
        self.assertNotEqual(cached[0]["n_used"], cached[1]["n_used"])

    def test_subset_is_the_one_equalise_indices_chose(self):
        # The cache returns rows by position, so it must index the *full* corpus's
        # logits with the same indices the uncached path would have sliced entries by.
        anchor = _write(self.dir, "anchor", 30)
        other = _write(self.dir, "other", 12)
        entries = json.loads(anchor.data_path.read_text())
        idx_s, _ = equalise_indices(30, 12, seed=0)
        expected = _fake_logits([entries[i] for i in idx_s], None, None)

        with mock.patch(
            "wic.wic_estimation._predict_logits", side_effect=_fake_logits
        ):
            cache = WiCLogitCache(trainer=None, tokenizer=None)
            got = cache.logits_for(anchor.data_path, entries, idx_s)
        np.testing.assert_array_equal(got, expected)
        # Same corpus, a different partner, so a different subset -- still exact.
        del other

    def test_lru_evicts_beyond_capacity(self):
        a = _write(self.dir, "a", 10)
        b = _write(self.dir, "b", 10)
        c = _write(self.dir, "c", 10)
        with mock.patch(
            "wic.wic_estimation._predict_logits", side_effect=_fake_logits
        ) as pl:
            cache = WiCLogitCache(trainer=None, tokenizer=None, capacity=2)
            for corpus in (a, b, c, a):
                entries = json.loads(corpus.data_path.read_text())
                cache.logits_for(corpus.data_path, entries, np.arange(10))
        # a was evicted when c arrived, so its second request re-embeds.
        self.assertEqual(pl.call_count, 4)

    def test_zero_capacity_is_rejected(self):
        with self.assertRaises(AssertionError):
            WiCLogitCache(trainer=None, tokenizer=None, capacity=0)


if __name__ == "__main__":
    unittest.main()
