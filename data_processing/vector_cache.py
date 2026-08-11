"""Memoise a corpus's extracted vectors across the pairs that share it.

Every scoring method compares corpus *pairs*, but a corpus belongs to several pairs:
the simulation's three comparison schemes put each variant in a ``primary``
comparison, an ``along_k`` one, and an ``along_slope`` one, so a corpus is scored
~4.3 times on average and the lemma's primary anchor once per sibling variant.
Extraction is a pure function of (corpus CSV, model) -- ``equalise_indices`` trims the
vectors *after* they are produced -- so those repeats are redundant work, and on the
largest dataset they account for 7278 extractions of 1634 distinct corpora.

The cache is per-run and in-memory. vMF and cosine are separate processes (see the
justfile's ``score-vmf-all`` / ``score-cosine-all``), so they do not share it.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from data_processing.vector_extraction import WordVectorExtractor

logger = logging.getLogger("div")

# Pairs arrive grouped by lemma and no lemma has more than ~33 corpora, so a capacity
# this size holds a lemma's whole working set and reaches the hit rate of an unbounded
# cache (measured: 1634 extractions either way, versus 7278 with no cache) for ~27 MB
# instead of ~880 MB. Bounded so an unlucky pair order degrades speed, not memory.
DEFAULT_CAPACITY = 64


class CorpusVectorCache:
    """Corpus CSV -> that corpus's target vectors, memoised with an LRU.

    Wraps the read-and-extract step both the vMF and cosine scorers used to repeat
    verbatim, so they share one implementation as well as one cache.
    """

    def __init__(
        self,
        extractor: WordVectorExtractor,
        capacity: int = DEFAULT_CAPACITY,
        batch_size: int | None = None,
    ):
        assert capacity > 0, "capacity must be positive; use the extractor directly"
        self.extractor = extractor
        self.capacity = capacity
        self.batch_size = batch_size
        self._entries: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def vectors(self, csv_path: Path) -> np.ndarray:
        """Return the (L2-normalised) target vectors for ``csv_path``.

        Occurrences are located by their stored gold span rather than re-found with a
        parser, so every scorer sees exactly the same set of occurrences.
        """
        key = Path(csv_path).resolve()
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return cached

        self.misses += 1
        contexts = pd.read_csv(key).to_dict("records")
        extra = {} if self.batch_size is None else {"batch_size": self.batch_size}
        vectors = self.extractor.get_word_vectors_from_spans(contexts, **extra)

        self._entries[key] = vectors
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return vectors

    def log_summary(self) -> None:
        """Log the hit rate, so a run shows whether the cache actually paid off."""
        total = self.hits + self.misses
        if not total:
            return
        logger.info(
            "vector cache: %d extractions for %d requests (%.1f%% hits, capacity %d)",
            self.misses,
            total,
            100.0 * self.hits / total,
            self.capacity,
        )
