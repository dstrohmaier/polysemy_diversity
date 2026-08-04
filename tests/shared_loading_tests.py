"""Tests for the corpus surface shared by both evaluations' on-disk layouts."""

import unittest
from pathlib import Path

from data_processing.dwug_loading import DwugCorpus
from data_processing.shared_loading import CorpusHandle
from data_processing.simulation_loading import Corpus


class CorpusHandleProtocolTestCase(unittest.TestCase):
    """Both corpus kinds satisfy the structural surface the scorers use.

    This guards the decision to keep ``DwugCorpus`` separate from ``Corpus``: if
    either grows or renames one of the four shared members, this fails here rather
    than at scoring time.
    """

    def test_simulated_corpus_satisfies_protocol(self):
        corpus = Corpus(
            lemma_pos="run_VERB", k=3, offset=0.0,
            csv_path=Path("a"), meta_path=Path("b"), data_path=Path("c"),
        )
        self.assertIsInstance(corpus, CorpusHandle)

    def test_dwug_corpus_satisfies_protocol(self):
        corpus = DwugCorpus(
            lemma_pos="bar_nn", grouping=1,
            csv_path=Path("a"), meta_path=Path("b"), data_path=Path("c"),
        )
        self.assertIsInstance(corpus, CorpusHandle)

    def test_dwug_corpus_has_no_simulation_axes(self):
        # The point of the separate dataclass: no unset k/offset travelling through
        # the diachronic path, where they have no meaning.
        corpus = DwugCorpus(
            lemma_pos="bar_nn", grouping=1,
            csv_path=Path("a"), meta_path=Path("b"), data_path=Path("c"),
        )
        self.assertFalse(hasattr(corpus, "k"))
        self.assertFalse(hasattr(corpus, "offset"))


if __name__ == "__main__":
    unittest.main()
