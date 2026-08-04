"""Tests for the DWUG EN diachronic evaluation (readme "Second Evaluation").

The DWUG path reuses the simulation's scorers and analysis wholesale; what is new is
the conversion into the shared on-disk layout, the single-pair-per-lemma enumeration,
and the ground-truth lookup over DWUG sidecars. These tests cover those seams plus
the two traps the conversion has to avoid: DWUG's unescaped quotes and its
fine-grained PoS tags.
"""

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.scored.stats import pair_ground_truth
from cosine.cosine_estimation import get_corpora_cosine_pairs
from data_processing.dwug_conversion import (
    MIN_GROUPING_USAGES,
    NOISE_CLUSTER,
    cluster_probs,
    dwug_lemma_frame,
    prepare_dwug_corpora,
    read_dwug_uses,
    write_dwug_lemma,
)
from data_processing.dwug_loading import (
    CorpusHandle,
    DwugCorpus,
    iter_dwug_corpora,
    parse_grouping,
)
from data_processing.simulation_loading import Corpus
from simulation.pairing import (
    CorpusPair,
    dwug_pairs,
    enumerate_dwug_pairs,
    simulated_pairs,
)
from vmf.vmf_estimation import get_corpora_vmf_pairs
from wic.wic_estimation import get_corpora_wic_pairs

# A context carrying an unescaped double quote, as DWUG's uses.csv does for quoted
# speech. The target token spans are char offsets into these strings.
_QUOTED_CONTEXT = 'He said "the bar was crowded" that night.'


def _dwug_corpus(lemma_pos: str = "bar_nn", grouping: int = 1) -> DwugCorpus:
    return DwugCorpus(
        lemma_pos=lemma_pos,
        grouping=grouping,
        csv_path=Path(f"{lemma_pos}/g{grouping}.csv"),
        meta_path=Path(f"{lemma_pos}/g{grouping}.meta.json"),
        data_path=Path(f"{lemma_pos}/g{grouping}.data"),
    )


def _uses_frame() -> pd.DataFrame:
    """A miniature uses.csv: two groupings, varying CLAWS tags, one noise row."""
    rows = [
        # (identifier, grouping, context, span, claws_pos)
        ("a1", 1, _QUOTED_CONTEXT, "13:16", "nn1"),
        ("a2", 1, "The bar closed early.", "4:7", "nnt1"),
        ("a3", 1, "A chocolate bar melted.", "12:15", "nn1"),
        ("b1", 2, "She raised the bar again.", "15:18", "nn1"),
        ("b2", 2, "The bar served drinks.", "4:7", "nnt1"),
        ("b3", 2, "Another bar entirely.", "8:11", "nn2"),
    ]
    return pd.DataFrame(
        [
            {
                "lemma": "bar_nn",
                "pos": claws,
                "grouping": g,
                "identifier": ident,
                "context": ctx,
                "indexes_target_token": span,
            }
            for ident, g, ctx, span, claws in rows
        ]
    )


def _clusters_frame() -> pd.DataFrame:
    """Clusters for ``_uses_frame``: g1 = {0,0,1}, g2 = {0,1,1}, plus one noise row."""
    return pd.DataFrame(
        [
            {"identifier": "a1", "cluster": 0},
            {"identifier": "a2", "cluster": 0},
            {"identifier": "a3", "cluster": NOISE_CLUSTER},
            {"identifier": "b1", "cluster": 0},
            {"identifier": "b2", "cluster": 1},
            {"identifier": "b3", "cluster": 1},
        ]
    )


class DwugLoadingTestCase(unittest.TestCase):
    def test_parse_grouping(self):
        self.assertEqual(parse_grouping("g1"), 1)
        self.assertEqual(parse_grouping("g2"), 2)

    def test_parse_grouping_rejects_other_stems(self):
        # A simulated variant stem must not be mistaken for a DWUG grouping.
        for stem in ("k3_offset_p0.00", "g3", "g", "grouping1"):
            with self.assertRaises(ValueError):
                parse_grouping(stem)

    def test_iter_dwug_corpora_finds_both_groupings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            word_dir = root / "bar_nn"
            word_dir.mkdir()
            for stem in ("g1", "g2"):
                (word_dir / f"{stem}.csv").write_text("lemma\n", encoding="utf-8")

            corpora = list(iter_dwug_corpora(root))
            self.assertEqual([c.grouping for c in corpora], [1, 2])
            self.assertTrue(all(c.lemma_pos == "bar_nn" for c in corpora))
            # Siblings are derived, not globbed, so they resolve even before writing.
            self.assertEqual(corpora[0].meta_path.name, "g1.meta.json")
            self.assertEqual(corpora[0].data_path.name, "g1.data")

    def test_iter_dwug_corpora_ignores_simulated_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_VERB").mkdir()
            (root / "run_VERB" / "k3_offset_p0.00.csv").write_text("x\n")
            self.assertEqual(list(iter_dwug_corpora(root)), [])


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
        self.assertIsInstance(_dwug_corpus(), CorpusHandle)

    def test_dwug_corpus_has_no_simulation_axes(self):
        # The point of the separate dataclass: no unset k/offset travelling through
        # the diachronic path, where they have no meaning.
        corpus = _dwug_corpus()
        self.assertFalse(hasattr(corpus, "k"))
        self.assertFalse(hasattr(corpus, "offset"))


class DwugReadingTestCase(unittest.TestCase):
    def test_quote_none_reading_preserves_unescaped_quotes(self):
        # DWUG embeds bare double quotes in `context`; the default QUOTE_MINIMAL
        # dialect mis-parses those rows, so the reader must disable quoting.
        with tempfile.TemporaryDirectory() as tmp:
            lemma_dir = Path(tmp)
            frame = _uses_frame()
            frame.to_csv(
                lemma_dir / "uses.csv", sep="\t", index=False, quoting=csv.QUOTE_NONE
            )
            out = read_dwug_uses(lemma_dir)
            self.assertEqual(len(out), len(frame))
            self.assertEqual(out.iloc[0]["context"], _QUOTED_CONTEXT)


class DwugFrameTestCase(unittest.TestCase):
    def setUp(self):
        self.frame = dwug_lemma_frame(_uses_frame(), _clusters_frame(), "bar_nn")

    def test_noise_cluster_dropped(self):
        # The one -1 row is gone, so g1 keeps 2 of its 3 usages.
        self.assertEqual(len(self.frame), 5)
        self.assertNotIn(str(NOISE_CLUSTER), set(self.frame["sense"]))
        self.assertEqual(int((self.frame["grouping"] == 1).sum()), 2)

    def test_pos_is_coarse_suffix_not_claws_tag(self):
        # DWUG's own `pos` varies within every lemma (nn1/nnt1/nn2 here), and
        # generate_comparison_pairs asserts paired rows agree on it -- so the coarse
        # suffix of lemma_pos is used instead.
        self.assertEqual(set(self.frame["pos"]), {"nn"})

    def test_lemma_is_lemma_pos(self):
        # A single constant value means generate_comparison_pairs sees one group.
        self.assertEqual(set(self.frame["lemma"]), {"bar_nn"})

    def test_target_span_selects_target_token(self):
        for _, row in self.frame.iterrows():
            self.assertEqual(row["sentence"][row["start"] : row["end"]], "bar")

    def test_unclustered_usage_raises(self):
        # A clustering that does not cover every usage would silently shrink the
        # ground truth, so it must fail loudly.
        partial = _clusters_frame().iloc[:-1]
        with self.assertRaises(AssertionError):
            dwug_lemma_frame(_uses_frame(), partial, "bar_nn")

    def test_cluster_probs_sum_to_one(self):
        for grouping in (1, 2):
            probs = cluster_probs(self.frame, grouping)
            self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_cluster_probs_reflect_full_grouping(self):
        # g1 keeps two usages, both cluster 0; g2 has one of cluster 0 and two of 1.
        self.assertEqual(cluster_probs(self.frame, 1), {"0": 1.0})
        self.assertEqual(
            cluster_probs(self.frame, 2), {"1": 2 / 3, "0": 1 / 3}
        )


class DwugWriteTestCase(unittest.TestCase):
    def setUp(self):
        self.frame = dwug_lemma_frame(_uses_frame(), _clusters_frame(), "bar_nn")
        self.raw_sizes = {1: 3, 2: 3}

    def test_writes_csv_meta_and_data_trio(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertTrue(
                write_dwug_lemma(self.frame, "bar_nn", out, self.raw_sizes)
            )
            for stem in ("g1", "g2"):
                for suffix in (".csv", ".meta.json", ".data"):
                    self.assertTrue((out / "bar_nn" / f"{stem}{suffix}").exists())

    def test_meta_carries_sense_probs_and_noise_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_dwug_lemma(self.frame, "bar_nn", out, self.raw_sizes)
            meta = json.loads((out / "bar_nn" / "g1.meta.json").read_text())
            # sense_probs is the key the ground-truth lookup reads.
            self.assertEqual(meta["sense_probs"], {"0": 1.0})
            self.assertEqual(meta["n_usages"], 2)
            self.assertEqual(meta["n_usages_raw"], 3)
            self.assertEqual(meta["n_noise_dropped"], 1)
            self.assertEqual(meta["role"], "source")
            # The simulation-only keys stay absent so a simulation-mode analysis
            # fails loudly rather than reporting nonsense.
            self.assertNotIn("baseline_slope", meta)

    def test_csv_roundtrips_embedded_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_dwug_lemma(self.frame, "bar_nn", out, self.raw_sizes)
            written = pd.read_csv(out / "bar_nn" / "g1.csv")
            self.assertIn(_QUOTED_CONTEXT, set(written["sentence"]))
            for _, row in written.iterrows():
                self.assertEqual(row["sentence"][row["start"] : row["end"]], "bar")

    def test_data_labels_match_cluster_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_dwug_lemma(self.frame, "bar_nn", out, self.raw_sizes)
            pairs = json.loads((out / "bar_nn" / "g1.data").read_text())
            # Both g1 usages share cluster 0, so every pair is same-sense.
            self.assertTrue(pairs)
            self.assertTrue(all(p["label"] == 1 for p in pairs))

    def test_undersized_grouping_skips_whole_lemma(self):
        # A pair needs both sides, so one thin grouping drops the lemma entirely.
        thin = self.frame[self.frame["grouping"] == 2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertFalse(write_dwug_lemma(thin, "bar_nn", out, self.raw_sizes))
            self.assertEqual(list(out.glob("bar_nn/*")), [])

    def test_min_grouping_usages_allows_pairing(self):
        # The floor exists because loo_centroid_distance needs >= 2 vectors.
        self.assertGreaterEqual(MIN_GROUPING_USAGES, 2)


class DwugPairingTestCase(unittest.TestCase):
    def test_one_pair_per_lemma_source_is_older_grouping(self):
        corpora = [
            _dwug_corpus("bar_nn", 1), _dwug_corpus("bar_nn", 2),
            _dwug_corpus("pin_vb", 2), _dwug_corpus("pin_vb", 1),
        ]
        pairs = enumerate_dwug_pairs(corpora)
        self.assertEqual(len(pairs), 2)
        for pair in pairs:
            # Source is the 1810-1860 corpus regardless of input order, and unlike
            # the simulation no diversity ordering is applied.
            self.assertEqual(pair.source.grouping, 1)
            self.assertEqual(pair.target.grouping, 2)
            self.assertEqual(pair.scheme, "diachronic")

    def test_lemma_missing_a_grouping_is_skipped(self):
        pairs = enumerate_dwug_pairs([_dwug_corpus("lone_nn", 1)])
        self.assertEqual(pairs, [])

    def test_corpus_pair_accepts_dwug_corpus(self):
        pair = CorpusPair(
            "bar_nn", "diachronic", _dwug_corpus("bar_nn", 1), _dwug_corpus("bar_nn", 2)
        )
        self.assertEqual(pair.source.csv_path.stem, "g1")


class DwugGroundTruthTestCase(unittest.TestCase):
    """The ground-truth lookup generalisation: same key shape, swapped iterator."""

    def _prepared(self, tmp: str) -> Path:
        """Write the miniature lemma into a DWUG-source layout and convert it."""
        root = Path(tmp)
        lemma_dir = root / "data" / "bar_nn"
        lemma_dir.mkdir(parents=True)
        _uses_frame().to_csv(
            lemma_dir / "uses.csv", sep="\t", index=False, quoting=csv.QUOTE_NONE
        )
        clusters_dir = root / "clusters" / "opt"
        clusters_dir.mkdir(parents=True)
        _clusters_frame().to_csv(clusters_dir / "bar_nn.csv", sep="\t", index=False)

        out = root / "corpora"
        prepare_dwug_corpora(root, out)
        return out

    def test_pair_ground_truth_with_dwug_iterator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(tmp)
            scores = pd.DataFrame(
                [{
                    "lemma_pos": "bar_nn", "scheme": "diachronic",
                    "source_variant": "g1", "target_variant": "g2",
                    "cosine_log_ratio": 0.5, "n_used": 2,
                }]
            )
            gt = pair_ground_truth(scores, out, iter_dwug_corpora)
            # g1 has one sense, g2 has two, so richness rises: log(2/1).
            self.assertAlmostEqual(gt["gt_shift_q0"].iloc[0], 0.6931471805599453)
            self.assertGreater(gt["gt_shift_q1"].iloc[0], 0)
            self.assertFalse(gt[["gt_shift_q0", "gt_shift_q1", "gt_shift_q2"]].isna().any().any())

    def test_identical_distributions_give_zero_shift(self):
        # The four DWUG lemmata with one sense in both groupings must score exactly
        # 0 at every order -- a legitimate tie, not a missing value.
        with tempfile.TemporaryDirectory() as tmp:
            out = self._prepared(tmp)
            scores = pd.DataFrame(
                [{
                    "lemma_pos": "bar_nn", "scheme": "diachronic",
                    "source_variant": "g1", "target_variant": "g1",
                    "cosine_log_ratio": 0.0, "n_used": 2,
                }]
            )
            gt = pair_ground_truth(scores, out, iter_dwug_corpora)
            for q in (0, 1, 2):
                self.assertEqual(gt[f"gt_shift_q{q}"].iloc[0], 0.0)

    def test_preparation_summary_reports_per_lemma(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lemma_dir = root / "data" / "bar_nn"
            lemma_dir.mkdir(parents=True)
            _uses_frame().to_csv(
                lemma_dir / "uses.csv", sep="\t", index=False, quoting=csv.QUOTE_NONE
            )
            clusters_dir = root / "clusters" / "opt"
            clusters_dir.mkdir(parents=True)
            _clusters_frame().to_csv(clusters_dir / "bar_nn.csv", sep="\t", index=False)

            summary = prepare_dwug_corpora(root, root / "corpora")
            row = summary.iloc[0]
            self.assertEqual(row["lemma_pos"], "bar_nn")
            self.assertTrue(row["written"])
            self.assertEqual(row["n_noise"], 1)
            self.assertEqual(row["n_g1"], 2)
            self.assertEqual(row["n_equalised"], 2)


class EnumeratorInjectionTestCase(unittest.TestCase):
    """The drivers' one simulation-specific line is now injectable.

    The defaults must stay on the simulated enumerator so the first evaluation is
    untouched by the DWUG work.
    """

    def test_drivers_default_to_simulated_pairs(self):
        for driver in (
            get_corpora_cosine_pairs, get_corpora_vmf_pairs, get_corpora_wic_pairs
        ):
            default = inspect.signature(driver).parameters[
                "enumerate_corpus_pairs"
            ].default
            self.assertIs(default, simulated_pairs, driver.__name__)

    def test_dwug_enumerator_reads_the_dwug_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            word_dir = root / "bar_nn"
            word_dir.mkdir()
            for stem in ("g1", "g2"):
                (word_dir / f"{stem}.csv").write_text("lemma\n", encoding="utf-8")
            pairs = dwug_pairs(root)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].scheme, "diachronic")


if __name__ == "__main__":
    unittest.main()
