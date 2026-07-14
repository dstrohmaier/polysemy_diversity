import random
import tempfile
import unittest
from pathlib import Path

from data_processing.loading_fews import (
    build_balanced_pairs,
    get_fews_wic_dsd,
    load_fews_occurrences,
    parse_fews_line,
)

_WIC_FIELDS = {
    "lemma", "sentence1", "sentence2", "label", "start1", "end1", "start2", "end2",
}


class ParseFewsLineTestCase(unittest.TestCase):
    def test_offsets_round_trip_to_target(self):
        line = "I sat on the <WSD>bank</WSD> of the river.\tbank.noun.1"
        rec = parse_fews_line(line)
        self.assertIsNotNone(rec)
        # The recorded span must slice the target back out of the cleaned sentence.
        self.assertEqual(rec["sentence"][rec["start"]:rec["end"]], "bank")
        self.assertNotIn("<WSD>", rec["sentence"])

    def test_pos_normalised_and_lemma_unslashed(self):
        rec = parse_fews_line("A <WSD>driving forces</WSD> example.\tdriving_force.noun.1")
        self.assertEqual(rec["pos"], "NOUN")
        self.assertEqual(rec["lemma"] if "lemma" in rec else rec["word"], "driving force")
        self.assertEqual(rec["sense_id"], "driving_force.noun.1")

    def test_first_wsd_span_used_when_multiple(self):
        line = "The <WSD>bank</WSD> near the river <WSD>bank</WSD>.\tbank.noun.1"
        rec = parse_fews_line(line)
        # Offsets point at the FIRST occurrence.
        self.assertEqual(rec["start"], len("The "))
        self.assertEqual(rec["sentence"][rec["start"]:rec["end"]], "bank")

    def test_malformed_lines_return_none(self):
        self.assertIsNone(parse_fews_line("no tab here"))
        self.assertIsNone(parse_fews_line("no tag\tbank.noun.1"))
        self.assertIsNone(parse_fews_line("<WSD>x</WSD>\tbadlabel"))
        self.assertIsNone(parse_fews_line("<WSD>x</WSD>\tx.gerund.0"))  # unknown pos


def _occ(word, pos, sense, sentence="the <WSD>%s</WSD> here"):
    # Build a raw FEWS line and parse it so offsets are realistic.
    raw = sentence % word if "%s" in sentence else sentence
    return parse_fews_line(f"{raw}\t{word}.{pos}.{sense}")


class BuildBalancedPairsTestCase(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)
        # 'bank': sense 1 has 4 occ (positives), sense 2 has 2 occ; two senses (negatives).
        # 'run': sense 0 has 3 occ.
        self.occ = []
        for i in range(4):
            self.occ.append(_occ("bank", "noun", "1", f"river <WSD>bank</WSD> {i}"))
        for i in range(2):
            self.occ.append(_occ("bank", "noun", "2", f"money <WSD>bank</WSD> {i}"))
        for i in range(3):
            self.occ.append(_occ("run", "verb", "0", f"they <WSD>run</WSD> {i}"))

    def test_pairs_are_balanced_50_50(self):
        pairs = build_balanced_pairs(self.occ, self.rng, cap_per_word=10)
        self.assertTrue(pairs)
        n_pos = sum(p["label"] for p in pairs)
        self.assertEqual(n_pos, len(pairs) - n_pos)  # exactly half positive

    def test_pairs_share_lemma_and_label_semantics(self):
        pairs = build_balanced_pairs(self.occ, self.rng, cap_per_word=10)
        for p in pairs:
            self.assertIn(p["lemma"], {"bank", "run"})
            # Both sentences must contain the shared surface word.
            self.assertIn(p["lemma"], p["sentence1"])
            self.assertIn(p["lemma"], p["sentence2"])
            # Offsets slice the target out of each sentence.
            self.assertEqual(
                p["sentence1"][p["start1"]:p["end1"]].strip(), p["lemma"]
            )
            self.assertEqual(
                p["sentence2"][p["start2"]:p["end2"]].strip(), p["lemma"]
            )

    def test_cap_limits_positive_pairs_per_word(self):
        pairs = build_balanced_pairs(self.occ, self.rng, cap_per_word=1)
        # With cap 1, 'bank' contributes at most 1 positive; 'run' at most 1. So <=2
        # positives total, and balance forces <=2 negatives.
        n_pos = sum(p["label"] for p in pairs)
        self.assertLessEqual(n_pos, 2)


class GetFewsWicDsdTestCase(unittest.TestCase):
    def _write_fews(self, root: Path):
        # Several multi-sense words so a lemma-disjoint validation split is non-empty.
        (root / "train").mkdir(parents=True)
        lines = []
        for w in ("bank", "plant", "run", "light", "spring", "bark"):
            for i in range(4):
                lines.append(f"a x <WSD>{w}</WSD> s{i}\t{w}.noun.1")
            for i in range(4):
                lines.append(f"a y <WSD>{w}</WSD> s{i}\t{w}.noun.2")
        (root / "train" / "train.txt").write_text("\n".join(lines) + "\n")

    def test_dsd_has_wic_schema_and_both_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_fews(root)
            dsd = get_fews_wic_dsd(root, seed=1, val_fraction=0.34)
            self.assertEqual(set(dsd.keys()), {"train", "validation"})
            self.assertTrue(len(dsd["train"]) > 0)
            self.assertTrue(len(dsd["validation"]) > 0)
            self.assertEqual(set(dsd["train"].column_names), _WIC_FIELDS)
            self.assertEqual(set(dsd["validation"].column_names), _WIC_FIELDS)

    def test_train_and_validation_are_lemma_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_fews(root)
            dsd = get_fews_wic_dsd(root, seed=1, val_fraction=0.34)
            train_lemmas = set(dsd["train"]["lemma"])
            val_lemmas = set(dsd["validation"]["lemma"])
            self.assertTrue(train_lemmas.isdisjoint(val_lemmas))

    def test_use_test_folds_validation_into_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_fews(root)
            base = get_fews_wic_dsd(root, seed=1, val_fraction=0.34)
            with_test = get_fews_wic_dsd(root, use_test=True, seed=1, val_fraction=0.34)
            self.assertGreater(len(with_test["train"]), len(base["train"]))

    def test_use_test_validation_is_a_true_holdout(self):
        # The use_test=True eval split must be lemma-disjoint from its own train split
        # AND from everything the hp search (use_test=False) ever saw.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._write_fews(root)
            base = get_fews_wic_dsd(root, seed=1, val_fraction=0.34)
            with_test = get_fews_wic_dsd(root, use_test=True, seed=1, val_fraction=0.34)
            test_lemmas = set(with_test["validation"]["lemma"])
            self.assertTrue(test_lemmas)
            self.assertTrue(test_lemmas.isdisjoint(set(with_test["train"]["lemma"])))
            self.assertTrue(test_lemmas.isdisjoint(set(base["train"]["lemma"])))
            self.assertTrue(test_lemmas.isdisjoint(set(base["validation"]["lemma"])))


if __name__ == "__main__":
    unittest.main()
