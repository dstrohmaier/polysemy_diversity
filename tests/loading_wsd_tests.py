import tempfile
import unittest
from pathlib import Path

from data_processing.loading_wsd import wsd_generator


def _word(text, break_level, *, lemma=None, pos=None, sense=None) -> str:
    attrs = [f'text="{text}"', f'break_level="{break_level}"']
    if lemma is not None:
        attrs.append(f'lemma="{lemma}"')
    if pos is not None:
        attrs.append(f'pos="{pos}"')
    if sense is not None:
        attrs.append(f'sense="{sense}"')
    return f"  <word {' '.join(attrs)}/>"


def _write_corpus(words: list[str]) -> Path:
    """Write the word elements into a single WSD-style XML file in a temp dir."""
    tmp = Path(tempfile.mkdtemp())
    body = "\n".join(words)
    (tmp / "doc.xml").write_text(f"<document>\n{body}\n</document>\n", encoding="utf-8")
    return tmp


class LoadingWsdTestCase(unittest.TestCase):
    def test_sentence_break_starts_new_sentence(self):
        """A word carrying SENTENCE_BREAK begins a new sentence, not ends the old one.

        In the Google WSD format break_level describes the break *before* the word,
        so the break-carrying word ("Then" here) is the first word of the next
        sentence. Each annotated target must therefore see the sentence it actually
        belongs to -- with no leading token lost and no trailing token bled in from
        the following sentence.
        """
        words = [
            _word("The", "NO_BREAK"),
            _word("dog", "SPACE_BREAK", lemma="dog", pos="NOUN", sense="dog.n.01"),
            _word("ran", "SPACE_BREAK"),
            _word(".", "NO_BREAK"),
            # First word of the second sentence carries the break.
            _word("Then", "SENTENCE_BREAK"),
            _word("it", "SPACE_BREAK"),
            _word("slept", "SPACE_BREAK", lemma="sleep", pos="VERB", sense="sleep.v.01"),
            _word(".", "NO_BREAK"),
        ]
        records = list(wsd_generator(_write_corpus(words)))

        by_lemma = {r["lemma"]: r for r in records}
        self.assertEqual(set(by_lemma), {"dog", "sleep"})

        # The "dog" target belongs to the first sentence; it must not include "Then".
        # (The final "." is NO_BREAK, so it attaches with no preceding space.)
        self.assertEqual(by_lemma["dog"]["sentence"], "The dog ran.")
        # The "sleep" target belongs to the second sentence; it must include "Then".
        self.assertEqual(by_lemma["sleep"]["sentence"], "Then it slept.")

    def test_recorded_offsets_match_the_target_span(self):
        """start/end index the target token within its (correctly split) sentence."""
        words = [
            _word("A", "NO_BREAK"),
            _word("bank", "SPACE_BREAK", lemma="bank", pos="NOUN", sense="bank.n.01"),
            _word("opened", "SPACE_BREAK"),
            _word(".", "NO_BREAK"),
            _word("Money", "SENTENCE_BREAK"),
            _word("flowed", "SPACE_BREAK"),
            _word(".", "NO_BREAK"),
        ]
        (record,) = [
            r for r in wsd_generator(_write_corpus(words)) if r["lemma"] == "bank"
        ]
        sentence, start, end = record["sentence"], record["start"], record["end"]
        self.assertEqual(sentence, "A bank opened.")
        self.assertEqual(sentence[start:end], "bank")


if __name__ == "__main__":
    unittest.main()
