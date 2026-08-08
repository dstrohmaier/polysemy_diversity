"""Methods to load the WSD dataset from which we can create a ground-truth polysemy dataset."""

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Generator
from dataclasses import dataclass

import pandas as pd  # type: ignore


@dataclass(frozen=True)
class AnnotatedExample:
    raw_verb: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class SenseCollection:
    verb: str
    sense: str
    pos: str
    examples: tuple[AnnotatedExample, ...]


_SENTENCE_BREAKS = {"SENTENCE_BREAK", "PARAGRAPH_BREAK"}


def _finalize_sentence(
    buffer: list[tuple[str, str, str | None, str | None, str | None]],
) -> list[dict]:
    """Reconstruct a sentence from a token buffer and return one record per annotated token."""
    if not buffer:
        return []

    chars: list[str] = []
    offsets: list[tuple[int, int]] = []

    for i, (text, break_level, *_) in enumerate(buffer):
        if i > 0 and break_level != "NO_BREAK":
            chars.append(" ")
        start = len(chars)
        chars.extend(text)
        offsets.append((start, start + len(text)))

    sentence = "".join(chars)

    records = []
    for i, (text, _, lemma, pos, sense) in enumerate(buffer):
        if lemma is not None and pos is not None and sense is not None:
            start, end = offsets[i]
            records.append(
                {
                    "lemma": lemma,
                    "pos": pos,
                    "sense": sense,
                    "sentence": sentence,
                    "start": start,
                    "end": end,
                }
            )
    return records


def wsd_generator(data_dir: Path) -> Generator[dict[str, Any], None, None]:
    for xml_fp in sorted(data_dir.glob("**/*.xml")):
        try:
            tree = ET.parse(xml_fp)
        except ET.ParseError:
            continue

        root = tree.getroot()
        buffer: list[tuple[str, str, str | None, str | None, str | None]] = []

        for word_el in root.iter("word"):
            text = word_el.get("text", "")
            break_level = word_el.get("break_level", "SPACE_BREAK")
            lemma = word_el.get("lemma")
            pos = word_el.get("pos")
            sense = word_el.get("sense")

            # break_level describes the break *before* this word, so a sentence/
            # paragraph break means this word starts a new sentence: flush the buffer
            # accumulated so far, then begin the new sentence with this word.
            if break_level in _SENTENCE_BREAKS and buffer:
                yield from _finalize_sentence(buffer)
                buffer = []

            buffer.append((text, break_level, lemma, pos, sense))

        yield from _finalize_sentence(buffer)


def load_wsd(data_dirs: list[Path]) -> pd.DataFrame:
    """Load all annotated WSD occurrences into one frame (no sense filtering).

    Returns every annotated ``(lemma, pos, sense)`` occurrence, including duplicate
    sentences. The Zipfian baseline slope is fitted from these raw counts; the
    minimum-examples filter is applied later, in ``simulate_word_corpus``, on the
    *distinct-sentence* count after deduplication.
    """
    frames = []
    for data_dir in data_dirs:
        part = pd.DataFrame(
            wsd_generator(data_dir),
            columns=["lemma", "pos", "sense", "sentence", "start", "end"],
        )
        part["source"] = data_dir.name
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["lemma", "pos", "sense"], ignore_index=True, inplace=True)
    return df
