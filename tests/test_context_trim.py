"""
Context compression (src/pipelines/rag/context.py): show the model each
retrieved chunk's best-matching sentences instead of all of it.

Why it exists: measured on 2026-10-07 (ADR-015), almost all of an answer's wait
is the model reading the prompt (13.2 s of 15.3 s on a CPU; answers about 37
tokens), so a shorter prompt is the one latency lever there is. (It then turned
out to cost answer quality, and is off and not recommended; see ADR-015.) What
these tests pin is that it can only ever shorten the PROMPT: the question's
best sentences survive, the budget holds, order is kept, gaps are marked, the
citations stay whole, and with the setting at 0 nothing changes at all.

A hashed bag-of-words stands in for MiniLM: no weights, no network.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import zlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag.context import MAX_PIECE_WORDS, split_pieces, trim_chunks

DIM = 64


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vector = [0.0] * DIM
        for word in text.lower().split():
            vector[zlib.crc32(word.strip(".,?!:;").encode()) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        out.append([v / norm for v in vector])
    return out


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(chunk_id=chunk_id, source="data/documents/notice.pdf", modality="pdf",
                 text=text, embedding_model="test", page=2, score=0.5)


FILLER = " ".join(f"Unrelated filler sentence number {i} about nothing in particular here." for i in range(30))
ANSWER = "The working prototype carries forty percent of the total marks."
LONG = f"{FILLER} {ANSWER} {FILLER}"


def _words(text: str) -> int:
    return len(text.split())


def test_the_sentence_that_matches_the_question_survives_and_most_filler_does_not():
    [trimmed] = trim_chunks([_chunk(LONG)], "how many marks does the working prototype carry", 40, embed=embed)
    assert ANSWER in trimmed.text
    assert _words(trimmed.text) < _words(LONG) / 4


def test_the_budget_is_respected_up_to_one_extra_piece_and_the_ellipsis_marks():
    [trimmed] = trim_chunks([_chunk(LONG)], "working prototype marks", 30, embed=embed)
    content = trimmed.text.replace("...", " ")
    assert _words(content) <= 30 + MAX_PIECE_WORDS
    assert trimmed.text.startswith("...") and trimmed.text.endswith("...")      # filler was cut on both sides


def test_kept_pieces_stay_in_their_original_order_even_when_the_later_one_matches_better():
    """The best match is LAST in the text and the second best FIRST: ranked order would put them the other way round."""
    first = "Weak match mentions prototype once."
    last = "Strong match: prototype prototype marks marks forty percent."
    text = f"{first} {FILLER} {last}"
    [trimmed] = trim_chunks([_chunk(text)], "prototype marks forty percent prototype marks", 15, embed=embed)
    assert first in trimmed.text and last in trimmed.text
    assert trimmed.text.index(first) < trimmed.text.index(last)


def test_a_gap_between_two_kept_pieces_is_marked_with_an_ellipsis():
    """First and last pieces kept, everything between dropped: the only marker is the one in the middle."""
    first = "Alpha prototype marks forty."
    last = "Omega prototype marks percent."
    text = f"{first} {FILLER} {last}"
    [trimmed] = trim_chunks([_chunk(text)], "prototype marks forty percent", 10, embed=embed)
    assert trimmed.text == f"{first} ... {last}"


def test_a_chunk_already_within_the_budget_is_returned_as_it_is():
    short = _chunk("Only a few words here.")
    assert trim_chunks([short], "anything", 50, embed=embed)[0] is short


def test_zero_disables_trimming_and_returns_the_same_list():
    chunks = [_chunk(LONG)]
    assert trim_chunks(chunks, "question", 0, embed=embed) is chunks
    assert trim_chunks(chunks, "question", -5, embed=embed) is chunks


def test_a_blank_question_cannot_pick_sentences_so_nothing_is_trimmed():
    chunks = [_chunk(LONG)]
    assert trim_chunks(chunks, "   ", 30, embed=embed) is chunks


def test_trimming_changes_only_the_text_and_never_the_original_chunk():
    original = _chunk(LONG)
    [trimmed] = trim_chunks([original], "working prototype marks", 30, embed=embed)
    assert original.text == LONG                                  # the caller's chunk is untouched (citations stay whole)
    assert (trimmed.chunk_id, trimmed.source, trimmed.modality, trimmed.page, trimmed.score) == \
           (original.chunk_id, original.source, original.modality, original.page, original.score)
    assert trimmed.text != original.text


def test_text_with_no_sentence_ends_is_cut_into_windows_so_it_can_still_be_trimmed():
    """OCR output often has no sentence punctuation; one 200-word run must not be an unselectable block."""
    run = " ".join(f"word{i}" for i in range(200)) + " prototype marks forty percent " + " ".join(f"other{i}" for i in range(200))
    assert all(_words(p) <= MAX_PIECE_WORDS for p in split_pieces(run))
    [trimmed] = trim_chunks([_chunk(run)], "prototype marks forty percent", 50, embed=embed)
    assert "prototype marks forty percent" in trimmed.text
    assert _words(trimmed.text) < 100


def test_every_chunk_is_embedded_in_one_batch():
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return embed(texts)

    trim_chunks([_chunk(LONG, "a"), _chunk(LONG, "b"), _chunk("short one", "c")], "prototype marks", 30, embed=counting)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# Wiring: only the prompt is shortened, the answer still cites full chunks
# --------------------------------------------------------------------------

@pytest.fixture
def pipeline(monkeypatch):
    """answer_query() with retrieval and the model replaced, recording the prompt."""
    chunk = _chunk(LONG)
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda query, **kwargs: [chunk])

    def fake_generate(prompt):
        seen["prompt"] = prompt
        return "Forty percent [1]."

    monkeypatch.setattr(answer_module, "generate", fake_generate)
    monkeypatch.setattr(answer_module, "trim_chunks",
                        lambda chunks, query, max_words: trim_chunks(chunks, query, max_words, embed=embed))
    return chunk, seen


def _set(monkeypatch, words: int):
    monkeypatch.setattr(answer_module, "settings", dataclasses.replace(answer_module.settings, CONTEXT_WORDS_PER_CHUNK=words))


def test_with_the_setting_on_the_prompt_is_shorter_but_the_citations_are_the_full_chunks(pipeline, monkeypatch):
    chunk, seen = pipeline
    _set(monkeypatch, 30)
    result = answer_module.answer_query("how many marks does the working prototype carry")
    assert ANSWER in seen["prompt"]
    assert FILLER not in seen["prompt"]
    assert result.citations[0].text == LONG


def test_with_the_setting_off_the_prompt_holds_the_whole_chunk(pipeline, monkeypatch):
    chunk, seen = pipeline
    _set(monkeypatch, 0)
    answer_module.answer_query("how many marks does the working prototype carry")
    assert LONG in seen["prompt"]
