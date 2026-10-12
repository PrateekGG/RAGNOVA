"""
Shorten what the model is shown, not what the user is cited (opt-in).

Measured on 2026-10-07 (data/eval/performance_latency_*, ADR-015): on this model
the wait is almost all the model READING the prompt, not writing the answer: on
CPU the default prompt (about 2000 tokens) takes 13.2 s to read and 2.2 s to
answer, and answers are short (about 37 tokens), so a lower token cap changes
nothing. A GPU reads the same prompt in 0.26 s. The one lever left is a
shorter prompt, and the cheapest honest way to get one is to show the model each
retrieved chunk's best-matching sentences instead of all ~300 words of it.

`trim_chunks()` keeps, per chunk, the pieces (sentences, or 40-word windows of
text that has no sentences, as OCR output often has) whose MiniLM vectors are
closest to the question, up to `max_words`, in their original order, with "..."
where text was left out. The chunk's id, source, page and modality are
untouched and `answer_query()` still returns the FULL chunks as its citations:
only the prompt is shortened. Off unless CONTEXT_WORDS_PER_CHUNK > 0.

Pure of I/O apart from the embedding model, which tests replace with a fake.
"""

from __future__ import annotations

import re
from dataclasses import replace

from src.core.schemas import Chunk

# Sentence ends, and line breaks (PDF text is full of them).
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# Text without sentence punctuation (OCR, tables) would otherwise be one
# unselectable piece, so a piece is never longer than this many words.
MAX_PIECE_WORDS = 40


def split_pieces(text: str) -> list[str]:
    """Sentences of `text`, each cut into windows of at most MAX_PIECE_WORDS words."""
    pieces: list[str] = []
    for sentence in _BOUNDARY.split(text):
        words = sentence.split()
        for start in range(0, len(words), MAX_PIECE_WORDS):
            pieces.append(" ".join(words[start:start + MAX_PIECE_WORDS]))
    return pieces


def _choose(pieces: list[str], similarities: list[float], max_words: int) -> str:
    """The best-scoring pieces that fit in `max_words`, in original order, "..." marking gaps.
    A piece that does not fit is skipped, not the end of the search: a shorter, lower-scoring
    one may still fit. The best piece is always kept (a piece is at most MAX_PIECE_WORDS)."""
    order = sorted(range(len(pieces)), key=lambda i: similarities[i], reverse=True)
    chosen: list[int] = []
    used = 0
    for i in order:
        words = len(pieces[i].split())
        if chosen and used + words > max_words:
            continue
        chosen.append(i)
        used += words
    chosen.sort()
    parts: list[str] = []
    if chosen[0] != 0:
        parts.append("...")
    for position, i in enumerate(chosen):
        if position and i != chosen[position - 1] + 1:
            parts.append("...")
        parts.append(pieces[i])
    if chosen[-1] != len(pieces) - 1:
        parts.append("...")
    return " ".join(parts)


def trim_chunks(chunks: list[Chunk], query: str, max_words: int, embed=None) -> list[Chunk]:
    """Copies of `chunks` whose text is cut to about `max_words` words of what matters for `query`.

    Returned unchanged (the same list) when `max_words` is 0 or less, or the question is blank.
    A chunk already within `max_words` is kept as it is. `embed(list[str]) -> list[vector]`
    defaults to MiniLM (src.core.embeddings.embed_texts); everything is embedded in one batch.
    """
    if max_words <= 0 or not query.strip():
        return chunks
    pieces_by_chunk = [split_pieces(c.text) if len(c.text.split()) > max_words else None for c in chunks]
    flat = [p for pieces in pieces_by_chunk if pieces for p in pieces]
    if not flat:
        return chunks
    if embed is None:
        from src.core.embeddings import embed_texts as embed
    vectors = embed([query] + flat)
    question, piece_vectors = vectors[0], vectors[1:]
    similarities = [sum(a * b for a, b in zip(question, v)) for v in piece_vectors]
    out: list[Chunk] = []
    offset = 0
    for chunk, pieces in zip(chunks, pieces_by_chunk):
        if not pieces:
            out.append(chunk)
            continue
        out.append(replace(chunk, text=_choose(pieces, similarities[offset:offset + len(pieces)], max_words)))
        offset += len(pieces)
    return out
