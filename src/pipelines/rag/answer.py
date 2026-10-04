"""The RAG core orchestrator: retrieve -> filter -> prompt -> generate ->
cite. This is what Chapter 7's search.py docstring meant by "Chapter 10 is
what orchestrates it into a full answer, not what reimplements it."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from typing import Iterator

from PIL import Image

from src.core.config import settings
from src.core.llm import generate, generate_stream
from src.core.schemas import Chunk
from src.pipelines.rag.prompt import build_prompt
from src.pipelines.rag.retrieve import filter_by_floor, retrieve

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

NOT_ENOUGH_INFO = (
    "I don't have enough information in the indexed documents to answer that."
)

# Stands in for the question when the user attached only an image (no
# typed words, no readable text in the picture) — Chapter 11's UI allows
# that, and "Question: " followed by nothing gives the model no task.
IMAGE_ONLY_QUESTION = (
    "The user attached an image without a written question. Say what the "
    "context shows that relates to it."
)


@dataclass
class RagAnswer:
    """The result of one end-to-end question. `citations` is the retrieved
    chunk list, in the exact order the prompt numbered them — index `n - 1`
    for citation "[n]" in `answer` text. Rendering a "Sources:" list from
    this (real Chunk metadata) rather than from anything the model itself
    wrote is deliberate: it decouples "did the model pick a valid citation
    number" (checked separately, see extract_citation_numbers) from "is the
    displayed source string correct" (always correct, since it never came
    from model output)."""

    query: str
    answer: str
    citations: list[Chunk]
    model: str


def extract_citation_numbers(text: str) -> set[int]:
    """Every bracketed integer the answer text cites, e.g. {1, 2} for
    "... supports this [1], and so does this [2]." Used only to detect and
    warn about a citation number outside the range of chunks actually
    shown to the model — never to build the displayed source list."""
    return {int(n) for n in _CITATION_RE.findall(text)}


def _filter_relevant(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks scoring below settings.MIN_RELEVANCE_SCORE (ADR-009).
    A pure function over already-retrieved Chunks, deliberately factored
    out of answer_query() so it's testable without a live ChromaDB/Ollama."""
    return filter_by_floor(chunks, settings.MIN_RELEVANCE_SCORE)


def answer_query(
    query: str,
    top_k: int | None = None,
    client=None,
    include_images: bool = False,
    query_image: Image.Image | None = None,
) -> RagAnswer:
    """Answer `query` using only relevant retrieved chunks.

    Chunks scoring below settings.MIN_RELEVANCE_SCORE are dropped before
    the LLM ever sees them (ADR-009) — Chroma's `.query()` always returns
    `top_k` nearest neighbours regardless of how irrelevant they are, so
    without this filter a negative-control question would still hand the
    model 5 "closest of a bad lot" chunks and no honest way to know they
    don't apply. If nothing survives, generation is skipped entirely and a
    fixed not-enough-information answer is returned — deterministic, and
    doesn't spend an LLM call on context that was never going to help.

    `include_images` / `query_image` (Chapter 12) add image_index to the
    search, merged by rank (ADR-007) — see retrieve(). Both default off,
    so Chapter 10's CLI and tests behave exactly as before.
    """
    relevant = retrieve(
        query, top_k=top_k, client=client,
        include_images=include_images, query_image=query_image,
    )

    if not relevant:
        return RagAnswer(
            query=query, answer=NOT_ENOUGH_INFO, citations=[], model=settings.OLLAMA_MODEL
        )

    answer_text = generate(build_prompt(query or IMAGE_ONLY_QUESTION, relevant))
    check_citations(answer_text, relevant, query)

    return RagAnswer(
        query=query, answer=answer_text, citations=relevant, model=settings.OLLAMA_MODEL
    )


def stream_answer(
    query: str,
    top_k: int | None = None,
    client=None,
    include_images: bool = False,
    query_image: Image.Image | None = None,
) -> tuple[list[Chunk], Iterator[str]]:
    """answer_query(), split in two for the Chapter 11 UI: retrieval runs
    now and its chunks come back straight away (so sources can be drawn
    before the answer finishes), and the answer comes back as an iterator
    of text pieces from generate_stream().

    Same ADR-009 short-circuit: nothing relevant -> no LLM call, and the
    iterator yields NOT_ENOUGH_INFO once. The caller should run
    check_citations() on the joined text when the stream ends — it can't
    run here, because the full answer doesn't exist yet.
    """
    relevant = retrieve(
        query, top_k=top_k, client=client,
        include_images=include_images, query_image=query_image,
    )
    if not relevant:
        return [], iter([NOT_ENOUGH_INFO])
    return relevant, generate_stream(build_prompt(query or IMAGE_ONLY_QUESTION, relevant))


def check_citations(answer_text: str, chunks: list[Chunk], query: str) -> set[int]:
    """Warn (never raise) about citation numbers outside 1..len(chunks) —
    Chapter 10 §2.5's check, factored out so the streaming path can run it
    too. Returns the out-of-range numbers so the UI can flag them."""
    cited = extract_citation_numbers(answer_text)
    out_of_range = {n for n in cited if n < 1 or n > len(chunks)}
    if out_of_range:
        logger.warning(
            "answer_query: model cited out-of-range number(s) %s for a "
            "%d-chunk context (query=%r)",
            sorted(out_of_range), len(chunks), query,
        )
    return out_of_range
