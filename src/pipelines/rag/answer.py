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
from src.pipelines.rag.context import trim_chunks
from src.pipelines.rag.followup import resolve as resolve_follow_up
from src.pipelines.rag.lexical import content_tokens
from src.pipelines.rag.prompt import build_prompt
from src.pipelines.rag.retrieve import filter_by_floor, retrieve

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

NOT_ENOUGH_INFO = (
    "I don't have enough information in the indexed documents to answer that."
)


def is_refusal(answer: str) -> bool:
    """The answer opens with the refusal sentence, as the model actually types it: small models mangle the
    apostrophe ("I don' have", "I don' t have", a curly ’) or write "do not". Letters only, so spacing and
    punctuation do not matter; the words that follow "I do(n't)" must be the refusal's own."""
    letters = re.sub(r"[^a-z]", "", answer.lower())
    return _REFUSAL_LETTERS.match(letters) is not None


# "I don't" / "I don' t" / "I don’t" -> idont, "I don' have" -> idon, "I do not" -> idonot; never "I do have".
_REFUSAL_LETTERS = re.compile(r"ido(?:nt|n|not)haveenoughinformationintheindexeddocuments")

# Stands in for the question when the user attached only an image (no
# typed words, no readable text in the picture) — Chapter 11's UI allows
# that, and "Question: " followed by nothing gives the model no task.
IMAGE_ONLY_QUESTION = (
    "The user attached an image without a written question. Say what the "
    "context shows that relates to it."
)


# Stands in for a question with no subject of its own ("explain", "summarize?") when files are picked under "Answer
# from": the files are the subject, but "Question: explain" alone gave the model no task and it refused (2026-10-11).
SCOPED_BARE_QUESTION = (
    'The user wrote "{query}" about the files they chose. Explain what the context from those files covers: '
    "give its key points."
)
# Words that ask for a kind of answer and name no subject, on top of lexical's filler ("explain", "tell", "more").
_ASK_ONLY_WORDS = frozenset("summarize summarise summary overview brief briefly elaborate simplify simply short".split())


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


def check_prompt_fits(prompt: str) -> bool:
    """Warn (never raise) when a prompt is close to Ollama's context window.

    Ollama silently truncates a prompt that does not fit in `num_ctx`
    (settings.LLM_NUM_CTX), and the part it cuts is not the part anyone would
    choose to lose. Measured 2026-10-05 (data/eval/ablations_2026-10-05.txt):
    at the shipped 4096 tokens, 600-word chunks or TOP_K=10 produced prompts
    of about 4100 tokens, and 2 of 25 prompts overflowed with no error. This
    makes that visible. The size is ESTIMATED at 1.3 tokens per word, the same
    ratio src/core/config.py's comment uses, so the 90% threshold leaves room
    for the estimate being off. Returns True if the prompt looks safe.
    """
    estimated = int(len(prompt.split()) * 1.3)
    limit = settings.LLM_NUM_CTX
    if estimated >= 0.9 * limit:
        logger.warning(
            "prompt is about %d tokens against a %d-token context window (LLM_NUM_CTX); Ollama "
            "will silently truncate it. Lower TOP_K / CHUNK_SIZE_WORDS or raise LLM_NUM_CTX.",
            estimated, limit,
        )
        return False
    return True


def _filter_relevant(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks scoring below settings.MIN_RELEVANCE_SCORE (ADR-009).
    A pure function over already-retrieved Chunks, deliberately factored
    out of answer_query() so it's testable without a live ChromaDB/Ollama."""
    return filter_by_floor(chunks, settings.MIN_RELEVANCE_SCORE)


def _for_prompt(chunks: list[Chunk], query: str) -> list[Chunk]:
    """What the model is shown: the retrieved chunks, shortened when
    CONTEXT_WORDS_PER_CHUNK is set (context.py). Only the prompt changes; the
    caller keeps citing the full chunks."""
    return trim_chunks(chunks, query, settings.CONTEXT_WORDS_PER_CHUNK)


def _gather(
    query: str,
    top_k: int | None,
    client,
    include_images: bool,
    query_image: Image.Image | None,
    sources: list[str] | None,
    attachments: list[Chunk] | None,
    search_query: str | None = None,
) -> list[Chunk]:
    """Everything the model may be shown, in citation order: the user's
    attachments first (they put them there; nothing filters them), then what
    retrieval found for the question as typed. The attachment's text is NOT added
    to the retrieval query: tried both ways on 2026-10-10 it made no difference to
    9 synthetic rows (9/9 each) and, on 4 questions about two real screenshots,
    gluing it on brought in unrelated corpus chunks and gave 0 usable answers
    against 2 without (one run each of a stochastic model; ADR-016)."""
    found = retrieve(
        search_query or query, top_k=top_k, client=client,
        include_images=include_images, query_image=query_image, sources=sources,
    )
    return [*(attachments or []), *found]


def search_question(query: str, history: list[tuple[str, str]] | None, attachments: list[Chunk] | None = None,
                    sources: list[str] | None = None) -> str:
    """What `query` is searched with: itself, or, for a follow-up, followup.resolve()'s question. The page stores it
    with the turn so the next follow-up builds on it ("tell me more", then "more")."""
    return _follow_up(query, history, attachments, sources)[0]


def _follow_up(query, history, attachments, sources=None) -> tuple[str, list[tuple[str, str]] | None]:
    """(question to search with, earlier exchange for the prompt or None). An attached image is what "this" means,
    so a question with an attachment is never treated as a follow-up; with files picked under "Answer from", "it" means
    those files, so only a question with no words of its own ("tell me more") is one."""
    if not history or attachments or not query.strip():
        return query, None
    if sources and content_tokens(query):
        return query, None
    search = resolve_follow_up(query, history[-1][0])
    return search, (history[-1:] if search != query else None)


def _plan(query, history, attachments, earlier_attachments, sources):
    """(question to search with, earlier exchange for the prompt or None, attachments to show)."""
    search_query, prompt_history = _follow_up(query, history, attachments, sources)
    if prompt_history and not attachments and earlier_attachments:
        attachments = list(earlier_attachments)
    return search_query, prompt_history, attachments


def _prompt(query: str, relevant: list[Chunk], history, sources: list[str] | None = None,
            search_query: str | None = None) -> str:
    question = query or IMAGE_ONLY_QUESTION
    if sources and not history and query.strip() and set(content_tokens(query)) <= _ASK_ONLY_WORDS:
        question = SCOPED_BARE_QUESTION.format(query=query.strip())
    return build_prompt(
        question, _for_prompt(relevant, search_query or query),
        settings.PROMPT_WHO_WHAT_RULE, settings.PROMPT_DETAIL_RULE, history, sources,
    )


def answer_query(
    query: str,
    top_k: int | None = None,
    client=None,
    include_images: bool = False,
    query_image: Image.Image | None = None,
    sources: list[str] | None = None,
    attachments: list[Chunk] | None = None,
    history: list[tuple[str, str]] | None = None,
    earlier_attachments: list[Chunk] | None = None,
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

    `sources` limits the search to the files the user picked (retrieve() says
    what that changes). `attachments` are items the user attached to this
    question (attachments.py): shown to the model first, cited like any chunk,
    never filtered, so an answer can come from them alone.

    `history` is the earlier conversation, [(search question, answer)] oldest first (the page passes the last
    exchange). Only a follow-up ("tell me more", "who made it?") uses it: followup.py builds the question to
    search with, and the prompt shows the exchange so the model knows what "it" is. A question that stands on its
    own is handled exactly as without it. `earlier_attachments` are the attachments of that earlier turn: a
    follow-up with no attachment of its own ("tell me more" after "what does this screenshot say?") is answered
    from them again, since a screenshot that is not indexed cannot be found by searching.
    """
    search_query, prompt_history, attachments = _plan(query, history, attachments, earlier_attachments, sources)
    relevant = _gather(query, top_k, client, include_images, query_image, sources, attachments, search_query)

    if not relevant:
        return RagAnswer(
            query=query, answer=NOT_ENOUGH_INFO, citations=[], model=settings.OLLAMA_MODEL
        )

    prompt = _prompt(query, relevant, prompt_history, sources, search_query)
    check_prompt_fits(prompt)
    answer_text = generate(prompt)
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
    sources: list[str] | None = None,
    attachments: list[Chunk] | None = None,
    history: list[tuple[str, str]] | None = None,
    earlier_attachments: list[Chunk] | None = None,
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
    search_query, prompt_history, attachments = _plan(query, history, attachments, earlier_attachments, sources)
    relevant = _gather(query, top_k, client, include_images, query_image, sources, attachments, search_query)
    if not relevant:
        return [], iter([NOT_ENOUGH_INFO])
    prompt = _prompt(query, relevant, prompt_history, sources, search_query)
    check_prompt_fits(prompt)
    return relevant, generate_stream(prompt)


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
