"""
Chapter 10's test suite: prompt construction and relevance filtering are
pure functions and always run; answer_query() end-to-end needs a real,
running Ollama, so those tests skip cleanly (not fail) when one isn't
reachable — same spirit as tests/test_retrieval.py's
`pytest.importorskip("chromadb")`.

Run with:  pytest tests/test_rag_core.py -v
(build the real index first: python scripts/build_index.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import ollama
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.schemas import Chunk
from src.core.vector_store import get_client, get_text_collection
from src.pipelines.documents.index import index_documents_directory
from src.pipelines.rag.answer import (
    NOT_ENOUGH_INFO,
    _filter_relevant,
    answer_query,
    extract_citation_numbers,
)
from src.pipelines.rag.prompt import build_prompt

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def _make_chunk(chunk_id: str, text: str, score: float | None, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, source="data/documents/notice.pdf", modality="pdf",
        text=text, embedding_model=settings.TEXT_EMBEDDING_MODEL, page=page, score=score,
    )


def _ollama_reachable() -> bool:
    try:
        ollama.Client(host=settings.OLLAMA_HOST).list()
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {settings.OLLAMA_HOST} — start `ollama serve` "
           f"and `ollama pull {settings.OLLAMA_MODEL}`",
)


# ---------------------------------------------------------------------------
# Part 1 — build_prompt(): pure, no I/O
# ---------------------------------------------------------------------------

def test_build_prompt_numbers_chunks_in_order_with_provenance():
    chunks = [
        _make_chunk("a", "First chunk text.", score=0.9, page=1),
        _make_chunk("b", "Second chunk text.", score=0.8, page=2),
    ]
    prompt = build_prompt("some question", chunks)
    assert "[1] data/documents/notice.pdf, page 1\nFirst chunk text." in prompt
    assert "[2] data/documents/notice.pdf, page 2\nSecond chunk text." in prompt
    assert "Question: some question" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")  # retrieval order preserved


def test_build_prompt_empty_chunks_uses_no_context_notice():
    prompt = build_prompt("some question", [])
    assert "No relevant context was found" in prompt
    assert "Question: some question" in prompt


def test_build_prompt_instructs_citation_and_refusal():
    prompt = build_prompt("q", [_make_chunk("a", "text", score=0.9)])
    assert "[1]" in prompt  # instructs the citation format, not just data
    assert "don't have enough information" in prompt.lower()


# ---------------------------------------------------------------------------
# Part 2 — relevance filtering (ADR-009): pure, no I/O
# ---------------------------------------------------------------------------

def test_filter_relevant_drops_low_score_keeps_high_score():
    low = _make_chunk("low", "irrelevant", score=settings.MIN_RELEVANCE_SCORE - 0.05)
    high = _make_chunk("high", "relevant", score=settings.MIN_RELEVANCE_SCORE + 0.05)
    result = _filter_relevant([low, high])
    assert result == [high]


def test_filter_relevant_keeps_chunk_exactly_at_threshold():
    at_threshold = _make_chunk("edge", "text", score=settings.MIN_RELEVANCE_SCORE)
    assert _filter_relevant([at_threshold]) == [at_threshold]


def test_filter_relevant_all_below_threshold_returns_empty():
    chunks = [_make_chunk("a", "x", score=0.0), _make_chunk("b", "y", score=0.1)]
    assert _filter_relevant(chunks) == []


# ---------------------------------------------------------------------------
# Part 3 — extract_citation_numbers(): pure, no I/O
# ---------------------------------------------------------------------------

def test_extract_citation_numbers_parses_multiple():
    text = "The prototype carries 40% [1], and the report carries 35% [2]."
    assert extract_citation_numbers(text) == {1, 2}


def test_extract_citation_numbers_ignores_non_citation_brackets():
    # [text] isn't a bracketed integer, so the regex should not match it at all.
    assert extract_citation_numbers("no citations here, just [text] in brackets") == set()


def test_extract_citation_numbers_empty_text_returns_empty_set():
    assert extract_citation_numbers("") == set()


# ---------------------------------------------------------------------------
# Part 4 — answer_query() end-to-end: real ChromaDB + real Ollama
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_client(tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("chroma_rag_module")
    client = get_client(persist_dir=persist_dir)
    count = index_documents_directory(DOCS_DIR, client=client)
    assert count > 0, "No chunks were indexed — run `python scripts/generate_sample_corpus.py` first."
    return client


@requires_ollama
def test_answer_query_cites_the_right_source_for_a_gold_question(indexed_client):
    result = answer_query(
        "If I don't get my system actually running by evaluation day, how many marks am I giving up?",
        client=indexed_client,
    )
    assert result.answer.strip() != ""
    assert result.citations, "expected at least one citation above the relevance threshold"
    cited_numbers = extract_citation_numbers(result.answer)
    assert cited_numbers, "expected the model to cite at least one [n]"
    assert all(1 <= n <= len(result.citations) for n in cited_numbers)
    assert any(c.source == "data/documents/notice.pdf" and c.page == 2 for c in result.citations)


@pytest.fixture
def empty_client(tmp_path):
    """A real ChromaDB client whose text_index has nothing in it — used to
    make the negative-control path deterministic without depending on how
    irrelevant a real off-topic question happens to score against the real
    corpus, and without needing Ollama reachable at all (this path never
    calls it)."""
    return get_client(persist_dir=tmp_path / "chroma_empty")


def test_answer_query_refuses_when_index_has_no_relevant_chunks(empty_client):
    # An empty text_index means search_text() has nothing to return, so
    # _filter_relevant() has nothing to filter and the LLM is never called
    # — this is the ADR-009 short-circuit, exercised directly.
    result = answer_query("What is the capital of France?", client=empty_client)
    assert result.answer == NOT_ENOUGH_INFO
    assert result.citations == []
