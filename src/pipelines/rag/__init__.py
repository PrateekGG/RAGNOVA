"""Chapter 10 — RAG core: retrieval + prompt construction + generation +
citations, orchestrated on top of Chapter 7's `search_text()`. Chapter 12
adds cross-modal retrieval (retrieve.py) and a streaming entry point."""

from src.pipelines.rag.answer import RagAnswer, answer_query, stream_answer

__all__ = ["RagAnswer", "answer_query", "stream_answer"]
