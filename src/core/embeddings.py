"""
Text embedding — the one function every track eventually calls to turn a
string into the vector ChromaDB actually searches over.

Shared, joint-owned (Ch4 §1.3), not Track A-private: documents (Chapter 6)
and audio transcripts (Chapter 9, ADR-005 — transcripts are indexed as
*text*) both need the identical model and the identical normalization
choice. Putting it here once means both tracks' vectors are guaranteed
comparable, which matters a great deal more for an embedding model than it
would for an ordinary utility function — two chunks embedded by two
independently-instantiated copies of "the same" model, with different
normalization, would silently produce vectors that don't rank the way
either track expects.
"""

from __future__ import annotations

from src.core.compat import tolerate_blocked_sklearn

# sentence-transformers imports scikit-learn, whose optional native accelerator Windows may block (compat.py).
tolerate_blocked_sklearn()

from sentence_transformers import SentenceTransformer  # noqa: E402  (after the line above on purpose)

from src.core.config import settings

# Loaded lazily, on first actual use, not at import time. This is the same
# principle Chapter 5 §2.1 taught about GGUF/mmap — loading a real model's
# weights is not free, and importing this module should not silently cost
# several seconds and a few hundred MB of RAM for code that ends up never
# calling embed_text() at all. Cached at module level after that first
# call, so the cost is paid at most once per process, not once per call.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one call.

    Prefer this over calling embed_text() in a loop when embedding more
    than one string — sentence-transformers batches the underlying model
    forward pass internally, which is meaningfully faster than one string
    at a time for anything beyond a handful of chunks (Chapter 7 §1.3).

    `normalize_embeddings=True` scales every vector to unit length. This
    is what makes cosine similarity and (squared) Euclidean distance
    produce IDENTICAL rankings (docs/GLOSSARY.md's "Cosine similarity"
    entry states this fact; this is where the project actually relies on
    it) — and it is also the input all-MiniLM-L6-v2 was trained to be
    compared under. Skipping this step wouldn't break the code, but it
    would silently make results modestly worse and inconsistent with the
    metric this project has told itself, in its own docs, that it's using.
    """
    model = _get_model()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    """Embed a single string. A thin convenience wrapper — see
    embed_texts()'s docstring before reaching for this in a loop."""
    return embed_texts([text])[0]
