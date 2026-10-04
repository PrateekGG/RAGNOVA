"""
Central configuration — the one place every track reads shared settings
from, so a value like "which embedding model" or "what chunk size" is
never hardcoded independently in three different files.

Usage, from any track's code:

    from src.core.config import settings
    print(settings.OLLAMA_MODEL)

Design rationale: we use a small dataclass loaded once from environment
variables (via python-dotenv reading .env), rather than a settings library
like Pydantic Settings. Same reasoning as schemas.py's dataclass-over-Pydantic
choice — fewer dependencies, and a beginner team can read every line of what
this does. If the project grows past Day 14, Pydantic Settings' automatic
type coercion and validation would be a reasonable upgrade.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Loads variables from a `.env` file in the project root into the process
# environment, if one exists. Does nothing (silently) if .env is missing —
# which is fine, because every setting below has a sensible default.
load_dotenv()


def _get_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.1)

    # Bounds Ollama's num_predict — how many tokens generation is allowed to
    # emit before it's cut off. Not a measured value; a safety cap so a
    # degenerate completion can't hang the CLI/UI indefinitely (Chapter 10).
    LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 512)

    # Bounds Ollama's num_ctx — the context window, in tokens, the model is
    # actually allowed to see. Ollama's own default (2048) is tight once
    # TOP_K (5) retrieved chunks are stuffed into the prompt: each chunk is
    # up to CHUNK_SIZE_WORDS (300) words, roughly ~1.3 tokens/word, so
    # 5 x 300 x 1.3 =~ 1950 tokens of context alone, before the system
    # instructions, provenance headers, the question, and the answer budget
    # (LLM_MAX_TOKENS) are counted. 4096 leaves real headroom; a starting
    # point, not a measured value — revisit if TOP_K or CHUNK_SIZE_WORDS grow.
    LLM_NUM_CTX: int = _get_int("LLM_NUM_CTX", 4096)

    # Cosine-similarity floor a retrieved chunk must clear before it's
    # allowed into the generation prompt at all (ADR-009). Below this, a
    # chunk is closer to noise than to an answer, and stuffing it into
    # context only invites the LLM to hallucinate a connection that isn't
    # there. 0.3 is a starting point picked by inspecting real scores
    # against the Chapter 6 corpus (Chapter 10 §3.x), not a literature
    # value — tune as the gold set (data/README.md) grows.
    MIN_RELEVANCE_SCORE: float = _get_float("MIN_RELEVANCE_SCORE", 0.3)

    # The image_index counterpart of MIN_RELEVANCE_SCORE (Chapter 12).
    # A separate number, not the same 0.3, because of the modality gap
    # (ADR-003/ADR-007): CLIP text-to-image cosine scores for a genuinely
    # correct match sit systematically lower than MiniLM text-to-text
    # scores, so reusing 0.3 here would silently filter out every image.
    # 0.2 is a starting point from CLIP's commonly reported range for real
    # caption matches, NOT measured against this project's corpus (which
    # has no images yet) -- measure it the same way Chapter 10 §3.2 did
    # once data/images/ and the I1-I3 gold rows exist.
    MIN_IMAGE_RELEVANCE_SCORE: float = _get_float("MIN_IMAGE_RELEVANCE_SCORE", 0.2)

    # Reciprocal Rank Fusion constant for merging text_index and
    # image_index results by rank (ADR-007): score = sum(1 / (RRF_K + rank)).
    # 60 is the value from Cormack et al.'s original RRF paper, used as-is.
    RRF_K: int = _get_int("RRF_K", 60)

    # Where the Chapter 11 UI appends one JSON line per piece of user
    # feedback (Chapter 12's human-feedback loop). Committed on purpose,
    # unlike chroma_db/ -- it is evidence for the mid-term report.
    FEEDBACK_LOG_PATH: str = os.environ.get("FEEDBACK_LOG_PATH", "./feedback/feedback.jsonl")

    TEXT_EMBEDDING_MODEL: str = os.environ.get("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CLIP_MODEL: str = os.environ.get("CLIP_MODEL", "ViT-B-32")
    CLIP_PRETRAINED: str = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
    WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "base")

    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

    CHUNK_SIZE_WORDS: int = _get_int("CHUNK_SIZE_WORDS", 300)
    CHUNK_OVERLAP_WORDS: int = _get_int("CHUNK_OVERLAP_WORDS", 50)
    TOP_K: int = _get_int("TOP_K", 5)


# A single shared instance — import this, don't instantiate Settings()
# yourself, so every track genuinely reads the same values.
settings = Settings()


if __name__ == "__main__":
    # Run `python -m src.core.config` to print the resolved settings —
    # useful for confirming your .env is actually being picked up.
    for field_name in Settings.__dataclass_fields__:
        print(f"{field_name} = {getattr(settings, field_name)!r}")
