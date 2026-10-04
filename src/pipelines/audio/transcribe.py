"""
Turn a spoken *question* into text — the mic / audio-query input of the
Chapter 11 UI.

Deliberately not AudioIngestor: that class chunks a long recording into
300-word, overlapping, timestamped pieces for the index. A spoken question
is one short sentence that just needs to become a string, which is then
asked exactly like a typed question.
"""

from __future__ import annotations

from pathlib import Path

from src.core.config import settings


def transcribe_query(path: str | Path, model=None) -> str:
    """Return the full transcript of a short audio clip as one string.

    Whisper is loaded for this call and dropped afterwards (Chapter 1
    §1.9.2's lazy loading): the chat app keeps the embedding model and the
    LLM warm, and holding Whisper too, all the time, is what would push an
    8 GB laptop over. Pass `model` to reuse one (tests pass a fake).
    """
    owns_model = model is None
    if owns_model:
        from faster_whisper import WhisperModel

        model = WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    try:
        segments, _info = model.transcribe(str(path), vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    finally:
        if owns_model:
            del model
