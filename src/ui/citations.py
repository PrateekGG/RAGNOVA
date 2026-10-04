"""
Turn the RAG core's citation list (real Chunk objects) into what the UI
draws for each "[n]" — pure data, no Streamlit, so it's unit-testable.

The rules each branch encodes come from decisions already on file:
  - ADR-008: a DOCX page number is only as good as the file's explicit page
    breaks, so it's shown with a warning a PDF page never needs.
  - ADR-003 / ADR-007: text and image similarity scores are on different
    scales, so no score is shown at all — the list order (which is the
    rank-merged order) is the only ranking the user sees.
  - Chapter 10 §2.3: every label comes from Chunk metadata, never from the
    model's answer text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.schemas import Chunk
from src.pipelines.rag.prompt import format_provenance

DOCX_PAGE_NOTE = (
    "Word files don't store page numbers. This page is counted from the "
    "document's manual page breaks only, so it can be off (ADR-008)."
)

_MODALITY_LABELS = {"pdf": "PDF", "docx": "Word", "image": "Image", "audio": "Audio"}


@dataclass
class CitationView:
    number: int
    title: str            # e.g. "[2] notice.pdf, page 2"
    modality_label: str   # "PDF" / "Word" / "Image" / "Audio"
    excerpt: str          # the chunk text the model was shown
    file_path: str        # the original file, for download / display
    file_exists: bool
    note: str | None = None          # a caveat to show under the excerpt
    audio_start_s: float | None = None


def citation_views(chunks: list[Chunk]) -> list[CitationView]:
    """One CitationView per chunk, numbered from 1 in the same order the
    prompt numbered them (so "[n]" in the answer matches view n)."""
    views = []
    for number, chunk in enumerate(chunks, start=1):
        note = DOCX_PAGE_NOTE if chunk.modality == "docx" else None
        excerpt = chunk.text
        if chunk.modality == "image" and not chunk.text.strip():
            excerpt = "(No readable text in this image.)"
        views.append(
            CitationView(
                number=number,
                title=f"[{number}] {_short_provenance(chunk)}",
                modality_label=_MODALITY_LABELS.get(chunk.modality, chunk.modality),
                excerpt=excerpt,
                file_path=chunk.source,
                file_exists=Path(chunk.source).is_file(),
                note=note,
                audio_start_s=chunk.start_s if chunk.modality == "audio" else None,
            )
        )
    return views


def _short_provenance(chunk: Chunk) -> str:
    """format_provenance() with just the file name instead of the full
    relative path — the full path is still one click away (download)."""
    full = format_provenance(chunk)
    return full.replace(chunk.source, Path(chunk.source).name, 1)
