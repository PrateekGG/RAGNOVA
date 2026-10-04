"""Prompt construction — the concrete implementation of ADR-001's mandate
that the model answer only from retrieved evidence and cite it.

Deliberately a pure function: no I/O, no imports of `src.core.llm` or
`src.core.vector_store`. Given the same `query` and `chunks`, it always
returns the same string — which is what makes it independently unit
testable without ChromaDB or Ollama running.
"""

from __future__ import annotations

from src.core.schemas import Chunk

_SYSTEM_INSTRUCTIONS = (
    "You answer questions using ONLY the numbered context below, taken from "
    "the user's own indexed files. Follow these rules exactly:\n"
    "1. Base your answer only on the context given. Never use outside "
    "knowledge, even if you happen to know the answer.\n"
    "2. Cite every claim with the matching bracketed number, e.g. [1] or "
    "[2], right after the sentence it supports.\n"
    "3. If the context does not contain the answer, say plainly: "
    '"I don\'t have enough information in the indexed documents to answer '
    'that." Do not guess, and do not cite a number if you are not using it.'
)

_NO_CONTEXT_NOTICE = "(No relevant context was found for this question.)"


def format_provenance(chunk: Chunk) -> str:
    """One-line "where this came from" description for a chunk — used both
    to label a numbered context block in the prompt, and (by
    scripts/ask.py) to render the human-facing "Sources:" footer.

    Documents cite a page (ADR-008's honest-best-effort DOCX page still
    applies here — this function doesn't distinguish pdf from docx). Audio
    would cite a timestamp range; no audio chunk exists in the vector store
    yet (Ch9 isn't wired in), but the branch is written now so this
    function doesn't need to change the day it is.
    """
    if chunk.modality in ("pdf", "docx") and chunk.page is not None:
        return f"{chunk.source}, page {chunk.page}"
    if chunk.modality == "audio" and chunk.start_s is not None and chunk.end_s is not None:
        return f"{chunk.source}, {chunk.start_s:.0f}s–{chunk.end_s:.0f}s"
    if chunk.modality == "image":
        return f"{chunk.source} (image)"
    return chunk.source


# What the model is shown for an image whose OCR found no text. The LLM
# reads text only, so it can't look at the picture — saying so plainly
# stops it from inventing a description, while still letting the image
# appear as a numbered, citable source ("see [3]").
_IMAGE_WITHOUT_TEXT = "(An image matching the question. No readable text was found in it.)"


def _context_text(chunk: Chunk) -> str:
    if chunk.modality == "image" and not chunk.text.strip():
        return _IMAGE_WITHOUT_TEXT
    if chunk.modality == "image":
        return f"Text read from the image (OCR): {chunk.text}"
    return chunk.text


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build the full prompt sent to the LLM: system instructions, numbered
    context blocks (best match first, matching `chunks`' own order), then
    the question.

    `chunks` is expected to already be filtered to ones worth showing the
    model (see settings.MIN_RELEVANCE_SCORE in answer.py) — this function
    itself applies no threshold, so an empty list here always means
    "nothing survived filtering," not "nothing was retrieved."
    """
    if chunks:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(f"[{i}] {format_provenance(chunk)}\n{_context_text(chunk)}")
        context_section = "\n\n".join(blocks)
    else:
        context_section = _NO_CONTEXT_NOTICE

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_section}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
