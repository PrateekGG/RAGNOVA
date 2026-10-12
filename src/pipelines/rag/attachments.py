"""
An image the user attaches to a question, as something the model can answer FROM.

Until 2026-10-10 an attached image was only a search key: CLIP found the indexed
images that looked like it, and its OCR text was glued onto the question to find
documents. The model was never shown what the picture said, so a screenshot that
was not already in the corpus got "I don't have enough information" even when
Tesseract had read every word of it. (Reproduced on two screenshots, 540 and
3539 characters of readable text, both refused.)

An attachment is now one more numbered context block, placed first and labelled
as the user's own attachment, so the answer can use and cite it. It is never
filtered: the user put it there on purpose. The model is still text-only, so
what an attachment contributes is the TEXT in the picture; one with none says so
(see prompt._context_text) instead of inviting an invented description.
"""

from __future__ import annotations

from src.core.schemas import Chunk
from src.pipelines.rag.prompt import ATTACHMENT_ID_PREFIX

# A screenshot of a busy page reads as thousands of characters. The prompt has a
# fixed context window (settings.LLM_NUM_CTX) shared with the retrieved chunks, so
# an attachment gets a fixed share of it: 400 words is about 520 tokens of 4096.
ATTACHMENT_MAX_WORDS = 400


def image_attachment(name: str, text: str, index: int = 1) -> Chunk:
    """The attached image `name` with the `text` read from it (may be empty)."""
    words = text.split()
    kept = " ".join(words[:ATTACHMENT_MAX_WORDS])
    if len(words) > ATTACHMENT_MAX_WORDS:
        kept += " ..."
    return Chunk(
        chunk_id=f"{ATTACHMENT_ID_PREFIX}{index}",
        source=f"attached image ({name})",
        modality="image",
        text=kept,
        embedding_model="none",
        score=None,
    )


def is_attachment(chunk: Chunk) -> bool:
    return chunk.chunk_id.startswith(ATTACHMENT_ID_PREFIX)
