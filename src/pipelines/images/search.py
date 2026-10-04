"""
Search image_index: the image-side twin of Chapter 7's search_text().

Two ways in, both landing in CLIP's shared space (ADR-003):
  - a text query  -> CLIP's *text* encoder  (text -> image search, "email screenshot")
  - an image query -> CLIP's *image* encoder (image -> image search, "find ones like this")

Never MiniLM: image_index holds 512-d CLIP vectors, and a 384-d MiniLM
query vector cannot even be compared against them (ADR-003's first hard
constraint). That is also why this function builds its own query vector
instead of letting ChromaDB embed it (same `query_embeddings=` rule
search_text() follows, same reason).
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from src.core.config import settings
from src.core.schemas import Chunk
from src.core.vector_store import get_client, get_image_collection
from src.pipelines.documents.search import _chunk_from_result
from src.pipelines.images.embedding import OpenCLIPEmbedder
from src.pipelines.images.models import ImageIngestionConfig

# Loaded lazily and cached per process, same pattern as
# src/core/embeddings.py's _model: CLIP's weights are ~600 MB, and the UI
# asks a question every few seconds — reloading them per query would make
# every answer wait on disk I/O for no reason.
_embedder: Optional[OpenCLIPEmbedder] = None


def get_clip_embedder() -> OpenCLIPEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OpenCLIPEmbedder(
            ImageIngestionConfig(model_name=settings.CLIP_MODEL, pretrained=settings.CLIP_PRETRAINED)
        )
    return _embedder


def search_images(
    query_text: str | None = None,
    query_image: Image.Image | None = None,
    top_k: int | None = None,
    client=None,
    embedder=None,
) -> list[Chunk]:
    """Return the `top_k` image chunks nearest to the query, best first,
    with `.score` set to cosine similarity (same `1 - distance` rule as
    search_text()).

    Pass exactly one of `query_text` / `query_image`. Returns [] when
    image_index is empty — the normal state until someone indexes
    data/images/ — rather than asking ChromaDB for `top_k` results out of
    zero and relying on how it happens to handle that.
    """
    if (query_text is None) == (query_image is None):
        raise ValueError("pass exactly one of query_text or query_image")

    top_k = top_k or settings.TOP_K
    client = client or get_client()
    collection = get_image_collection(client)
    count = collection.count()
    if count == 0:
        return []

    embedder = embedder or get_clip_embedder()
    if query_text is not None:
        query_vector = embedder.embed_text(query_text)
    else:
        query_vector = embedder.embed_image(query_image)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        _chunk_from_result(chunk_id, text, metadata, distance)
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        )
    ]
