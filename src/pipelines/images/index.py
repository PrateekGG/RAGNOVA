"""
Write images into ChromaDB's image_index — the write path Chapter 10 noted
was missing ("image_index has no write path into ChromaDB yet").

Chapter 8's ImageIngestionPipeline already does the hard work (load, OCR,
CLIP-embed) but stores results in an in-memory ChunkStore that disappears
when the process ends. This module is the Chapter 12 integration glue:
run that pipeline, then hand its chunks to the same add_chunks() the text
side uses (src/core/vector_store.py), into the image collection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.core.config import settings
from src.core.schemas import Chunk
from src.core.text_normalize import normalize_text
from src.core.vector_store import (
    get_client,
    get_image_collection,
    get_image_text_collection,
    replace_source_chunks,
)
from src.pipelines.images.ingest import SUPPORTED_EXTENSIONS, ImageIngestionPipeline
from src.pipelines.images.models import ImageChunk


def index_image_files(paths: list[str | Path], client=None, pipeline=None, embed_texts_fn=None) -> int:
    """Ingest the given image files and upsert them into image_index, and the
    MiniLM vectors of the text read from them into image_text_index (ADR-014).
    Returns how many images were indexed. Unreadable files are skipped by the
    pipeline itself (Chapter 8's per-file try/except), not here.

    `pipeline` defaults to a real ImageIngestionPipeline (Tesseract + CLIP);
    tests pass one built with a fake embedder so they run without
    downloading CLIP's weights. `embed_texts_fn` defaults to MiniLM
    (src.core.embeddings.embed_texts) and is injected the same way.
    """
    if not paths:
        return 0
    client = client or get_client()
    collection = get_image_collection(client)
    pipeline = pipeline or ImageIngestionPipeline()

    image_chunks = pipeline.ingest_batch([Path(p) for p in paths], store=False)
    # An image with no vector can't be searched at all — only possible
    # when the pipeline was built with embedding_enabled=False.
    image_chunks = [c for c in image_chunks if c.embedding is not None]
    if not image_chunks:
        return 0

    chunks = [_to_index_chunk(c) for c in image_chunks]
    # One replace per image file, so a re-index also drops any chunk that file no
    # longer produces (vector_store.replace_source_chunks).
    by_source: dict[str, tuple[list[Chunk], list[list[float]]]] = {}
    for chunk, image_chunk in zip(chunks, image_chunks):
        group_chunks, group_vectors = by_source.setdefault(chunk.source, ([], []))
        group_chunks.append(chunk)
        group_vectors.append(image_chunk.embedding)
    for source, (group_chunks, group_vectors) in by_source.items():
        replace_source_chunks(collection, source, group_chunks, group_vectors)
    index_ocr_text(chunks, client=client, embed_texts_fn=embed_texts_fn)
    return len(chunks)


def index_ocr_text(chunks: list[Chunk], client=None, embed_texts_fn=None) -> int:
    """Write the OCR text of already-built image chunks into image_text_index
    (ADR-014): one MiniLM vector per image that has readable text, under the
    SAME chunk id as its image_index chunk so the two searches can be fused.

    Every source in `chunks` is replaced, including an image whose text is now
    empty, so a re-index whose OCR changed (or found nothing) cannot leave the
    old text searchable (vector_store.replace_source_chunks). An image with no
    readable text gets no vector here: a photo is found by CLIP or not at all.
    Returns how many images got a text vector.
    """
    client = client or get_client()
    collection = get_image_text_collection(client)
    with_text = [c for c in chunks if c.text.strip()]
    vectors: list[list[float]] = []
    if with_text:
        if embed_texts_fn is None:
            from src.core.embeddings import embed_texts as embed_texts_fn
        vectors = embed_texts_fn([c.text for c in with_text])
    by_source: dict[str, tuple[list[Chunk], list[list[float]]]] = {c.source: ([], []) for c in chunks}
    for chunk, vector in zip(with_text, vectors):
        group_chunks, group_vectors = by_source[chunk.source]
        group_chunks.append(replace(chunk, embedding_model=settings.TEXT_EMBEDDING_MODEL))
        group_vectors.append(vector)
    for source, (group_chunks, group_vectors) in by_source.items():
        replace_source_chunks(collection, source, group_chunks, group_vectors)
    return len(with_text)


def index_images_directory(
    directory: str | Path,
    client=None,
    pipeline=None,
    recursive: bool = False,
) -> int:
    """Index every supported image under `directory`.

    Parameters
    ----------
    directory:
        Root directory to search.
    client:
        ChromaDB client instance.
    pipeline:
        ImageIngestionPipeline instance.
    recursive:
        Whether to search subdirectories too. Defaults to False (flat):
        scripts/build_index.py and the UI index data/images/ this way, and
        index_documents_directory() / index_audio_directory() are flat for
        the same reason — one folder per modality, nothing nested is
        indexed by accident. The image CLI opts in with recursive=True,
        which is the behaviour its old ingest_directory() call had.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    pattern = "**/*" if recursive else "*"
    paths = [
        p for p in sorted(directory.glob(pattern))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return index_image_files(paths, client=client, pipeline=pipeline)


def _to_index_chunk(image_chunk: ImageChunk) -> Chunk:
    """Turn Chapter 8's ImageChunk into a plain, contract-shaped Chunk.

    Two real fixes happen here, both found while wiring this up:

    1. `source`. ImageIngestionPipeline stores the *absolute* path
       (`Path(...).resolve()`), e.g. "/home/alice/RAGNova/data/images/x.png".
       That bakes one person's checkout folder into the citation — the
       exact bug Chapter 7's _relative_to_cwd() already fixed for
       documents. Made relative to the project root here, the same way.

    2. OCR text. Tesseract output is full of stray newlines and double
       spaces. normalize_text() is the shared cleanup Chapter 6 made for
       exactly this ("Chapter 8's OCR output ... need[s] the identical
       cleanup" — docs/GLOSSARY.md, Normalization (of text)).
    """
    return Chunk(
        chunk_id=image_chunk.chunk_id,
        source=_relative_to_cwd(image_chunk.source),
        modality="image",
        text=normalize_text(image_chunk.text or ""),
        embedding_model=image_chunk.embedding_model,
    )


def _relative_to_cwd(source: str) -> str:
    try:
        return Path(source).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return source
