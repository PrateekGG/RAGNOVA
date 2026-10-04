"""
Write audio transcripts into ChromaDB's text_index — the audio write path
Chapter 10 noted was missing.

Audio goes into text_index, not a third collection: ADR-005 chose
transcription over native audio embeddings, so a transcript chunk is just
text with a timestamp range instead of a page number. It is embedded by
the same MiniLM model as documents (src/core/embeddings.py), which is
what lets one text question find a PDF page and a spoken sentence in the
same search.
"""

from __future__ import annotations

from pathlib import Path

from src.core.embeddings import embed_texts
from src.core.schemas import Chunk, validate_chunk
from src.core.text_normalize import normalize_text
from src.core.vector_store import add_chunks, get_client, get_text_collection
from src.pipelines.audio.ingestion import SUPPORTED_EXTENSIONS


def index_audio_file(path: str | Path, client=None, ingestor=None) -> int:
    """Transcribe one audio file (Chapter 9's AudioIngestor), check every
    chunk against the contract, embed, and upsert into text_index.
    Returns the number of chunks indexed (0 for a file with no speech).

    `ingestor` defaults to a real AudioIngestor, created here and closed
    right after — Whisper is only held in memory while a file is being
    transcribed (Chapter 1 §1.9.2's lazy-loading mitigation for the 8 GB
    RAM budget). Tests pass a fake with a `process_file()` method.

    Raises ValueError if a chunk fails validate_chunk(): an invalid chunk
    in the index would break citation rendering later, far from the cause.
    """
    client = client or get_client()
    collection = get_text_collection(client)
    source = _relative_to_cwd(Path(path))

    if ingestor is None:
        from src.pipelines.audio.ingestion import AudioIngestor

        with AudioIngestor() as real_ingestor:
            raw_chunks = real_ingestor.process_file(source)
    else:
        raw_chunks = ingestor.process_file(source)

    chunks = []
    for raw in raw_chunks:
        chunk = Chunk(
            chunk_id=raw.chunk_id,
            source=source.as_posix(),
            modality="audio",
            text=normalize_text(raw.text),
            embedding_model=raw.embedding_model,
            start_s=raw.start_s,
            end_s=raw.end_s,
        )
        errors = validate_chunk(chunk)
        if errors:
            raise ValueError(f"audio chunk {chunk.chunk_id!r} breaks the contract: {errors}")
        if chunk.text:
            chunks.append(chunk)

    if not chunks:
        return 0
    add_chunks(collection, chunks, embed_texts([c.text for c in chunks]))
    return len(chunks)


def index_audio_directory(directory: str | Path, client=None, ingestor=None) -> int:
    """Index every supported audio file directly under `directory`."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    total = 0
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            total += index_audio_file(path, client=client, ingestor=ingestor)
    return total


def _relative_to_cwd(path: Path) -> Path:
    """Same portability fix as src/pipelines/documents/index.py's."""
    try:
        return path.resolve().relative_to(Path.cwd())
    except ValueError:
        return path
