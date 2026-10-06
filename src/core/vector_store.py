"""
ChromaDB connection and collection management — the shared plumbing both
Track A (documents, audio → text_index) and Track B (images → image_index)
write into (ADR-003, ADR-004).

`tests/test_contract.py` (Chapter 5) already proved a chunk survives a raw
ChromaDB round-trip with placeholder zero-vectors. This module is what
Chapter 4 §4.1 meant by "the storage layer is a shared dependency" —
before today, every piece of code that wanted to talk to ChromaDB had to
know its exact API shape itself; from here on, it calls this module
instead.

A `client` is passed explicitly into every function here, rather than
hidden behind a module-level global the way `src.core.config.settings` is.
A database CONNECTION is not the same kind of thing as read-only
configuration: tests need to point at a throwaway temp directory instead
of the real `chroma_db/`, and a hidden global would make that impossible
without monkeypatching. Explicit is worth the one extra argument.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.core.schemas import Chunk, IMAGE_COLLECTION, TEXT_COLLECTION

logger = logging.getLogger(__name__)

# Both collections are created with cosine distance explicitly, rather than
# accepting Chroma's default (squared Euclidean, "l2"). For UNIT-LENGTH
# vectors the two give identical rankings (see embeddings.py's docstring
# and docs/GLOSSARY.md's Cosine similarity entry) — but cosine distance is
# additionally something a human can read directly: Chroma reports cosine
# distance as `1 - cosine_similarity`, a value in [0, 2] where 0 means
# "identical direction," rather than a raw squared-Euclidean number with
# no such intuitive anchor. `search.py` relies on this exact relationship
# to turn a returned distance back into a similarity score.
_COSINE_SPACE = {"hnsw:space": "cosine"}

# Approximate-search effort. ChromaDB's search is HNSW, an APPROXIMATE
# nearest-neighbour index: how hard it looks is set by `search_ef` (how many
# candidates it keeps while searching), `construction_ef` and `M` (how well
# the graph is built). The defaults (search_ef 10, construction_ef 100,
# M 16) are tuned for millions of vectors where speed matters. Measured on
# this project's 623-chunk text index with the version requirements.txt pins
# (chromadb 0.5.23): over 29 gold questions the default index returned a
# DIFFERENT top 5 from exact brute-force search for 9-10 of them, and a wrong
# top 1 once or twice, varying run to run. With the values below it returned
# the exact top 5 for all 29 (chromadb 1.5.9 was already exact at the
# defaults, and stays exact with these). At a few thousand vectors the extra
# effort costs milliseconds; a retrieval system that silently misses the best
# chunk is the expensive failure. These apply when a collection is CREATED:
# an index built before this change must be rebuilt (delete chroma_db/ and
# run scripts/build_index.py) to pick them up.
_HNSW_EFFORT = {"hnsw:search_ef": 200, "hnsw:construction_ef": 400, "hnsw:M": 32}
_COLLECTION_METADATA = {**_COSINE_SPACE, **_HNSW_EFFORT}


def get_client(persist_dir: str | Path | None = None) -> chromadb.ClientAPI:
    """A persistent ChromaDB client. Defaults to settings.CHROMA_PERSIST_DIR
    (the real, on-disk index); pass a `tmp_path`-derived directory in tests
    so nothing ever writes into the real index during a test run.

    Anonymous telemetry is switched off: this project is offline by design
    (Objective O6), so it must not report usage to a third party, and with
    some version combinations the telemetry call also fails noisily."""
    return chromadb.PersistentClient(
        path=str(persist_dir or settings.CHROMA_PERSIST_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _open_collection(client: chromadb.ClientAPI, name: str):
    try:
        return client.get_or_create_collection(name, metadata=_COLLECTION_METADATA)
    except Exception as exc:  # noqa: BLE001 - re-raised below with the likely cause attached
        raise RuntimeError(
            f"Could not open the {name!r} collection ({type(exc).__name__}: {exc}). The most common cause: "
            f"chroma_db/ was written by a different ChromaDB version than the one installed here "
            f"({chromadb.__version__}); an index is not portable across versions (0.5.x and 1.x differ). "
            f"It is rebuildable: delete chroma_db/ and run `python scripts/build_index.py`."
        ) from exc


def get_text_collection(client: chromadb.ClientAPI):
    return _open_collection(client, TEXT_COLLECTION)


def get_image_collection(client: chromadb.ClientAPI):
    return _open_collection(client, IMAGE_COLLECTION)


def add_chunks(collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Write a batch of chunks and their already-computed embedding vectors
    into one collection.

    Deliberately takes embeddings as a separate, explicit argument rather
    than computing them internally — this function has no opinion about
    *which* model produced them (a text collection gets MiniLM vectors, an
    image collection gets CLIP vectors), it only knows how to store
    whatever it's handed, which is exactly the separation of concerns
    ADR-003 requires between "how a modality is embedded" and "where the
    result is stored."

    Does not call `validate_chunk()` itself — by the time a chunk reaches
    this function it should already have passed that check at ingestion
    time (Chapter 6); re-checking it here would be validating the same
    fact twice for no new information.

    Uses `collection.upsert()`, not `collection.add()` — checked directly
    against a real ChromaDB rather than assumed, because the two behave
    very differently on a chunk_id that already exists: `add()` silently
    KEEPS the original stored content and drops the new call entirely (no
    error, no update — confirmed by writing deliberately different text to
    an existing id and reading back the original, unchanged text).
    `upsert()` correctly replaces it. Since this function's whole purpose
    is "make the index reflect the current state of the files," `add()`'s
    silent-no-op behaviour would mean re-running scripts/build_index.py
    after fixing a typo in a source document leaves the index quietly
    serving the pre-fix text forever — exactly the kind of "looks like it
    worked" failure this project's whole testing philosophy exists to
    catch before it becomes a Day-12 surprise.

    Upsert alone cannot REMOVE a chunk, though: a file that now yields fewer
    chunks would keep its old trailing ones. The indexing functions therefore
    call replace_source_chunks() (below), which uses this and then deletes the
    stale ids.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — "
            f"these must be the same length and in the same order."
        )
    if not chunks:
        return

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        record = chunk.to_chroma_record()
        ids.append(record["id"])
        documents.append(record["document"])
        metadatas.append(record["metadata"])

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def replace_source_chunks(
    collection, source: str, chunks: list[Chunk], embeddings: list[list[float]]
) -> int:
    """Make the collection hold exactly `chunks` for file `source`: write them
    (add_chunks' upsert), then delete whichever chunks of that file the index
    still holds under ids that are not in the new set. Returns how many stale
    chunks were removed.

    Why add_chunks() alone was not enough: upsert replaces a chunk with the
    same id, but a re-ingest that now produces FEWER chunks (a shortened
    document, a re-transcription with fewer segments) leaves the old trailing
    chunks in the index forever, still searchable and citable, text the file no
    longer contains. This was a disclosed limitation until 2026-10-05.

    Order matters: the new chunks are written BEFORE the stale ones are
    removed, so there is never a moment when the file has no chunks at all, and
    a failure while embedding or writing (which raises before anything is
    deleted) cannot lose the old content. With an empty `chunks` list (the file
    now yields nothing) every old chunk of `source` is removed.
    """
    existing = set(collection.get(where={"source": source}, include=[])["ids"])
    add_chunks(collection, chunks, embeddings)
    stale = sorted(existing - {chunk.chunk_id for chunk in chunks})
    if stale:
        collection.delete(ids=stale)
    return len(stale)


def prune_missing_sources(collection, root: str | Path) -> dict[str, int]:
    """Delete the chunks of every source file that no longer exists on disk.
    Returns {source: chunks removed}. Relative sources (the portable form every
    pipeline stores) are resolved against `root`, the project root.

    Safety valve: if EVERY source appears to be missing, nothing is deleted. That
    pattern is almost never "all my files were deleted" and almost always "this
    was run from the wrong folder", and wiping the whole index on a path mistake
    is the one outcome worth refusing to risk; it logs a warning instead.
    """
    root = Path(root)
    records = collection.get(include=["metadatas"])
    sources = {m["source"] for m in records["metadatas"]}
    missing = [src for src in sorted(sources)
               if not (Path(src) if Path(src).is_absolute() else root / src).exists()]
    if sources and len(missing) == len(sources):
        logger.warning("prune_missing_sources: all %d sources look missing from %s; "
                       "refusing to delete anything (wrong working directory?)", len(sources), root)
        return {}
    removed: dict[str, int] = {}
    for src in missing:
        ids = collection.get(where={"source": src}, include=[])["ids"]
        collection.delete(ids=ids)
        removed[src] = len(ids)
    return removed

