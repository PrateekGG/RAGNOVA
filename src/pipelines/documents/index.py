"""
Build the real text index: walk a directory of documents, run each one
through Chapter 6's ingestion pipeline, embed every chunk, and write the
result into ChromaDB's text_index collection.

This is the function that turns "we can parse a PDF into Chunk objects"
(Chapter 6) into "the corpus is actually searchable" (this chapter) —
scripts/build_index.py is the thin, runnable script that calls this
against the real data/documents/ directory.
"""

from __future__ import annotations

from pathlib import Path

from src.core.embeddings import embed_texts
from src.core.vector_store import add_chunks, get_client, get_text_collection
from src.pipelines.documents.ingest import ingest_document

# Kept in sync with ingest.py's own _PARSERS — duplicating just the set of
# supported extensions here (not the parser functions themselves) avoids
# this module needing to know anything about *how* a file is parsed.
_SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def index_documents_directory(directory: str | Path, client=None) -> int:
    """Ingest and embed every supported file directly under `directory`,
    writing all resulting chunks into text_index. Returns the total number
    of chunks indexed.

    Not recursive on purpose — data/documents/ is a flat folder (see
    data/README.md's own layout), and silently descending into
    subdirectories a team member created for their own organisation could
    index files nobody meant to include.

    `client` defaults to the real, on-disk ChromaDB (settings.CHROMA_PERSIST_DIR)
    — pass a throwaway client in tests, exactly as vector_store.py's own
    docstring describes.
    """
    client = client or get_client()
    directory = Path(directory)

    total_chunks = 0
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        total_chunks += index_document_file(path, client=client)

    return total_chunks


def index_document_file(path: str | Path, client=None) -> int:
    """Ingest, embed and upsert ONE document; returns its chunk count.
    Split out of index_documents_directory() in Chapter 11 so the UI can
    index a single uploaded file without re-embedding the whole folder."""
    client = client or get_client()
    collection = get_text_collection(client)

    chunks = ingest_document(_relative_to_cwd(Path(path)))
    if not chunks:
        # A real, if unlikely, case: every page of this file produced
        # zero extractable text (Chapter 6 §1.4). Nothing to embed or
        # store, and not an error.
        return 0

    vectors = embed_texts([c.text for c in chunks])
    add_chunks(collection, chunks, vectors)
    return len(chunks)


def _relative_to_cwd(path: Path) -> Path:
    """ingest_document() stores whatever it's given, verbatim, as each
    chunk's `source` (Chapter 6) — so if `directory` above was constructed
    as an absolute path (e.g. via `Path(__file__).resolve()`, a completely
    natural thing for a caller to do), every resulting citation would bake
    in this one machine's exact checkout location, breaking the moment a
    teammate with a different checkout path re-indexes the same file. This
    is a real bug this function shipped with initially — caught by
    scripts/evaluate_retrieval.py's T1 gold question genuinely failing to
    match on `source`, not by inspection.

    Converting here, once, means index_documents_directory() produces a
    correct, portable `source` regardless of what path form its own caller
    happened to pass in — rather than relying on every future caller to
    independently remember to pass a relative path.
    """
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path  # already relative, or genuinely outside the cwd
