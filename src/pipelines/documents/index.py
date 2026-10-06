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
from src.core.vector_store import get_client, get_text_collection, replace_source_chunks
from src.pipelines.documents.ingest import SUPPORTED_EXTENSIONS as _SUPPORTED_EXTENSIONS
from src.pipelines.documents.ingest import ingest_document


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

    relative = _relative_to_cwd(Path(path))
    # Ingest and embed FIRST: if the file is corrupt or embedding fails, this
    # raises before anything in the index is touched, so the old chunks survive.
    chunks = ingest_document(relative)
    vectors = embed_texts([c.text for c in chunks]) if chunks else []
    # Replace, not just add: a re-ingest that now yields fewer chunks (a shortened
    # file, or one whose pages all lost their text, Chapter 6 §1.4) must not leave
    # the old trailing chunks behind. replace_source_chunks() documents why.
    replace_source_chunks(collection, relative.as_posix(), chunks, vectors)
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
