"""
Build the real ChromaDB indexes from data/: documents and audio
transcripts into text_index, images into image_index (Chapter 12 added
the last two — before that, this script indexed data/documents/ only).

Run with:  python scripts/build_index.py

Safe to re-run any time data/documents/ changes: index_documents_directory()
writes via ChromaDB's upsert (add-or-replace-by-id — see vector_store.py's
add_chunks() docstring for why plain add() was tried first and rejected),
so re-running this after editing a source file correctly refreshes that
file's chunks in place rather than either erroring or silently keeping
stale content. A NEW file just adds new chunks. Deleting chroma_db/ first
is only needed for a genuinely from-scratch rebuild (e.g. after changing
CHUNK_SIZE_WORDS, which changes every chunk_id's page-relative numbering, or
the HNSW settings, which apply only when a collection is created).

Re-running keeps the index in step with the files: a file that now yields
fewer chunks has its stale ones removed, and chunks of deleted files are
pruned (vector_store.replace_source_chunks / prune_missing_sources).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.vector_store import (
    get_client,
    get_image_collection,
    get_image_text_collection,
    get_text_collection,
    prune_missing_sources,
)
from src.pipelines.audio.index import index_audio_directory
from src.pipelines.documents.index import index_documents_directory
from src.pipelines.images.index import index_images_directory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"


def refuse_unresolved_lfs_pointers() -> None:
    """The downloaded corpus files live in Git LFS (data/SOURCES.md). If
    `git lfs install` was never run, a clone gets ~130-byte text pointers in
    their place, and every parser then fails with a confusing error about a
    corrupt PDF or image. Say what is actually wrong instead."""
    marker = b"version https://git-lfs.github.com/spec/v1"
    pointers = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.stat().st_size < 300 and p.read_bytes().startswith(marker)
    ]
    if pointers:
        names = ", ".join(p.name for p in pointers[:5])
        sys.exit(
            f"{len(pointers)} corpus file(s) are Git LFS pointers, not real files (e.g. {names}).\n"
            "Run:  git lfs install  &&  git lfs pull   and then re-run this script."
        )


def main() -> None:
    refuse_unresolved_lfs_pointers()
    print(f"Indexing {DATA_DIR} into {settings.CHROMA_PERSIST_DIR} ...")
    count = index_documents_directory(DOCS_DIR)
    print(f"Indexed {count} document chunks into text_index.")

    # Each of these loads its model (CLIP / Whisper) only if its folder
    # actually has files in it, so a documents-only corpus stays fast.
    count = index_audio_directory(AUDIO_DIR)
    print(f"Indexed {count} audio transcript chunks into text_index.")
    count = index_images_directory(IMAGES_DIR)
    print(f"Indexed {count} images into image_index (and the text read from them into image_text_index).")

    # Chunks of files that have been deleted since the last build would otherwise
    # stay searchable and citable (the index only ever added or replaced).
    client = get_client()
    for name, collection in (
        ("text_index", get_text_collection(client)),
        ("image_index", get_image_collection(client)),
        ("image_text_index", get_image_text_collection(client)),
    ):
        removed = prune_missing_sources(collection, DATA_DIR.parent)
        for source, n in removed.items():
            print(f"Pruned {n} chunk(s) of {source} from {name}: the file no longer exists.")

    print("\nNext: python scripts/evaluate_retrieval.py, or streamlit run src/app.py")


if __name__ == "__main__":
    main()
