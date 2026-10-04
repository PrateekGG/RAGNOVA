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
CHUNK_SIZE_WORDS, which changes every chunk_id's page-relative numbering).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.pipelines.audio.index import index_audio_directory
from src.pipelines.documents.index import index_documents_directory
from src.pipelines.images.index import index_images_directory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"


def main() -> None:
    print(f"Indexing {DATA_DIR} into {settings.CHROMA_PERSIST_DIR} ...")
    count = index_documents_directory(DOCS_DIR)
    print(f"Indexed {count} document chunks into text_index.")

    # Each of these loads its model (CLIP / Whisper) only if its folder
    # actually has files in it, so a documents-only corpus stays fast.
    count = index_audio_directory(AUDIO_DIR)
    print(f"Indexed {count} audio transcript chunks into text_index.")
    count = index_images_directory(IMAGES_DIR)
    print(f"Indexed {count} images into image_index.")

    print("\nNext: python scripts/evaluate_retrieval.py, or streamlit run src/app.py")


if __name__ == "__main__":
    main()
