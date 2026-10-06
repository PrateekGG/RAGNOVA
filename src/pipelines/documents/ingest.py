"""
Track A's top-level document-ingestion entry point: one file in, real
`Chunk` objects out.

Named `ingest_document`, not `ingest_pdf` — Chapter 4 §4.1's produces/
consumes table already named Track B's entry points `ingest_image(file)`
and `ingest_audio(file)`. Matching that convention here means all three
tracks' top-level functions read the same way without anyone having to
agree on it in a meeting.

This module is deliberately thin: it owns dispatch-by-extension and
`Chunk` assembly, and delegates the actual work to pdf_parser.py,
docx_parser.py, and chunker.py. Each of those is independently testable;
this file is what wires them together the way a real caller would.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.core.config import settings
from src.core.schemas import Chunk
from src.core.text_normalize import normalize_text
from src.pipelines.documents.chunker import chunk_page_text
from src.pipelines.documents.docx_parser import extract_docx_pages
from src.pipelines.documents.pdf_parser import extract_pdf_pages

# The zero-extractable-text warning below prints a real em-dash. On an
# unconfigured Windows console this renders as "?" instead — the exact bug
# Chapter 5's verify_setup.py already found and fixed for stdout (Ch5 §6.3's
# troubleshooting table). That fix didn't cover stderr, which is what this
# warning actually uses, so it slipped through in Chapter 6 unnoticed until
# Chapter 7's evaluate_retrieval.py hit the identical symptom on stdout.
# `hasattr` guards against environments where stderr has been replaced with
# something that doesn't support `.reconfigure()` — pytest's own output
# capturing is exactly such an environment, and this import runs under it
# on every test collection.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# extension -> (modality, per-page text extractor). Adding a third document
# type later (e.g. .txt, .md) means adding one entry here, not branching
# logic inside ingest_document() itself.
_PARSERS = {
    ".pdf": ("pdf", extract_pdf_pages),
    ".docx": ("docx", extract_docx_pages),
}

# The file types this pipeline can read. Exported so the indexer and the UI's
# upload gate read the same list instead of each keeping a copy.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_PARSERS)


def ingest_document(path: str | Path) -> list[Chunk]:
    """Parse, normalize, and chunk one PDF or DOCX file into real `Chunk`s.

    Every chunk returned here satisfies `validate_chunk()` with zero errors
    — that is the whole point of this chapter (Chapter 5 built the
    contract; this is the first code that has to actually honour it against
    real, not hand-written, data).

    `path` is expected relative to the project root (e.g.
    "data/documents/notice.pdf"), matching every existing fixture's
    convention in tests/test_contract.py and Ch5's "run pytest from the
    project root" assumption — no path-root-detection magic is attempted.

    Raises:
        ValueError: `path`'s extension isn't one this pipeline supports.
            Checked before touching the filesystem at all — cheap, and a
            clearer error than whatever the parser library would raise on
            a file type it wasn't designed for.
        FileNotFoundError: `path` doesn't exist. Re-raised with the path
            included, rather than left as a lower-level library exception.
        Whatever the underlying parser raises on a genuinely corrupt or
            unreadable file — deliberately not caught. There is no sensible
            degraded behaviour for a file that cannot be opened at all, so
            this fails loudly instead of returning a misleadingly empty
            chunk list.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _PARSERS:
        raise ValueError(
            f"Unsupported file extension {ext!r} for {path} — "
            f"ingest_document() only supports {sorted(_PARSERS)}."
        )
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    modality, extract_pages = _PARSERS[ext]
    source = path.as_posix()  # forward slashes even on Windows, so `source`
                               # is identical across every team member's OS
    id_prefix = path.name.replace(".", "_")  # NOT Path.stem — see note below

    chunks: list[Chunk] = []
    for page_num, raw_text in extract_pages(path):
        clean = normalize_text(raw_text)
        if not clean:
            # A page can genuinely have nothing extractable (most often: a
            # scanned page with no text layer — see pdf_parser.py). Warn
            # loudly, but don't abort the rest of the document over it —
            # one bad page in a 20-page file shouldn't cost the other 19.
            print(
                f"[ingest_document] WARNING: {source} page {page_num} has "
                f"no extractable text — skipping.",
                file=sys.stderr,
            )
            continue

        windows = chunk_page_text(
            clean, settings.CHUNK_SIZE_WORDS, settings.CHUNK_OVERLAP_WORDS
        )
        for seq, window in enumerate(windows, start=1):
            # chunk_id convention (Ch4 §4.3): "<file>__p<page>__c<seq>".
            # `path.name.replace(".", "_")`, not `path.stem`, because
            # `.stem` drops the extension — "report.pdf" and "report.docx"
            # would otherwise collide on the same id prefix. Sequence
            # numbers reset to c001 at the start of every page: page 2's
            # ids then stay stable even if page 1 is edited later, and
            # "3rd chunk of page 2" reads as c003 without needing to
            # cross-reference how many chunks page 1 had.
            chunks.append(Chunk(
                chunk_id=f"{id_prefix}__p{page_num}__c{seq:03d}",
                source=source,
                modality=modality,
                text=window,
                embedding_model=settings.TEXT_EMBEDDING_MODEL,
                page=page_num,
            ))

    return chunks


if __name__ == "__main__":
    # `python -m src.pipelines.documents.ingest <path>` — ingests one real
    # file and prints every resulting chunk's id and word count. The
    # fastest way to eyeball "did this actually work" on a new file.
    if len(sys.argv) != 2:
        print("Usage: python -m src.pipelines.documents.ingest <path-to-pdf-or-docx>")
        sys.exit(1)
    for chunk in ingest_document(sys.argv[1]):
        print(f"{chunk.chunk_id}  (page {chunk.page}, {len(chunk.text.split())} words)")
