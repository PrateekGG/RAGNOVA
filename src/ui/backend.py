"""
Every action the chat app takes that isn't drawing on screen: saving an
uploaded file, indexing it, turning a voice clip or a photo into query
text, checking what's in the index.

Kept out of the Streamlit page (src/app.py) for two reasons. First,
testability: every function here runs under plain pytest, with fakes in
place of Whisper, Tesseract and CLIP. Second, Chapter 4's rule for Track C:
the UI calls Track A/B's entry points, it does not re-implement them —
every function below is a thin wrapper over one of theirs.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from PIL import Image

from src.core.config import settings

DOCUMENT_EXTS = {".pdf", ".docx"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

# Where an uploaded file lands, by kind — the same flat folders
# data/README.md defines and scripts/build_index.py reads.
DATA_DIRS = {
    "document": Path("data/documents"),
    "image": Path("data/images"),
    "audio": Path("data/audio"),
}


def kind_of(filename: str) -> str | None:
    """"document" / "image" / "audio", or None for an unsupported file."""
    ext = Path(filename).suffix.lower()
    if ext in DOCUMENT_EXTS:
        return "document"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def safe_filename(filename: str) -> str:
    """The uploaded name, reduced to letters, digits, dot, dash and
    underscore. A browser hands over whatever the user's file was called —
    including "../" or characters Windows can't store — and this name
    becomes a path on disk and a citation label."""
    name = Path(filename).name
    stem = re.sub(r"[^\w\-]+", "_", Path(name).stem).strip("_") or "upload"
    return f"{stem}{Path(name).suffix.lower()}"


def save_upload(filename: str, data: bytes, data_root: Path | None = None) -> Path:
    """Write an uploaded file into its data/ folder and return the path,
    relative to the project root (so it becomes a portable citation).
    Raises ValueError for an unsupported type."""
    kind = kind_of(filename)
    if kind is None:
        raise ValueError(f"unsupported file type: {filename}")
    folder = (data_root / DATA_DIRS[kind].relative_to("data")) if data_root else DATA_DIRS[kind]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / safe_filename(filename)
    path.write_bytes(data)
    return path


def index_file(path: Path, client=None) -> int:
    """Index one saved file into the right collection; returns how many
    chunks it produced. Heavy imports stay inside each branch, so indexing
    a PDF never loads CLIP or Whisper."""
    kind = kind_of(path.name)
    if kind == "document":
        from src.pipelines.documents.index import index_document_file
        return index_document_file(path, client=client)
    if kind == "image":
        from src.pipelines.images.index import index_image_files
        return index_image_files([path], client=client)
    if kind == "audio":
        from src.pipelines.audio.index import index_audio_file
        return index_audio_file(path, client=client)
    raise ValueError(f"unsupported file type: {path.name}")


def index_counts(client=None) -> dict[str, int]:
    """How many chunks each collection holds — shown in the sidebar, and
    what decides whether image search is worth running at all."""
    from src.core.vector_store import get_client, get_image_collection, get_text_collection

    client = client or get_client()
    return {
        "text": get_text_collection(client).count(),
        "image": get_image_collection(client).count(),
    }


def transcribe_audio_bytes(data: bytes, suffix: str = ".wav") -> str:
    """Speech -> question text, for the mic button and audio-clip queries.
    faster-whisper reads from a file path, so the bytes go to a temp file
    that is deleted afterwards."""
    from src.pipelines.audio.transcribe import transcribe_query

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"query{suffix}"
        path.write_bytes(data)
        return transcribe_query(path)


def ocr_image(image: Image.Image) -> str:
    """Text printed inside a query image (a screenshot, a photographed
    notice), used as extra query text for image -> document search.
    Returns "" when Tesseract isn't installed rather than failing the
    whole question — the image-to-image half still works without it."""
    from src.core.text_normalize import normalize_text
    from src.pipelines.images.ocr import TesseractOCREngine

    engine = TesseractOCREngine()
    if not engine.is_available:
        return ""
    return normalize_text(engine.extract_text(image).text)


def ollama_ready() -> bool:
    """Is `ollama serve` reachable? Checked once per page load so the app
    can say so up front, instead of failing on the first question."""
    import ollama

    try:
        ollama.Client(host=settings.OLLAMA_HOST).list()
        return True
    except Exception:
        return False
