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
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.core.config import settings

# What the upload gate accepts is exactly what each pipeline can read: derived
# from the pipelines' own lists, never copied (the audio copy had drifted to 6
# formats while the pipeline read 14). Only light modules are imported here, so
# the UI still starts fast and never loads CLIP or Whisper to check a file name.
from src.pipelines.audio.formats import SUPPORTED_EXTENSIONS as AUDIO_EXTS
from src.pipelines.documents.ingest import SUPPORTED_EXTENSIONS as DOCUMENT_EXTS
from src.pipelines.images.ingest import SUPPORTED_EXTENSIONS as _IMAGE_EXTENSIONS

IMAGE_EXTS = frozenset(_IMAGE_EXTENSIONS)

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


def list_sources(client=None) -> list[str]:
    """Every file that has something in the index, as the `source` path the index
    stores (sorted), for the "answer from these files only" picker. Metadata only:
    no vectors are read, so it is cheap enough to run on every page rerun."""
    from src.core.vector_store import get_client, get_image_collection, get_text_collection

    client = client or get_client()
    found: set[str] = set()
    for collection in (get_text_collection(client), get_image_collection(client)):
        if collection.count():
            found.update(m["source"] for m in collection.get(include=["metadatas"])["metadatas"])
    return sorted(found)


def sources_for_upload(saved: Path, listed: list[str]) -> list[str]:
    """The index entry for a file that was just saved and indexed. The index stores
    the path relative to the folder the app was started from, which this does not
    assume: it matches on the file name, so an upload still selects itself when the
    app was started from somewhere unexpected. [] when nothing matches (a file
    that produced no chunks is not in the index, and there is nothing to select)."""
    return [s for s in listed if Path(s).name == saved.name]


@dataclass
class UploadResult:
    ok: bool
    message: str                # what the page shows the user
    focus: list[str]            # the index entries to answer from next ([] = leave the scope alone)


def add_to_corpus(filename: str, data: bytes, data_root: Path | None = None) -> UploadResult:
    """Save an uploaded file, index it, and say what happened.

    A file that indexes to nothing (a scanned PDF with no text layer, a picture
    with no text) is reported as a failure: the old page said "Indexed (0 chunks
    added)" in green, and the file then could never be asked about. A file that
    does index is selected as the scope for the next questions, because corpus-wide
    search hides a freshly added file from most questions about it (a "who is this
    person?" about a CV scores below the relevance floor; measured 2026-10-10)."""
    saved = save_upload(filename, data, data_root=data_root)
    chunks = index_file(saved)
    if not chunks:
        return UploadResult(
            False,
            f"`{saved.name}` was saved but nothing could be read from it (0 chunks), so it cannot be asked about. "
            "A scanned PDF with no text layer, or a picture with no text, produces this.",
            [],
        )
    return UploadResult(
        True,
        f"✅ Indexed `{saved.name}` ({chunks} chunks). Answers now come from this file only; "
        "clear **Answer from** to use everything again.",
        sources_for_upload(saved, list_sources()),
    )


def attachment_note(name: str, text: str, tesseract_installed: bool) -> str:
    """The line the page shows about an image attached to a question: what was
    read from it, or why nothing was. The three cases are different facts (the
    picture has text / the picture has none / this machine cannot read pictures)
    and the old page reported the last two identically, as "no readable text"."""
    from src.pipelines.rag.attachments import ATTACHMENT_MAX_WORDS

    if text.strip():
        words = len(text.split())
        cut = f" Only the first {ATTACHMENT_MAX_WORDS} words are used." if words > ATTACHMENT_MAX_WORDS else ""
        return f"📎 Attached image `{name}`: read {words} words of text from it. The answer can use and cite that text.{cut}"
    if tesseract_installed:
        return (
            f"📎 Attached image `{name}`: no text could be read from it. RAGNova reads text in pictures; "
            "it cannot describe what a photo shows."
        )
    return (
        f"📎 Attached image `{name}`: Tesseract is not installed, so text in pictures cannot be read. "
        "Only look-alike images already in the index can be found."
    )


def ocr_available() -> bool:
    """Whether Tesseract is installed. ocr_image() answers "" both for a picture
    with no text and for a machine with no Tesseract, and the page has to tell the
    two apart: one is a fact about the picture, the other a fact about the setup."""
    from src.pipelines.images.ocr import TesseractOCREngine

    return TesseractOCREngine().is_available


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


def _listed_model_names(listing) -> set[str]:
    """Model names out of whatever `Client.list()` returned (an object with
    `.models` in ollama-python 0.4+, a plain dict before)."""
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])
    names = set()
    for entry in models or []:
        name = getattr(entry, "model", None)
        if name is None and isinstance(entry, dict):
            name = entry.get("model") or entry.get("name")
        if name:
            names.add(name)
    return names


def ollama_status() -> tuple[str, str]:
    """What can be known about the language model WITHOUT running it:
    ("ready" | "missing_model" | "unreachable", detail).

    "ready" means the server answers and the model is pulled. It does not
    mean the model can start: on 2026-10-07 a half-applied Ollama update left
    the server answering while every generate call failed with "llama-server
    binary not found". Only `check_model_generates()` or an actual answer
    proves that, so the page must not call "ready" "online".
    """
    from src.core.llm import ollama_client

    try:
        listing = ollama_client().list()
    except Exception as exc:
        return "unreachable", str(exc)
    names = _listed_model_names(listing)
    wanted = settings.OLLAMA_MODEL
    if wanted in names or f"{wanted}:latest" in names:
        return "ready", wanted
    return "missing_model", f"{wanted} is not among: {', '.join(sorted(names)) or 'no models'}"


def ollama_ready() -> bool:
    """Server reachable and model pulled (see `ollama_status` for what that
    does and does not prove)."""
    return ollama_status()[0] == "ready"


def check_model_generates() -> tuple[bool, str]:
    """Ask the model for a few tokens: (worked, message). An on-demand check
    behind a button, not run on every page load, because it loads the model
    into memory. It is the one check that fails when the model cannot start."""
    from src.core.llm import generation_options, ollama_client

    try:
        reply = ollama_client().generate(
            model=settings.OLLAMA_MODEL,
            prompt="Reply with the single word: OK",
            options={**generation_options(), "num_predict": 8},
        )
    except Exception as exc:
        return False, str(exc)
    return True, f"{settings.OLLAMA_MODEL} answered: {str(reply['response']).strip()[:40]!r}"
