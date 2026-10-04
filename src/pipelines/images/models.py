"""Data models and configuration for the image ingestion pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.core.schemas import Chunk


@dataclass
class ImageIngestionConfig:
    """Configuration for ImageIngestionPipeline."""

    # OCR Settings
    ocr_enabled: bool = True
    tesseract_cmd: Optional[str] = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD")
    )
    ocr_lang: str = "eng"
    ocr_config: str = "--psm 3"
    preprocess_for_ocr: bool = True

    # OpenCLIP Embedding Settings
    embedding_enabled: bool = True
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: Optional[str] = None  # None = auto-detect ('cuda', 'mps', 'cpu')
    normalize_embeddings: bool = True

    # Batch ingestion settings. Maximum images decoded and embedded together
    # in one sub-batch during ingest_batch()/ingest_directory() — a real
    # Greptile finding on the original implementation: sending every image
    # discovered in a directory through a single, unbounded batch call
    # meant a large directory, or a directory of high-resolution images,
    # could hold every decoded image and every preprocessed OpenCLIP tensor
    # in memory simultaneously before any of it was freed. 16 is a starting
    # point, not a measured value — lower it for high-resolution corpora on
    # constrained (e.g. CPU-only) hardware, raise it if memory genuinely
    # isn't the bottleneck.
    batch_size: int = 16

    @property
    def embedding_model_id(self) -> str:
        """Returns standard embedding model identifier string."""
        return f"{self.model_name}/{self.pretrained}"


@dataclass
class OCRResult:
    """Result of OCR text extraction."""

    text: str = ""
    confidence: float = 0.0
    words_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageMetadata:
    """Extracted metadata for an input image."""

    source_path: Optional[str] = None
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    mode: str = "RGB"
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    exif: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "exif": self.exif,
        }


@dataclass
class ImageChunk(Chunk):
    """The canonical :class:`~src.core.schemas.Chunk` (Chapter 5's frozen,
    cross-track contract), extended with the two fields this pipeline
    actually needs to carry per image: the OpenCLIP vector, and the
    combined image/OCR metadata dict (dimensions, format, sha256, OCR
    confidence, ...).

    This is a **subclass**, not a change to ``Chunk`` itself. Every base
    field (``chunk_id``, ``source``, ``modality``, ``text``,
    ``embedding_model``, ``page``, ``start_s``, ``end_s``, ``bbox``,
    ``score``) is inherited unchanged, so an ``ImageChunk`` satisfies
    ``validate_chunk()`` exactly like a plain ``Chunk`` would, and every
    other track's code that only ever sees ``src.core.schemas.Chunk``
    is completely unaffected. Adding ``embedding``/``metadata`` directly
    to the base schema instead would have required the whole-team sign-off
    ``schemas.py``'s own "Changing this file" note asks for, for something
    that is genuinely this one pipeline's concern, not a cross-track one.

    ``as_dict()`` (inherited from ``Chunk``) already does the right thing
    here for free: it calls ``dataclasses.asdict(self)``, which walks the
    *actual* instance's fields — base and subclass both — so it returns
    every field below, embedding and metadata included, with no override
    needed.
    """

    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Alias for :meth:`as_dict` — kept because it's the name this
        pipeline's CLI, example script, and smoke test were all written
        against."""
        return self.as_dict()

    def to_json(self, **kwargs: Any) -> str:
        """JSON-serialise this chunk. Any keyword args are forwarded to
        :func:`json.dumps` (e.g. ``indent=2``)."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageChunk":
        """Inverse of :meth:`to_dict` — reconstructs an ``ImageChunk`` from
        a plain dict with exactly this class's field names, e.g. one just
        read back from :meth:`to_json`."""
        return cls(**data)


__all__ = [
    "ImageIngestionConfig",
    "OCRResult",
    "ImageMetadata",
    "ImageChunk",
]
