"""Image ingestion orchestrator — creates and stores Chunks.

This module is the **top-level entry point** for the image RAG pipeline.
It wires together:

- :class:`~src.pipelines.images.ocr.TesseractOCREngine`  — OCR text extraction
- :class:`~src.pipelines.images.embedding.OpenCLIPEmbedder` — visual embeddings
- :class:`~src.core.schemas.Chunk`                          — output data model
- :class:`ChunkStore`                                        — in-process storage

Designed for extensibility
--------------------------
Each pipeline component is injected via constructor parameters so you can
swap them out (e.g. replace Tesseract with a cloud OCR service) without
touching this file.  The ``ChunkStore`` interface is also intentionally thin
so it can be replaced with a vector-database adapter.

Typical usage
-------------
>>> pipeline = ImageIngestionPipeline()
>>> chunks   = pipeline.ingest_directory("./photos")
>>> for chunk in chunks:
...     print(chunk.chunk_id, len(chunk.text), "chars of OCR text")
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from PIL import Image

from src.core.schemas import Chunk
from src.pipelines.images.embedding import OpenCLIPEmbedder
from src.pipelines.images.models import (
    ImageChunk,
    ImageIngestionConfig,
    ImageMetadata,
    OCRResult,
)
from src.pipelines.images.ocr import TesseractOCREngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image file extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
)


# ---------------------------------------------------------------------------
# ChunkStore — lightweight in-process storage
# ---------------------------------------------------------------------------

class ChunkStore:
    """In-memory store for ingested :class:`~src.core.schemas.Chunk` objects.

    This serves as the default persistence layer during development and
    testing.  Replace with a vector-DB adapter (e.g. Qdrant, Weaviate,
    Chroma) for production by passing a custom ``store`` to
    :class:`ImageIngestionPipeline`.

    Parameters
    ----------
    deduplicate:
        When *True*, re-inserting a chunk whose ``chunk_id`` already exists
        in the store is silently ignored.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        self._store: Dict[str, Chunk] = {}
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(self, chunk: Chunk) -> bool:
        """Persists a single chunk.

        Parameters
        ----------
        chunk:
            The :class:`~src.core.schemas.Chunk` to store.

        Returns
        -------
        bool
            *True* when the chunk was stored; *False* when a duplicate was
            detected and ``deduplicate=True``.
        """
        if self.deduplicate and chunk.chunk_id in self._store:
            logger.debug("Skipping duplicate chunk_id=%s", chunk.chunk_id)
            return False
        self._store[chunk.chunk_id] = chunk
        return True

    def add_many(self, chunks: List[Chunk]) -> int:
        """Persists a list of chunks.

        Returns
        -------
        int
            Number of chunks actually stored (duplicates excluded).
        """
        return sum(self.add(c) for c in chunks)

    def remove(self, chunk_id: str) -> bool:
        """Removes a chunk by its ``chunk_id``.

        Returns
        -------
        bool
            *True* when the chunk existed and was removed.
        """
        if chunk_id in self._store:
            del self._store[chunk_id]
            return True
        return False

    def clear(self) -> None:
        """Removes all stored chunks."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieves a chunk by ``chunk_id``; returns *None* if not found."""
        return self._store.get(chunk_id)

    def all(self) -> List[Chunk]:
        """Returns all stored chunks as a list."""
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"ChunkStore(count={len(self)})"


# ---------------------------------------------------------------------------
# ImageIngestionPipeline — main orchestrator
# ---------------------------------------------------------------------------

class ImageIngestionPipeline:
    """Ingests images via OCR + OpenCLIP embeddings, producing one Chunk each.

    Parameters
    ----------
    config:
        Pipeline settings.  Defaults to :class:`ImageIngestionConfig` which
        selects ``ViT-B-32/laion2b_s34b_b79k``.
    ocr_engine:
        Custom OCR engine instance.  Pass *None* to auto-create a
        :class:`~src.pipelines.images.ocr.TesseractOCREngine`.
    embedder:
        Custom embedder instance.  Pass *None* to auto-create an
        :class:`~src.pipelines.images.embedding.OpenCLIPEmbedder`.
    store:
        Chunk storage backend.  Pass *None* to use an ephemeral
        :class:`ChunkStore`.

    Notes
    -----
    Both ``ocr_engine`` and ``embedder`` are only instantiated when the
    corresponding feature is enabled in ``config``
    (``ocr_enabled`` / ``embedding_enabled``).  This lets you disable either
    step for testing without importing their heavyweight dependencies.
    """

    def __init__(
        self,
        config: Optional[ImageIngestionConfig] = None,
        ocr_engine: Optional[TesseractOCREngine] = None,
        embedder: Optional[OpenCLIPEmbedder] = None,
        store: Optional[ChunkStore] = None,
    ) -> None:
        self.config = config or ImageIngestionConfig()

        # OCR engine — created lazily only when OCR is enabled
        self.ocr_engine: Optional[TesseractOCREngine] = ocr_engine
        if self.ocr_engine is None and self.config.ocr_enabled:
            self.ocr_engine = TesseractOCREngine(self.config)

        # Embedding model — created lazily only when embeddings are enabled
        self.embedder: Optional[OpenCLIPEmbedder] = embedder
        if self.embedder is None and self.config.embedding_enabled:
            self.embedder = OpenCLIPEmbedder(self.config)

        # Chunk store — default to in-memory store.
        # `store or ChunkStore()` was tried first and is wrong: ChunkStore
        # defines __len__ but not __bool__, so Python falls back to
        # len(store) != 0 for truthiness — an empty (but real, caller-
        # supplied) store is falsy, so `or` silently discards it and
        # substitutes a brand-new, private one. Every chunk then gets
        # stored where the caller can never see it. Verified directly:
        # smoke_test.py passes a fresh ChunkStore and asserts len(store)==1
        # after one ingest_image() call — it failed with len(store)==0
        # before this fix. `is not None` checks identity, not truthiness,
        # so a real empty store is correctly kept.
        self.store: ChunkStore = store if store is not None else ChunkStore()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_bytes(raw_bytes: bytes, source_desc: str) -> Image.Image:
        """`Image.open()`, with a clean, catchable error instead of a raw
        traceback for the corrupt/not-actually-an-image case.

        Verified directly (Chapter 7 /verify pass): a text file renamed
        `.png` crashed with a bare `PIL.UnidentifiedImageError` traceback.
        Raising `ValueError` here instead lets callers (ingest_batch below,
        and any CLI/script built on this pipeline) catch it and report on
        it the same clean way they already handle a missing file.

        Message is plain ASCII on purpose — no em-dash. This is a library
        module, not an entry point: it has no business reconfiguring
        stdout/stderr encoding itself, so any caller who imports this
        directly, on an unreconfigured Windows console, would otherwise
        see this exact text mangled to "?" — confirmed live in the
        previous /verify pass.

        Also calls `.load()` immediately, still inside this try block —
        not just `Image.open()`. Pillow opens lazily: it reads only enough
        of the header to identify the format and report size/mode, and
        defers actually decoding pixel data until something needs it.
        A truncated file can pass `Image.open()` cleanly and only fail
        later, inside OpenCLIP's `img.convert("RGB")` during
        `embed_batch()` — a real Greptile finding on this exact fix,
        confirmed directly: `Image.open()` succeeded on a deliberately
        truncated PNG, and only `.load()`/`.convert()` raised. That later
        failure happens outside ingest_batch()'s per-file try/except
        below, so it would abort the *entire* batch — defeating the very
        fix that try/except exists for. Forcing the decode here, inside
        the block ingest_batch() already catches, means a truncated file
        is caught and skipped at load time, before it ever reaches a
        batched embedding call it could take down with it.
        """
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
            return image
        except OSError as exc:
            raise ValueError(
                f"Cannot identify image file: {source_desc} - the file may be "
                f"corrupt, empty, or not actually an image."
            ) from exc

    def _load_image(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    ) -> tuple[Image.Image, ImageMetadata]:
        """Loads a PIL Image and extracts initial file-level metadata.

        Accepts a file path, raw bytes, a file-like object, or an
        already-decoded :class:`PIL.Image.Image`.
        """
        source_path: Optional[str] = None
        raw_bytes: Optional[bytes] = None

        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input).expanduser().resolve()
            if not path_obj.is_file():
                raise FileNotFoundError(f"Image not found: {path_obj}")
            source_path = str(path_obj)
            raw_bytes = path_obj.read_bytes()
            pil_img = self._open_bytes(raw_bytes, source_path)

        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            pil_img = self._open_bytes(raw_bytes, "<bytes>")

        elif hasattr(image_input, "read"):
            raw_bytes = image_input.read()  # type: ignore[union-attr]
            pil_img = self._open_bytes(raw_bytes, "<file-like object>")

        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            buf = io.BytesIO()
            fmt = pil_img.format or "PNG"
            try:
                pil_img.save(buf, format=fmt)
                raw_bytes = buf.getvalue()
            except Exception:
                raw_bytes = None

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Collect image properties before any colour-space conversion
        fmt = pil_img.format
        width, height = pil_img.size
        mode = pil_img.mode

        sha256 = (
            hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
        )
        file_size = len(raw_bytes) if raw_bytes is not None else None

        # Extract EXIF data when available (JPEG / TIFF)
        exif: Dict[str, Any] = {}
        try:
            raw_exif = pil_img.getexif()
            if raw_exif:
                exif = {str(k): str(v) for k, v in raw_exif.items()}
        except Exception:
            pass

        metadata = ImageMetadata(
            source_path=source_path,
            width=width,
            height=height,
            format=fmt,
            mode=mode,
            file_size_bytes=file_size,
            sha256=sha256,
            exif=exif,
        )

        return pil_img, metadata

    @staticmethod
    def _make_chunk_id(img_meta: ImageMetadata) -> str:
        """Derives a deterministic chunk identifier from image metadata."""
        if img_meta.sha256:
            return f"img_{img_meta.sha256[:16]}"
        if img_meta.source_path:
            stem = Path(img_meta.source_path).stem
            return f"img_{stem}_{os.urandom(4).hex()}"
        return f"img_{os.urandom(8).hex()}"

    # ------------------------------------------------------------------
    # Core ingestion — single image
    # ------------------------------------------------------------------

    def ingest_image(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
        extra_metadata: Optional[Dict[str, Any]] = None,
        store: bool = True,
    ) -> ImageChunk:
        """Processes a single image and produces exactly one :class:`ImageChunk`.

        Processing steps
        ~~~~~~~~~~~~~~~~
        1. Load image and extract file-level metadata.
        2. Run Tesseract OCR → populate ``chunk.text`` with extracted text.
        3. Run OpenCLIP inference → populate ``chunk.embedding``.
        4. Assemble metadata dict (image properties + OCR stats).
        5. Construct and optionally persist the Chunk.

        Parameters
        ----------
        image_input:
            File path (``str`` or :class:`pathlib.Path`), raw ``bytes``,
            a file-like object with a ``read()`` method, or a
            :class:`PIL.Image.Image`.
        extra_metadata:
            Optional dictionary of caller-supplied metadata to merge into
            the chunk's ``metadata`` field.
        store:
            When *True* (default), the resulting chunk is stored via
            :attr:`self.store`.

        Returns
        -------
        ImageChunk
            An :class:`ImageChunk` (a :class:`~src.core.schemas.Chunk` plus
            ``embedding``/``metadata`` — see that class's docstring) with:
            - ``chunk_id``       — deterministic identifier derived from image hash / path.
            - ``source``         — absolute file path, or ``"<bytes>"`` for in-memory images.
            - ``text``           — OCR-extracted text (empty string when OCR is disabled).
            - ``modality``       — always ``"image"``.
            - ``embedding``      — 512-d float vector (``None`` when embeddings are disabled).
            - ``embedding_model``— e.g. ``"ViT-B-32/laion2b_s34b_b79k"``.
            - ``metadata``       — image properties + OCR stats + *extra_metadata*.
        """
        pil_img, img_meta = self._load_image(image_input)

        # Step 1 — OCR
        ocr_result: OCRResult = OCRResult()
        if self.ocr_engine and self.config.ocr_enabled:
            logger.debug("Running OCR on %s", img_meta.source_path or "<bytes>")
            ocr_result = self.ocr_engine.extract_text(pil_img)

        # Step 2 — Visual embedding
        embedding: Optional[List[float]] = None
        if self.embedder and self.config.embedding_enabled:
            logger.debug("Generating embedding for %s", img_meta.source_path or "<bytes>")
            embedding = self.embedder.embed_image(pil_img)

        # Step 3 — Assemble metadata
        combined_meta: Dict[str, Any] = {
            **img_meta.to_dict(),
            "ocr": {
                "text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "words_count": ocr_result.words_count,
                **ocr_result.metadata,
            },
        }
        if extra_metadata:
            combined_meta.update(extra_metadata)

        # Step 4 — Build an ImageChunk (Chunk + embedding/metadata — see
        # models.py's ImageChunk docstring for why plain Chunk(embedding=...,
        # metadata=...) raised TypeError here before this fix).
        #   chunk_id  ← deterministic hash/path-based identifier
        #   source    ← absolute file path or "<bytes>" for in-memory images
        #   text      ← OCR-extracted text (searchable content)
        #   embedding ← dense visual vector stored directly on the ImageChunk
        chunk = ImageChunk(
            chunk_id=self._make_chunk_id(img_meta),
            source=img_meta.source_path or "<bytes>",
            text=ocr_result.text,
            modality="image",
            embedding=embedding,
            embedding_model=self.config.embedding_model_id,
            metadata=combined_meta,
        )

        # Step 5 — Persist
        if store:
            self.store.add(chunk)

        logger.info(
            "Ingested chunk_id=%s  source=%s  ocr_words=%d  embedding=%s",
            chunk.chunk_id,
            chunk.source,
            ocr_result.words_count,
            "yes" if embedding else "no",
        )
        return chunk

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    def ingest_batch(
        self,
        image_inputs: List[Union[str, Path, bytes, BinaryIO, Image.Image]],
        extra_metadata_list: Optional[List[Dict[str, Any]]] = None,
        store: bool = True,
    ) -> List[ImageChunk]:
        """Processes a batch of images, producing one ImageChunk per image.

        OCR is run sequentially per image; OpenCLIP inference batches
        within each sub-batch of :attr:`ImageIngestionConfig.batch_size`
        images for efficiency, without holding every decoded image and
        every preprocessed tensor for the *entire* input in memory at
        once — a real Greptile finding on this exact method: sending
        every discovered image through a single `ingest_batch` call meant
        a large directory, or a directory of high-resolution images,
        could exhaust host or accelerator memory before ever running.
        Each sub-batch's PIL images and tensors go out of scope (eligible
        for garbage collection) before the next sub-batch is loaded.

        Parameters
        ----------
        image_inputs:
            List of image sources (paths, bytes, file-like objects, PIL Images).
        extra_metadata_list:
            Optional list of extra metadata dicts, one per image (positionally
            aligned with ``image_inputs``).
        store:
            When *True*, all produced chunks are stored via :attr:`self.store`.

        Returns
        -------
        List[ImageChunk]
            One ImageChunk per input image (skipped/unreadable ones aside),
            in the same order.
        """
        if not image_inputs:
            return []

        batch_size = max(1, self.config.batch_size)
        all_chunks: List[ImageChunk] = []
        for start in range(0, len(image_inputs), batch_size):
            sub_inputs = image_inputs[start : start + batch_size]
            sub_extra = (
                extra_metadata_list[start : start + batch_size]
                if extra_metadata_list
                else None
            )
            all_chunks.extend(self._ingest_batch_chunk(sub_inputs, sub_extra, store))
        return all_chunks

    def _ingest_batch_chunk(
        self,
        image_inputs: List[Union[str, Path, bytes, BinaryIO, Image.Image]],
        extra_metadata_list: Optional[List[Dict[str, Any]]],
        store: bool,
    ) -> List[ImageChunk]:
        """Processes ONE sub-batch (at most `config.batch_size` images) —
        the original, un-chunked `ingest_batch()` body. Kept as a separate
        method so `ingest_batch()`'s job is purely "split the input and
        call this repeatedly," not interleaved with the actual per-image
        work.
        """
        if not image_inputs:
            return []

        # Load every image individually, catching per-file failures rather
        # than letting one corrupt/unreadable file abort the whole batch —
        # the same "one bad item shouldn't cost the rest" policy Chapter 6's
        # document pipeline already uses for a zero-text PDF page. The
        # original single list comprehension loaded eagerly and would raise
        # on the first bad file, losing every other image in the batch too.
        loaded: List[tuple[Image.Image, ImageMetadata]] = []
        surviving_extra_metadata: List[Optional[Dict[str, Any]]] = []
        for i, inp in enumerate(image_inputs):
            try:
                loaded.append(self._load_image(inp))
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping unreadable image %r: %s", inp, exc)
                continue
            if extra_metadata_list and i < len(extra_metadata_list):
                surviving_extra_metadata.append(extra_metadata_list[i])
            else:
                surviving_extra_metadata.append(None)
        extra_metadata_list = surviving_extra_metadata

        if not loaded:
            return []

        pil_images = [img for img, _ in loaded]

        # --- OCR (sequential; Tesseract is single-threaded) ---
        ocr_results: List[OCRResult] = []
        for pil_img in pil_images:
            if self.ocr_engine and self.config.ocr_enabled:
                ocr_results.append(self.ocr_engine.extract_text(pil_img))
            else:
                ocr_results.append(OCRResult())

        # --- Embeddings (single batched forward pass) ---
        embeddings: List[Optional[List[float]]]
        if self.embedder and self.config.embedding_enabled:
            raw_embeddings = self.embedder.embed_batch(pil_images)
            embeddings = raw_embeddings  # type: ignore[assignment]
        else:
            embeddings = [None] * len(pil_images)

        # --- Assemble Chunks ---
        chunks: List[ImageChunk] = []
        for i, ((_, img_meta), ocr_res, emb) in enumerate(
            zip(loaded, ocr_results, embeddings)
        ):
            meta: Dict[str, Any] = {
                **img_meta.to_dict(),
                "ocr": {
                    "text": ocr_res.text,
                    "confidence": ocr_res.confidence,
                    "words_count": ocr_res.words_count,
                    **ocr_res.metadata,
                },
            }
            if extra_metadata_list and i < len(extra_metadata_list) and extra_metadata_list[i]:
                meta.update(extra_metadata_list[i])

            chunk = ImageChunk(
                chunk_id=self._make_chunk_id(img_meta),
                source=img_meta.source_path or "<bytes>",
                text=ocr_res.text,
                modality="image",
                embedding=emb,
                embedding_model=self.config.embedding_model_id,
                metadata=meta,
            )
            chunks.append(chunk)

        if store:
            stored_count = self.store.add_many(chunks)
            logger.info("Stored %d / %d chunks.", stored_count, len(chunks))

        return chunks

    # ------------------------------------------------------------------
    # Directory ingestion
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS,
        store: bool = True,
    ) -> List[ImageChunk]:
        """Discovers and ingests all images in a directory.

        Parameters
        ----------
        directory_path:
            Root directory to search.
        recursive:
            If *True*, searches subdirectories recursively.
        extensions:
            Tuple of lower-case file extensions to include
            (e.g. ``(".png", ".jpg")``).
        store:
            When *True*, all produced chunks are stored via :attr:`self.store`.

        Returns
        -------
        List[ImageChunk]
            All chunks produced from the discovered images.

        Raises
        ------
        NotADirectoryError
            When ``directory_path`` does not point to an existing directory.
        """
        dir_obj = Path(directory_path).expanduser().resolve()
        if not dir_obj.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_obj}")

        pattern = "**/*" if recursive else "*"
        image_files = sorted(
            f
            for f in dir_obj.glob(pattern)
            if f.is_file() and f.suffix.lower() in extensions
        )

        logger.info(
            "Found %d image(s) in '%s' (recursive=%s).",
            len(image_files),
            dir_obj,
            recursive,
        )

        return self.ingest_batch(
            image_files,  # type: ignore[arg-type]
            store=store,
        )


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

def ingest_image(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    config: Optional[ImageIngestionConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ImageChunk:
    """Convenience function: ingest a single image into an :class:`ImageChunk`.

    Creates a throw-away :class:`ImageIngestionPipeline` and returns the
    resulting chunk.  The chunk is **not** stored in any persistent store.

    Parameters
    ----------
    image_input:
        File path, raw bytes, file-like object, or PIL Image.
    config:
        Optional pipeline configuration.
    extra_metadata:
        Extra metadata to attach to the chunk.

    Returns
    -------
    ImageChunk
    """
    pipeline = ImageIngestionPipeline(config=config)
    return pipeline.ingest_image(image_input, extra_metadata=extra_metadata, store=False)


def ingest_directory(
    directory_path: Union[str, Path],
    config: Optional[ImageIngestionConfig] = None,
    recursive: bool = True,
) -> List[ImageChunk]:
    """Convenience function: ingest all images in a directory.

    Parameters
    ----------
    directory_path:
        Root directory path.
    config:
        Optional pipeline configuration.
    recursive:
        Whether to search subdirectories.

    Returns
    -------
    List[ImageChunk]
    """
    pipeline = ImageIngestionPipeline(config=config)
    return pipeline.ingest_directory(directory_path, recursive=recursive, store=False)
