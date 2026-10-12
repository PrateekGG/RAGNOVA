"""
The shared chunk schema — the interface contract every RAGNova pipeline codes against.

This module is the real implementation of the strawman sketched in
docs/chapters/ch04-timeline-and-team-split.md §4.3. It exists so that Track A
(text + RAG core), Track B (vision + audio), and Track C (interface) can be
built in parallel without agreeing on anything else.

Design rationale (see docs/decisions/adr-003-two-vector-collections.md and
docs/chapters/ch04-timeline-and-team-split.md §1.3, §1.6):

  - This is a plain @dataclass, not a Pydantic model. We chose this
    deliberately: dataclasses are in the Python standard library (no extra
    dependency, one less thing to `pip install` and one less thing to
    version-pin), and Chapter 0 §A.6.16 already taught this exact pattern.
    Pydantic gives you automatic runtime validation "for free," which is a
    real advantage for a larger team or a longer-lived project — but it is
    also a steeper learning curve for a beginner team, and it hides the
    validation logic inside a library instead of in code the team wrote and
    can read. We get the validation anyway, explicitly, via `validate_chunk()`
    below — which is also exactly what the Chapter 4 §1.6 contract test calls.
    If the team later wants Pydantic's automatic validation and JSON schema
    export, that is a reasonable future-work upgrade (note it as such, don't
    silently swap it in mid-project — that would violate the interface freeze
    this whole file exists to protect).

  - Every field a track needs is here because Chapter 4 §4.2 said a track
    needs it, not because it seemed like a good idea at implementation time.
    If you need a new field, that is an interface change — see the note at
    the bottom of this file before adding one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal

# The four modalities this project supports. Ch1 §2.4 scopes video out
# deliberately — do not add "video" here without updating that scope decision.
Modality = Literal["pdf", "docx", "image", "audio"]

# The two ChromaDB collections defined in ADR-003. A chunk belongs to exactly
# one of these — see `Chunk.default_collection()` below.
TEXT_COLLECTION = "text_index"
IMAGE_COLLECTION = "image_index"

# A third collection (ADR-014), NOT a home for a new kind of chunk: it holds
# MiniLM vectors of the text read from each image, keyed by the same chunk ids
# as IMAGE_COLLECTION, so an image can also be found by what it says. No Chunk
# belongs to it by default (`default_collection()` is unchanged).
IMAGE_TEXT_COLLECTION = "image_text_index"


@dataclass
class Chunk:
    """One retrievable, citable unit of content.

    A Chunk is produced once, at ingestion time, by exactly one pipeline
    (Track A for documents and — via transcription, ADR-005 — audio; Track B
    for images). It is stored in a ChromaDB collection and retrieved, unchanged,
    at query time. Provenance fields (source, modality, page, start_s/end_s)
    are what make citation possible (Ch1 §1.8) — they are set once at
    ingestion and never recomputed later.

    Required for every chunk, regardless of modality:
        chunk_id:        Unique identifier. Convention:
                          "<source_stem>__<location>__c<NNN>", e.g.
                          "notice_pdf__p2__c003" or "lecture3_mp3__t872__c001".
        source:           Path to the original file, relative to the project's
                          `data/` directory (e.g. "data/documents/notice.pdf").
        modality:         Which pipeline produced this chunk. Determines which
                          collection it lives in (see `default_collection`).
        text:             The retrievable, displayable content. For a
                          document chunk, the extracted passage. For an audio
                          chunk, the transcript segment. For an image record,
                          any OCR text found (may be empty string, never None
                          — see `validate_chunk`).
        embedding_model:  Which model produced this chunk's vector, e.g.
                          "all-MiniLM-L6-v2" or "ViT-B-32/laion2b_s34b_b79k".
                          Required so retrieval code never accidentally
                          compares vectors from different models (ADR-003).

    Modality-specific (populate the pair that applies; leave the rest None):
        page:             1-indexed page number. Documents only.
        start_s, end_s:   Segment boundaries in seconds. Audio only.
                          Invariant: start_s < end_s when both are set.
        bbox:             Optional (x0, y0, x1, y1) region on a page or
                          image, for future fine-grained citation. Rarely
                          populated in the initial implementation — present
                          in the schema now so adding it later is not an
                          interface change.

    Populated only at query time, never at ingestion (see the note above
    `score` below):
        score:            Similarity or fused rank score from retrieval.
                          None for a freshly-ingested chunk.
    """

    chunk_id: str
    source: str
    modality: Modality
    text: str
    embedding_model: str

    page: Optional[int] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None

    # NOT set by any ingestion pipeline. Chapter 10's retrieval code attaches
    # this after a query; a Chunk with score=None simply hasn't been through
    # retrieval yet. Keeping this on the same dataclass (rather than a
    # separate `RetrievedChunk` wrapper type) is a deliberate simplicity
    # trade-off for a 9-day sprint — revisit if it starts causing confusion
    # about which fields are "always there" versus "only after retrieval".
    score: Optional[float] = None

    def default_collection(self) -> str:
        """Which ChromaDB collection this chunk belongs in (ADR-003)."""
        return IMAGE_COLLECTION if self.modality == "image" else TEXT_COLLECTION

    def to_chroma_record(self) -> dict:
        """Convert to the three parallel structures ChromaDB's `.add()` wants:
        one id, one document string, and one metadata dict. Called by each
        track's ingestion code right before storage — this is the one place
        the schema and the vector database actually meet.

        Note: this method does NOT include the embedding vector itself.
        Each track computes its own vector (with its own model — ADR-003)
        and passes it to `collection.add(embeddings=[...])` alongside what
        this method returns.
        """
        metadata = {
            "source": self.source,
            "modality": self.modality,
            "embedding_model": self.embedding_model,
        }
        # ChromaDB metadata values must be str/int/float/bool — None is not
        # accepted, so modality-specific fields are only included when set.
        # This is exactly the kind of small, easy-to-forget rule a contract
        # test (see tests/test_contract.py) catches before it becomes a
        # Day-12 surprise.
        if self.page is not None:
            metadata["page"] = self.page
        if self.start_s is not None:
            metadata["start_s"] = self.start_s
        if self.end_s is not None:
            metadata["end_s"] = self.end_s
        if self.bbox is not None:
            metadata["bbox"] = list(self.bbox)  # tuples aren't JSON-native

        return {
            "id": self.chunk_id,
            "document": self.text,
            "metadata": metadata,
        }

    def as_dict(self) -> dict:
        """Plain-dict form, for JSON serialisation (e.g. the corpus manifest
        or gold-set annotations — Chapter 3 §3.8's reproducibility record)."""
        return asdict(self)


def validate_chunk(chunk: Chunk) -> list[str]:
    """Check a Chunk against the interface contract from Ch4 §4.2/§4.4.

    Returns a list of human-readable problems. An empty list means the chunk
    is valid. This function IS the contract test's core logic — Chapter 4
    §1.6 sketched it as pseudocode; this is the real implementation, and
    tests/test_contract.py calls this exact function against each track's
    fixture.

    Deliberately returns a list rather than raising on the first problem:
    when a chunk is malformed, seeing every issue at once (not just the
    first) saves a debugging round-trip — a small usability decision worth
    naming, not an accident.
    """
    errors: list[str] = []

    if not chunk.chunk_id or not chunk.chunk_id.strip():
        errors.append("chunk_id must be a non-empty string")

    if not chunk.source or not chunk.source.strip():
        errors.append("source must be a non-empty string")

    if chunk.modality not in ("pdf", "docx", "image", "audio"):
        errors.append(
            f"modality must be one of pdf/docx/image/audio, got {chunk.modality!r}"
        )

    if chunk.text is None:
        errors.append("text must not be None (use an empty string if there is no text)")

    if not chunk.embedding_model or not chunk.embedding_model.strip():
        errors.append("embedding_model must be a non-empty string")

    # Modality-specific requirements, straight from Ch4 §4.2's requirements
    # table: documents need a page, audio needs a validated time range.
    if chunk.modality in ("pdf", "docx"):
        if chunk.page is None:
            errors.append(f"modality={chunk.modality!r} requires page to be set")
        elif chunk.page < 1:
            errors.append(f"page must be >= 1, got {chunk.page}")

    if chunk.modality == "audio":
        if chunk.start_s is None or chunk.end_s is None:
            errors.append("modality='audio' requires both start_s and end_s")
        elif chunk.start_s < 0:
            errors.append(f"start_s must be >= 0, got {chunk.start_s}")
        elif chunk.end_s <= chunk.start_s:
            errors.append(
                f"end_s ({chunk.end_s}) must be greater than start_s ({chunk.start_s})"
            )

    # A chunk should not carry provenance fields that belong to a different
    # modality — this catches the "wrong field name, nobody noticed" failure
    # mode described in Ch4 §1.7's counterfactual.
    if chunk.modality != "audio" and (chunk.start_s is not None or chunk.end_s is not None):
        errors.append(
            f"modality={chunk.modality!r} should not set start_s/end_s (audio only)"
        )
    if chunk.modality not in ("pdf", "docx") and chunk.page is not None:
        errors.append(
            f"modality={chunk.modality!r} should not set page (documents only)"
        )

    return errors


# ---------------------------------------------------------------------------
# Changing this file
# ---------------------------------------------------------------------------
# This schema is the interface contract (Ch4 §1.3, §1.6). A field's NAME and
# TYPE are not any single track's to change unilaterally once another track
# is coding against it — that is exactly the kind of change Chapter 4 §1.7
# describes as expensive if it happens silently on Day 12 instead of being
# agreed on Day 5. If you need a new field:
#   1. Propose it in the team's sync channel (Ch4 §3.3) — not a silent edit.
#   2. Add it here with a clear docstring, matching the style above.
#   3. Add or extend the relevant fixture in tests/test_contract.py so the
#      contract test actually exercises it.
#   4. Tell the other two tracks it exists before you depend on it.
