"""
Unit and integration tests for Track C UI components:
  - backend.py: AUDIO_EXTS alignment, kind_of, safe_filename, save_upload
  - citations.py: CitationView, citation_views rules (ADR-003, ADR-007, ADR-008)
  - feedback.py: record_feedback, load_feedback, summarize
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.pipelines.audio.ingestion import SUPPORTED_EXTENSIONS as AUDIO_INGESTION_SUPPORTED_EXTS
from src.ui import backend
from src.ui.citations import DOCX_PAGE_NOTE, CitationView, citation_views
from src.ui.feedback import FeedbackEntry, load_feedback, record_feedback, summarize


def test_audio_exts_mismatch_resolved():
    """Verify that backend.AUDIO_EXTS contains every extension supported by

    src.pipelines.audio.ingestion.SUPPORTED_EXTENSIONS.
    """
    assert backend.AUDIO_EXTS == set(AUDIO_INGESTION_SUPPORTED_EXTS), (
        f"AUDIO_EXTS mismatch! Missing from backend: {set(AUDIO_INGESTION_SUPPORTED_EXTS) - backend.AUDIO_EXTS}, "
        f"Unexpected in backend: {backend.AUDIO_EXTS - set(AUDIO_INGESTION_SUPPORTED_EXTS)}"
    )


def test_kind_of_all_supported_audio_extensions():
    """Ensure kind_of correctly classifies every supported audio extension."""
    for ext in AUDIO_INGESTION_SUPPORTED_EXTS:
        filename = f"sample_recording{ext}"
        assert backend.kind_of(filename) == "audio", f"Failed to classify {filename} as audio"
        filename_upper = f"SAMPLE_RECORDING{ext.upper()}"
        assert backend.kind_of(filename_upper) == "audio", f"Failed to classify {filename_upper} as audio"


def test_kind_of_documents_and_images():
    """Ensure kind_of correctly classifies documents and images."""
    assert backend.kind_of("report.pdf") == "document"
    assert backend.kind_of("notes.docx") == "document"
    assert backend.kind_of("screenshot.png") == "image"
    assert backend.kind_of("photo.jpeg") == "image"
    assert backend.kind_of("malware.exe") is None
    assert backend.kind_of("script.py") is None


def test_safe_filename():
    """Ensure safe_filename cleans paths and unsafe characters."""
    assert backend.safe_filename("../../etc/My File (1).PDF") == "My_File_1.pdf"
    assert backend.safe_filename("C:\\Windows\\system32\\evil.exe") == "evil.exe"
    assert backend.safe_filename("audio clip [interview].wav") == "audio_clip_interview.wav"


def test_citation_views_rules():
    """Verify citation view formatting adheres to ADR-003, ADR-007, ADR-008."""
    chunks = [
        Chunk(
            chunk_id="doc1",
            source="data/documents/handbook.pdf",
            modality="pdf",
            text="Campus library rules.",
            page=4,
            embedding_model="test-embedder",
        ),
        Chunk(
            chunk_id="doc2",
            source="data/documents/policy.docx",
            modality="docx",
            text="Attendance policy rules.",
            page=2,
            embedding_model="test-embedder",
        ),
        Chunk(
            chunk_id="aud1",
            source="data/audio/briefing.wav",
            modality="audio",
            text="The evaluation carries 40 percent.",
            start_s=12.5,
            end_s=25.0,
            embedding_model="test-embedder",
        ),
        Chunk(
            chunk_id="img1",
            source="data/images/diagram.png",
            modality="image",
            text="",
            embedding_model="test-embedder",
        ),
    ]

    views = citation_views(chunks)
    assert len(views) == 4

    # 1. PDF: no docx warning note, no audio start time
    assert views[0].number == 1
    assert views[0].modality_label == "PDF"
    assert views[0].note is None
    assert views[0].audio_start_s is None
    assert "handbook.pdf" in views[0].title

    # 2. DOCX: has DOCX_PAGE_NOTE (ADR-008)
    assert views[1].number == 2
    assert views[1].modality_label == "Word"
    assert views[1].note == DOCX_PAGE_NOTE

    # 3. Audio: has audio_start_s
    assert views[2].number == 3
    assert views[2].modality_label == "Audio"
    assert views[2].audio_start_s == 12.5

    # 4. Image with no OCR text: shows placeholder notice
    assert views[3].number == 4
    assert views[3].modality_label == "Image"
    assert views[3].excerpt == "(No readable text in this image.)"


def test_feedback_loop_roundtrip(tmp_path):
    """Verify human feedback recording, loading, and summarizing."""
    log_file = tmp_path / "feedback.jsonl"
    entry1 = FeedbackEntry(
        query="What is the wifi network?",
        answer="Connect to RAGNOVA-STUDENT.",
        rating=5,
        sources=["[1] it_onboarding.docx, Page 1"],
        tester="Alice",
    )
    entry2 = FeedbackEntry(
        query="What is the canteen menu?",
        answer="I do not have information about that.",
        rating=2,
        sources=[],
        tester="Bob",
    )

    record_feedback(entry1, path=log_file)
    record_feedback(entry2, path=log_file)

    loaded = load_feedback(path=log_file)
    assert len(loaded) == 2
    assert loaded[0].tester == "Alice"
    assert loaded[1].rating == 2

    summary = summarize(loaded)
    assert summary["count"] == 2
    assert summary["average"] == 3.5
    assert summary["low"] == 1
    assert summary["histogram"][5] == 1
    assert summary["histogram"][2] == 1
