"""
Tests for Task 1 (Corpus Inventory) and Task 4 (UI Scaffold).
Verifies that:
  - data/images/ has >= 10 valid image files readable by PIL
  - data/audio/ has >= 3 valid .wav audio files with valid header and duration
  - data/documents/ has the required starter documents
  - src/app.py exists and contains the required TODO hook and query processing logic
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
APP_FILE = PROJECT_ROOT / "src" / "app.py"


def test_documents_corpus_present():
    assert DOCS_DIR.exists(), "data/documents/ directory missing"
    docs = list(DOCS_DIR.glob("*.*"))
    assert len(docs) >= 3, f"Expected at least 3 starter documents, found {len(docs)}"
    expected_filenames = {"notice.pdf", "library_hours.pdf", "it_onboarding.docx"}
    actual_filenames = {d.name for d in docs}
    assert expected_filenames.issubset(actual_filenames), f"Missing starter docs: {expected_filenames - actual_filenames}"


def test_images_corpus_target_reached():
    assert IMAGES_DIR.exists(), "data/images/ directory missing"
    images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
    assert 10 <= len(images) <= 15, f"Expected 10-15 images, found {len(images)}"

    screenshots = [img for img in images if "screenshot" in img.name.lower()]
    assert len(screenshots) >= 3, f"Expected >=3 screenshots, found {len(screenshots)}"

    photos = [img for img in images if "photo" in img.name.lower()]
    assert len(photos) >= 3, f"Expected >=3 photo assets with visible text, found {len(photos)}"

    for img_path in images:
        assert img_path.stat().st_size > 0, f"Image {img_path.name} is empty"
        with Image.open(img_path) as img:
            assert img.width > 100, f"Image {img_path.name} width too small: {img.width}"
            assert img.height > 100, f"Image {img_path.name} height too small: {img.height}"


def test_audio_corpus_target_reached():
    assert AUDIO_DIR.exists(), "data/audio/ directory missing"
    audio_files = list(AUDIO_DIR.glob("*.wav")) + list(AUDIO_DIR.glob("*.mp3"))
    assert len(audio_files) >= 3, f"Expected 3-5 audio clips, found {len(audio_files)}"

    for audio_path in audio_files:
        assert audio_path.stat().st_size > 1000, f"Audio {audio_path.name} unexpectedly small ({audio_path.stat().st_size} bytes)"
        if audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as w:
                n_channels = w.getnchannels()
                sample_rate = w.getframerate()
                n_frames = w.getnframes()
                duration = n_frames / float(sample_rate)
                assert n_channels == 1, f"Audio {audio_path.name} expected mono (1 channel), got {n_channels}"
                assert sample_rate == 16000, f"Audio {audio_path.name} expected 16000 Hz, got {sample_rate}"
                assert duration >= 5.0, f"Audio {audio_path.name} duration {duration:.1f}s is less than 5s"


def test_app_scaffold_present_and_has_todo():
    assert APP_FILE.exists(), "src/app.py does not exist"
    content = APP_FILE.read_text(encoding="utf-8")
    assert "# TODO: to wire the real call here" in content, (
        "src/app.py missing explicit '# TODO: to wire the real call here' marker"
    )
    assert "process_query" in content, "src/app.py missing process_query function"


def test_app_process_query_mock_and_citations():
    from src.app import process_query

    # Query matching evaluation marks
    ans1, cit1 = process_query("How many marks does the prototype demo carry?")
    assert "40%" in ans1
    assert len(cit1) >= 2
    assert any(c["modality"] == "pdf" and "notice.pdf" in c["source"] for c in cit1)
    assert any(c["modality"] == "audio" for c in cit1)

    # Query matching wifi
    ans2, cit2 = process_query("How do I connect to wifi?")
    assert "RAGNOVA-STUDENT" in ans2
    assert "certificate" in ans2.lower()
    assert any(c["modality"] == "docx" for c in cit2)
    assert any(c["modality"] == "image" for c in cit2)

    # General supported query
    ans3, cit3 = process_query("What is this project?")
    assert len(cit3) >= 1

    # Photo query
    ans_photo, cit_photo = process_query("Where is the AI research lab room 302?")
    assert "room 302" in ans_photo.lower()
    assert any("photo" in c["source"] for c in cit_photo)

    # Negative controls (P2: must refuse and must NOT claim false grounding or emit citations)
    ans_neg1, cit_neg1 = process_query("What is the hostel mess menu for Wednesday lunch?")
    assert "could not find" in ans_neg1.lower()
    assert len(cit_neg1) == 0, "Negative control N1 emitted false citations"

    ans_neg2, cit_neg2 = process_query("How do I apply for a refund on tuition fees?")
    assert "could not find" in ans_neg2.lower()
    assert len(cit_neg2) == 0, "Negative control N2 emitted false citations"

    # Broad keywords that previously bypassed refusal (must NOT match wifi, library, etc.)
    ans_neural, cit_neural = process_query("What is a neural network?")
    assert "could not find" in ans_neural.lower()
    assert len(cit_neural) == 0, "Query 'What is a neural network?' falsely routed to Wi-Fi"

    ans_book_lab, cit_book_lab = process_query("How do I book the AI lab?")
    assert "could not find" in ans_book_lab.lower()
    assert len(cit_book_lab) == 0, "Query 'How do I book the AI lab?' falsely routed to Library"

    # Unsupported query (P2: must NOT claim false grounding or invent citations)
    ans_unsupported, cit_unsupported = process_query("What is the population of Jupiter?")
    assert "could not find" in ans_unsupported.lower()
    assert len(cit_unsupported) == 0, "Unsupported query emitted false citations"


