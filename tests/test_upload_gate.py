"""
The UI's upload gate (src/ui/backend.py) must accept exactly the file types
the pipelines can read. Its audio list was a hand-copied 6 formats while the
audio pipeline read 14, so an .opus or .aac upload was refused although it could
have been transcribed. The gate now derives its lists from the pipelines'.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.audio.ingestion import SUPPORTED_EXTENSIONS as AUDIO_PIPELINE
from src.pipelines.documents.ingest import SUPPORTED_EXTENSIONS as DOCUMENT_PIPELINE
from src.pipelines.images.ingest import SUPPORTED_EXTENSIONS as IMAGE_PIPELINE
from src.ui import backend


@pytest.mark.parametrize("kind, extensions", [
    ("document", DOCUMENT_PIPELINE), ("image", IMAGE_PIPELINE), ("audio", AUDIO_PIPELINE),
])
def test_every_file_type_a_pipeline_can_read_is_accepted_by_the_gate(kind, extensions):
    refused = [ext for ext in sorted(extensions) if backend.kind_of(f"upload{ext}") != kind]
    assert not refused, f"the {kind} pipeline reads {refused} but the upload gate does not route them as {kind}"


def test_the_gate_reads_the_pipelines_own_lists_so_they_cannot_drift():
    assert backend.AUDIO_EXTS == AUDIO_PIPELINE
    assert backend.DOCUMENT_EXTS == DOCUMENT_PIPELINE
    assert backend.IMAGE_EXTS == frozenset(IMAGE_PIPELINE)


def test_no_file_type_is_claimed_by_two_kinds():
    sets = [backend.DOCUMENT_EXTS, backend.IMAGE_EXTS, backend.AUDIO_EXTS]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not (set(a) & set(b)), f"ambiguous extension(s): {sorted(set(a) & set(b))}"


def test_unsupported_types_are_refused_and_case_is_ignored():
    assert backend.kind_of("notes.txt") is None
    assert backend.kind_of("setup.exe") is None
    assert backend.kind_of("no_extension") is None
    assert backend.kind_of("SONG.OPUS") == "audio"
    assert backend.kind_of("Scan.PDF") == "document"


def test_an_opus_upload_which_the_old_gate_refused_is_saved_into_data_audio(tmp_path):
    path = backend.save_upload("voice note.opus", b"OggS fake", data_root=tmp_path / "data")
    assert path.parent == tmp_path / "data" / "audio"
    assert path.suffix == ".opus"


def test_importing_the_gate_does_not_load_whisper_or_clip():
    # the gate only checks a file NAME; it must not pay for the heavy models
    probe = ("import sys; sys.path.insert(0, r'%s'); import src.ui.backend;"
             "print(sorted(m for m in ('faster_whisper', 'ctranslate2', 'torch', 'open_clip') if m in sys.modules))") % PROJECT_ROOT
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().splitlines()[-1] == "[]"
