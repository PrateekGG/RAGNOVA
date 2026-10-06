"""
A transcript cache for audio files, keyed by the audio's own bytes.

Why this exists. Transcribing speech with Whisper is not reproducible across
machines and library versions: on 2026-10-05 the pinned requirements
environment and the newer-library environment transcribed the same eight clips
slightly differently ("Overdue fines" against "Overdoophines" in one
build), and the retrieval numbers moved with them (text MRR 0.75 against
0.81 on the same questions and corpus). A number in the report that depends
on which laptop built the index is not a result. Storing each file's
transcript, keyed by the sha256 of the audio, makes the transcript part of the
corpus: every machine and every rebuild indexes the same text, and a rebuild
no longer re-runs Whisper for files it has already seen.

The key is the audio's content hash plus the Whisper model size, so editing or
replacing a file (new bytes) or switching model size triggers a fresh
transcription instead of serving a stale one. Delete a cache file to force
re-transcription of that clip. Nothing here is a database: one small JSON
file per transcript, readable and diffable, committed with the corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = 1


def file_sha256(path: str | Path) -> str:
    """sha256 of the file's bytes, read in 1 MiB pieces (audio can be large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_file(cache_dir: str | Path, digest: str, model_size: str) -> Path:
    safe_model = re.sub(r"[^\w.-]", "_", model_size)
    return Path(cache_dir) / f"{digest[:16]}.{safe_model}.json"


def load(cache_dir: str | Path, digest: str, model_size: str) -> dict | None:
    """The cached transcript for these exact audio bytes and model size, or
    None. A missing, unreadable, malformed or mismatched file all return None
    (and a corrupt one logs a warning), so the caller simply re-transcribes."""
    path = cache_file(cache_dir, digest, model_size)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA or data["sha256"] != digest or data["model_size"] != model_size:
            return None
        segments = data["segments"]
        for seg in segments:
            if not isinstance(seg["text"], str) or not isinstance(seg["start"], (int, float)) \
                    or not isinstance(seg["end"], (int, float)):
                raise TypeError("segment fields have the wrong types")
        return data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("ignoring unreadable transcript cache %s (%s); re-transcribing", path.name, exc)
        return None


def store(
    cache_dir: str | Path,
    digest: str,
    model_size: str,
    source_name: str,
    duration: float,
    compute_type: str,
    segments: list[dict],
) -> Path:
    """Write the transcript atomically (a temp file, then a rename), so an
    interrupted run can never leave a half-written cache file behind."""
    path = cache_file(cache_dir, digest, model_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from importlib.metadata import version

        whisper_version = version("faster-whisper")
    except Exception:  # noqa: BLE001 - provenance only; never block a transcript on it
        whisper_version = "unknown"
    payload = {
        "schema": SCHEMA,
        "sha256": digest,
        "source_name": source_name,
        "model_size": model_size,
        "compute_type": compute_type,
        "faster_whisper_version": whisper_version,
        "created": date.today().isoformat(),
        "duration": round(float(duration), 2),
        "segments": segments,
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path
