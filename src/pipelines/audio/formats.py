"""
The audio file types the pipeline can read.

A module of its own, with no imports, so that code which only needs to know
"is this an audio file?" (the UI's upload gate, src/ui/backend.py) does not
have to import src/pipelines/audio/ingestion.py, which pulls in faster-whisper
and CTranslate2 (about 1.6 s at import, measured 2026-10-05). The upload gate
used to keep its own hand-copied list of 6 formats while the pipeline read 14,
so an .aac or .opus upload was refused although the pipeline could transcribe
it; both now read this one list.
"""

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".opus", ".mpeg", ".mp4", ".webm", ".wma", ".mka",
    ".3gp", ".amr",
})
