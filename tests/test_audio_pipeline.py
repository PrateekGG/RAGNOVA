from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.schemas import Chunk, validate_chunk
from src.core.vector_store import get_client, get_text_collection
from src.pipelines.rag.prompt import format_provenance


@pytest.fixture(autouse=True)
def _no_transcript_cache_by_default(monkeypatch):
    """The transcript cache (src/pipelines/audio/transcript_cache.py) is OFF
    for every test in this module unless a test turns it on with `use_cache`.
    Left on, these tests would write fake transcripts into the real
    data/transcripts/ folder, and one test's fake transcript would be served
    to the next test that happens to use the same fake audio bytes."""
    from src.pipelines.audio import ingestion

    monkeypatch.setattr(ingestion, "settings", dataclasses.replace(ingestion.settings, WHISPER_TRANSCRIPT_CACHE_DIR=""))


@pytest.fixture
def use_cache(tmp_path, monkeypatch):
    """Turn the transcript cache on, in a throwaway folder; returns that folder."""
    from src.pipelines.audio import ingestion

    cache_dir = tmp_path / "transcripts"
    monkeypatch.setattr(
        ingestion, "settings", dataclasses.replace(ingestion.settings, WHISPER_TRANSCRIPT_CACHE_DIR=str(cache_dir))
    )
    return cache_dir


def fake_embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * 384
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def fake_embed_text(text: str) -> list[float]:
    return fake_embed_texts([text])[0]


@pytest.fixture
def fake_text_model(monkeypatch):
    monkeypatch.setattr("src.pipelines.documents.search.embed_text", fake_embed_text)
    monkeypatch.setattr("src.pipelines.audio.index.embed_texts", fake_embed_texts)


class FakeIngestor:
    def __init__(self, embedding_model=settings.TEXT_EMBEDDING_MODEL):
        self.embedding_model = embedding_model

    def process_file(self, path):
        return [
            Chunk(
                chunk_id="talk_wav__t0__c000",
                source=str(path),
                modality="audio",
                text="Welcome  to the library orientation.\nThe fine is five rupees per day.",
                embedding_model=self.embedding_model,
                start_s=0.0,
                end_s=12.5,
            ),
            Chunk(
                chunk_id="talk_wav__t12__c001",
                source=str(path),
                modality="audio",
                text="The wifi password is on the back of your ID card.",
                embedding_model=self.embedding_model,
                start_s=12.5,
                end_s=20.0,
            ),
        ]


def test_audio_is_indexed_into_text_index_and_cited_by_timestamp(
    tmp_path,
    monkeypatch,
    fake_text_model,
):
    from src.pipelines.audio.index import index_audio_file
    from src.pipelines.documents.search import search_text

    monkeypatch.chdir(tmp_path)
    audio = Path("data/audio")
    audio.mkdir(parents=True)
    (audio / "talk.wav").write_bytes(b"RIFF fake")

    client = get_client(persist_dir=tmp_path / "chroma")
    assert index_audio_file(
        (audio / "talk.wav").resolve(),
        client=client,
        ingestor=FakeIngestor(),
    ) == 2

    top = search_text("what is the library fine per day", client=client)[0]
    assert top.modality == "audio"
    assert top.source == "data/audio/talk.wav"
    assert (top.start_s, top.end_s) == (0.0, 12.5)
    assert top.text.startswith("Welcome to the library")
    assert format_provenance(top) == "data/audio/talk.wav, 0s–12s"


def test_audio_chunks_that_break_the_contract_are_refused(tmp_path, fake_text_model):
    from src.pipelines.audio.index import index_audio_file

    client = get_client(persist_dir=tmp_path / "chroma")
    with pytest.raises(ValueError, match="embedding_model"):
        index_audio_file(
            tmp_path / "talk.wav",
            client=client,
            ingestor=FakeIngestor(embedding_model=None),
        )
    assert get_text_collection(client).count() == 0


def test_transcribe_query_joins_segments(tmp_path):
    from src.pipelines.audio.transcribe import transcribe_query

    class FakeWhisper:
        def transcribe(self, path, vad_filter=True):
            segments = [
                SimpleNamespace(text=" How late is "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="the library open? "),
            ]
            return iter(segments), None

    assert transcribe_query(tmp_path / "q.wav", model=FakeWhisper()) == "How late is the library open?"


# ---------------------------------------------------------------------------
# The REAL AudioIngestor, with only Whisper faked.
#
# FakeIngestor above stands in for the whole ingestor, so none of the tests
# before this point ever ran AudioIngestor's own code. That let the same bugs
# come back several times while PR #5 was reviewed: embedding_model resolving
# to None (every chunk then fails validate_chunk), a `metadata=` argument that
# Chunk doesn't accept (TypeError on every chunk), chunk ids that changed when
# Whisper's timestamps jittered, and segments duplicated by a retry. These
# tests build the real AudioIngestor and fake only faster-whisper.
# ---------------------------------------------------------------------------

class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel. `attempts` has one entry per
    transcribe() call: a list of segments, optionally ending with an Exception
    that is raised after the segments before it have been yielded (a failure
    part-way through a lazy transcription)."""

    def __init__(self, attempts):
        self.attempts = list(attempts)

    def transcribe(self, path, vad_filter=True, vad_parameters=None):
        script = self.attempts.pop(0)

        def generate():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return generate(), SimpleNamespace(duration=20.0)


def _segment(text, start, end):
    return SimpleNamespace(text=text, start=start, end=end)


def _real_ingestor(monkeypatch, attempts, **kwargs):
    from src.pipelines.audio import ingestion

    model = FakeWhisperModel(attempts)
    monkeypatch.setattr(ingestion, "WhisperModel", lambda *args, **kw: model)
    monkeypatch.setattr(ingestion.time, "sleep", lambda *_: None)  # skip retry back-off
    return ingestion.AudioIngestor(**kwargs)


def _audio_file(directory, name="lecture.wav", content=b"RIFF fake"):
    path = directory / name
    path.write_bytes(content)
    return path


def test_real_audio_ingestor_chunks_satisfy_the_contract(tmp_path, monkeypatch):
    ingestor = _real_ingestor(monkeypatch, [[
        _segment("Welcome to the library orientation.", 0.0, 4.0),
        _segment("The fine is five rupees per day.", 4.0, 9.0),
    ]])

    chunks = ingestor.process_file(_audio_file(tmp_path))

    assert chunks
    for chunk in chunks:
        assert validate_chunk(chunk) == []
        assert chunk.modality == "audio"
        assert chunk.embedding_model == settings.TEXT_EMBEDDING_MODEL
    assert (chunks[0].start_s, chunks[-1].end_s) == (0.0, 9.0)


def test_real_audio_chunk_ids_ignore_timestamp_jitter_and_differ_per_file(tmp_path, monkeypatch):
    # Whisper's timestamps can shift slightly between two runs over the SAME
    # file; the id must not, or re-indexing adds new chunks instead of
    # replacing the old ones.
    def ids_for(path, start):
        ingestor = _real_ingestor(monkeypatch, [[_segment("Same words every time.", start, start + 3.0)]])
        return [chunk.chunk_id for chunk in ingestor.process_file(path)]

    lecture = _audio_file(tmp_path, "lecture.wav")
    assert ids_for(lecture, 3.99) == ids_for(lecture, 4.02)

    # A different file (here: different folder, same name, different size)
    # must not collide with it.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _audio_file(other_dir, "lecture.wav", content=b"RIFF a different, longer fake")
    assert ids_for(lecture, 3.99) != ids_for(other, 3.99)


def test_real_audio_ingestor_does_not_duplicate_segments_when_a_retry_restarts_transcription(tmp_path, monkeypatch):
    # Attempt 1 yields two segments, then fails. Attempt 2 starts from the
    # beginning of the file again, so it yields those same two plus a new one.
    first = _segment("Alpha beta gamma.", 0.0, 3.0)
    second = _segment("Delta epsilon zeta.", 3.0, 6.0)
    third = _segment("Eta theta iota.", 6.0, 9.0)
    ingestor = _real_ingestor(monkeypatch, [
        [first, second, RuntimeError("transient failure")],
        [first, second, third],
    ])

    chunks = ingestor.process_file(_audio_file(tmp_path))

    text = " ".join(chunk.text for chunk in chunks)
    for phrase in ("Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."):
        assert text.count(phrase) == 1


# ---------------------------------------------------------------------------
# The transcript cache: the same audio bytes always give the same chunks.
# `use_cache` (above) switches it on in a throwaway folder; in every other test
# it is off.
# ---------------------------------------------------------------------------

TALK = [_segment("Welcome to the library orientation.", 1.234567, 4.0), _segment("The fine is five rupees.", 4.0, 9.987654)]


def _chunk_view(chunks):
    return [(c.chunk_id, c.text, c.start_s, c.end_s) for c in chunks]


def test_a_second_ingest_of_the_same_audio_is_served_from_the_cache_and_is_identical(tmp_path, monkeypatch, use_cache):
    audio = _audio_file(tmp_path)
    first = _real_ingestor(monkeypatch, [TALK]).process_file(audio)
    assert len(list(use_cache.glob("*.json"))) == 1

    # attempts=[] means any call to Whisper would raise: a cache hit must not call it.
    second = _real_ingestor(monkeypatch, []).process_file(audio)

    assert _chunk_view(second) == _chunk_view(first)
    assert first[0].start_s == 1.23          # timestamps are rounded identically on a miss and a hit


def test_a_cache_hit_never_loads_whisper_at_all(tmp_path, monkeypatch, use_cache):
    from src.pipelines.audio import ingestion

    audio = _audio_file(tmp_path)
    _real_ingestor(monkeypatch, [TALK]).process_file(audio)

    def must_not_load(*args, **kwargs):
        raise AssertionError("Whisper was loaded although the transcript was cached")

    monkeypatch.setattr(ingestion, "WhisperModel", must_not_load)
    assert ingestion.AudioIngestor().process_file(audio)


def test_changing_the_audio_bytes_triggers_a_fresh_transcription(tmp_path, monkeypatch, use_cache):
    audio = _audio_file(tmp_path, content=b"RIFF the first recording")
    _real_ingestor(monkeypatch, [TALK]).process_file(audio)

    audio.write_bytes(b"RIFF a different recording")
    other = [_segment("Completely different speech.", 0.0, 5.0)]
    chunks = _real_ingestor(monkeypatch, [other]).process_file(audio)

    assert "Completely different speech." in chunks[0].text
    assert len(list(use_cache.glob("*.json"))) == 2


def test_the_whisper_model_size_is_part_of_the_cache_key(tmp_path, monkeypatch, use_cache):
    audio = _audio_file(tmp_path)
    _real_ingestor(monkeypatch, [TALK], model_size="base").process_file(audio)
    other = [_segment("Heard by the bigger model.", 0.0, 5.0)]
    chunks = _real_ingestor(monkeypatch, [other], model_size="small").process_file(audio)

    assert "bigger model" in chunks[0].text
    assert sorted(f.name.split(".")[1] for f in use_cache.glob("*.json")) == ["base", "small"]


def test_a_corrupt_cache_file_is_ignored_and_rewritten(tmp_path, monkeypatch, use_cache):
    audio = _audio_file(tmp_path)
    _real_ingestor(monkeypatch, [TALK]).process_file(audio)
    (cache_file,) = use_cache.glob("*.json")
    cache_file.write_text("{ this is not json", encoding="utf-8")

    chunks = _real_ingestor(monkeypatch, [TALK]).process_file(audio)

    assert chunks and "library orientation" in chunks[0].text
    import json
    assert json.loads(cache_file.read_text(encoding="utf-8"))["segments"]      # rewritten, valid again


def test_with_the_cache_disabled_nothing_is_written_and_whisper_runs_every_time(tmp_path, monkeypatch):
    from src.pipelines.audio import ingestion

    assert ingestion._transcript_cache_dir() is None          # the autouse fixture's default
    audio = _audio_file(tmp_path)
    _real_ingestor(monkeypatch, [TALK, TALK]).process_file(audio)
    _real_ingestor(monkeypatch, [TALK]).process_file(audio)
    assert not (tmp_path / "transcripts").exists()


def test_a_segment_never_ends_after_the_audio_does(tmp_path, monkeypatch):
    # FakeWhisperModel reports duration=20.0. A garbled decode once put a clip's
    # last segment at 80.6 s on a 72.6 s file.
    ingestor = _real_ingestor(monkeypatch, [[_segment("Intro words here.", 0.0, 5.0),
                                            _segment("Garbled tail.", 18.0, 25.5)]])
    chunks = ingestor.process_file(_audio_file(tmp_path))
    assert max(c.end_s for c in chunks) <= 20.0


def test_an_interrupted_transcription_does_not_leave_a_partial_cache(tmp_path, monkeypatch, use_cache):
    ingestor = _real_ingestor(monkeypatch, [TALK])
    stream = ingestor._transcribe_with_retry(_audio_file(tmp_path))
    next(stream)                 # take one segment, then abandon the rest
    stream.close()
    assert not list(use_cache.glob("*.json"))


def test_a_cache_write_failure_does_not_lose_the_transcript_or_retry_it(tmp_path, monkeypatch, use_cache):
    from src.pipelines.audio import transcript_cache

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(transcript_cache, "store", fail)
    # one attempt only: if the failed write triggered a retry, FakeWhisperModel would run out of attempts
    chunks = _real_ingestor(monkeypatch, [TALK]).process_file(_audio_file(tmp_path))
    assert chunks and "library orientation" in chunks[0].text


def test_real_whisper_on_a_real_clip_is_reproduced_exactly_from_the_cache(tmp_path, use_cache):
    # The point of the whole feature, with the real model: transcribe a real
    # corpus clip, then ingest it again, and get the identical chunks without
    # Whisper being loaded a second time.
    from src.pipelines.audio import ingestion

    clip = Path(__file__).resolve().parent.parent / "data" / "audio" / "it_helpdesk_wifi_instructions.wav"
    loads = []
    real_model = ingestion.WhisperModel
    ingestion.WhisperModel = lambda *a, **k: (loads.append(1), real_model(*a, **k))[1]
    try:
        try:
            first = ingestion.AudioIngestor().process_file(clip)
        except Exception as exc:  # the Whisper weights are not available on this machine
            pytest.skip(f"real Whisper model not available: {exc}")
        second = ingestion.AudioIngestor().process_file(clip)
    finally:
        ingestion.WhisperModel = real_model

    assert first and _chunk_view(second) == _chunk_view(first)
    assert len(loads) == 1, "Whisper was loaded again for a clip whose transcript was cached"
