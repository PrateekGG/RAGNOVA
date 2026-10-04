"""
Chapter 12's integration suite: the seams where the three tracks meet.

Run with:  pytest tests/test_integration.py -v

Unlike tests/test_retrieval.py (which runs the real MiniLM model), every
model here is replaced by a small, deterministic fake: a bag-of-words
"embedder" for text, fixed vectors for CLIP, a canned transcript for
Whisper, a recorder for Ollama. That is on purpose. These tests check
the WIRING (does an image land in image_index with a portable source?
does an audio chunk come back with its timestamps? does the merge
interleave by rank?), and wiring bugs should fail in two seconds on any
laptop, with no model downloads. Whether the real models give good
answers is what scripts/evaluate_answers.py and the human feedback loop
measure.

ChromaDB is real throughout, in a pytest tmp_path — never the real chroma_db/.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.schemas import Chunk, validate_chunk
from src.core.vector_store import get_client, get_image_collection, get_text_collection
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.answer import NOT_ENOUGH_INFO, answer_query, check_citations, stream_answer
from src.pipelines.rag.prompt import build_prompt, format_provenance
from src.pipelines.rag.retrieve import filter_by_floor, retrieve, rrf_merge
from src.ui import backend
from src.ui.citations import DOCX_PAGE_NOTE, citation_views
from src.ui.feedback import FeedbackEntry, load_feedback, record_feedback, summarize

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def fake_embed_texts(texts: list[str]) -> list[list[float]]:
    """384-d bag-of-words vectors: each word adds 1 to a hashed dimension,
    then the vector is scaled to unit length. Texts sharing words get a
    high cosine score; texts sharing none score 0. Enough to make
    retrieval deterministic without MiniLM."""
    vectors = []
    for text in texts:
        v = [0.0] * 384
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    return vectors


def fake_embed_text(text: str) -> list[float]:
    return fake_embed_texts([text])[0]


@pytest.fixture
def fake_text_model(monkeypatch):
    """Swap MiniLM for the fake everywhere a module imported it by name."""
    monkeypatch.setattr("src.pipelines.documents.search.embed_text", fake_embed_text)
    monkeypatch.setattr("src.pipelines.documents.index.embed_texts", fake_embed_texts)
    monkeypatch.setattr("src.pipelines.audio.index.embed_texts", fake_embed_texts)


def _unit(i: int, dim: int = 512) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


class FakeClip:
    """Stands in for OpenCLIPEmbedder. Red images -> axis 0, blue -> axis 1;
    the text "red" -> axis 0, anything else -> axis 1."""
    model_id = "fake-clip/test"

    def embed_batch(self, images):
        return [self.embed_image(img) for img in images]

    def embed_image(self, image):
        r, _g, b = image.convert("RGB").getpixel((0, 0))
        return _unit(0) if r > b else _unit(1)

    def embed_text(self, text):
        return _unit(0) if "red" in text.lower() else _unit(1)


def _chunk(cid, modality="pdf", score=0.5, text="some text", source="data/documents/notice.pdf", **kw):
    if modality in ("pdf", "docx"):
        kw.setdefault("page", 1)
    return Chunk(chunk_id=cid, source=source, modality=modality, text=text,
                 embedding_model="m", score=score, **kw)


# ---------------------------------------------------------------------------
# Part 1 — rrf_merge() and filter_by_floor(): pure, no I/O (ADR-007, ADR-009)
# ---------------------------------------------------------------------------

def test_rrf_merge_interleaves_two_lists_by_rank_not_score():
    # Text scores are all much higher than image scores (the modality gap).
    # A raw-score sort would put every text chunk first; rank merge must not.
    text = [_chunk("t1", score=0.9), _chunk("t2", score=0.8)]
    images = [_chunk("i1", modality="image", score=0.25, source="data/images/a.png"),
              _chunk("i2", modality="image", score=0.21, source="data/images/b.png")]
    merged = rrf_merge([text, images])
    assert [c.chunk_id for c in merged] == ["t1", "i1", "t2", "i2"]


def test_rrf_merge_rewards_a_chunk_found_in_both_lists():
    a = [_chunk("x"), _chunk("shared")]
    b = [_chunk("y"), _chunk("shared")]
    merged = rrf_merge([a, b])
    assert merged[0].chunk_id == "shared"
    assert len(merged) == 3


def test_rrf_merge_keeps_each_chunks_original_score():
    merged = rrf_merge([[_chunk("t1", score=0.42)]])
    assert merged[0].score == 0.42


def test_rrf_merge_of_nothing_is_empty():
    assert rrf_merge([[], []]) == []


def test_filter_by_floor_is_inclusive_and_drops_unscored():
    kept = filter_by_floor([_chunk("a", score=0.2), _chunk("b", score=0.19), _chunk("c", score=None)], 0.2)
    assert [c.chunk_id for c in kept] == ["a"]


# ---------------------------------------------------------------------------
# Part 2 — retrieve(): per-collection floors, then merge
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_searches(monkeypatch):
    calls = {}

    def fake_search_text(query, top_k=None, client=None):
        calls["text"] = query
        return [_chunk("t_hi", score=0.5), _chunk("t_lo", score=0.25)]

    def fake_search_images(query_text=None, query_image=None, top_k=None, client=None):
        calls["image"] = query_image if query_image is not None else query_text
        return [_chunk("i_hi", modality="image", score=0.25, source="data/images/a.png"),
                _chunk("i_lo", modality="image", score=0.1, source="data/images/b.png")]

    monkeypatch.setattr(retrieve_module, "search_text", fake_search_text)
    return calls, fake_search_images


def test_retrieve_text_only_matches_chapter_10_behaviour(fake_searches):
    calls, _ = fake_searches
    assert [c.chunk_id for c in retrieve("q")] == ["t_hi"]
    assert "image" not in calls


def test_retrieve_uses_a_separate_floor_for_images(fake_searches):
    # 0.25 fails the text floor (0.3) but passes the image floor (0.2):
    # exactly the modality-gap case one shared number would get wrong.
    _, image_search = fake_searches
    ids = [c.chunk_id for c in retrieve("q", include_images=True, image_search=image_search)]
    assert ids == ["t_hi", "i_hi"]


def test_retrieve_searches_images_by_the_uploaded_image_when_given(fake_searches):
    calls, image_search = fake_searches
    img = Image.new("RGB", (4, 4))
    retrieve("q", include_images=True, query_image=img, image_search=image_search)
    assert calls["image"] is img


def test_retrieve_searches_images_when_given_an_image_even_without_include_images(fake_searches):
    # A passed-in query_image must never be silently dropped just because
    # the caller forgot include_images=True — the image itself is enough
    # of a signal that image search was wanted.
    calls, image_search = fake_searches
    img = Image.new("RGB", (4, 4))
    ids = [c.chunk_id for c in retrieve("q", query_image=img, image_search=image_search)]
    assert calls["image"] is img
    assert "i_hi" in ids


def test_retrieve_skips_text_search_for_an_empty_question(fake_searches):
    calls, image_search = fake_searches
    ids = [c.chunk_id for c in retrieve("  ", include_images=True, image_search=image_search)]
    assert "text" not in calls
    assert ids == ["i_hi"]


# ---------------------------------------------------------------------------
# Part 3 — the RAG core with images in the prompt, and the streaming path
# ---------------------------------------------------------------------------

def test_prompt_labels_images_and_explains_an_image_without_text():
    with_text = _chunk("i1", modality="image", text="LIBRARY FINES Rs 5 per day", source="data/images/fines.png")
    without = _chunk("i2", modality="image", text="", source="data/images/map.png")
    prompt = build_prompt("q", [with_text, without])
    assert "[1] data/images/fines.png (image)\nText read from the image (OCR): LIBRARY FINES" in prompt
    assert "[2] data/images/map.png (image)\n(An image matching the question. No readable text" in prompt


def test_answer_query_with_images_sends_merged_context_to_the_llm(monkeypatch, fake_searches):
    _, image_search = fake_searches
    monkeypatch.setattr(
        answer_module, "retrieve",
        lambda q, **kw: retrieve(q, image_search=image_search, **kw),
    )
    seen = {}
    monkeypatch.setattr(answer_module, "generate", lambda p: seen.setdefault("prompt", p) and "Answer [1][2].")
    result = answer_query("q", include_images=True)
    assert [c.chunk_id for c in result.citations] == ["t_hi", "i_hi"]
    assert "[2] data/images/a.png (image)" in seen["prompt"]


def test_stream_answer_short_circuits_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [])
    monkeypatch.setattr(answer_module, "generate_stream", lambda p: pytest.fail("LLM must not be called"))
    citations, pieces = stream_answer("What is the capital of France?")
    assert citations == []
    assert "".join(pieces) == NOT_ENOUGH_INFO


def test_stream_answer_streams_the_llm_output(monkeypatch):
    chunk = _chunk("t1")
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [chunk])
    monkeypatch.setattr(answer_module, "generate_stream", lambda p: iter(["Forty ", "percent [1]."]))
    citations, pieces = stream_answer("q")
    assert citations == [chunk]
    assert "".join(pieces) == "Forty percent [1]."


def test_image_only_question_gets_a_real_question_in_the_prompt(monkeypatch):
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [_chunk("i1", modality="image", source="x.png")])
    seen = {}
    monkeypatch.setattr(answer_module, "generate", lambda p: seen.setdefault("prompt", p))
    answer_query("", include_images=True)
    assert "Question: The user attached an image without a written question" in seen["prompt"]


def test_check_citations_reports_out_of_range_numbers():
    assert check_citations("A [1], B [3].", [_chunk("a"), _chunk("b")], "q") == {3}
    assert check_citations("A [1][2].", [_chunk("a"), _chunk("b")], "q") == set()


# ---------------------------------------------------------------------------
# Part 4 — image_index: the Chapter 8 pipeline, written into ChromaDB
# ---------------------------------------------------------------------------

@pytest.fixture
def image_pipeline():
    from src.pipelines.images.ingest import ImageIngestionPipeline
    from src.pipelines.images.models import ImageIngestionConfig

    return ImageIngestionPipeline(config=ImageIngestionConfig(ocr_enabled=False), embedder=FakeClip())


def test_images_are_indexed_with_portable_sources_and_found_by_text(tmp_path, monkeypatch, image_pipeline):
    from src.pipelines.images.index import index_images_directory
    from src.pipelines.images.search import search_images

    monkeypatch.chdir(tmp_path)  # tmp_path plays the project root
    images = Path("data/images")
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(images / "red.png")
    Image.new("RGB", (8, 8), (0, 0, 255)).save(images / "blue.png")
    (images / "notes.txt").write_text("not an image")

    client = get_client(persist_dir=tmp_path / "chroma")
    assert index_images_directory(images, client=client, pipeline=image_pipeline) == 2

    stored = get_image_collection(client).get(include=["metadatas"])
    sources = sorted(m["source"] for m in stored["metadatas"])
    assert sources == ["data/images/blue.png", "data/images/red.png"]  # relative, not /tmp/...

    hits = search_images(query_text="a red square", client=client, embedder=FakeClip())
    assert hits[0].source == "data/images/red.png"
    assert hits[0].score == pytest.approx(1.0)
    assert validate_chunk(hits[0]) == []


def test_image_search_on_an_empty_index_returns_nothing_without_loading_clip(tmp_path):
    from src.pipelines.images.search import search_images

    client = get_client(persist_dir=tmp_path / "chroma")
    assert search_images(query_text="anything", client=client, embedder=object()) == []


def test_image_search_by_image(tmp_path, monkeypatch, image_pipeline):
    from src.pipelines.images.index import index_image_files
    from src.pipelines.images.search import search_images

    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (8, 8), (0, 0, 255)).save("blue.png")
    Image.new("RGB", (8, 8), (255, 0, 0)).save("red.png")
    client = get_client(persist_dir=tmp_path / "chroma")
    index_image_files(["blue.png", "red.png"], client=client, pipeline=image_pipeline)
    hits = search_images(query_image=Image.new("RGB", (8, 8), (10, 0, 200)), client=client, embedder=FakeClip())
    assert hits[0].source == "blue.png"


# ---------------------------------------------------------------------------
# Part 5 — audio: transcripts into text_index, next to documents
# ---------------------------------------------------------------------------

class FakeIngestor:
    """Stands in for AudioIngestor: returns two transcript chunks."""

    def __init__(self, embedding_model=settings.TEXT_EMBEDDING_MODEL):
        self.embedding_model = embedding_model

    def process_file(self, path):
        return [
            Chunk(chunk_id="talk_wav__t0__c000", source=str(path), modality="audio",
                  text="Welcome  to the library orientation.\nThe fine is five rupees per day.",
                  embedding_model=self.embedding_model, start_s=0.0, end_s=12.5),
            Chunk(chunk_id="talk_wav__t12__c001", source=str(path), modality="audio",
                  text="The wifi password is on the back of your ID card.",
                  embedding_model=self.embedding_model, start_s=12.5, end_s=20.0),
        ]


def test_audio_is_indexed_into_text_index_and_cited_by_timestamp(tmp_path, monkeypatch, fake_text_model):
    from src.pipelines.audio.index import index_audio_file
    from src.pipelines.documents.search import search_text

    monkeypatch.chdir(tmp_path)
    audio = Path("data/audio")
    audio.mkdir(parents=True)
    (audio / "talk.wav").write_bytes(b"RIFF fake")

    client = get_client(persist_dir=tmp_path / "chroma")
    assert index_audio_file((audio / "talk.wav").resolve(), client=client, ingestor=FakeIngestor()) == 2

    hits = search_text("what is the library fine per day", client=client)
    top = hits[0]
    assert top.modality == "audio"
    assert top.source == "data/audio/talk.wav"
    assert (top.start_s, top.end_s) == (0.0, 12.5)
    assert top.text.startswith("Welcome to the library")  # whitespace normalized
    assert format_provenance(top) == "data/audio/talk.wav, 0s–12s"


def test_audio_chunks_that_break_the_contract_are_refused(tmp_path, fake_text_model):
    # The exact bug PR #5 fixes in AudioIngestor: embedding_model=None.
    from src.pipelines.audio.index import index_audio_file

    client = get_client(persist_dir=tmp_path / "chroma")
    with pytest.raises(ValueError, match="embedding_model"):
        index_audio_file(tmp_path / "talk.wav", client=client, ingestor=FakeIngestor(embedding_model=None))
    assert get_text_collection(client).count() == 0


def test_transcribe_query_joins_segments(tmp_path):
    from types import SimpleNamespace

    from src.pipelines.audio.transcribe import transcribe_query

    class FakeWhisper:
        def transcribe(self, path, vad_filter=True):
            segs = [SimpleNamespace(text=" How late is "), SimpleNamespace(text=""), SimpleNamespace(text="the library open? ")]
            return iter(segs), None

    assert transcribe_query(tmp_path / "q.wav", model=FakeWhisper()) == "How late is the library open?"


# ---------------------------------------------------------------------------
# Part 6 — uploads: save -> index -> searchable, the path the UI drives
# ---------------------------------------------------------------------------

def test_kind_of_and_safe_filename():
    assert backend.kind_of("Notice.PDF") == "document"
    assert backend.kind_of("shot.jpeg") == "image"
    assert backend.kind_of("clip.m4a") == "audio"
    assert backend.kind_of("virus.exe") is None
    assert backend.safe_filename("../../etc/My File (1).PDF") == "My_File_1.pdf"


def test_uploaded_pdf_becomes_searchable(tmp_path, monkeypatch, fake_text_model):
    monkeypatch.chdir(tmp_path)
    data = (DOCS_DIR / "library_hours.pdf").read_bytes()
    path = backend.save_upload("Library Hours.pdf", data, data_root=Path("data"))
    assert path == Path("data/documents/Library_Hours.pdf")

    client = get_client(persist_dir=tmp_path / "chroma")
    assert backend.index_file(path, client=client) > 0
    assert backend.index_counts(client=client) == {"text": get_text_collection(client).count(), "image": 0}
    from src.pipelines.documents.search import search_text

    hit = search_text("library borrowing books fine", client=client)[0]
    assert hit.source == "data/documents/Library_Hours.pdf"


def test_save_upload_refuses_unsupported_types(tmp_path):
    with pytest.raises(ValueError):
        backend.save_upload("script.sh", b"echo", data_root=tmp_path)


# ---------------------------------------------------------------------------
# Part 7 — what the UI shows for each citation, and the feedback log
# ---------------------------------------------------------------------------

def test_citation_views_follow_the_adrs():
    views = citation_views([
        _chunk("p", modality="pdf", page=2),
        _chunk("d", modality="docx", source="data/documents/it_onboarding.docx", page=2),
        _chunk("i", modality="image", text="", source="data/images/map.png"),
        Chunk(chunk_id="a", source="data/audio/talk.wav", modality="audio", text="hi",
              embedding_model="m", start_s=12.5, end_s=20.0),
    ])
    assert [v.number for v in views] == [1, 2, 3, 4]
    assert views[0].title == "[1] notice.pdf, page 2" and views[0].note is None
    assert views[0].file_exists  # the real generated corpus file
    assert views[1].note == DOCX_PAGE_NOTE  # ADR-008
    assert views[2].excerpt == "(No readable text in this image.)"
    assert views[3].audio_start_s == 12.5 and views[3].title == "[4] talk.wav, 12s–20s"
    # ADR-003: no score is ever part of what's displayed
    assert all("0.5" not in v.title for v in views)


def test_feedback_round_trip_and_summary(tmp_path):
    log = tmp_path / "fb" / "feedback.jsonl"
    record_feedback(FeedbackEntry(query="q1", answer="a1", rating=5, sources=["[1] x"], tester="team"), path=log)
    record_feedback(FeedbackEntry(query="q2", answer="a2", rating=2, sources=[], comment="wrong page"), path=log)
    with log.open("a") as f:
        f.write("{not json\n")  # one corrupt line must not lose the others
    entries = load_feedback(log)
    assert [e.rating for e in entries] == [5, 2]
    assert json.loads(log.read_text().splitlines()[0])["tester"] == "team"
    s = summarize(entries)
    assert s == {"count": 2, "average": 3.5, "histogram": {1: 0, 2: 1, 3: 0, 4: 0, 5: 1}, "low": 1}


def test_feedback_rejects_out_of_scale_ratings(tmp_path):
    with pytest.raises(ValueError):
        record_feedback(FeedbackEntry(query="q", answer="a", rating=0, sources=[]), path=tmp_path / "f.jsonl")


def test_missing_feedback_log_is_empty():
    assert load_feedback("/nonexistent/feedback.jsonl") == []
    assert summarize([])["count"] == 0
