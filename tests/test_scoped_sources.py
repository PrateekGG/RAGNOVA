"""
Answering from the files the user picked (2026-10-10).

Why it exists: a CV was uploaded through "Add to Corpus" and indexed (2 chunks), and
5 of 6 questions about it still got "I don't have enough information", because every
question is gated on a corpus-wide relevance floor (0.3) and "who is <the person>?"
scores below it against a long CV chunk. The gate is what refuses questions the corpus
cannot answer (the negative controls), so it stays the default. What the user may do is
say where the answer lives: with a scope, search is limited to those files and the floor
and the image gate are not applied. These tests pin what that must and must not change:

  - the filter really limits the search (real ChromaDB, fake vectors),
  - the DEFAULT, no scope, is exactly what it was (floor still drops weak chunks),
  - a scope turns the floor and the gate off, and only then,
  - an empty scope matches nothing instead of silently meaning "everything",
  - the helpers the page uses to list files and to select an upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.core.vector_store import add_chunks, get_client, get_image_collection, get_text_collection
from src.pipelines.documents import search as text_search
from src.pipelines.images import search as image_search_module
from src.pipelines.rag import retrieve as retrieve_module
from src.ui import backend


def _chunk(chunk_id: str, source: str, modality: str = "pdf", score: float | None = None, text: str = "words") -> Chunk:
    return Chunk(chunk_id=chunk_id, source=source, modality=modality, text=text, embedding_model="m",
                 page=1 if modality == "pdf" else None, score=score)


# ---------------------------------------------------------------------------
# The filter, against a real ChromaDB
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    return get_client(persist_dir=tmp_path / "chroma")


@pytest.fixture
def two_files(client, monkeypatch):
    """a.pdf is FAR from the query vector, b.pdf is next to it: unscoped, b wins every time."""
    collection = get_text_collection(client)
    add_chunks(collection, [_chunk("a1", "data/documents/a.pdf"), _chunk("a2", "data/documents/a.pdf")],
               [[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]])
    add_chunks(collection, [_chunk("b1", "data/documents/b.pdf")], [[1.0, 0.0, 0.0]])
    monkeypatch.setattr(text_search, "embed_text", lambda text: [1.0, 0.0, 0.0])
    return client


def _ids(chunks):
    return [c.chunk_id for c in chunks]


def test_a_scope_limits_the_text_search_to_those_files_even_when_another_file_is_nearer(two_files):
    assert _ids(text_search.search_text("q", top_k=5, client=two_files)) == ["b1", "a2", "a1"]
    assert set(_ids(text_search.search_text("q", top_k=5, client=two_files, sources=["data/documents/a.pdf"]))) == {"a1", "a2"}


def test_a_scope_of_several_files_takes_each_of_them(two_files):
    both = ["data/documents/a.pdf", "data/documents/b.pdf"]
    assert set(_ids(text_search.search_text("q", top_k=5, client=two_files, sources=both))) == {"a1", "a2", "b1"}


def test_a_scope_naming_a_file_that_is_not_indexed_finds_nothing(two_files):
    assert text_search.search_text("q", top_k=5, client=two_files, sources=["data/documents/nope.pdf"]) == []


def test_an_empty_scope_matches_nothing_and_does_not_mean_everything(two_files):
    assert text_search.search_text("q", top_k=5, client=two_files, sources=[]) == []


class _Clip:
    """Stands in for CLIP: every text and every image is the same vector."""

    def embed_text(self, text):
        return [1.0, 0.0]

    def embed_image(self, image):
        return [1.0, 0.0]


@pytest.fixture
def two_images(client):
    collection = get_image_collection(client)
    add_chunks(collection, [_chunk("i1", "data/images/one.png", "image"), _chunk("i2", "data/images/two.png", "image")],
               [[1.0, 0.0], [0.0, 1.0]])
    return client


def test_a_scope_limits_the_image_search_too(two_images):
    found = image_search_module.search_images(query_text="q", top_k=5, client=two_images, embedder=_Clip(),
                                              sources=["data/images/two.png"])
    assert _ids(found) == ["i2"]
    assert image_search_module.search_images(query_text="q", top_k=5, client=two_images, embedder=_Clip(), sources=[]) == []


def test_a_scope_limits_the_ocr_text_image_search_too(two_images):
    from src.core.vector_store import get_image_text_collection

    add_chunks(get_image_text_collection(two_images),
               [_chunk("i1", "data/images/one.png", "image", text="alpha"), _chunk("i2", "data/images/two.png", "image", text="beta")],
               [[1.0, 0.0], [0.9, 0.1]])
    found = image_search_module.search_image_text("q", top_k=5, client=two_images, embed=lambda t: [1.0, 0.0],
                                                  sources=["data/images/two.png"])
    assert _ids(found) == ["i2"]


# ---------------------------------------------------------------------------
# retrieve(): the default is unchanged, a scope removes the floor and the gate
# ---------------------------------------------------------------------------

WEAK = 0.05          # far below MIN_RELEVANCE_SCORE (0.3) and MIN_IMAGE_RELEVANCE_SCORE (0.2)


@pytest.fixture
def seen(monkeypatch):
    """retrieve() with both searches replaced, recording the `sources` each one was given."""
    calls = {"text": [], "image": []}
    weak_text = _chunk("t1", "data/documents/cv.pdf", score=WEAK)
    weak_image = _chunk("i1", "data/images/shot.png", "image", score=WEAK, text="some words")

    def fake_text(query, top_k=None, client=None, sources=None):
        calls["text"].append(sources)
        return [weak_text]

    def fake_image(query_text=None, query_image=None, top_k=None, client=None, sources=None):
        calls["image"].append(sources)
        return [weak_image]

    monkeypatch.setattr(retrieve_module, "search_text", fake_text)
    return calls, fake_image


def test_without_a_scope_the_floor_still_drops_a_weak_chunk(seen):
    calls, _ = seen
    assert retrieve_module.retrieve("who is this person") == []
    assert calls["text"] == [None]                    # nothing was restricted


def test_with_a_scope_a_weak_chunk_is_kept_and_the_search_was_restricted(seen):
    calls, _ = seen
    found = retrieve_module.retrieve("who is this person", sources=["data/documents/cv.pdf"])
    assert _ids(found) == ["t1"]
    assert calls["text"] == [["data/documents/cv.pdf"]]


def test_without_a_scope_the_image_gate_still_drops_a_weak_image(seen):
    _, fake_image = seen
    found = retrieve_module.retrieve("what is on the notice", include_images=True, image_search=fake_image,
                                     image_agreement=lambda q, c: 0.0)
    assert found == []


def test_with_a_scope_a_weak_image_is_kept_and_the_image_search_was_restricted(seen):
    calls, fake_image = seen
    found = retrieve_module.retrieve("what is on the notice", include_images=True, image_search=fake_image,
                                     image_agreement=lambda q, c: 0.0, sources=["data/images/shot.png"])
    assert "i1" in _ids(found)
    assert calls["image"] == [["data/images/shot.png"]]


def test_with_a_scope_an_attached_query_image_searches_only_the_scope_and_has_no_floor(seen):
    calls, fake_image = seen
    found = retrieve_module.retrieve("what is this", query_image=object(), image_search=fake_image,
                                     sources=["data/images/shot.png"])
    assert "i1" in _ids(found)
    assert calls["image"] == [["data/images/shot.png"]]


def test_with_a_scope_and_a_blank_question_no_image_search_runs_on_nothing(seen):
    calls, fake_image = seen
    assert retrieve_module.retrieve("   ", include_images=True, image_search=fake_image, sources=["data/images/shot.png"]) == []
    assert calls["image"] == []


def test_an_empty_scope_reaches_the_search_as_an_empty_scope_not_as_none(seen):
    calls, _ = seen
    retrieve_module.retrieve("anything", sources=[])
    assert calls["text"] == [[]]


# ---------------------------------------------------------------------------
# What the page calls: listing the indexed files, selecting an upload
# ---------------------------------------------------------------------------

def test_list_sources_is_every_indexed_file_once_sorted_from_both_collections(client):
    add_chunks(get_text_collection(client),
               [_chunk("a1", "data/documents/a.pdf"), _chunk("a2", "data/documents/a.pdf"), _chunk("w1", "data/audio/w.wav", "audio")],
               [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    add_chunks(get_image_collection(client), [_chunk("i1", "data/images/i.png", "image")], [[1.0, 0.0]])
    assert backend.list_sources(client) == ["data/audio/w.wav", "data/documents/a.pdf", "data/images/i.png"]


def test_list_sources_of_an_empty_index_is_empty(client):
    assert backend.list_sources(client) == []


def test_an_upload_is_matched_to_its_index_entry_by_file_name():
    listed = ["data/documents/a.pdf", "data/documents/cv.pdf", "data/images/cv.png"]
    assert backend.sources_for_upload(Path("X:/anywhere/data/documents/cv.pdf"), listed) == ["data/documents/cv.pdf"]
    assert backend.sources_for_upload(Path("cv.docx"), listed) == []


def test_a_file_that_indexes_is_saved_and_becomes_the_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "index_file", lambda path, client=None: 2)
    monkeypatch.setattr(backend, "list_sources", lambda client=None: ["data/documents/other.pdf", "data/documents/cv.pdf"])

    result = backend.add_to_corpus("cv.pdf", b"%PDF", data_root=tmp_path)

    assert (tmp_path / "documents" / "cv.pdf").read_bytes() == b"%PDF"
    assert result.ok and "2 chunks" in result.message
    assert result.focus == ["data/documents/cv.pdf"]


def test_a_file_that_indexes_to_nothing_is_reported_as_a_failure_and_changes_no_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "index_file", lambda path, client=None: 0)
    monkeypatch.setattr(backend, "list_sources", lambda client=None: [])

    result = backend.add_to_corpus("scan.pdf", b"%PDF", data_root=tmp_path)

    assert not result.ok and "0 chunks" in result.message
    assert result.focus == []


def test_an_unsupported_file_is_still_refused_before_anything_is_saved(tmp_path):
    with pytest.raises(ValueError):
        backend.add_to_corpus("notes.exe", b"MZ", data_root=tmp_path)
    assert not list(tmp_path.rglob("*.exe"))


# ---------------------------------------------------------------------------
# What the page says about an attached picture
# ---------------------------------------------------------------------------

def test_the_note_for_a_picture_with_text_says_how_much_was_read():
    note = backend.attachment_note("shot.png", "one two three four", tesseract_installed=True)
    assert "`shot.png`" in note and "read 4 words" in note and "Only the first" not in note


def test_the_note_for_a_very_long_picture_text_says_it_was_cut():
    from src.pipelines.rag.attachments import ATTACHMENT_MAX_WORDS

    note = backend.attachment_note("busy.png", " ".join(["w"] * (ATTACHMENT_MAX_WORDS + 1)), tesseract_installed=True)
    assert f"Only the first {ATTACHMENT_MAX_WORDS} words are used" in note


def test_a_picture_with_no_text_and_a_machine_with_no_tesseract_are_told_apart():
    no_text = backend.attachment_note("photo.jpg", "  ", tesseract_installed=True)
    no_tool = backend.attachment_note("photo.jpg", "", tesseract_installed=False)
    assert "no text could be read" in no_text and "Tesseract" not in no_text
    assert "Tesseract is not installed" in no_tool and no_text != no_tool
