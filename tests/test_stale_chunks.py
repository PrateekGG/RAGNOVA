"""
The index used to be upsert-only: re-ingesting a file that now yielded FEWER
chunks left its old trailing chunks behind, still searchable and citable, text
the file no longer contained (a disclosed limitation until 2026-10-05).
replace_source_chunks() and prune_missing_sources() close it. These tests use a
real ChromaDB in a temp folder; embeddings are fake (the subject here is which
chunks exist, not what they mean).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.schemas import Chunk
from src.core.vector_store import get_client, get_text_collection, prune_missing_sources, replace_source_chunks


def _chunk(chunk_id: str, source: str, text: str = "words", page: int = 1) -> Chunk:
    return Chunk(chunk_id=chunk_id, source=source, modality="pdf", text=text, embedding_model="m", page=page)


def _vectors(n: int) -> list[list[float]]:
    return [[1.0, float(i + 1)] for i in range(n)]


@pytest.fixture
def collection(tmp_path):
    client = get_client(persist_dir=tmp_path / "chroma")
    return client.get_or_create_collection("stale_probe", metadata={"hnsw:space": "cosine"})


def _ids(collection, source=None) -> set[str]:
    where = {"source": source} if source else None
    return set(collection.get(where=where, include=[])["ids"])


# ---------------------------------------------------------------------------
# replace_source_chunks
# ---------------------------------------------------------------------------

def test_a_file_that_shrinks_loses_its_old_trailing_chunks(collection):
    three = [_chunk(f"a__c{i}", "data/documents/a.pdf") for i in range(3)]
    replace_source_chunks(collection, "data/documents/a.pdf", three, _vectors(3))
    assert _ids(collection) == {"a__c0", "a__c1", "a__c2"}

    removed = replace_source_chunks(collection, "data/documents/a.pdf", three[:1], _vectors(1))

    assert removed == 2
    assert _ids(collection) == {"a__c0"}


def test_a_file_that_now_yields_nothing_loses_every_chunk_but_other_files_are_untouched(collection):
    replace_source_chunks(collection, "a.pdf", [_chunk("a0", "a.pdf"), _chunk("a1", "a.pdf")], _vectors(2))
    replace_source_chunks(collection, "b.pdf", [_chunk("b0", "b.pdf")], _vectors(1))

    assert replace_source_chunks(collection, "a.pdf", [], []) == 2

    assert _ids(collection) == {"b0"}


def test_replacing_with_the_same_chunks_twice_changes_nothing(collection):
    chunks = [_chunk("a0", "a.pdf"), _chunk("a1", "a.pdf")]
    replace_source_chunks(collection, "a.pdf", chunks, _vectors(2))
    assert replace_source_chunks(collection, "a.pdf", chunks, _vectors(2)) == 0
    assert _ids(collection) == {"a0", "a1"}


def test_new_chunks_are_written_before_stale_ones_are_deleted(collection, monkeypatch):
    # If deleting the stale chunks fails, the file must still have its NEW
    # chunks (add-then-delete), not be left empty (delete-then-add).
    replace_source_chunks(collection, "a.pdf", [_chunk("old0", "a.pdf"), _chunk("old1", "a.pdf")], _vectors(2))

    def broken_delete(*args, **kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(collection, "delete", broken_delete)
    with pytest.raises(RuntimeError):
        replace_source_chunks(collection, "a.pdf", [_chunk("new0", "a.pdf")], _vectors(1))

    assert "new0" in _ids(collection, "a.pdf")


# ---------------------------------------------------------------------------
# through the real index functions
# ---------------------------------------------------------------------------

def _make_pdf(path: Path, words: int) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(36, 36, 560, 800), " ".join(f"word{i}" for i in range(words)), fontsize=6)
    doc.save(path)
    doc.close()


def test_reindexing_a_shorter_pdf_removes_the_chunks_it_no_longer_has(tmp_path, monkeypatch):
    from src.pipelines.documents import index as doc_index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doc_index, "embed_texts", lambda texts: [[1.0, float(i + 1)] for i, _ in enumerate(texts)])
    folder = Path("data/documents")
    folder.mkdir(parents=True)
    client = get_client(persist_dir=tmp_path / "chroma")

    _make_pdf(folder / "report.pdf", words=900)
    before = doc_index.index_document_file(folder / "report.pdf", client=client)
    assert before >= 3

    _make_pdf(folder / "report.pdf", words=80)
    after = doc_index.index_document_file(folder / "report.pdf", client=client)

    assert after == 1
    assert len(_ids(get_text_collection(client), "data/documents/report.pdf")) == 1


def test_a_failed_ingest_leaves_the_old_chunks_in_place(tmp_path, monkeypatch):
    from src.pipelines.documents import index as doc_index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doc_index, "embed_texts", lambda texts: [[1.0, float(i + 1)] for i, _ in enumerate(texts)])
    folder = Path("data/documents")
    folder.mkdir(parents=True)
    client = get_client(persist_dir=tmp_path / "chroma")
    _make_pdf(folder / "report.pdf", words=900)
    kept = doc_index.index_document_file(folder / "report.pdf", client=client)

    (folder / "report.pdf").write_bytes(b"this is not a pdf at all")
    with pytest.raises(Exception):
        doc_index.index_document_file(folder / "report.pdf", client=client)

    assert len(_ids(get_text_collection(client), "data/documents/report.pdf")) == kept


def test_reindexing_a_shorter_transcript_removes_the_chunks_it_no_longer_has(tmp_path, monkeypatch):
    from src.pipelines.audio import index as audio_index

    class Scripted:
        def __init__(self, n):
            self.n = n

        def process_file(self, path):
            return [Chunk(chunk_id=f"talk__c{i}", source=str(path), modality="audio", text=f"segment {i} text",
                          embedding_model="m", start_s=float(i), end_s=float(i + 1)) for i in range(self.n)]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audio_index, "embed_texts", lambda texts: [[1.0, float(i + 1)] for i, _ in enumerate(texts)])
    Path("data/audio").mkdir(parents=True)
    (Path("data/audio") / "talk.wav").write_bytes(b"RIFF fake")
    client = get_client(persist_dir=tmp_path / "chroma")

    assert audio_index.index_audio_file("data/audio/talk.wav", client=client, ingestor=Scripted(3)) == 3
    assert audio_index.index_audio_file("data/audio/talk.wav", client=client, ingestor=Scripted(1)) == 1

    assert _ids(get_text_collection(client), "data/audio/talk.wav") == {"talk__c0"}


# ---------------------------------------------------------------------------
# prune_missing_sources
# ---------------------------------------------------------------------------

def test_prune_removes_the_chunks_of_deleted_files_and_keeps_the_rest(collection, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "kept.pdf").write_bytes(b"x")
    replace_source_chunks(collection, "data/kept.pdf", [_chunk("k0", "data/kept.pdf")], _vectors(1))
    replace_source_chunks(collection, "data/gone.pdf", [_chunk("g0", "data/gone.pdf"), _chunk("g1", "data/gone.pdf")], _vectors(2))

    removed = prune_missing_sources(collection, tmp_path)

    assert removed == {"data/gone.pdf": 2}
    assert _ids(collection) == {"k0"}


def test_prune_refuses_to_delete_anything_when_every_source_looks_missing(collection, tmp_path, caplog):
    # Almost always a wrong working directory, not "I deleted everything".
    replace_source_chunks(collection, "data/a.pdf", [_chunk("a0", "data/a.pdf")], _vectors(1))
    replace_source_chunks(collection, "data/b.pdf", [_chunk("b0", "data/b.pdf")], _vectors(1))

    with caplog.at_level(logging.WARNING):
        removed = prune_missing_sources(collection, tmp_path / "an_empty_wrong_folder")

    assert removed == {}
    assert _ids(collection) == {"a0", "b0"}
    assert "refusing to delete" in caplog.text


def test_prune_on_an_empty_index_does_nothing(collection, tmp_path):
    assert prune_missing_sources(collection, tmp_path) == {}
