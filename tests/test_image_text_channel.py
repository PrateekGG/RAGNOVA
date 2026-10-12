"""
The OCR-text channel for images (ADR-014): finding an image by what it says,
not only by what CLIP thinks it looks like.

Before it, images were searched by CLIP alone and the text read from them was
used only to corroborate a weak CLIP match (ADR-011) and to build the prompt.
So an image whose text answered the question was findable only if CLIP
happened to rank it (gold rows I3 and I10, and a canteen notice uploaded
through the page on 2026-10-07 that ranked 6th for "When is breakfast
served?"). The channel adds a second ranking, the question against MiniLM
vectors of every image's OCR text, fuses the two by rank, and (optionally)
lets a strong text match keep an image CLIP did not convince.

Fakes throughout: a hashed bag-of-words embedder for MiniLM, a colour-keyed
fake CLIP and OCR engine. No model weights, no network.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.core.vector_store import get_client, get_image_text_collection, prune_missing_sources
from src.pipelines.images.index import index_image_files, index_ocr_text
from src.pipelines.images.models import OCRResult
from src.pipelines.images.search import search_image_text
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.retrieve import filter_images, gate_image_channels, retrieve, shares_content_word

DIM = 64


def _embed_one(text: str) -> list[float]:
    """A hashed bag of words, unit length: two texts that share words are close."""
    vector = [0.0] * DIM
    for word in text.lower().split():
        vector[zlib.crc32(word.strip(".,?!:").encode()) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_many(texts: list[str]) -> list[list[float]]:
    return [_embed_one(t) for t in texts]


def _image(chunk_id: str, text: str = "words", score: float | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, source=f"data/images/{chunk_id}.png", modality="image",
        text=text, embedding_model="test", score=score,
    )


def agree_always(query, chunk):      # the shipped ADR-011 corroboration, injected
    return 1.0


def agree_never(query, chunk):
    return 0.0


# --------------------------------------------------------------------------
# The gate: gate_image_channels()
# --------------------------------------------------------------------------

def test_with_no_text_hits_and_no_text_rule_the_gate_is_exactly_the_shipped_one():
    clip = [
        _image("a", score=0.35), _image("b", score=0.25), _image("c", score=0.22),
        _image("d", score=0.15), _image("e", "", score=0.27),
        _image("floor", score=0.20), _image("confident", score=0.30),      # both boundaries are inclusive
    ]
    for agreement in (agree_always, agree_never):
        shipped = filter_images(clip, "any question", agreement)
        new = gate_image_channels(clip, [], "any question", agreement)
        assert [c.chunk_id for c in new] == [c.chunk_id for c in shipped]


def test_an_image_only_the_text_search_found_is_dropped_unless_the_text_rule_is_set():
    ocr = [_image("canteen", "breakfast is served 7 to 9", score=0.62)]
    assert gate_image_channels([], ocr, "when is breakfast served", agree_always) == []


def test_the_text_rule_keeps_an_image_clip_did_not_find_and_its_boundary_is_inclusive():
    ocr = [_image("canteen", "breakfast is served 7 to 9", score=0.50)]
    kept = gate_image_channels([], ocr, "when is breakfast served", agree_never, ocr_only_min_agreement=0.50)
    assert [c.chunk_id for c in kept] == ["canteen"]
    dropped = gate_image_channels([], ocr, "when is breakfast served", agree_never, ocr_only_min_agreement=0.51)
    assert dropped == []


def test_a_clip_confident_image_is_still_kept_when_the_text_rule_is_set_and_the_text_search_missed_it():
    clip = [_image("photo", "", score=0.35)]
    kept = gate_image_channels(clip, [], "show me the building", agree_never, ocr_only_min_agreement=0.5)
    assert [c.chunk_id for c in kept] == ["photo"]


def test_the_shared_word_requirement_vetoes_a_text_match_with_no_word_in_common():
    """The lab-door-sign leak: a high text match for a question that has no word in common with it."""
    sign = _image("lab_sign", "AI and Machine Learning Research Laboratory. Location: Room 302", score=0.55)
    without = gate_image_channels([], [sign], "Who won the Turing Award in 2018?", agree_never,
                                  ocr_only_min_agreement=0.5)
    assert [c.chunk_id for c in without] == ["lab_sign"]                 # admitted on the score alone
    vetoed = gate_image_channels([], [sign], "Who won the Turing Award in 2018?", agree_never,
                                 ocr_only_min_agreement=0.5, require_shared_word=True)
    assert vetoed == []
    kept = gate_image_channels([], [sign], "Where is the AI research laboratory?", agree_never,
                               ocr_only_min_agreement=0.5, require_shared_word=True)
    assert [c.chunk_id for c in kept] == ["lab_sign"]


def test_a_text_score_does_not_rescue_an_image_that_clip_found_but_failed_the_clip_rule():
    """The text rule is an alternative way in for images the text search returned; it does not
    lower the CLIP floor, and an image the text search did NOT return gets no help from it."""
    clip = [_image("weak", "unrelated words", score=0.21)]          # above the floor, not confident, text disagrees
    assert gate_image_channels(clip, [], "question", agree_never, ocr_only_min_agreement=0.1) == []


def test_an_image_both_searches_like_is_ranked_ahead_of_one_only_clip_likes():
    """Gold row I3: CLIP ranks the seminar poster third, the text search ranks it first."""
    clip = [_image("a", score=0.35), _image("b", score=0.34), _image("poster", score=0.33)]
    ocr = [_image("poster", "seminar venue", score=0.6)]
    kept = gate_image_channels(clip, ocr, "seminar venue", agree_always)
    assert [c.chunk_id for c in kept] == ["poster", "a", "b"]
    assert [c.chunk_id for c in filter_images(clip, "seminar venue", agree_always)] == ["a", "b", "poster"]


def test_an_image_in_both_lists_keeps_its_clip_chunk_and_clip_score():
    clip = [_image("poster", score=0.31)]
    ocr = [_image("poster", score=0.9)]
    [kept] = gate_image_channels(clip, ocr, "q", agree_always)
    assert kept.score == 0.31


# --------------------------------------------------------------------------
# shares_content_word()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query,text,expected", [
    ("Who won the Turing Award in 2018?", "AI Research Laboratory Room 302 Operating Hours", False),
    ("Where is the AI laboratory?", "AI Research Laboratory Room 302", True),
    ("How many books can I borrow?", "Undergraduate quota: 4 book for 14 days", True),      # plural
    ("Show me a photo of a cat on a sofa", "PHOTOGRAPH CORPUS photo of the library", False),  # generic words never count
    ("What is the capital of France?", "", False),
    ("", "anything at all", False),
])
def test_shares_content_word(query, text, expected):
    assert shares_content_word(query, text) is expected


# --------------------------------------------------------------------------
# The write path and the search
# --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    return get_client(persist_dir=tmp_path / "chroma")


def test_text_vectors_are_written_only_for_images_with_readable_text_under_the_same_ids(client):
    chunks = [_image("poster", "seminar venue is Auditorium B"), _image("photo", "")]
    assert index_ocr_text(chunks, client=client, embed_texts_fn=embed_many) == 1
    stored = get_image_text_collection(client).get(include=["metadatas"])
    assert stored["ids"] == ["poster"]
    assert stored["metadatas"][0]["source"] == "data/images/poster.png"


def test_a_reindexed_image_whose_text_is_now_empty_loses_its_old_text_vector(client):
    index_ocr_text([_image("poster", "seminar venue is Auditorium B")], client=client, embed_texts_fn=embed_many)
    index_ocr_text([_image("poster", "")], client=client, embed_texts_fn=embed_many)
    assert get_image_text_collection(client).get()["ids"] == []


def test_reindexing_one_image_leaves_the_others_text_vectors_alone(client):
    index_ocr_text([_image("a", "alpha words"), _image("b", "beta words")], client=client, embed_texts_fn=embed_many)
    index_ocr_text([_image("a", "alpha changed")], client=client, embed_texts_fn=embed_many)
    assert sorted(get_image_text_collection(client).get()["ids"]) == ["a", "b"]


def test_pruning_a_deleted_image_removes_its_text_vector_too(client, tmp_path):
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "images" / "kept.png").write_bytes(b"x")
    index_ocr_text([_image("kept", "some words"), _image("gone", "other words")], client=client, embed_texts_fn=embed_many)
    removed = prune_missing_sources(get_image_text_collection(client), tmp_path)
    assert removed == {"data/images/gone.png": 1}
    assert get_image_text_collection(client).get()["ids"] == ["kept"]


def test_the_text_search_finds_an_image_by_what_it_says(client):
    chunks = [_image("canteen", "breakfast is served 7 to 9 AM daily"), _image("bus", "last bus leaves at 6:30 PM")]
    index_ocr_text(chunks, client=client, embed_texts_fn=embed_many)
    hits = search_image_text("when is breakfast served", client=client, embed=_embed_one)
    assert hits[0].chunk_id == "canteen"
    assert hits[0].modality == "image"
    assert hits[0].score > hits[1].score


def test_the_text_search_is_empty_for_a_blank_question_and_for_an_index_without_text_vectors(client):
    assert search_image_text("anything", client=client, embed=_embed_one) == []          # nothing indexed yet
    index_ocr_text([_image("a", "alpha words")], client=client, embed_texts_fn=embed_many)
    assert search_image_text("   ", client=client, embed=_embed_one) == []


class _FakeClip:
    model_id = "fake-clip/test"

    def embed_batch(self, images):
        return [self.embed_image(i) for i in images]

    def embed_image(self, image):
        vector = [0.0] * 512
        vector[0 if image.convert("RGB").getpixel((0, 0))[0] > 128 else 1] = 1.0
        return vector


class _FakeOcr:
    """Reads 'text' off an image by its colour: red pictures say something, blue ones say nothing."""
    is_available = True

    def extract_text(self, image):
        red = image.convert("RGB").getpixel((0, 0))[0] > 128
        return OCRResult(text="breakfast is served 7 to 9 AM" if red else "", confidence=90.0, words_count=7)


def test_indexing_an_image_file_writes_both_collections(client, tmp_path, monkeypatch):
    from src.pipelines.images.ingest import ImageIngestionPipeline
    from src.pipelines.images.models import ImageIngestionConfig
    from src.core.vector_store import get_image_collection

    monkeypatch.chdir(tmp_path)
    Path("data/images").mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save("data/images/red_notice.png")
    Image.new("RGB", (8, 8), (0, 0, 255)).save("data/images/blue_photo.png")
    pipeline = ImageIngestionPipeline(config=ImageIngestionConfig(ocr_enabled=True), ocr_engine=_FakeOcr(), embedder=_FakeClip())

    count = index_image_files(["data/images/red_notice.png", "data/images/blue_photo.png"],
                              client=client, pipeline=pipeline, embed_texts_fn=embed_many)

    assert count == 2
    clip_ids = get_image_collection(client).get()["ids"]
    text_ids = get_image_text_collection(client).get()["ids"]
    assert len(clip_ids) == 2
    assert len(text_ids) == 1 and text_ids[0] in clip_ids           # same id, only for the image that has text


# --------------------------------------------------------------------------
# retrieve() wiring
# --------------------------------------------------------------------------

@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(retrieve_module, "search_text", lambda query, top_k=None, client=None: [])
    monkeypatch.setattr(retrieve_module, "_ocr_agreement", lambda query, chunk: 0.0)

    def clip_search(query_text=None, query_image=None, top_k=None, client=None):
        return [_image("clip_only", "x", score=0.35)]

    def text_search(query_text=None, top_k=None, client=None):
        return [_image("text_only", "breakfast is served", score=0.62)]

    return clip_search, text_search


def _with_settings(monkeypatch, **changes):
    monkeypatch.setattr(retrieve_module, "settings", dataclasses.replace(retrieve_module.settings, **changes))


def test_with_the_channel_off_retrieve_never_runs_the_text_search(wired, monkeypatch):
    clip_search, _ = wired

    def must_not_run(**kwargs):
        raise AssertionError("the text search ran although IMAGE_TEXT_SEARCH is off")

    _with_settings(monkeypatch, IMAGE_TEXT_SEARCH=False)
    ids = [c.chunk_id for c in retrieve("when is breakfast", include_images=True,
                                        image_search=clip_search, image_text_search=must_not_run)]
    assert ids == ["clip_only"]


def test_with_the_channel_on_but_no_text_rule_the_text_search_only_reorders(wired, monkeypatch):
    clip_search, text_search = wired
    _with_settings(monkeypatch, IMAGE_TEXT_SEARCH=True, IMAGE_OCR_ONLY_MIN_AGREEMENT=None)
    ids = [c.chunk_id for c in retrieve("when is breakfast", include_images=True,
                                        image_search=clip_search, image_text_search=text_search)]
    assert ids == ["clip_only"]                          # text_only is not admitted without the rule


def test_with_the_channel_and_the_text_rule_on_an_image_clip_missed_is_retrieved(wired, monkeypatch):
    clip_search, text_search = wired
    _with_settings(monkeypatch, IMAGE_TEXT_SEARCH=True, IMAGE_OCR_ONLY_MIN_AGREEMENT=0.5)
    ids = [c.chunk_id for c in retrieve("when is breakfast", include_images=True,
                                        image_search=clip_search, image_text_search=text_search)]
    assert sorted(ids) == ["clip_only", "text_only"]


def test_an_uploaded_image_query_never_uses_the_text_search(wired, monkeypatch):
    clip_search, _ = wired

    def must_not_run(**kwargs):
        raise AssertionError("image-to-image search must not use the OCR-text channel")

    _with_settings(monkeypatch, IMAGE_TEXT_SEARCH=True, IMAGE_OCR_ONLY_MIN_AGREEMENT=0.5)
    retrieve("", query_image=Image.new("RGB", (4, 4)), image_search=clip_search, image_text_search=must_not_run)
