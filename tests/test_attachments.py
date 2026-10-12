"""
An image attached to a question as something the model can answer from (2026-10-10).

Reproduced the day it was reported: two screenshots that were not in the corpus were
attached, Tesseract read 540 and 3539 characters from them, and the answer was still
"I don't have enough information", because the picture was used only as a search key
and its text was never shown to the model. An attachment is now one more numbered
context block, first, labelled as the user's attachment, never filtered. These tests
pin that, and the limits that go with it: a bounded share of the context window, the
question shown to the model as typed, and an honest line for a picture with no text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag.attachments import (
    ATTACHMENT_MAX_WORDS,
    image_attachment,
    is_attachment,
)
from src.pipelines.rag.prompt import build_prompt, format_provenance
from src.ui.citations import citation_views

SHOT_TEXT = "Evaluation dates October fourteenth to sixteenth, Conference Room 3"


def _found(text: str = "Corpus passage about something else.") -> Chunk:
    return Chunk(chunk_id="n1", source="data/documents/notice.pdf", modality="pdf", text=text,
                 embedding_model="m", page=2, score=0.5)


# ---------------------------------------------------------------------------
# The attachment itself
# ---------------------------------------------------------------------------

def test_an_attachment_is_an_image_chunk_that_says_it_is_an_attachment():
    attachment = image_attachment("shot.png", SHOT_TEXT)
    assert is_attachment(attachment) and attachment.modality == "image"
    assert attachment.text == SHOT_TEXT
    assert not is_attachment(_found())


def test_a_long_screenshot_keeps_only_its_first_words_and_says_so():
    words = [f"w{i}" for i in range(ATTACHMENT_MAX_WORDS + 150)]
    kept = image_attachment("busy.png", " ".join(words)).text
    assert kept.endswith(" ...")
    assert kept.split()[:-1] == words[:ATTACHMENT_MAX_WORDS]


def test_a_screenshot_within_the_limit_is_kept_whole():
    words = " ".join(f"w{i}" for i in range(ATTACHMENT_MAX_WORDS))
    assert image_attachment("ok.png", words).text == words


# ---------------------------------------------------------------------------
# What the model is shown, and what the user is shown
# ---------------------------------------------------------------------------

def test_the_prompt_labels_an_attachment_as_such_and_shows_its_text():
    prompt = build_prompt("what does it say", [image_attachment("shot.png", SHOT_TEXT), _found()])
    assert f"[1] attached image (shot.png)\nText read from the image (OCR): {SHOT_TEXT}" in prompt
    assert "[2] data/documents/notice.pdf, page 2" in prompt


def test_a_picture_with_no_text_is_described_as_unreadable_not_left_blank():
    prompt = build_prompt("what is this", [image_attachment("photo.jpg", "")])
    assert "[1] attached image (photo.jpg)" in prompt
    assert "No readable text was found in it" in prompt
    assert "cannot look at pictures" in prompt


def test_an_indexed_image_with_no_text_keeps_its_own_wording():
    indexed = Chunk(chunk_id="i1", source="data/images/p.png", modality="image", text="", embedding_model="m")
    assert "An image matching the question" in build_prompt("q", [indexed])


def test_provenance_of_an_attachment_is_its_label_without_a_second_image_suffix():
    assert format_provenance(image_attachment("shot.png", "x")) == "attached image (shot.png)"
    assert format_provenance(Chunk(chunk_id="i1", source="data/images/p.png", modality="image", text="", embedding_model="m")) \
        == "data/images/p.png (image)"


def test_the_citation_list_shows_an_attachment_without_trying_to_open_it_as_a_file():
    [view] = citation_views([image_attachment("shot.png", SHOT_TEXT)])
    assert view.title == "[1] attached image (shot.png)"
    assert view.excerpt == SHOT_TEXT and view.file_exists is False


# ---------------------------------------------------------------------------
# answer_query / stream_answer
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(monkeypatch):
    """The pipeline with retrieval and the model replaced; records what each was given."""
    seen = {"retrieve": [], "prompts": []}

    def fake_retrieve(query, **kwargs):
        seen["retrieve"].append({"query": query, **kwargs})
        return list(seen.get("returns", [_found()]))

    def fake_generate(prompt):
        seen["prompts"].append(prompt)
        return "It says October fourteenth [1]."

    def fake_stream(prompt):
        seen["prompts"].append(prompt)
        return iter(["It says ", "October fourteenth [1]."])

    monkeypatch.setattr(answer_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(answer_module, "generate", fake_generate)
    monkeypatch.setattr(answer_module, "generate_stream", fake_stream)
    return seen


def test_the_attachment_comes_first_in_the_prompt_and_in_the_citations(pipeline):
    attachment = image_attachment("shot.png", SHOT_TEXT)
    result = answer_module.answer_query("what does it say", attachments=[attachment])
    assert result.citations[0] is attachment and len(result.citations) == 2      # then the one chunk retrieval found
    assert "[1] attached image (shot.png)" in pipeline["prompts"][0]


def test_the_model_and_retrieval_both_get_the_question_as_typed_and_the_picture_text_is_not_glued_on(pipeline):
    """Glued on, the picture's generic words pulled unrelated corpus chunks into the context (ADR-016)."""
    answer_module.answer_query("what does it say", attachments=[image_attachment("shot.png", SHOT_TEXT)])
    assert "Question: what does it say\n" in pipeline["prompts"][0]
    assert pipeline["retrieve"][0]["query"] == "what does it say"


def test_an_answer_can_come_from_the_attachment_alone(pipeline):
    pipeline["returns"] = []
    attachment = image_attachment("shot.png", SHOT_TEXT)
    result = answer_module.answer_query("what does it say", attachments=[attachment])
    assert result.answer != answer_module.NOT_ENOUGH_INFO
    assert result.citations == [attachment]
    assert len(pipeline["prompts"]) == 1


def test_with_nothing_retrieved_and_no_attachment_the_model_is_not_called(pipeline):
    pipeline["returns"] = []
    result = answer_module.answer_query("what does it say")
    assert result.answer == answer_module.NOT_ENOUGH_INFO and result.citations == []
    assert pipeline["prompts"] == []


def test_the_scope_reaches_retrieval(pipeline):
    answer_module.answer_query("who is this", sources=["data/documents/cv.pdf"])
    assert pipeline["retrieve"][0]["sources"] == ["data/documents/cv.pdf"]


def test_no_scope_is_passed_as_none_so_the_default_path_is_the_old_one(pipeline):
    answer_module.answer_query("who is this")
    assert pipeline["retrieve"][0]["sources"] is None


def test_streaming_gets_the_same_attachment_and_scope_handling(pipeline):
    attachment = image_attachment("shot.png", SHOT_TEXT)
    chunks, stream = answer_module.stream_answer("what does it say", attachments=[attachment], sources=["data/documents/cv.pdf"])
    assert chunks[0] is attachment
    assert "".join(stream).endswith("[1].")
    assert pipeline["retrieve"][0]["sources"] == ["data/documents/cv.pdf"]
    assert "[1] attached image (shot.png)" in pipeline["prompts"][0]


def test_streaming_with_nothing_retrieved_and_no_attachment_refuses_without_the_model(pipeline):
    pipeline["returns"] = []
    chunks, stream = answer_module.stream_answer("what does it say")
    assert chunks == [] and list(stream) == [answer_module.NOT_ENOUGH_INFO]
    assert pipeline["prompts"] == []


# ---------------------------------------------------------------------------
# "this image" must have one referent (2026-10-10)
# ---------------------------------------------------------------------------

def _other_image() -> Chunk:
    return Chunk(chunk_id="i1", source="data/images/notice_library_fines.png", modality="image",
                 text="Late Return Fine: Rs 2.00 per day", embedding_model="m", score=0.4)


def test_the_prompt_says_which_block_is_the_attached_image_and_that_this_image_means_it():
    prompt = build_prompt("What does this image say?", [image_attachment("shot.png", SHOT_TEXT), _other_image()])
    assert 'The user attached an image to this question: [1].' in prompt
    assert '"this image", "the image" or "the picture", it means [1], not any other image in the context.' in prompt
    assert prompt.index("The user attached an image") < prompt.index("Question: What does this image say?")
    assert prompt.index("[2] data/images/notice_library_fines.png") < prompt.index("The user attached an image")


def test_the_note_names_the_attachments_real_number_when_it_is_not_first():
    prompt = build_prompt("What does this say?", [_found(), _other_image(), image_attachment("shot.png", SHOT_TEXT)])
    assert "The user attached an image to this question: [3]." in prompt


def test_without_an_attachment_the_prompt_has_no_note_and_is_exactly_what_it_was():
    chunks = [_found(), _other_image()]
    prompt = build_prompt("What does this image say?", chunks)
    assert "attached" not in prompt
    assert prompt.endswith("\n\nQuestion: What does this image say?\n\nAnswer:")
    assert "Context:\n[1] data/documents/notice.pdf, page 2\n" in prompt


def test_the_note_is_in_the_prompt_the_model_actually_gets(pipeline):
    answer_module.answer_query("What does this image say?", attachments=[image_attachment("shot.png", SHOT_TEXT)])
    assert "The user attached an image to this question: [1]." in pipeline["prompts"][0]
