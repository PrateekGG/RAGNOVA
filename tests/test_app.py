"""
The chat page itself (src/app.py), run headlessly with Streamlit's AppTest.

tests/test_ui.py checks the modules the page calls (backend, citations,
feedback) but never runs the page, and the page is where the PR #10 review
(2026-10-07) found the defects these tests pin:

  - When the live call failed, the page silently served the hand-written
    canned answer in its place, so a failure looked like a success and a
    tester's rating of it was logged under the real model's name.
  - The error it did show was removed by the st.rerun() on the very next line.
  - Streamlit's defaults listen on every network interface and look up the
    machine's public IP, which an offline, one-machine app should not do.

The model, the index and the feedback log are replaced by fakes; the script
under test is the real src/app.py. No Ollama, no ChromaDB, no network.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.schemas import Chunk
from src.pipelines.rag import answer as answer_module
from src.ui import backend, feedback

APP = str(PROJECT_ROOT / "src" / "app.py")
ENGINE_CANNED = "Canned demo answers (not the real system)"

# A question the canned scaffold has a hand-written answer for, and a fragment
# that only that hand-written answer contains.
CANNED_QUESTION = "How many marks does the prototype carry?"
CANNED_FRAGMENT = "40% of the total marks"
SOURCES = ["data/documents/notice.pdf", "data/documents/cv.pdf", "data/images/notice_midterm_schedule.png"]
RUNNER_ERROR = "llama-server binary not found"      # what the broken Ollama install said on 2026-10-07


class Page:
    """The running app, plus what it logged to the feedback file."""

    def __init__(self, at: AppTest, logged: list):
        self.at = at
        self.logged = logged

    def ask(self, question: str) -> dict:
        self.at.chat_input[0].set_value(question).run()
        assert not self.at.exception, [e.value for e in self.at.exception]
        return self.at.session_state["messages"][-1]

    def pick_engine(self, label: str) -> None:
        self.at.radio[0].set_value(label).run()

    def rate_last_answer(self) -> None:
        [button] = [b for b in self.at.button if b.label == "Submit Rating"]
        button.click().run()

    def has_rating_form(self) -> bool:
        return any("Rate this answer" in x.label for x in self.at.expander)


@pytest.fixture
def page(monkeypatch) -> Page:
    monkeypatch.setattr(backend, "ollama_status", lambda: ("ready", settings.OLLAMA_MODEL))
    monkeypatch.setattr(backend, "index_counts", lambda client=None: {"text": 1, "image": 1})
    monkeypatch.setattr(backend, "list_sources", lambda client=None: SOURCES)
    logged: list = []
    monkeypatch.setattr(feedback, "record_feedback", lambda entry, path=None: logged.append(entry))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return Page(at, logged)


def _live_answer(text: str, calls: list | None = None):
    """A stand-in for stream_answer() that retrieves one real-looking chunk and streams `text`.
    Each call's arguments are appended to `calls` when it is given."""
    chunk = Chunk(
        chunk_id="n1", source="data/documents/notice.pdf", modality="pdf",
        text="The working prototype will carry forty percent.", page=2, embedding_model="test",
    )

    def fake(query, top_k=None, client=None, include_images=False, query_image=None, sources=None, attachments=None,
             history=None, earlier_attachments=None):
        if calls is not None:
            calls.append({"query": query, "sources": sources, "attachments": attachments, "query_image": query_image,
                          "history": history, "earlier_attachments": earlier_attachments})
        return [chunk], iter([word + " " for word in text.split(" ")])

    return fake


def test_a_failed_live_call_is_shown_and_never_replaced_by_the_canned_answer(page, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError(RUNNER_ERROR)

    monkeypatch.setattr(answer_module, "stream_answer", broken)

    reply = page.ask(CANNED_QUESTION)

    assert CANNED_FRAGMENT not in reply["content"], "a canned answer was substituted for the failure"
    assert reply["engine"] == "live"
    assert RUNNER_ERROR in reply["error"]
    # the error must still be on screen after the page's rerun, not only in the history list
    assert any(RUNNER_ERROR in e.value for e in page.at.error)
    # there is no answer to rate, so no rating form (and no log line for a failure)
    assert not page.has_rating_form()
    assert page.logged == []


def test_a_live_answer_keeps_its_citations_and_is_logged_under_the_real_model(page, monkeypatch):
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1]."))

    reply = page.ask(CANNED_QUESTION)

    assert reply["engine"] == "live" and reply["error"] is None
    assert reply["content"].strip() == "Forty percent [1]."
    assert [v.number for v in reply["views"]] == [1]
    assert page.has_rating_form()

    page.rate_last_answer()
    assert [entry.model for entry in page.logged] == [settings.OLLAMA_MODEL]
    assert page.logged[0].sources[0].startswith("[1] notice.pdf")


def test_canned_mode_is_labelled_on_the_answer_and_logged_as_scaffold(page, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("canned mode must not call the live pipeline")

    monkeypatch.setattr(answer_module, "stream_answer", must_not_run)
    page.pick_engine(ENGINE_CANNED)

    reply = page.ask(CANNED_QUESTION)

    assert reply["engine"] == "scaffold"
    assert CANNED_FRAGMENT in reply["content"]
    assert any("Canned demo answer" in w.value for w in page.at.warning)

    page.rate_last_answer()
    assert [entry.model for entry in page.logged] == [feedback.SCAFFOLD_MODEL_LABEL]
    assert page.logged[0].model != settings.OLLAMA_MODEL


def test_test_the_model_button_reports_a_model_that_cannot_start(page, monkeypatch):
    monkeypatch.setattr(backend, "check_model_generates", lambda: (False, RUNNER_ERROR))

    [button] = [b for b in page.at.button if "Test the model" in b.label]
    button.click().run()

    assert any(RUNNER_ERROR in e.value for e in page.at.error)


def test_clearing_the_conversation_leaves_the_greeting_not_a_blank_page(page, monkeypatch):
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1]."))
    page.ask(CANNED_QUESTION)
    assert len(page.at.session_state["messages"]) == 3

    [clear] = [b for b in page.at.button if b.label == "Clear Conversation"]
    clear.click().run()

    messages = page.at.session_state["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "assistant"
    assert "RAGNova" in messages[0]["content"]


def test_streamlit_config_keeps_the_app_on_this_machine_and_telemetry_off():
    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "127.0.0.1"
    assert config["browser"]["gatherUsageStats"] is False


def test_streamlit_config_is_headless_so_a_fresh_machine_does_not_stop_at_the_email_prompt():
    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["headless"] is True


# ---------------------------------------------------------------------------
# Answering from the files the user picked (2026-10-10)
# ---------------------------------------------------------------------------

CV = "data/documents/cv.pdf"


def _banners(at) -> list[str]:
    return [i.value for i in at.info if "Answering from" in i.value]


def test_the_picker_offers_every_indexed_file(page):
    [picker] = page.at.sidebar.multiselect
    assert len(picker.options) == len(SOURCES)


def test_without_a_pick_questions_run_over_everything_and_no_banner_is_shown(page, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1].", calls))

    page.ask(CANNED_QUESTION)

    assert calls[-1]["sources"] is None
    assert _banners(page.at) == []
    assert not [b for b in page.at.button if b.label.startswith("Give me a short summary")]


def test_with_a_pick_the_banner_names_the_file_and_the_question_is_limited_to_it(page, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("He is a student [1].", calls))
    page.at.session_state["focus_sources"] = [CV]
    page.at.run()

    assert any("cv.pdf" in text for text in _banners(page.at))
    page.ask("who is this person")

    assert calls[-1]["sources"] == [CV]


def test_use_everything_drops_the_scope_for_the_next_question(page, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1].", calls))
    page.at.session_state["focus_sources"] = [CV]
    page.at.run()

    [clear] = [b for b in page.at.button if b.label == "Use everything"]
    clear.click().run()
    assert not page.at.exception, [e.value for e in page.at.exception]
    assert page.at.session_state["focus_sources"] == [] and _banners(page.at) == []

    page.ask(CANNED_QUESTION)
    assert calls[-1]["sources"] is None


def test_a_suggested_question_is_asked_over_the_picked_file(page, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Skills: Python [1].", calls))
    page.at.session_state["focus_sources"] = [CV]
    page.at.run()

    [chip] = [b for b in page.at.button if b.label == "What are the key points?"]
    chip.click().run()

    assert not page.at.exception, [e.value for e in page.at.exception]
    messages = page.at.session_state["messages"]
    assert messages[-2]["content"] == "What are the key points?" and messages[-1]["engine"] == "live"
    assert calls[-1]["sources"] == [CV]


def test_a_picked_file_that_is_no_longer_indexed_is_dropped_instead_of_crashing_the_page(page):
    page.at.session_state["focus_sources"] = ["data/documents/gone.pdf"]
    page.at.run()

    assert not page.at.exception, [e.value for e in page.at.exception]
    assert page.at.session_state["focus_sources"] == [] and _banners(page.at) == []


def test_a_question_with_no_picture_attached_passes_no_attachment(page, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1].", calls))
    page.ask(CANNED_QUESTION)
    assert calls[-1]["attachments"] == [] and calls[-1]["query_image"] is None


def test_an_answer_citing_a_source_that_was_not_shown_is_flagged_on_the_page(page, monkeypatch):
    """The model cites [7] but one chunk was retrieved: the page must warn (the check runs on the question as typed)."""
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [7]."))

    reply = page.ask(CANNED_QUESTION)

    assert "Citation Alert" in (reply["warning"] or "")
    assert any("Citation Alert" in w.value for w in page.at.warning)


def test_a_refusal_lists_no_sources_even_though_chunks_were_retrieved(page, monkeypatch):
    """With a scope or an attached image, chunks are retrieved before the model decides they do not hold the answer."""
    from src.pipelines.rag.answer import NOT_ENOUGH_INFO

    monkeypatch.setattr(answer_module, "stream_answer", _live_answer(NOT_ENOUGH_INFO))
    reply = page.ask("what is the capital of France")

    assert reply["content"].strip() == NOT_ENOUGH_INFO
    assert not reply["views"]
    assert not any("Sources & Citations" in x.label for x in page.at.expander)


def test_an_answer_that_merely_mentions_the_refusal_phrase_midway_keeps_its_sources(page, monkeypatch):
    from src.pipelines.rag.answer import NOT_ENOUGH_INFO

    monkeypatch.setattr(answer_module, "stream_answer", _live_answer(f"Forty percent [1]. Note: {NOT_ENOUGH_INFO}"))
    reply = page.ask(CANNED_QUESTION)

    assert [v.number for v in reply["views"]] == [1]


def test_a_follow_up_is_sent_with_the_previous_exchange_and_a_fresh_question_with_it_too(page, monkeypatch):
    """"tell me more" was refused on 2026-10-11: the page asked every question on its own (ADR-018)."""
    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer("Forty percent [1].", calls))
    page.ask("who is raiden?")
    assert calls[-1]["history"] is None                                    # nothing before it

    page.ask("tell me more")
    [(question, answer)] = calls[-1]["history"]
    assert question == "who is raiden?" and answer.strip() == "Forty percent [1]."
    assert page.at.session_state["messages"][-2]["search"] == "who is raiden?"

    page.ask("and more")                                                   # chains on what the last turn searched
    assert calls[-1]["history"][0][0] == "who is raiden?"


def test_a_refused_answer_gives_the_next_question_no_history(page, monkeypatch):
    from src.pipelines.rag.answer import NOT_ENOUGH_INFO

    calls: list = []
    monkeypatch.setattr(answer_module, "stream_answer", _live_answer(NOT_ENOUGH_INFO, calls))
    page.ask("who is nobody?")
    page.ask("tell me more")
    assert calls[-1]["history"] is None


def test_only_the_sources_the_answer_cites_are_listed(page, monkeypatch):
    two = [
        Chunk(chunk_id="g", source="data/documents/greedy.pdf", modality="pdf", text="Boston Pool interns.", page=14,
              embedding_model="test"),
        Chunk(chunk_id="f", source="data/images/fair.png", modality="image", text="Internship Fair 3 December",
              embedding_model="test"),
    ]
    monkeypatch.setattr(answer_module, "stream_answer",
                        lambda query, **kw: (two, iter(["The fair is on 3 December [2]."])))
    reply = page.ask("tell me about internship fair")
    assert [v.title for v in reply["views"]] == ["[2] fair.png (image)"]
