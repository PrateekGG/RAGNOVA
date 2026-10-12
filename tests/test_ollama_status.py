"""
What the sidebar may claim about the language model, and the on-demand check
that proves it.

The old `ollama_ready()` listed the models and nothing else, and the page
called a True result "Online". On 2026-10-07 a half-applied Ollama update left
the server answering `list()` while every generate call failed with
"llama-server binary not found": the page said Online, and the first
question failed. `ollama_status()` now says only what it knows, and
`check_model_generates()` is the one check that fails when the model cannot
start.

A fake `ollama.Client` stands in for the server: no Ollama needed.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import ollama
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import llm
from src.ui import backend, feedback


class FakeClient:
    """Configured per test through class attributes; remembers what generate() was asked."""

    listing = SimpleNamespace(models=[])
    list_error: Exception | None = None
    generate_error: Exception | None = None
    generate_calls: list[dict] = []

    def __init__(self, host=None):
        self.host = host

    def list(self):
        if FakeClient.list_error:
            raise FakeClient.list_error
        return FakeClient.listing

    def generate(self, **kwargs):
        FakeClient.generate_calls.append(kwargs)
        if FakeClient.generate_error:
            raise FakeClient.generate_error
        return {"response": "OK"}


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    FakeClient.listing = SimpleNamespace(models=[])
    FakeClient.list_error = None
    FakeClient.generate_error = None
    FakeClient.generate_calls = []
    monkeypatch.setattr(ollama, "Client", FakeClient)
    monkeypatch.setattr(backend, "settings", dataclasses.replace(backend.settings, OLLAMA_MODEL="llama3.2:3b"))


def _models(*names):
    return SimpleNamespace(models=[SimpleNamespace(model=n) for n in names])


def test_ready_means_the_server_answers_and_the_model_is_pulled():
    FakeClient.listing = _models("nomic-embed-text:latest", "llama3.2:3b")
    assert backend.ollama_status() == ("ready", "llama3.2:3b")
    assert backend.ollama_ready() is True


def test_an_older_client_that_returns_plain_dicts_is_understood_too():
    FakeClient.listing = {"models": [{"name": "llama3.2:3b"}]}
    assert backend.ollama_status()[0] == "ready"


def test_an_untagged_model_name_matches_its_latest_tag(monkeypatch):
    monkeypatch.setattr(backend, "settings", dataclasses.replace(backend.settings, OLLAMA_MODEL="llama3.2"))
    FakeClient.listing = _models("llama3.2:latest")
    assert backend.ollama_status()[0] == "ready"


def test_a_running_server_without_the_model_is_not_ready_and_names_what_it_has():
    FakeClient.listing = _models("mistral:7b")
    state, detail = backend.ollama_status()
    assert state == "missing_model"
    assert "mistral:7b" in detail
    assert backend.ollama_ready() is False


def test_an_unreachable_server_is_reported_with_the_reason():
    FakeClient.list_error = ConnectionError("Failed to connect to Ollama")
    state, detail = backend.ollama_status()
    assert state == "unreachable"
    assert "Failed to connect" in detail
    assert backend.ollama_ready() is False


def test_the_model_check_passes_when_the_model_answers_and_asks_for_only_a_few_tokens():
    FakeClient.listing = _models("llama3.2:3b")
    worked, message = backend.check_model_generates()
    assert worked is True and "llama3.2:3b" in message
    [call] = FakeClient.generate_calls
    assert call["model"] == "llama3.2:3b"
    assert call["options"]["num_predict"] == 8


def test_the_model_check_fails_with_the_real_error_when_the_model_cannot_start():
    """The 2026-10-07 case: the server lists the model, but starting it fails."""
    FakeClient.listing = _models("llama3.2:3b")
    FakeClient.generate_error = RuntimeError("error starting llama-server: llama-server binary not found")
    assert backend.ollama_status()[0] == "ready"            # the listing alone cannot see this...
    worked, message = backend.check_model_generates()
    assert worked is False                                  # ...the check can
    assert "llama-server binary not found" in message


def test_the_model_check_uses_the_same_generation_options_as_real_answers(monkeypatch):
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, OLLAMA_NUM_GPU=0))
    backend.check_model_generates()
    [call] = FakeClient.generate_calls
    assert call["options"]["num_gpu"] == 0                  # a CPU-only setup is checked as CPU-only


def test_canned_answers_are_logged_under_a_label_that_is_not_the_real_model(monkeypatch):
    monkeypatch.setattr(feedback, "settings", dataclasses.replace(feedback.settings, OLLAMA_MODEL="llama3.2:3b"))
    assert feedback.model_label("scaffold") == "scaffold-mock"
    assert feedback.model_label("live") == "llama3.2:3b"
    assert feedback.model_label(None) == "llama3.2:3b"
