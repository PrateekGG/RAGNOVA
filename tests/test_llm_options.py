"""
OLLAMA_NUM_GPU: the opt-in that forces CPU-only inference so the project's
stated target (a laptop with no discrete GPU) can be measured honestly. It
must be absent from the request unless set (Ollama's own choice is the
default), reach BOTH the streaming and non-streaming calls, and a malformed
value must fail loudly instead of being silently ignored. A fake Ollama client
stands in for the server.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import config, llm


class FakeClient:
    calls: list[dict] = []

    def __init__(self, host=None):
        pass

    def generate(self, **kwargs):
        FakeClient.calls.append(kwargs)
        if kwargs["stream"]:
            return iter([{"response": "a"}, {"response": "b"}])
        return {"response": "ok"}


@pytest.fixture
def fake_ollama(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(llm.ollama, "Client", FakeClient)
    return FakeClient


def _with_gpu(monkeypatch, value):
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, OLLAMA_NUM_GPU=value))


def test_by_default_no_gpu_option_is_sent_so_ollama_chooses(monkeypatch, fake_ollama):
    _with_gpu(monkeypatch, None)
    llm.generate("hello")
    options = fake_ollama.calls[0]["options"]
    assert "num_gpu" not in options
    assert set(options) == {"temperature", "num_predict", "num_ctx"}


def test_num_gpu_zero_reaches_the_non_streaming_call(monkeypatch, fake_ollama):
    _with_gpu(monkeypatch, 0)
    llm.generate("hello")
    assert fake_ollama.calls[0]["options"]["num_gpu"] == 0


def test_num_gpu_zero_reaches_the_streaming_call_too(monkeypatch, fake_ollama):
    _with_gpu(monkeypatch, 0)
    assert "".join(llm.generate_stream("hello")) == "ab"
    assert fake_ollama.calls[0]["stream"] is True
    assert fake_ollama.calls[0]["options"]["num_gpu"] == 0


def test_the_setting_is_read_from_the_environment_and_absent_by_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_GPU", raising=False)
    assert config._get_optional_int("OLLAMA_NUM_GPU") is None
    monkeypatch.setenv("OLLAMA_NUM_GPU", "")
    assert config._get_optional_int("OLLAMA_NUM_GPU") is None
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")
    assert config._get_optional_int("OLLAMA_NUM_GPU") == 0          # zero is a real value, not "unset"
    monkeypatch.setenv("OLLAMA_NUM_GPU", " 33 ")
    assert config._get_optional_int("OLLAMA_NUM_GPU") == 33


def test_a_malformed_value_fails_loudly_instead_of_being_ignored(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_GPU", "cpu")
    with pytest.raises(ValueError):
        config._get_optional_int("OLLAMA_NUM_GPU")


# ---------------------------------------------------------------------------
# LLM_THINK (2026-10-11): gemma4:12b thinks by default and returned '' with a 512-token cap.
# ---------------------------------------------------------------------------

def _think(monkeypatch, value, tokens=1536):
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, LLM_THINK=value, LLM_THINK_TOKENS=tokens,
                                                             OLLAMA_NUM_GPU=None))


def test_unset_sends_no_think_and_keeps_the_budgets(monkeypatch, fake_ollama):
    _think(monkeypatch, None)
    llm.generate("hello")
    call = fake_ollama.calls[0]
    assert "think" not in call
    assert call["options"]["num_predict"] == llm.settings.LLM_MAX_TOKENS
    assert call["options"]["num_ctx"] == llm.settings.LLM_NUM_CTX


def test_thinking_on_is_sent_and_widens_both_budgets_by_the_think_tokens(monkeypatch, fake_ollama):
    _think(monkeypatch, True, tokens=1000)
    llm.generate("hello")
    call = fake_ollama.calls[0]
    assert call["think"] is True
    assert call["options"]["num_predict"] == llm.settings.LLM_MAX_TOKENS + 1000
    assert call["options"]["num_ctx"] == llm.settings.LLM_NUM_CTX + 1000


def test_thinking_off_is_sent_and_changes_no_budget(monkeypatch, fake_ollama):
    _think(monkeypatch, False)
    assert "".join(llm.generate_stream("hello")) == "ab"
    call = fake_ollama.calls[0]
    assert call["think"] is False and call["options"]["num_predict"] == llm.settings.LLM_MAX_TOKENS


def test_the_streamed_answer_never_contains_the_reasoning(monkeypatch):
    class Thinking(FakeClient):
        def generate(self, **kwargs):
            return iter([{"response": "", "thinking": "2+2 is 4"}, {"response": "Four", "thinking": ""}])

    monkeypatch.setattr(llm.ollama, "Client", Thinking)
    _think(monkeypatch, True)
    assert list(llm.generate_stream("q")) == ["Four"]


@pytest.mark.parametrize("raw, expected", [("", None), ("1", True), ("0", False)])
def test_llm_think_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_THINK", raw)
    value = None if raw == "" else config._get_bool("LLM_THINK", False)
    assert value is expected
