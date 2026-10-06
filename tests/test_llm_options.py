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
