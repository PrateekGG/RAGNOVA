"""
One shared Ollama client, and a host that is an IP literal.

Measured on 2026-10-07 on Windows: a NEW connection to http://localhost:11434
costs about 2.2 s (IPv6 is tried first), to http://127.0.0.1:11434 about 0.2 s,
and a reused connection 4 ms. The code built a new client for every answer, so
every answer paid 2.2 s that had nothing to do with the model (the model call was
0.68 s on a GPU and the answer 2.9 s), and the chat page paid it again on every
click for its status check. These tests pin the fix: one client per host, used by
every path, and a default host that is not a name.

A counting fake stands in for `ollama.Client`: no server, no network.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import ollama
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import llm
from src.ui import backend


class CountingClient:
    """Counts how many clients were built; answers every call like a working server."""

    built = 0

    def __init__(self, host=None):
        CountingClient.built += 1
        self.host = host

    def generate(self, **kwargs):
        if kwargs.get("stream"):
            return iter([{"response": "a"}, {"response": "b"}])
        return {"response": "ok"}

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=llm.settings.OLLAMA_MODEL)])


@pytest.fixture(autouse=True)
def counting(monkeypatch):
    CountingClient.built = 0
    monkeypatch.setattr(ollama, "Client", CountingClient)
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(backend, "settings", dataclasses.replace(backend.settings, OLLAMA_MODEL=llm.settings.OLLAMA_MODEL))


def test_the_same_host_gets_the_same_client_and_another_host_another():
    first = llm.ollama_client("http://127.0.0.1:11434")
    assert llm.ollama_client("http://127.0.0.1:11434") is first
    assert llm.ollama_client("http://127.0.0.1:9999") is not first
    assert CountingClient.built == 2


def test_the_default_host_is_the_settings_host():
    assert llm.ollama_client().host == llm.settings.OLLAMA_HOST


def test_a_patched_client_class_never_gets_a_cached_real_client(monkeypatch):
    """The cache is keyed by class too, so a test that patches ollama.Client is not handed a stale instance."""
    real_looking = llm.ollama_client("http://127.0.0.1:11434")

    class OtherClient(CountingClient):
        pass

    monkeypatch.setattr(ollama, "Client", OtherClient)
    assert llm.ollama_client("http://127.0.0.1:11434") is not real_looking


def test_many_answers_build_one_client_not_one_each():
    for _ in range(5):
        assert llm.generate("prompt") == "ok"
    assert CountingClient.built == 1


def test_streaming_and_blocking_answers_share_the_same_client():
    llm.generate("prompt")
    assert "".join(llm.generate_stream("prompt")) == "ab"
    assert CountingClient.built == 1


def test_the_pages_status_check_and_the_model_test_share_it_too():
    """The status check runs on EVERY click of the chat page: a new client each time cost 2.2 s per click."""
    for _ in range(4):
        backend.ollama_status()
    backend.check_model_generates()
    llm.generate("prompt")
    assert CountingClient.built == 1


def test_the_default_host_in_the_source_is_an_ip_literal_not_the_name_localhost():
    """Read from the source, not the live settings, so a developer's own OLLAMA_HOST cannot change the answer."""
    source = (PROJECT_ROOT / "src" / "core" / "config.py").read_text(encoding="utf-8")
    match = re.search(r'OLLAMA_HOST: str = os\.environ\.get\("OLLAMA_HOST", "([^"]+)"\)', source)
    assert match, "the OLLAMA_HOST setting line moved"
    assert match.group(1) == "http://127.0.0.1:11434"
    assert "OLLAMA_HOST=http://127.0.0.1:11434" in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
