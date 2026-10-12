"""
The one function every track eventually calls to turn a prompt string into
an LLM answer: a thin wrapper around Ollama's local HTTP API.

ADR-002 fixes the model (Llama 3.2 3B Instruct, 4-bit) and the serving
mechanism (`ollama serve` + REST at settings.OLLAMA_HOST, `POST /api/generate`
specifically — never `/api/chat`, since this project sends single-shot
prompts with the retrieved context baked in, not a multi-turn conversation).

`ollama==0.4.4` has been sitting in requirements.txt since Day 4, pinned but
never imported by any code — scripts/verify_setup.py talks to the same HTTP
endpoint via raw urllib instead, deliberately, because that script has to
run *before* requirements.txt can be trusted installed. This module is what
finally exercises the pinned client, now that we're past that bootstrap
concern.
"""

from __future__ import annotations

from typing import Iterator

import ollama

from src.core.config import settings

# One Ollama client per (host, client class), reused. A fresh client means a fresh
# connection, and on Windows a connection to the NAME "localhost" costs about 2.2 s
# (IPv6 is tried first), against 0.2 s for 127.0.0.1 and 4 ms on a connection that
# is already open (measured 2026-10-07). The old code built a new client for every
# answer, so every answer paid that, and the page paid it again on every click for
# the status check. The class is part of the key so a test that patches
# `ollama.Client` gets its own client, never a cached real one.
_clients: dict[tuple[str, type], "ollama.Client"] = {}


def ollama_client(host: str | None = None) -> "ollama.Client":
    """The shared client for `host` (default settings.OLLAMA_HOST)."""
    host = host or settings.OLLAMA_HOST
    key = (host, ollama.Client)
    if key not in _clients:
        _clients[key] = ollama.Client(host=host)
    return _clients[key]


def generation_options() -> dict:
    """The Ollama `options` for every generation call, in one place (generate()
    and generate_stream() used to each spell them out). `num_gpu` is included
    only when OLLAMA_NUM_GPU is set, so by default Ollama chooses as before."""
    extra = settings.LLM_THINK_TOKENS if settings.LLM_THINK else 0     # room for the hidden reasoning
    options = {
        "temperature": settings.LLM_TEMPERATURE,
        "num_predict": settings.LLM_MAX_TOKENS + extra,
        "num_ctx": settings.LLM_NUM_CTX + extra,
    }
    if settings.OLLAMA_NUM_GPU is not None:
        options["num_gpu"] = settings.OLLAMA_NUM_GPU
    return options


def think_kwargs() -> dict:
    """`think=` for the Ollama call, only when LLM_THINK is set (see config.py). The model's reasoning comes back in
    a separate `thinking` field, never in `response`, so it is never shown as part of the answer."""
    return {} if settings.LLM_THINK is None else {"think": settings.LLM_THINK}


def generate(prompt: str) -> str:
    """Send `prompt` to the configured Ollama model and return its response
    text.

    Uses `ollama.Client(host=...).generate(...)`, the client's wrapper
    around `POST /api/generate` — the same endpoint ch05 §2.2 documents as
    "what Chapter 10's RAG core calls." `stream=False` so this function
    returns one complete string rather than a generator of tokens; see
    generate_stream() below for the streaming variant Chapter 11's UI uses.

    `options` bounds generation deliberately: `temperature` from
    settings.LLM_TEMPERATURE (already low, 0.1, so answers stay close to
    the retrieved evidence rather than creative); `num_predict` caps how
    many tokens the model can emit, so a degenerate repeating completion
    can't hang the caller forever; `num_ctx` widens Ollama's own default
    (2048) enough to hold TOP_K retrieved chunks plus prompt scaffolding
    (see settings.LLM_NUM_CTX's docstring for the arithmetic).
    """
    client = ollama_client()
    try:
        response = client.generate(
            model=settings.OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options=generation_options(),
            **think_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.OLLAMA_HOST} — "
            f"is `ollama serve` running, and is {settings.OLLAMA_MODEL!r} "
            f"pulled (`ollama pull {settings.OLLAMA_MODEL}`)? "
            f"Underlying error: {exc}"
        ) from exc

    return response["response"]


def generate_stream(prompt: str) -> Iterator[str]:
    """Same call as generate(), with `stream=True`: yields the answer a
    piece at a time as Ollama produces it, instead of one string at the end.

    This is the streaming variant Chapter 10 deferred to Chapter 11's UI.
    A CPU-only 3B model can take 10+ seconds for a full answer;
    showing words as they arrive makes the UI feel alive instead of frozen,
    even though the total time is identical.

    Ollama is only contacted when the caller starts iterating, so the
    connection error is raised from inside the loop, re-wrapped in the
    same RuntimeError message generate() uses.
    """
    client = ollama_client()
    try:
        for part in client.generate(
            model=settings.OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            options=generation_options(),
            **think_kwargs(),
        ):
            if part["response"]:                    # thinking-only pieces carry an empty response
                yield part["response"]
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {settings.OLLAMA_HOST} — "
            f"is `ollama serve` running, and is {settings.OLLAMA_MODEL!r} "
            f"pulled (`ollama pull {settings.OLLAMA_MODEL}`)? "
            f"Underlying error: {exc}"
        ) from exc
