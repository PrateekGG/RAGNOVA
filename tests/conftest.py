"""Shared test setup.

The lexical admit and the spelling fix are ON by default since 2026-10-11 (ADR-018), and both read the real index
(./chroma_db) when nothing is injected. A test that fakes the similarity searches must not get chunks or a rewritten
question from whatever index happens to be on the machine, so both start OFF here; the tests of those two features
switch them on themselves.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.pipelines.rag import retrieve as retrieve_module


@pytest.fixture(autouse=True)
def _no_index_reads_behind_a_faked_search(monkeypatch):
    monkeypatch.setattr(
        retrieve_module, "settings",
        dataclasses.replace(retrieve_module.settings, LEXICAL_ADMIT=False, QUERY_SPELLING_FIX=False),
    )
