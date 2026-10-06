"""
An index written by one ChromaDB version cannot be opened by another (0.5.x
and 1.x differ), and the failure used to surface as a bare `KeyError: '_type'`
from deep inside ChromaDB. The error now says what the likely cause is and how
to fix it, and keeps the original exception attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import vector_store


class _IncompatibleClient:
    def get_or_create_collection(self, name, metadata=None):
        raise KeyError("_type")


@pytest.mark.parametrize("opener", [vector_store.get_text_collection, vector_store.get_image_collection])
def test_an_unreadable_index_raises_a_clear_error_with_the_fix_and_the_original_cause(opener):
    with pytest.raises(RuntimeError) as raised:
        opener(_IncompatibleClient())
    message = str(raised.value)
    assert "different ChromaDB version" in message
    assert "scripts/build_index.py" in message
    assert "KeyError" in message                       # the original error is still named
    assert isinstance(raised.value.__cause__, KeyError)


def test_a_normal_index_still_opens(tmp_path):
    client = vector_store.get_client(persist_dir=tmp_path / "chroma")
    assert vector_store.get_text_collection(client).count() == 0
