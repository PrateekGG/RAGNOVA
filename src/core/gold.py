"""
The gold-standard evaluation set, loaded from data/gold_set.json.

Why a JSON file and a loader instead of the lists each script used to carry:
data/README.md, scripts/evaluate_retrieval.py, scripts/evaluate_answers.py,
scripts/evaluate_cross_modal.py and tests/test_retrieval.py each kept their
own hand-copied version of the same questions, and the copies drifted twice
(one test checked 3 of 13 questions, one script still held 5 of 13 and the
old negative controls). One file read by all of them cannot drift; a test
(tests/test_gold_set.py) keeps the human-readable README tables honest too.

    from src.core.gold import load_gold_set
    gold = load_gold_set()
    for row in gold["text"]:
        row["id"], row["question"], row["expected_source"], row["expected_pages"]
"""

from __future__ import annotations

import json
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "gold_set.json"

# Out-of-corpus questions used in earlier chapters (ADR-009/ADR-010's original
# numbers). Not gold rows, so they stay out of the JSON and the README tables,
# but the measurement scripts add them to the negatives so those older
# figures stay comparable. One list here, not one per script.
LEGACY_NEGATIVES = [
    "How much is the tuition fee for one semester?",
    "What is the capital of France?",
]

# Section name -> the keys every row in it must carry.
_REQUIRED = {
    "text": {"id", "question", "expected_source", "expected_pages"},
    "text_to_image": {"id", "query", "expected_sources"},
    "image_to_doc": {"id", "query_image", "expected_sources"},
    "audio_topic": {"id", "clip", "query", "expected_sources"},
    "negatives": {"id", "question"},
}


def load_gold_set(path: Path | None = None) -> dict:
    """Read and sanity-check the gold set. Raises ValueError on a missing
    section or key, or a duplicated id, so a typo in the JSON fails loudly
    here rather than as a confusing KeyError halfway through an evaluation."""
    data = json.loads((path or GOLD_PATH).read_text(encoding="utf-8"))
    seen: set[str] = set()
    for section, required in _REQUIRED.items():
        if section not in data:
            raise ValueError(f"gold set is missing the {section!r} section")
        for row in data[section]:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"gold row {row.get('id')!r} in {section!r} is missing {sorted(missing)}")
            if row["id"] in seen:
                raise ValueError(f"gold id {row['id']!r} appears more than once")
            seen.add(row["id"])
    return data


def cross_modal_rows(gold: dict) -> list[dict]:
    """Every I/M/A row, each with a `kind` key (`text_to_image`, `image_to_doc`
    or `audio_topic`) — the shape scripts/evaluate_cross_modal.py reports on."""
    rows: list[dict] = []
    for kind in ("text_to_image", "image_to_doc", "audio_topic"):
        rows.extend({**row, "kind": kind} for row in gold[kind])
    return rows
