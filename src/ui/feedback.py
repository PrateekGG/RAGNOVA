"""
The human-feedback log (Chapter 12): one JSON line per rating a user gives
an answer in the chat UI, appended to settings.FEEDBACK_LOG_PATH.

JSON Lines, not a spreadsheet or a database: appending one line never has
to read or rewrite the rest of the file, two people's ratings can't
corrupt each other's rows, and `git diff` shows exactly which ratings
were added. scripts/summarize_feedback.py turns it into numbers for
docs/feedback-log.md and the mid-term report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import settings


@dataclass
class FeedbackEntry:
    query: str
    answer: str
    rating: int                      # 1-5, the same scale as Chapter 10's answer check
    sources: list[str]               # citation titles, in order
    comment: str = ""
    tester: str = ""                 # optional name/initials, so outside testers can be counted separately
    model: str = settings.OLLAMA_MODEL
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def record_feedback(entry: FeedbackEntry, path: str | Path | None = None) -> Path:
    """Append one entry. Creates the file (and its folder) on first use."""
    if not 1 <= entry.rating <= 5:
        raise ValueError(f"rating must be 1-5, got {entry.rating}")
    path = Path(path or settings.FEEDBACK_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return path


def load_feedback(path: str | Path | None = None) -> list[FeedbackEntry]:
    """Every entry in the log, oldest first. A missing file is an empty
    log, not an error. A corrupt line is skipped rather than losing every
    other rating because of one bad row."""
    path = Path(path or settings.FEEDBACK_LOG_PATH)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(FeedbackEntry(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return entries


def summarize(entries: list[FeedbackEntry]) -> dict:
    """Count, average rating, rating histogram, and how many answers were
    rated 2 or lower (the ones worth reading first)."""
    if not entries:
        return {"count": 0, "average": None, "histogram": {r: 0 for r in range(1, 6)}, "low": 0}
    ratings = [e.rating for e in entries]
    return {
        "count": len(ratings),
        "average": round(sum(ratings) / len(ratings), 2),
        "histogram": {r: ratings.count(r) for r in range(1, 6)},
        "low": sum(1 for r in ratings if r <= 2),
    }
