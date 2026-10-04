"""
Summarize the chat UI's feedback log (Chapter 12) into numbers you can
paste into docs/feedback-log.md and the mid-term report.

Run with:  python scripts/summarize_feedback.py [path/to/feedback.jsonl]

Prints the count, average rating, a 1-5 histogram, the same numbers split
by tester (so the team's own ratings and outside testers' ratings can be
reported separately — Chapter 3 §3.5.3's rule for not grading yourself),
and every answer rated 2 or lower in full, since those are the ones that
point at something to fix.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.config import settings
from src.ui.feedback import load_feedback, summarize


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(settings.FEEDBACK_LOG_PATH)
    entries = load_feedback(path)
    if not entries:
        print(f"No feedback yet in {path}. Rate some answers in the chat UI first.")
        return

    overall = summarize(entries)
    print(f"Feedback log: {path}")
    print(f"Ratings: {overall['count']}   average: {overall['average']}/5   rated <= 2: {overall['low']}")
    print("Histogram: " + "  ".join(f"{r}*: {n}" for r, n in overall["histogram"].items()))

    by_tester = defaultdict(list)
    for e in entries:
        by_tester[e.tester or "(anonymous)"].append(e)
    print("\nBy tester:")
    for tester, rows in sorted(by_tester.items()):
        s = summarize(rows)
        print(f"  {tester:<20} {s['count']:>3} rating(s), average {s['average']}/5")

    low = [e for e in entries if e.rating <= 2]
    if low:
        print("\nAnswers rated 2 or lower (read these first):")
        for e in low:
            print(f"\n  [{e.rating}*] {e.timestamp}  {e.tester or '(anonymous)'}")
            print(f"  Q: {e.query}")
            print(f"  A: {e.answer}")
            if e.comment:
                print(f"  Comment: {e.comment}")
            print(f"  Sources: {', '.join(e.sources) or '(none)'}")


if __name__ == "__main__":
    main()
