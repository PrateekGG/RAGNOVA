"""
Measure Recall@5 and MRR against the gold-standard text queries — the
measurement data/README.md has been pointing at since Day 1 ("Chapter 7,
when we first measure Recall@5 and MRR").

Run with:  python scripts/evaluate_retrieval.py
(after scripts/build_index.py has built the real index at least once)

KNOWN LIMITATION, stated plainly rather than hidden: the questions below
are a manual copy of data/README.md's T1-T13 rows, not a parse of that
file. data/README.md is written for humans (prose, a callout box, a table
meant to be edited by hand) and thirteen rows still isn't enough to
justify a markdown-table parser. If you add or change a gold-set row in
data/README.md, update GOLD_QUESTIONS below to match — nothing enforces
that the two stay in sync automatically. A future chapter could promote
the gold set to a small machine-readable file (JSON/CSV) that
data/README.md renders from instead of hand-duplicating; noted here as
real, deferred future work, not silently worked around.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same fix as scripts/verify_setup.py (Ch5 §6.3) for the same reason: the
# HIT/MISS lines below print an em-dash, which an unconfigured Windows
# console renders as "?" instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipelines.documents.search import search_text

# Mirrors data/README.md's "Text -> text/document queries" table, T1-T13.
# Each entry: the question, and the (source, page) that should appear
# somewhere in the top-K results for the question to count as a hit.
GOLD_QUESTIONS = [
    {
        "id": "T1",
        "question": "If I don't get my system actually running by evaluation day, how many marks am I giving up?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 2,
    },
    {
        "id": "T2",
        "question": "As an undergrad, how many items can I check out from the library at once, and for how long?",
        "expected_source": "data/documents/library_hours.pdf",
        "expected_page": 1,
    },
    {
        "id": "T3",
        "question": "What happens the first time someone gets caught sharing their login with a friend?",
        "expected_source": "data/documents/it_onboarding.docx",
        "expected_page": 2,
    },
    {
        "id": "T4",
        "question": "What software must students set up if they want to read digital journals from home?",
        "expected_source": "data/documents/it_onboarding.docx",
        "expected_page": 1,
    },
    {
        "id": "T5",
        "question": "Where will the guide allotment list be announced?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 1,
    },
    {
        "id": "T6",
        "question": "How many people are allowed to work together on this project, at most?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 1,
    },
    {
        "id": "T7",
        "question": "Is there a maximum length for the synopsis, not counting the cover page or references?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 1,
    },
    {
        "id": "T8",
        "question": "If someone misses their individual viva without approval beforehand, what mark do they get for that part?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 2,
    },
    {
        "id": "T9",
        "question": "What happens if part of our submission turns out to be copied from somewhere else, even just a small section?",
        "expected_source": "data/documents/notice.pdf",
        "expected_page": 2,
    },
    {
        "id": "T10",
        "question": "On a Sunday, what time does the library close?",
        "expected_source": "data/documents/library_hours.pdf",
        "expected_page": 1,
    },
    {
        "id": "T11",
        "question": "Can I extend my library loan if nobody else wants that book?",
        "expected_source": "data/documents/library_hours.pdf",
        "expected_page": 1,
    },
    {
        "id": "T12",
        "question": "Before I can start using my new institute email account, what do I need to do first?",
        "expected_source": "data/documents/it_onboarding.docx",
        "expected_page": 1,
    },
    {
        "id": "T13",
        "question": "Am I allowed to download movies or paid software through the campus network without a license?",
        "expected_source": "data/documents/it_onboarding.docx",
        "expected_page": 2,
    },
]

TOP_K = 5


def evaluate() -> tuple[float, float]:
    """Returns (recall_at_k, mrr). Prints a per-question breakdown as it goes."""
    hits = 0
    reciprocal_ranks = []

    for item in GOLD_QUESTIONS:
        results = search_text(item["question"], top_k=TOP_K)
        rank = None
        for i, chunk in enumerate(results, start=1):
            if chunk.source == item["expected_source"] and chunk.page == item["expected_page"]:
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            print(f"[{item['id']}] HIT  at rank {rank}  (score={results[rank-1].score:.3f})  "
                  f"— {item['question']}")
        else:
            reciprocal_ranks.append(0.0)
            top1 = f"{results[0].source} p{results[0].page}" if results else "(no results)"
            print(f"[{item['id']}] MISS (expected {item['expected_source']} p{item['expected_page']}, "
                  f"top hit was {top1})  — {item['question']}")

    recall_at_k = hits / len(GOLD_QUESTIONS)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall_at_k, mrr


def main() -> None:
    print(f"Evaluating {len(GOLD_QUESTIONS)} gold questions at top_k={TOP_K}...\n")
    recall_at_k, mrr = evaluate()
    print(f"\nRecall@{TOP_K}: {recall_at_k:.2f}")
    print(f"MRR:       {mrr:.2f}")
    print(f"\nAdd a row to data/README.md's Results log:")
    print(f"| {date.today().isoformat()} | Ch7 | {recall_at_k:.2f} | {mrr:.2f} | N/A (no images/audio yet) | first real semantic search, {len(GOLD_QUESTIONS)} text queries |")


if __name__ == "__main__":
    main()
