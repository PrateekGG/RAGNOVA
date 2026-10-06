"""
Measure Recall@5 and MRR against the gold-standard text queries — the
measurement data/README.md has been pointing at since Day 1 ("Chapter 7,
when we first measure Recall@5 and MRR").

Run with:  python scripts/evaluate_retrieval.py
(after scripts/build_index.py has built the real index at least once)

The questions are NOT defined here: they come from data/gold_set.json via
src/core/gold.py, the single source every evaluation script and
tests/test_retrieval.py share. (They used to be a hand-copied list per
file, and the copies drifted: one held 3 of 13 questions, another 5 of
13.) A hit means the expected source appears in the top K on any of the
row's expected pages.
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

from src.core.gold import load_gold_set
from src.pipelines.documents.search import search_text

GOLD_QUESTIONS = load_gold_set()["text"]

TOP_K = 5


def evaluate() -> tuple[float, float]:
    """Returns (recall_at_k, mrr). Prints a per-question breakdown as it goes."""
    hits = 0
    reciprocal_ranks = []

    for item in GOLD_QUESTIONS:
        results = search_text(item["question"], top_k=TOP_K)
        rank = None
        for i, chunk in enumerate(results, start=1):
            if chunk.source == item["expected_source"] and chunk.page in item["expected_pages"]:
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
            print(f"[{item['id']}] MISS (expected {item['expected_source']} p{item['expected_pages']}, "
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
    print(f"| {date.today().isoformat()} | Ch12 | {recall_at_k:.2f} | {mrr:.2f} | N/A (text only) | text retrieval, {len(GOLD_QUESTIONS)} gold questions |")


if __name__ == "__main__":
    main()
