"""
Run the full gold-set (every T and N row of data/gold_set.json) through the real
RAG core, and print a side-by-side comparison for a human to rate.

Run with:  python scripts/evaluate_answers.py            (text only, the Chapter 10 behaviour)
           python scripts/evaluate_answers.py --images   (also search images, ADR-010/011)
(after scripts/build_index.py, with `ollama serve` running)

docs/ROADMAP.md's own stated Ch10 evaluation bar is "10 test questions,
humans rate answers 1-5" — an answer's quality (is it actually correct,
readable, appropriately hedged) is a judgment call this script cannot make
for you, unlike Recall@5/MRR's pure numbers. What IS automatable is
running every question through the pipeline and laying the result out
next to what's expected, so rating it is a five-minute read instead of a
five-minute setup, per question. Same division of labour as
scripts/evaluate_retrieval.py: the script measures/formats, a human judges.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.gold import load_gold_set
from src.pipelines.rag import answer_query
from src.pipelines.rag.prompt import format_provenance

# The questions come from data/gold_set.json (src/core/gold.py), shared with
# every evaluation script. A negative control has no expected source: the
# right outcome is a refusal.
_GOLD = load_gold_set()
GOLD_QUESTIONS = [
    {"id": r["id"], "question": r["question"], "expected_source": r["expected_source"],
     "expected_page": "/".join(str(n) for n in r["expected_pages"])}
    for r in _GOLD["text"]
] + [
    {"id": r["id"], "question": r["question"], "expected_source": None, "expected_page": None}
    for r in _GOLD["negatives"]
]


def run(include_images: bool = False) -> list[dict]:
    rows = []
    for item in GOLD_QUESTIONS:
        result = answer_query(item["question"], include_images=include_images)
        rows.append({**item, "result": result})
    return rows


def print_report(rows: list[dict]) -> None:
    for row in rows:
        result = row["result"]
        print(f"\n{'=' * 70}")
        print(f"[{row['id']}] {row['question']}")
        print(f"{'-' * 70}")
        if row["expected_source"] is None:
            print(f"Expected: refusal (not covered by the corpus)")
        else:
            print(f"Expected: {row['expected_source']}, page {row['expected_page']}")
        print(f"{'-' * 70}")
        print(f"Answer: {result.answer}")
        if result.citations:
            print("Sources:")
            for i, chunk in enumerate(result.citations, start=1):
                print(f"  [{i}] {format_provenance(chunk)}  (score={chunk.score:.3f})")
        else:
            print("Sources: (none — below relevance threshold, no LLM call made)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold questions through the real RAG core.")
    parser.add_argument("--images", action="store_true",
                        help="also search image_index (include_images=True); default is text only")
    args = parser.parse_args()
    mode = "text + images (include_images=True)" if args.images else "text only"
    print(f"Running {len(GOLD_QUESTIONS)} gold questions through answer_query(), {mode}...")
    rows = run(include_images=args.images)
    print_report(rows)

    print(f"\n{'=' * 70}")
    print("Rate each answer 1-5 by hand (docs/ROADMAP.md's Ch10 bar), then "
          "add a row to data/README.md's Results log, e.g.:")
    print(f"| {date.today().isoformat()} | Ch10 | - | - | - | Answer-quality check, "
          f"{len(GOLD_QUESTIONS)} questions (T + N rows), human-rated 1-5: "
          f"<fill in average and notes> |")


if __name__ == "__main__":
    main()
