"""
Run the full gold-set (T1-T5, N1-N2 from data/README.md) through the real
RAG core, and print a side-by-side comparison for a human to rate.

Run with:  python scripts/evaluate_answers.py
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

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipelines.rag import answer_query
from src.pipelines.rag.prompt import format_provenance

# Mirrors data/README.md's gold-set tables. expected_source=None marks a
# negative control (N1/N2) — there is no "right" citation, only a correct
# refusal. Kept as a manual copy, not a parse of that file, for the exact
# reason scripts/evaluate_retrieval.py's own docstring gives: update both
# by hand if you edit a gold-set row.
GOLD_QUESTIONS = [
    {"id": "T1", "question": "If I don't get my system actually running by evaluation day, how many marks am I giving up?",
     "expected_source": "data/documents/notice.pdf", "expected_page": 2},
    {"id": "T2", "question": "As an undergrad, how many items can I check out from the library at once, and for how long?",
     "expected_source": "data/documents/library_hours.pdf", "expected_page": 1},
    {"id": "T3", "question": "What happens the first time someone gets caught sharing their login with a friend?",
     "expected_source": "data/documents/it_onboarding.docx", "expected_page": 2},
    {"id": "T4", "question": "How many earlier projects are we expected to briefly cover for context before explaining what makes ours different?",
     "expected_source": "data/documents/notice.pdf", "expected_page": 1},
    {"id": "T5", "question": "If I return a book really late, what's the most I could end up owing for it?",
     "expected_source": "data/documents/library_hours.pdf", "expected_page": 1},
    {"id": "N1", "question": "How much is the tuition fee for one semester?",
     "expected_source": None, "expected_page": None},
    {"id": "N2", "question": "What is the capital of France?",
     "expected_source": None, "expected_page": None},
]


def run() -> list[dict]:
    rows = []
    for item in GOLD_QUESTIONS:
        result = answer_query(item["question"])
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
    print(f"Running {len(GOLD_QUESTIONS)} gold questions through answer_query()...")
    rows = run()
    print_report(rows)

    print(f"\n{'=' * 70}")
    print("Rate each answer 1-5 by hand (docs/ROADMAP.md's Ch10 bar), then "
          "add a row to data/README.md's Results log, e.g.:")
    print(f"| {date.today().isoformat()} | Ch10 | - | - | - | Answer-quality check, "
          f"{len(GOLD_QUESTIONS)} questions (T1-T5 + N1-N2), human-rated 1-5: "
          f"<fill in average and notes> |")


if __name__ == "__main__":
    main()
