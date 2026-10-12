"""
Run the full gold-set (every T and N row of data/gold_set.json) through the real
RAG core, and print a side-by-side comparison for a human to rate.

Run with:  python scripts/evaluate_answers.py            (text only, the Chapter 10 behaviour)
           python scripts/evaluate_answers.py --images   (also search images, ADR-010/011)
           python scripts/evaluate_answers.py --images --image-rows
                                                         (also the text-to-image gold rows, as questions)
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
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.gold import load_gold_set
from src.pipelines.rag import answer_query
from src.pipelines.rag.answer import is_refusal
from src.pipelines.rag.prompt import format_provenance

# The questions come from data/gold_set.json (src/core/gold.py), shared with
# every evaluation script. A negative control has no expected source: the
# right outcome is a refusal.
_GOLD = load_gold_set()
GOLD_QUESTIONS = [
    {"id": r["id"], "question": r["question"], "expected_source": r["expected_source"],
     "expected_page": "/".join(str(n) for n in r["expected_pages"]), "tier": r.get("tier")}
    for r in _GOLD["text"]
] + [
    {"id": r["id"], "question": r["question"], "expected_source": None, "expected_page": None, "tier": r.get("tier")}
    for r in _GOLD["negatives"]
]


def select_rows(items: list[dict], rows: str) -> list[dict]:
    """--rows: 'regression' = rows without a held-out text tier (T1-T25, N1-N16), 'heldout' = only the
    held-out TEXT tier (T26-T33, N17-N20), 'all' = both. Text-to-image rows are added separately."""
    held = lambda item: (item.get("tier") or "").startswith("heldout-text")
    if rows == "regression":
        return [i for i in items if not held(i)]
    if rows == "heldout":
        return [i for i in items if held(i)]
    return items


# The text-to-image rows, asked as questions: does the answer cite the expected image,
# and is it an answer rather than a refusal? (Only with --image-rows.)
IMAGE_QUESTIONS = [
    {"id": r["id"], "question": r["query"], "expected_source": " or ".join(r["expected_sources"]),
     "expected_page": None, "expected_images": r["expected_sources"], "tier": r.get("tier")}
    for r in _GOLD["text_to_image"]
]


def _refused(result) -> bool:
    return is_refusal(result.answer)


def _bare(result) -> bool:
    """An answer with fewer than three real words once its citation markers are removed (a refusal
    always has more, so it is never bare): the model returned "[2]" and nothing else. Counted as 'answered' by a
    refusal check, but it answers nothing (seen 2026-10-07: 2 of 25 at the default, 3 of 8 with
    trimmed context), so the tallies report it separately."""
    return len(re.findall(r"[A-Za-z0-9]+", re.sub(r"\[\d+\]", " ", result.answer))) < 3


def run(include_images: bool = False, image_rows: bool = False, which: str = "all") -> list[dict]:
    rows = []
    for item in select_rows(GOLD_QUESTIONS, which) + (IMAGE_QUESTIONS if image_rows else []):
        result = answer_query(item["question"], include_images=include_images or image_rows)
        rows.append({**item, "result": result})
    return rows


def print_tallies(rows: list[dict]) -> None:
    """Counts a person would otherwise do by eye. 'Refused' = the model said it lacks the information
    (or no context survived retrieval); 'cites' = the expected image is among the numbered sources."""
    negatives = [r for r in rows if r["id"].startswith("N")]
    images = [r for r in rows if r.get("expected_images")]
    positives = [r for r in rows if r["id"].startswith("T")]
    print("\n" + "=" * 70)
    print("TALLIES (automatic; a person still judges whether each answer is right)")
    def tier_of(row) -> str:
        tier = row.get("tier") or ""
        return "held-out text" if tier.startswith("heldout-text") else "held-out image" if tier.startswith("heldout") else "earlier"

    def split(group, count) -> str:
        parts = []
        for name in ("earlier", "held-out image", "held-out text"):
            members = [r for r in group if tier_of(r) == name]
            if members:
                parts.append(f"{name} {sum(count(r) for r in members)}/{len(members)}")
        return ", ".join(parts)

    if positives:
        print(f"  text rows answered (not refused):        {sum(not _refused(r['result']) for r in positives)}/{len(positives)}"
              f"   ({split(positives, lambda r: not _refused(r['result']))})")
        print(f"  ... of which substantive (not bare [n]): {sum(not _refused(r['result']) and not _bare(r['result']) for r in positives)}/{len(positives)}"
              f"   bare: {[r['id'] for r in positives if _bare(r['result'])] or 'none'}")
    if negatives:
        print(f"  negatives refused:                       {sum(_refused(r['result']) for r in negatives)}/{len(negatives)}"
              f"   ({split(negatives, lambda r: _refused(r['result']))})")
        leaked = [r["id"] for r in negatives if not _refused(r["result"])]
        print(f"  negatives that were answered instead:    {leaked if leaked else 'none'}")
    if images:
        cited = [r for r in images if any(c.source in r["expected_images"] for c in r["result"].citations)]
        answered = [r for r in images if not _refused(r["result"])]
        print(f"  image questions: expected image cited:   {len(cited)}/{len(images)}   ({split(images, lambda r: r in cited)})")
        print(f"  image questions answered (not refused):  {len(answered)}/{len(images)}")
        print(f"  image questions NOT citing the image:    {[r['id'] for r in images if r not in cited]}")


def print_report(rows: list[dict]) -> None:
    for row in rows:
        result = row["result"]
        print(f"\n{'=' * 70}")
        print(f"[{row['id']}] {row['question']}")
        print(f"{'-' * 70}")
        if row["expected_source"] is None:
            print(f"Expected: refusal (not covered by the corpus)")
        else:
            page = f", page {row['expected_page']}" if row["expected_page"] is not None else ""
            print(f"Expected: {row['expected_source']}{page}")
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
    parser.add_argument("--rows", choices=("all", "regression", "heldout"), default="all",
                        help="which T/N rows to run: regression = T1-T25 and N1-N16 only, heldout = T26-T33 and N17-N20 only")
    parser.add_argument("--image-rows", action="store_true",
                        help="also run the text-to-image gold rows (I*) as questions; implies --images")
    args = parser.parse_args()
    mode = "text + images (include_images=True)" if (args.images or args.image_rows) else "text only"
    total = len(select_rows(GOLD_QUESTIONS, args.rows)) + (len(IMAGE_QUESTIONS) if args.image_rows else 0)
    print(f"Running {total} gold questions through answer_query(), {mode}...")
    rows = run(include_images=args.images, image_rows=args.image_rows, which=args.rows)
    print_report(rows)
    print_tallies(rows)

    print(f"\n{'=' * 70}")
    print("Rate each answer 1-5 by hand (docs/ROADMAP.md's Ch10 bar), then "
          "add a row to data/README.md's Results log, e.g.:")
    print(f"| {date.today().isoformat()} | Ch10 | - | - | - | Answer-quality check, "
          f"{len(select_rows(GOLD_QUESTIONS, args.rows))} questions (T + N rows), human-rated 1-5: "
          f"<fill in average and notes> |")


if __name__ == "__main__":
    main()
