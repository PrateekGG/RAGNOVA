"""
Sanity-check data/gold_set.json against the real corpus, and MEASURE how much
each text question's wording overlaps with the passage that answers it.

Run with:  python scripts/check_gold_set.py

Why the overlap number matters: the gold set exists to show that *semantic*
search beats Ctrl+F (data/README.md). A question that copies the passage's
words proves nothing about meaning. The synthetic starter questions (T1-T13)
were written to share no keywords at all. Real documents make that harder, since a
question about BERT has to say "BERT", so this script reports the overlap
instead of promising zero: the share of the question's content words (4+
letters, minus common words) that also appear on the answer page's best
matching chunk. A high share is a flag to reword, not an error.

It also checks, against the real files, that every expected source exists and
that every expected page actually exists in the ingested document, so a typo
in the gold set fails here, not as a mysterious "MISS" in an evaluation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.gold import load_gold_set
from src.pipelines.documents.ingest import ingest_document

FLAG_ABOVE = 0.5

_STOP = set("""
about above after again against also among because been before being between both
can't cannot could does doing during each from have having here into just like
many more most much must only other over same should some such than that their
them then there these they this those through under until very want were what
when where which while will with without would your yours
""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def best_overlap(question: str, chunks: list, pages: list[int]) -> tuple[float, set[str]]:
    """The share of the question's content words found in the best matching
    chunk on one of `pages`, and which words those were."""
    q_words = content_words(question)
    best_share, best_shared = 0.0, set()
    for chunk in chunks:
        if chunk.page in pages:
            shared = q_words & content_words(chunk.text)
            share = len(shared) / len(q_words) if q_words else 0.0
            if share > best_share:
                best_share, best_shared = share, shared
    return best_share, best_shared


def question_overlaps(gold: dict) -> dict[str, float]:
    """id -> wording overlap for every text question whose expected file and
    pages exist. scripts/evaluate_bm25_baseline.py uses it to split the
    questions into low- and high-overlap groups."""
    cache: dict[str, list] = {}
    out: dict[str, float] = {}
    for row in gold["text"]:
        path = PROJECT_ROOT / row["expected_source"]
        if not path.exists():
            continue
        chunks = cache.setdefault(row["expected_source"], ingest_document(path))
        out[row["id"]] = best_overlap(row["question"], chunks, row["expected_pages"])[0]
    return out


def main() -> int:
    gold = load_gold_set()
    problems = 0
    cache: dict[str, list] = {}

    print(f"{'id':5s} {'overlap':>8s}  shared content words (best chunk on an expected page)")
    overlaps = []
    for row in gold["text"]:
        path = PROJECT_ROOT / row["expected_source"]
        if not path.exists():
            print(f"{row['id']:5s} MISSING FILE {row['expected_source']}")
            problems += 1
            continue
        chunks = cache.setdefault(row["expected_source"], ingest_document(path))
        pages_present = {c.page for c in chunks}
        absent = [p for p in row["expected_pages"] if p not in pages_present]
        if absent:
            print(f"{row['id']:5s} PAGE(S) {absent} DO NOT EXIST in {row['expected_source']} (has {sorted(pages_present)})")
            problems += 1
            continue
        best_share, best_shared = best_overlap(row["question"], chunks, row["expected_pages"])
        overlaps.append((row["id"], best_share))
        flag = "  <-- high" if best_share > FLAG_ABOVE else ""
        print(f"{row['id']:5s} {best_share:8.0%}  {sorted(best_shared)}{flag}")

    for section in ("text_to_image", "image_to_doc", "audio_topic"):
        for row in gold[section]:
            for src in row["expected_sources"] + [row.get("query_image"), row.get("clip")]:
                if src and not (PROJECT_ROOT / src).exists():
                    print(f"{row['id']:5s} MISSING FILE {src}")
                    problems += 1

    if overlaps:
        values = sorted(v for _, v in overlaps)
        median = values[len(values) // 2]
        print(f"\nText questions: {len(overlaps)}. Overlap median {median:.0%}, max {values[-1]:.0%}; "
              f"{sum(1 for v in values if v > FLAG_ABOVE)} above the {FLAG_ABOVE:.0%} flag.")
    print("Gold-set integrity: " + ("OK, every expected file and page exists." if not problems else f"{problems} PROBLEM(S)."))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
