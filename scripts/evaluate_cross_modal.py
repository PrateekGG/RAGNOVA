"""
Run the cross-modal gold-set (I1-I4, M1-M2, A1-A2 from data/README.md)
against the real indexed corpus and report Recall@5 per category and
overall — this project's first real measurement of cross-modal
retrieval, closing the gap ADR-010 names as still open ("provisional...
not yet measured on this corpus").

Run with:  python scripts/evaluate_cross_modal.py
(after scripts/build_index.py has indexed documents + audio + images)

Three different retrieval paths, one per row category, because there
isn't one API call that covers all three:
  - I1-I4 (text -> image): retrieve(question, include_images=True)
    directly.
  - M1-M2 (image -> document): there's no CLIP-image-to-MiniLM-text
    bridge (ADR-003's first hard constraint), so this goes through the
    same OCR step the real UI uses (src/ui/backend.py's ocr_image()) to
    turn the query image into text, then an ordinary text_index search.
    If Tesseract isn't installed, ocr_image() returns "" and the row
    correctly misses rather than erroring — reported plainly, not hidden.
  - A1-A2 (audio topic -> anything): the audio clips are already indexed
    as ordinary text_index chunks at build time (ADR-005) — these rows
    document a spoken clip's *content*, not a live microphone query.
    Each gets a real, freshly-written question about that content (same
    paraphrase spirit as T1-T5), checked against either of its two
    acceptable expected sources (the clip's own topic overlaps a
    document AND an image).

KNOWN LIMITATION, stated plainly: 8 questions is fragile evidence — one
miss moves any category's Recall@5 by a lot (Chapter 7 §3.3 already
made this exact point for the 3-question text-only case). Measuring for
the first time is still worth doing; reporting it as a strong result
would not be.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from src.pipelines.rag.retrieve import retrieve
from src.ui.backend import ocr_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5

# Mirrors data/README.md's three cross-modal tables (a manual copy, not a
# parse — same tradeoff scripts/evaluate_retrieval.py's docstring already
# names: update both by hand if a gold-set row changes). expected_sources
# is a list because A1/A2 each accept either of two sources.
GOLD_QUESTIONS = [
    {"id": "I1", "kind": "text_to_image",
     "query": "Where can I see the Wi-Fi authentication screen for campus wireless?",
     "expected_sources": ["data/images/screenshot_wifi_setup.png"]},
    {"id": "I2", "kind": "text_to_image",
     "query": "Find the diagram showing how ChromaDB and OpenCLIP connect together",
     "expected_sources": ["data/images/diagram_rag_architecture.png"]},
    {"id": "I3", "kind": "text_to_image",
     "query": "What poster shows the upcoming AI and Machine Learning seminar venue?",
     "expected_sources": ["data/images/notice_seminar_poster.png"]},
    {"id": "I4", "kind": "text_to_image",
     "query": "Where is the AI & Machine Learning Research Laboratory located and who heads it?",
     "expected_sources": ["data/images/photo_lab_door_sign.png"]},
    {"id": "M1", "kind": "image_to_doc",
     "query_image": "data/images/screenshot_synopsis_portal.png",
     "expected_sources": ["data/documents/notice.pdf"]},
    {"id": "M2", "kind": "image_to_doc",
     "query_image": "data/images/screenshot_wifi_setup.png",
     "expected_sources": ["data/documents/it_onboarding.docx"]},
    {"id": "A1", "kind": "audio_topic",
     "query": "What did the announcement say about how the prototype demo and viva are weighted?",
     "expected_sources": ["data/documents/notice.pdf", "data/images/notice_midterm_schedule.png"]},
    {"id": "A2", "kind": "audio_topic",
     "query": "What did the orientation talk say about library hours and overdue fines?",
     "expected_sources": ["data/documents/library_hours.pdf", "data/images/notice_library_fines.png"]},
]


def _hit_rank(chunks, expected_sources: list[str]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.source in expected_sources:
            return rank
    return None


def evaluate() -> list[dict]:
    """Runs every gold question through the real retrieve() (or, for
    image_to_doc rows, real OCR first). Returns each row plus its hit
    rank (None = miss) and the actual results, for the report below."""
    rows = []
    for item in GOLD_QUESTIONS:
        if item["kind"] == "image_to_doc":
            image_path = PROJECT_ROOT / item["query_image"]
            with Image.open(image_path) as img:
                ocr_text = ocr_image(img.convert("RGB"))
            results = retrieve(ocr_text, top_k=TOP_K)
            note = "" if ocr_text.strip() else " (OCR found no text — Tesseract unavailable?)"
        else:
            results = retrieve(item["query"], top_k=TOP_K, include_images=True)
            note = ""

        rank = _hit_rank(results, item["expected_sources"])
        rows.append({**item, "rank": rank, "results": results, "note": note})
    return rows


def print_report(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Prints a HIT/MISS line per question (matching evaluate_retrieval.py's
    format) and returns {category: (hits, total)} for the summary."""
    per_category: dict[str, list[bool]] = {}
    for row in rows:
        per_category.setdefault(row["kind"], []).append(row["rank"] is not None)
        label = {"text_to_image": "text->image", "image_to_doc": "image->doc",
                  "audio_topic": "audio-topic"}[row["kind"]]
        query_desc = row.get("query", row.get("query_image", ""))
        if row["rank"] is not None:
            top = row["results"][row["rank"] - 1]
            print(f"[{row['id']}] HIT  at rank {row['rank']}  (score={top.score:.3f}, {label})  "
                  f"— {query_desc}{row['note']}")
        else:
            top1 = f"{row['results'][0].source}" if row["results"] else "(no results)"
            print(f"[{row['id']}] MISS (expected one of {row['expected_sources']}, top hit was {top1}, {label})  "
                  f"— {query_desc}{row['note']}")
    return {k: (sum(v), len(v)) for k, v in per_category.items()}


def main() -> None:
    print(f"Evaluating {len(GOLD_QUESTIONS)} cross-modal gold questions at top_k={TOP_K}...\n")
    rows = evaluate()
    per_category = print_report(rows)

    print()
    total_hits = sum(h for h, _ in per_category.values())
    total_n = sum(n for _, n in per_category.values())
    for kind, (hits, n) in per_category.items():
        print(f"Recall@{TOP_K} ({kind}): {hits}/{n} = {hits / n:.2f}")
    overall = total_hits / total_n
    print(f"Recall@{TOP_K} (overall, cross-modal): {total_hits}/{total_n} = {overall:.2f}")
    print(f"\nSample size is small (n={total_n}) — one miss swings any category's Recall@5 by a lot "
          f"(same caveat Chapter 7 §3.3 already made for the 3-question text-only case). "
          f"This is a first real measurement, not yet strong evidence either way.")

    print(f"\nAdd a row to data/README.md's Results log:")
    print(f"| {date.today().isoformat()} | Ch12 | N/A (cross-modal only, see next column) | N/A | "
          f"{overall:.2f} | First real cross-modal measurement, {total_n} questions "
          f"({', '.join(f'{k}: {h}/{n}' for k, (h, n) in per_category.items())}) "
          f"against the real corpus (real CLIP + real Tesseract OCR + real Whisper-indexed audio). "
          f"Small sample — not yet strong evidence. |")


if __name__ == "__main__":
    main()
