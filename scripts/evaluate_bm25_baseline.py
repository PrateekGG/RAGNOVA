"""
Does dense (MiniLM) retrieval actually beat keyword search on this corpus?

Run with:  python scripts/evaluate_bm25_baseline.py
(after scripts/build_index.py; set CHROMA_PERSIST_DIR to measure a scratch index)

The project's central claim (docs/ROADMAP.md, data/README.md, the synopsis) is
that semantic search finds what keyword search misses. That needs a keyword
baseline to be measured against. This ranks the SAME chunks the dense search
ranks (everything in text_index: documents and audio transcripts), with the
SAME hit rule as scripts/evaluate_retrieval.py (the expected source on one of
its expected pages, within the top 5), using Okapi BM25 (src/core/bm25.py).

It reports three rankers, on all 25 text gold questions and split by how much
of each question's wording also appears in its answer chunk (the overlap that
scripts/check_gold_set.py measures), because the claim is specifically about
questions that do NOT repeat the document's words:
  - dense:   MiniLM cosine, what the product uses
  - bm25:    keyword ranking
  - hybrid:  reciprocal rank fusion of the two (k=60). Informational only; it
             is not part of the product, and nothing here adopts it.

Honest limits of the baseline: BM25 here does no stemming ("retrieve" does
not match "retrieval"), which is also true of Ctrl+F but not of a tuned
search engine; and it is evaluated on 25 questions, so one question moves a
figure by 0.04.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.bm25 import BM25, tokenize
from src.core.gold import load_gold_set
from src.core.vector_store import get_client, get_text_collection
from src.pipelines.documents.search import _chunk_from_result, search_text
from src.pipelines.rag.retrieve import rrf_merge

TOP_K = 5
POOL = 50                   # how deep each ranker looks, so the hybrid has something to fuse
HIGH_OVERLAP = 0.5          # "above 50%" is the group scripts/check_gold_set.py flags


def _load_overlaps(gold: dict) -> dict[str, float]:
    spec = importlib.util.spec_from_file_location("check_gold_set", PROJECT_ROOT / "scripts" / "check_gold_set.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.question_overlaps(gold)


def _rank(chunks: list, row: dict) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if c.source == row["expected_source"] and c.page in row["expected_pages"]:
            return i
    return None


def _summary(ranks: list[int | None]) -> tuple[float, float]:
    hits = [r for r in ranks if r is not None and r <= TOP_K]
    recall = len(hits) / len(ranks) if ranks else 0.0
    mrr = sum(1.0 / r for r in hits) / len(ranks) if ranks else 0.0
    return recall, mrr


def main() -> None:
    gold = load_gold_set()
    rows = gold["text"]
    overlaps = _load_overlaps(gold)

    client = get_client()
    stored = get_text_collection(client).get(include=["documents", "metadatas"])
    chunks = [_chunk_from_result(i, d, m, 0.0) for i, d, m in zip(stored["ids"], stored["documents"], stored["metadatas"])]
    bm25 = BM25([tokenize(c.text) for c in chunks])
    print(f"Ranking {len(chunks)} chunks (text_index: documents + audio transcripts) for {len(rows)} questions.\n")

    results: dict[str, dict[str, int | None]] = {}
    print(f"{'id':4s} {'overlap':>7s}   {'dense':>6s} {'bm25':>6s} {'hybrid':>6s}   (rank of the expected chunk, '-' = not in the top {POOL})")
    for row in rows:
        dense = search_text(row["question"], top_k=POOL, client=client)
        order = bm25.top(tokenize(row["question"]), POOL)
        keyword = [chunks[i] for i in order]
        hybrid = rrf_merge([dense, keyword])
        r = {"dense": _rank(dense, row), "bm25": _rank(keyword, row), "hybrid": _rank(hybrid, row)}
        results[row["id"]] = r
        show = lambda v: "-" if v is None else str(v)
        print(f"{row['id']:4s} {overlaps.get(row['id'], float('nan')):7.0%}   {show(r['dense']):>6s} {show(r['bm25']):>6s} {show(r['hybrid']):>6s}")

    def block(title: str, ids: list[str]) -> None:
        print(f"\n{title} (n={len(ids)})")
        for method in ("dense", "bm25", "hybrid"):
            recall, mrr = _summary([results[i][method] for i in ids])
            print(f"   {method:7s} Recall@{TOP_K} {recall:.2f}   MRR {mrr:.2f}")

    all_ids = [r["id"] for r in rows]
    low = [i for i in all_ids if overlaps.get(i, 0) <= HIGH_OVERLAP]
    high = [i for i in all_ids if overlaps.get(i, 0) > HIGH_OVERLAP]
    block("ALL QUESTIONS", all_ids)
    block(f"LOWER wording overlap (<= {HIGH_OVERLAP:.0%} of the question's words appear in its answer chunk)", low)
    block(f"HIGHER wording overlap (> {HIGH_OVERLAP:.0%})", high)

    def rank_value(v):  # a miss ranks worse than anything found
        return POOL + 1 if v is None else v
    dense_better = sum(1 for r in results.values() if rank_value(r["dense"]) < rank_value(r["bm25"]))
    bm25_better = sum(1 for r in results.values() if rank_value(r["bm25"]) < rank_value(r["dense"]))
    print(f"\nHead to head over {len(rows)} questions: dense ranks the answer higher on {dense_better}, "
          f"BM25 on {bm25_better}, tied on {len(rows) - dense_better - bm25_better}.")
    print("\nOne question moves any figure above by 0.04; read differences smaller than that as noise.")


if __name__ == "__main__":
    main()
