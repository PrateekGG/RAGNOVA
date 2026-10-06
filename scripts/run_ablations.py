"""
The ablation studies methodology section 8.4 commits to: chunk size, top-K,
and the cross-modal merge policy. Each answers "was this parameter chosen or
guessed?" with a measurement.

Run with:  python scripts/run_ablations.py [--only chunk topk merge]   (default: all three)
           python scripts/run_ablations.py --no-llm                  (retrieval numbers only)
(after scripts/build_index.py; needs `ollama serve` for the end-to-end parts)

How it is built, so the numbers can be trusted:
  - Every chunk-size variant is a THROWAWAY index built from the real
    documents; the audio chunks are copied (text, vectors, metadata) from the
    real index, so transcripts are identical across variants and Whisper is
    not re-run. Images do not take part in the text ablations. The real
    index is only read.
  - Chunk size varies with the overlap held at one sixth of it (150/25,
    300/50, 600/100), so size is the only thing that changes. Audio chunk
    size is not varied: each clip is a single chunk already.
  - "End to end" means the real model (Ollama) answers the 25 gold text
    questions and 6 out-of-corpus questions; a positive counts as answered
    unless the reply is the not-enough-information refusal. That is "did it
    answer", NOT "was it right". The script also records the largest prompt
    the model was actually given, because chunks of 600 words, or ten chunks,
    can exceed the context window (LLM_NUM_CTX) and be silently truncated.
  - One run per variant of a stochastic model: a one-question difference
    is noise. 25 questions: one question moves Recall@5 by 0.04.

A shipped default (CHUNK_SIZE_WORDS, TOP_K, the merge policy) should change
only on a clear win that does not hurt the other metric. These numbers are
evidence for that decision, not an automatic one.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.config import settings
from src.core.gold import LEGACY_NEGATIVES, cross_modal_rows, load_gold_set
from src.core.llm import generation_options
from src.core.vector_store import get_client, get_text_collection
from src.pipelines.documents import ingest as ingest_module
from src.pipelines.documents.index import index_documents_directory
from src.pipelines.documents.search import search_text
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.answer import answer_query

DOCS_DIR = PROJECT_ROOT / "data" / "documents"
REFUSAL = "don't have enough information"
GOLD = load_gold_set()
NEGATIVES = [r["question"] for r in GOLD["negatives"]] + LEGACY_NEGATIVES

_CTX = {"num_ctx": settings.LLM_NUM_CTX}      # read by the recording wrapper below
_PROMPT_TOKENS: list[int] = []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _generate_recording(prompt: str) -> str:
    """Same call as src/core/llm.generate(), but with a settable context size
    and a record of the prompt's size. The size is ESTIMATED from the word
    count at 1.3 tokens per word (the ratio src/core/config.py's own comment
    uses): Ollama's own prompt_eval_count can exclude a cached prefix, so it
    would under-report exactly the truncation this is meant to catch."""
    import ollama

    response = ollama.Client(host=settings.OLLAMA_HOST).generate(
        model=settings.OLLAMA_MODEL, prompt=prompt, stream=False,
        options={**generation_options(), "num_ctx": _CTX["num_ctx"]},
    )
    _PROMPT_TOKENS.append(int(len(prompt.split()) * 1.3))
    return response["response"]


def _real_client():
    return get_client()


def build_variant_index(size: int, overlap: int, real_client, directory: str) -> tuple[object, int, float]:
    """A throwaway index with documents chunked at `size`/`overlap` and the
    real index's audio chunks copied in. Returns (client, text chunks, seconds)."""
    client = get_client(persist_dir=Path(directory) / f"chroma_{size}")
    saved = ingest_module.settings
    ingest_module.settings = dataclasses.replace(settings, CHUNK_SIZE_WORDS=size, CHUNK_OVERLAP_WORDS=overlap)
    try:
        t0 = time.perf_counter()
        index_documents_directory(DOCS_DIR, client=client)
        seconds = time.perf_counter() - t0
    finally:
        ingest_module.settings = saved
    audio = get_text_collection(real_client).get(where={"modality": "audio"},
                                                 include=["documents", "metadatas", "embeddings"])
    if audio["ids"]:
        get_text_collection(client).upsert(ids=audio["ids"], documents=audio["documents"],
                                           metadatas=audio["metadatas"], embeddings=audio["embeddings"])
    return client, get_text_collection(client).count(), seconds


def retrieval_metrics(client, ks=(3, 5, 10)) -> dict[int, tuple[float, float]]:
    """Recall@K and MRR@K over the 25 text gold questions: {K: (recall, mrr)}."""
    ranks = []
    for row in GOLD["text"]:
        hits = search_text(row["question"], top_k=max(ks), client=client)
        rank = next((i for i, c in enumerate(hits, 1)
                     if c.source == row["expected_source"] and c.page in row["expected_pages"]), None)
        ranks.append(rank)
    out = {}
    for k in ks:
        found = [r for r in ranks if r is not None and r <= k]
        out[k] = (len(found) / len(ranks), sum(1 / r for r in found) / len(ranks))
    return out


def end_to_end(client, top_k: int | None, num_ctx: int) -> dict:
    """Real-model behaviour: how many positives are answered, negatives refused,
    how often the expected chunk reached the prompt, and how big prompts got."""
    _CTX["num_ctx"] = num_ctx
    _PROMPT_TOKENS.clear()
    saved = answer_module.generate
    answer_module.generate = _generate_recording
    try:
        answered = expected_in_context = 0
        for row in GOLD["text"]:
            result = answer_query(row["question"], client=client, top_k=top_k)
            answered += REFUSAL not in result.answer.lower()
            expected_in_context += any(c.source == row["expected_source"] and c.page in row["expected_pages"]
                                       for c in result.citations)
        refused = sum(REFUSAL in answer_query(q, client=client, top_k=top_k).answer.lower() for q in NEGATIVES)
    finally:
        answer_module.generate = saved
    tokens = [t for t in _PROMPT_TOKENS if t]
    return {"answered": answered, "n_pos": len(GOLD["text"]), "refused": refused, "n_neg": len(NEGATIVES),
            "expected_in_context": expected_in_context,
            "max_prompt_tokens": max(tokens) if tokens else 0,
            "near_full_window": sum(1 for t in tokens if t >= 0.95 * num_ctx), "n_prompts": len(tokens),
            "num_ctx": num_ctx}


def _e2e_line(r: dict) -> str:
    warn = f"  <-- {r['near_full_window']} of {r['n_prompts']} prompts filled >=95% of the {r['num_ctx']}-token window (likely truncated)" \
        if r["near_full_window"] else ""
    return (f"answered {r['answered']}/{r['n_pos']}, refused {r['refused']}/{r['n_neg']}, expected chunk in context "
            f"{r['expected_in_context']}/{r['n_pos']}, largest prompt ~{r['max_prompt_tokens']} tokens (window {r['num_ctx']}){warn}")


# ---------------------------------------------------------------------------
# the three ablations
# ---------------------------------------------------------------------------

def ablate_chunk_size(real_client, use_llm: bool, tmp: str) -> dict[int, object]:
    print("\n=== ABLATION 1: chunk size (overlap held at one sixth of it) ===")
    clients: dict[int, object] = {}
    for size, overlap in ((150, 25), (300, 50), (600, 100)):
        client, n_chunks, seconds = build_variant_index(size, overlap, real_client, tmp)
        clients[size] = client
        m = retrieval_metrics(client)
        recall, mrr = m[5]
        print(f"\n  {size} words / overlap {overlap}: {n_chunks} text chunks (documents + 8 audio), built in {seconds:.0f}s")
        print(f"     retrieval  Recall@5 {recall:.2f}   MRR {mrr:.2f}")
        if use_llm:
            print("     end to end ", _e2e_line(end_to_end(client, None, settings.LLM_NUM_CTX)))
            if size == 600:
                print("     end to end ", _e2e_line(end_to_end(client, None, 8192)), "  [window raised to 8192]")
    return clients


def ablate_top_k(clients: dict[int, object], use_llm: bool) -> None:
    print("\n=== ABLATION 2: top-K, on the shipped 300-word index ===")
    client = clients[300]
    m = retrieval_metrics(client, ks=(3, 5, 10))
    for k in (3, 5, 10):
        print(f"\n  K = {k}: retrieval Recall@{k} {m[k][0]:.2f}   MRR@{k} {m[k][1]:.2f}")
        if use_llm:
            print("     end to end ", _e2e_line(end_to_end(client, k, settings.LLM_NUM_CTX)))
            if k == 10:
                print("     end to end ", _e2e_line(end_to_end(client, k, 8192)), "  [window raised to 8192]")


def _score_merge(ranked_lists, k=None):
    """Merge by raw score: the policy ADR-007 rejected, kept here to measure the rejection."""
    return sorted((c for lst in ranked_lists for c in lst), key=lambda c: c.score or 0.0, reverse=True)


def ablate_merge() -> None:
    n_rows = len(cross_modal_rows(load_gold_set()))
    print(f"\n=== ABLATION 3: cross-modal merge policy (the {n_rows} cross-modal gold rows, real index) ===")
    spec = importlib.util.spec_from_file_location("evaluate_cross_modal", PROJECT_ROOT / "scripts" / "evaluate_cross_modal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(label: str) -> dict[str, int | None]:
        rows = module.evaluate()
        by_kind: dict[str, list[bool]] = {}
        for r in rows:
            by_kind.setdefault(r["kind"], []).append(r["rank"] is not None)
        summary = ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by_kind.items())
        total = sum(1 for r in rows if r["rank"] is not None)
        print(f"  {label:34s} {total}/{len(rows)} hit   ({summary})")
        return {r["id"]: r["rank"] for r in rows}

    rrf = run("rank-based RRF, k=60 (shipped)")
    original = retrieve_module.rrf_merge
    retrieve_module.rrf_merge = _score_merge
    try:
        by_score = run("raw-score merge (rejected, ADR-007)")
    finally:
        retrieve_module.rrf_merge = original
    changed = {i: (rrf[i], by_score[i]) for i in rrf if rrf[i] != by_score[i]}
    print("  rows whose rank changed (RRF -> score): " + (", ".join(f"{i} {a}->{b}" for i, (a, b) in changed.items()) or "none"))
    print("  I3 is the question to watch: it misses because its image is outranked by text chunks in the merged top 5.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=["chunk", "topk", "merge"], help="run just these (default: all)")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    which = set(args.only or ["chunk", "topk", "merge"])
    use_llm = not args.no_llm
    print(f"shipped defaults: CHUNK_SIZE_WORDS={settings.CHUNK_SIZE_WORDS}, CHUNK_OVERLAP_WORDS={settings.CHUNK_OVERLAP_WORDS}, "
          f"TOP_K={settings.TOP_K}, LLM_NUM_CTX={settings.LLM_NUM_CTX}, RRF_K={settings.RRF_K}")
    real_client = _real_client()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        clients: dict[int, object] = {}
        if which & {"chunk", "topk"}:
            clients = ablate_chunk_size(real_client, use_llm, tmp) if "chunk" in which else {
                300: build_variant_index(300, 50, real_client, tmp)[0]}
            if "topk" in which:
                ablate_top_k(clients, use_llm)
    if "merge" in which:
        ablate_merge()


if __name__ == "__main__":
    main()
