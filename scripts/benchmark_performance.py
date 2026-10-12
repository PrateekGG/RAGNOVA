"""
Measure the system-performance numbers the methodology (section 8.3) asks for:
retrieval latency, end-to-end latency, indexing throughput per modality, and
Whisper's word error rate, each as median AND worst case (never best case).

Run with:  python scripts/benchmark_performance.py            (needs `ollama serve` for the end-to-end part)
           python scripts/benchmark_performance.py --no-llm   (everything except end-to-end latency)
           python scripts/benchmark_performance.py --latency-only
                                                              (retrieval + end-to-end + where the time goes;
                                                               skips the indexing and Whisper sections)
(after scripts/build_index.py; set CHROMA_PERSIST_DIR to benchmark a scratch index)

READ THIS BEFORE QUOTING ANY NUMBER. What the numbers do and do not mean:
  - They describe ONE machine, printed at the top of the output. The project's
    stated target is a CPU-only laptop. If this machine has a GPU, Ollama will
    use it, and the end-to-end latency and the 3B model's answers are then
    GPU numbers, not CPU-only ones. The embedding models (MiniLM, CLIP), OCR
    and Whisper run on the CPU either way.
  - Latencies are WARM: one throw-away call first, so model loading (a one-off
    cost of seconds) is not folded into a per-question figure. Cold start is
    reported separately by the indexing section (which includes model load).
  - Word error rate is measured only on the four synthetic text-to-speech
    clips, because only they have a known script. Clean synthetic speech is
    easier than real speech, so this UNDERSTATES the real error rate. The
    scorer does no number normalisation, so "forty percent" heard as "40%"
    counts as an error: that overstates it. The two effects pull opposite
    ways; neither is corrected for.
  - Audio throughput and word error rate are measured with the transcript cache
    (ADR-013) bypassed, so Whisper actually runs; they describe Whisper in
    THIS environment, not the cached transcripts the index normally uses.
  - Peak memory is not measured.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.gold import load_gold_set
from src.core.vector_store import get_client, get_text_collection
from src.pipelines.documents.search import search_text

DATA = PROJECT_ROOT / "data"


def _stats(seconds: list[float]) -> str:
    ordered = sorted(seconds)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return (f"median {statistics.median(seconds) * 1000:8.0f} ms   p90 {p90 * 1000:8.0f} ms   "
            f"worst {ordered[-1] * 1000:8.0f} ms   (n={len(seconds)})")


def _timed(fn, args_list) -> list[float]:
    fn(*args_list[0])                       # warm-up, discarded
    out = []
    for args in args_list:
        t0 = time.perf_counter()
        fn(*args)
        out.append(time.perf_counter() - t0)
    return out


def machine() -> None:
    print("MACHINE")
    print(f"  {platform.platform()}  |  Python {platform.python_version()}  |  {platform.processor()}")
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("p", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]

        mem = _Mem()
        mem.l = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        print(f"  RAM {mem.total / 2**30:.1f} GiB")
    except Exception:  # non-Windows: just skip the RAM line
        pass
    print("  Check `ollama ps` for CPU vs GPU inference before quoting the end-to-end figure.\n")


def retrieval_latency(gold: dict) -> None:
    print("RETRIEVAL LATENCY (warm)")
    texts = [(r["question"],) for r in gold["text"]]
    print("  text_index search, top 5, MiniLM:   ", _stats(_timed(lambda q: search_text(q, top_k=5), texts)))
    from src.pipelines.rag.retrieve import retrieve
    imgs = [(r["query"],) for r in gold["text_to_image"]]
    print("  retrieve(include_images=True):      ", _stats(_timed(lambda q: retrieve(q, include_images=True), imgs)))
    print("  (the second includes CLIP's text encoder, the image search, the ADR-011 gate and the merge)\n")


def end_to_end_latency(gold: dict) -> None:
    print("END-TO-END LATENCY (retrieve + prompt + generate, warm)")
    from src.pipelines.rag import answer_query
    questions = [(r["question"],) for r in gold["text"][::2]]       # every other question (17 of the 33 text questions)
    print("  answer_query, text only:            ", _stats(_timed(lambda q: answer_query(q), questions)))
    print()


def install_cold_prompts() -> None:
    """Make every measured request a COLD prompt. Ollama's runner keeps the KV cache of recent
    prompts, so asking the same prompt twice (or two prompts that share their start, such as the
    same retrieved chunks) skips the model's reading of it: on CPU a ~1800-token prompt costs about
    12 s the first time and 0.08 s the second (measured 2026-10-07 with Ollama's own timings). A
    new question normally retrieves different chunks, so cold is what a person waits for. A unique
    marker at the very start of the prompt defeats prefix reuse entirely; it adds about 8 tokens."""
    import uuid

    from src.pipelines.rag import answer as answer_module
    real_build_prompt = answer_module.build_prompt
    answer_module.build_prompt = lambda query, chunks: f"[request {uuid.uuid4().hex[:8]}]\n" + real_build_prompt(query, chunks)


def generation_breakdown(gold: dict) -> None:
    """Where an answer's time goes, from Ollama's own accounting of each request: reading the prompt
    (prefill), writing the answer, loading the model. Time to first token on the chat page is about
    load + prefill. Same 13 questions as end_to_end_latency(), cold prompts (install_cold_prompts)."""
    print("WHERE THE TIME GOES (Ollama's own timings per request; cold prompts)")
    from src.core.config import settings
    from src.core.llm import generation_options, ollama_client
    from src.pipelines.rag import answer as answer_module
    from src.pipelines.rag.retrieve import retrieve
    client = ollama_client()
    questions = [r["question"] for r in gold["text"][::2]]

    def one(question: str) -> dict:
        relevant = retrieve(question)
        prompt = answer_module.build_prompt(question, answer_module._for_prompt(relevant, question))
        started = time.perf_counter()
        reply = client.generate(model=settings.OLLAMA_MODEL, prompt=prompt, options=generation_options())
        return {
            "prompt_tokens": reply["prompt_eval_count"], "prefill": reply["prompt_eval_duration"] / 1e9,
            "gen_tokens": reply["eval_count"], "gen": reply["eval_duration"] / 1e9,
            "load": reply["load_duration"] / 1e9, "wall": time.perf_counter() - started,
        }

    one(questions[0])                                # warm-up, discarded
    runs = [one(q) for q in questions]

    def med(key: str) -> float:
        return statistics.median(r[key] for r in runs)

    print(f"  prompt tokens:   median {med('prompt_tokens'):6.0f}   longest {max(r['prompt_tokens'] for r in runs):5d}   (n={len(runs)})")
    print(f"  prefill (model reads the prompt):   median {med('prefill'):6.2f} s   worst {max(r['prefill'] for r in runs):6.2f} s")
    print(f"  answer tokens:   median {med('gen_tokens'):6.0f}   longest {max(r['gen_tokens'] for r in runs):5d}")
    print(f"  generation (model writes it):       median {med('gen'):6.2f} s   worst {max(r['gen'] for r in runs):6.2f} s")
    print(f"  model load:                         median {med('load'):6.2f} s")
    print(f"  time to first token on the page ~ load + prefill:  median "
          f"{statistics.median(r['load'] + r['prefill'] for r in runs):6.2f} s")
    print(f"  model call, wall clock:             median {med('wall'):6.2f} s   p90 "
          f"{sorted(r['wall'] for r in runs)[min(len(runs) - 1, int(0.9 * len(runs)))]:6.2f} s   worst {max(r['wall'] for r in runs):6.2f} s")
    print()


def configuration() -> None:
    from src.core.config import settings
    print("CONFIGURATION OF THIS RUN (so a number can always be traced to its settings)")
    print(f"  OLLAMA_MODEL={settings.OLLAMA_MODEL}  OLLAMA_NUM_GPU={settings.OLLAMA_NUM_GPU}  TOP_K={settings.TOP_K}  "
          f"LLM_MAX_TOKENS={settings.LLM_MAX_TOKENS}  LLM_NUM_CTX={settings.LLM_NUM_CTX}  "
          f"CONTEXT_WORDS_PER_CHUNK={getattr(settings, 'CONTEXT_WORDS_PER_CHUNK', 'n/a')}")
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(f"{settings.OLLAMA_HOST}/api/version", timeout=5) as response:
            print(f"  ollama server version: {json.load(response)['version']}")
    except Exception as exc:                                  # a missing server is reported where it matters, in the e2e section
        print(f"  ollama server version: unknown ({type(exc).__name__})")
    print()


def indexing_throughput() -> dict[str, str]:
    """Index into a THROWAWAY Chroma dir (never the real one) and time each modality."""
    print("INDEXING THROUGHPUT (throw-away index; includes first-use model loading)")
    from src.pipelines.audio.index import index_audio_file
    from src.pipelines.documents.index import index_document_file
    from src.pipelines.documents.ingest import ingest_document
    from src.pipelines.images.index import index_images_directory

    transcripts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        client = get_client(persist_dir=Path(tmp) / "chroma")
        # documents: pages per second, parse + embed + store
        docs = ["data/documents/acl_bert.pdf", "data/documents/algorithms_ch04_greedy.pdf"]
        pages = sum(len({c.page for c in ingest_document(PROJECT_ROOT / d)}) for d in docs)
        t0 = time.perf_counter()
        chunks = sum(index_document_file(PROJECT_ROOT / d, client=client) for d in docs)
        dt = time.perf_counter() - t0
        print(f"  PDF: {pages} pages / {chunks} chunks in {dt:.1f} s  ->  {pages / dt:.2f} pages/s   (target >= 1 page/s)")
        # images: CLIP embedding + Tesseract OCR + store
        t0 = time.perf_counter()
        n = index_images_directory(DATA / "images", client=client)
        dt = time.perf_counter() - t0
        print(f"  images: {n} images in {dt:.1f} s  ->  {n / dt:.2f} images/s   (CLIP + OCR)")
        # audio: Whisper base on CPU
        import wave
        wavs = sorted((DATA / "audio").glob("*.wav"))
        seconds = 0.0
        for w in wavs:
            with wave.open(str(w), "rb") as f:
                seconds += f.getnframes() / f.getframerate()
        # The transcript cache (ADR-013) would turn this into a timing of file reads
        # (it once printed "996x real time"); bypass it so Whisper really runs.
        from src.pipelines.audio import ingestion
        saved_settings = ingestion.settings
        ingestion.settings = dataclasses.replace(saved_settings, WHISPER_TRANSCRIPT_CACHE_DIR="")
        try:
            t0 = time.perf_counter()
            for w in wavs:
                index_audio_file(w, client=client)
            dt = time.perf_counter() - t0
        finally:
            ingestion.settings = saved_settings
        print(f"  audio: {len(wavs)} clips, {seconds:.0f} s of speech in {dt:.1f} s  ->  {seconds / dt:.1f}x real time (Whisper `base`, CPU)")
        got = get_text_collection(client).get(include=["documents", "metadatas"], where={"modality": "audio"})
        for doc, md in zip(got["documents"], got["metadatas"]):
            transcripts[Path(md["source"]).name] = doc
    print()
    return transcripts


def _wer(reference: str, hypothesis: str) -> float:
    import re
    norm = lambda t: re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()
    r, h = norm(reference), norm(hypothesis)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def whisper_wer(transcripts: dict[str, str]) -> None:
    print("WHISPER WORD ERROR RATE (4 synthetic clips with a known script; see the caveats at the top)")
    spec = importlib.util.spec_from_file_location("gen", PROJECT_ROOT / "scripts" / "generate_multimodal_corpus.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rates = []
    for filename, script in gen.AUDIO_CLIPS:
        if filename in transcripts:
            rate = _wer(script, transcripts[filename])
            rates.append(rate)
            print(f"  {filename:36s} WER {rate:6.1%}   ({len(script.split())} words in the script)")
    if rates:
        print(f"  mean {sum(rates) / len(rates):.1%}, worst {max(rates):.1%}   (target < 15%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip the end-to-end latency (needs Ollama)")
    parser.add_argument("--allow-prompt-cache", action="store_true",
                        help="do NOT defeat Ollama's prompt cache (shows the warm, repeated-prompt case)")
    parser.add_argument("--latency-only", action="store_true",
                        help="retrieval, end-to-end and streamed latency only; skip indexing throughput and Whisper")
    args = parser.parse_args()
    gold = load_gold_set()
    if not args.allow_prompt_cache:
        install_cold_prompts()
    machine()
    configuration()
    print(f"  prompt cache: {'ALLOWED (warm case)' if args.allow_prompt_cache else 'defeated, every request is a cold prompt'}\n")
    retrieval_latency(gold)
    if not args.no_llm:
        end_to_end_latency(gold)
        generation_breakdown(gold)
    if args.latency_only:
        return
    transcripts = indexing_throughput()
    whisper_wer(transcripts)


if __name__ == "__main__":
    main()
