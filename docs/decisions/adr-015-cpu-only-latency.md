# ADR-015: CPU-only latency: where the time goes, one 2.2-second fix, and an opt-in profile

## Status
Accepted: 2026-10-07. The **shipped defaults are unchanged**; a CPU-only profile (`TOP_K=3`) is documented in `.env.example`. A fix to how the app connects to Ollama (shared client, `127.0.0.1`) ships on for everyone. Follows ADR-009 (what the model is shown) and the 2026-10-05 measurement that missed the target.

## Context

The methodology (8.3) sets a target of under 15 s end to end on the project's stated hardware, a CPU-only laptop. On 2026-10-05, on Ollama 0.20.6, the median was **29.2 s** (14.1 s with `TOP_K=3`) and the target was missed. Since then Ollama was reinstalled (0.40.0, a different runner), so every old figure needed re-measuring, and the levers needed a cause to aim at.

## What was measured

Machine: Intel Core i5-14600K, 15.7 GiB, Windows 11; `llama3.2:3b`; CPU-only runs use `OLLAMA_NUM_GPU=0` (checked with `ollama ps`: 100% CPU); n = 17 questions (every other text question); **every request a cold prompt** (see 2). Transcripts: `data/eval/performance_latency_*_2026-10-07.txt`.

**1. Where the time goes (Ollama's own timings per request, CPU, default `TOP_K=5`).** A median prompt of 2036 tokens; the model reads it (prefill) in **13.2 s** (about 150 tokens per second); writing the answer takes **2.2 s** (median 37 tokens; the 512-token cap is never reached by a normal answer); loading is 0. Reading the prompt is 85% of the model call. On the GPU the same call is 0.72 s (prefill 0.26 s). So a lower token cap (`LLM_MAX_TOKENS`) cannot help, and the only lever is how much the model is made to read.

**2. Ollama's prompt cache made my first numbers wrong.** A repeated prompt (or one that starts with the same text, such as the same retrieved chunks) costs 0.08 s of prefill on CPU instead of about 12 s. My first "time to first token" (2.5 s) timed a cache hit, because I asked the same 13 prompts twice in a row, and was discarded (`..._first-run_prompt-cache-contaminated_*.txt` are kept, named so). The benchmark now puts a unique marker at the start of every prompt (`--allow-prompt-cache` shows the warm case). A new question normally retrieves different chunks, so cold is what a person waits for; a follow-up on the same chunks is much faster.

**3. A 2.2-second cost that had nothing to do with the model.** `answer_query` took 2.9 s on the GPU while the model call took 0.68 s. The cause: every call built a new `ollama.Client`, and on Windows a **new connection to the name `localhost` costs about 2.2 s** (IPv6 is tried first), against 0.21 s for `127.0.0.1` and 4 ms on a connection already open. The chat page built another client on every click for its status check, so every click paid it too. Fixed: one shared client per host (`src/core/llm.py::ollama_client`, used by `generate`, `generate_stream` and the page) and a default host of `http://127.0.0.1:11434`. On the GPU, `answer_query` went from **2.9 s to 0.50 s** (p90 0.87 s); on CPU, from 17.8 s to 15.8 s.

**4. The levers** (`answer_query` after the fix; behaviour from `scripts/evaluate_answers.py`, one run each of a stochastic model on the GPU, since CPU and GPU answered alike on 2026-10-05; "substantive" = at least three real words besides citation markers):

| Setting | CPU median / p90 / worst | Prompt tokens | Earlier rows T1-T25: substantive / refused / bare | Negatives refused | Held-out T26-T33: substantive / refused / bare |
|---|---|---|---|---|---|
| default, `TOP_K=5` | 15.8 s / 22.9 s / 25.3 s | 2036 | 21 / 2 / 2 | 16 / 16 | 6 / 1 / 1 |
| **`TOP_K=3`** | **9.2 s / 14.9 s / 15.5 s** | 1362 | **23 / 2 / 0** | 16 / 16 | **6 / 2 / 0** |
| trim each chunk to 120 words | 8.2 s / 12.6 s / 14.6 s | 1105 | 23 / 2 / 0 | 16 / 16 | **4 / 1 / 3** |
| trim to 80 words | not timed | | 21 / 4 / 0 | 16 / 16 | not run |
| `TOP_K=3` and trim to 120 | not timed | | 22 / 3 / 0 | 16 / 16 | not run |

Retrieval itself is 0.02 s (text) to 0.2 s (with images), so it is not a lever. `LLM_MAX_TOKENS` is not a lever (point 1).

**5. The held-out confirmation was worth running.** The eight held-out text questions (T26-T33, four documents no earlier question used, committed before any model ran on them) were run once, for three settings. Counting "answered, not refused" would have scored the default and trimming 7 of 8 and `TOP_K=3` 6 of 8. Reading the answers against the facts the questions were written from says otherwise: with trimmed context the model returned a bare `[3]`, `[1]` and `[2]` on three questions, gave BERT's parameter counts (110M, 340M) for the Fusion-in-Decoder reader, and invented that the maze search is "memoized"; with `TOP_K=3` six of eight were correct (T26 and T27 refused) against four at the default, which itself returned a bare `[2]` on T30 and 770M for the base model on T26. One run of eight on a 3B model is noise-sized in either direction; the direction (trimming hurts, `TOP_K=3` does not) matches the 25 earlier rows. The evaluation script now reports bare answers separately, because the old tally could not see this.

**6. One runaway.** In one cold benchmark pass with trimmed context a single answer ran to the 512-token cap (33.75 s). It did not recur in the three later trimmed runs. Cause not found; noted as a risk of that setting.

## Decision

1. **Ship the connection fix for everyone** (shared client, `127.0.0.1`). It is free and helps the GPU path most.
2. **Do not change `TOP_K` or any default.** The GPU path answers in 0.5 s at `TOP_K=5`, where more context costs nothing, and the retrieval ablation found `TOP_K=5` best (Recall@5 1.00, @3 0.96).
3. **Document `TOP_K=3` as the CPU-only profile** in `.env.example`. Measured: median 9.2 s and p90 14.9 s against the 15 s target, behaviour no worse on 33 questions (substantive answers 29 of 33 against 27 at the default), at a retrieval cost of 0.04 in Recall@K. The worst case, 15.5 s, is just over.
4. **Keep context trimming (`CONTEXT_WORDS_PER_CHUNK`, `src/pipelines/rag/context.py`) off, and recommend against it.** It saves about a second more than `TOP_K=3` and cost answer quality on the held-out rows. It stays as a documented knob for a machine so slow that a second matters more, and as the reason the measurement exists.

## Alternatives considered

1. **Lower the default `TOP_K` to 3 for everyone.** Rejected under the rule that a default changes only on a clear held-out win: it helps CPU, is neutral on the GPU, and the retrieval ablation prefers 5.
2. **Cap the answer length.** Pointless: answers are about 30 tokens.
3. **A smaller model (`llama3.2:1b`).** Not tried: it needs a download of about 1.3 GB, which was not requested, and it would be judged on answer quality, not speed. Recorded as untested.
4. **Ollama-level settings (flash attention, thread count).** Not tried: they change the user's Ollama environment, not this project.
5. **Judge the levers on "answered, not refused".** Rejected after point 5: it scored a setting that returned bare citation markers as a success.

## Consequences

**Positive**
+ Every answer is 2.2 s faster on every machine; the GPU path is 5.8 times faster (2.9 s to 0.50 s) and the page no longer pays 2.2 s per click.
+ The CPU-only target is met at the median (9.2 s) and the 90th percentile (14.9 s) with a documented profile; the default CPU figure is 15.8 s, 0.8 s over.
+ The benchmark is honest about cold prompts, reports where the time goes, and records its configuration; the evaluation reports bare answers.

**Negative / limits**
- The profile's worst case (15.5 s) and the default's median (15.8 s) are just over 15 s; "met" means the median and p90 of 17 questions on one machine.
- One CPU, one run per behaviour setting, 33 questions: differences of one or two rows are noise.
- The 15 s target was for a laptop; a slower laptop than this desktop CPU will be slower in proportion to its prefill speed (about 150 tokens per second here).
- The held-out text questions are worded closer to their sources than the earlier ones (overlap about 75% against 50%).

## Revisit if

A smaller model is approved for download, the Ollama runner changes again (the 29.2 s to 17.8 s drop on the same hardware most likely came from the engine, not from this project; the benchmark's question set and its handling of cold prompts also changed, so the two are not strictly comparable), or a real laptop is measured: its prefill speed sets everything.
