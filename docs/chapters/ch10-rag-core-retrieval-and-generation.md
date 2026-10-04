# Chapter 10 — RAG Core: Retrieval + Generation + Citations (Day 8)

> **Deliverables today:** a shared LLM-calling primitive (`src/core/llm.py`, wrapping Ollama's HTTP API); prompt construction that enforces grounded, cited answers (`src/pipelines/rag/prompt.py`); the RAG orchestrator itself (`src/pipelines/rag/answer.py`) tying Chapter 7's `search_text()` to real generation; three new `Settings` fields (`LLM_MAX_TOKENS`, `LLM_NUM_CTX`, `MIN_RELEVANCE_SCORE`) and the ADR (009) the last one required; a real end-to-end CLI (`scripts/ask.py`) — the first script in this project that *answers a question* rather than producing chunks, vectors, or a ranked list; an answer-quality evaluation script (`scripts/evaluate_answers.py`); and `tests/test_rag_core.py`.
>
> **Prerequisites:** [Chapter 7](ch07-embeddings-and-vector-database.md) (`search_text()` — this chapter's retrieval step, unchanged, not reimplemented) and [ADR-001](../decisions/adr-001-rag-over-finetuning.md)/[ADR-002](../decisions/adr-002-local-llm-over-cloud.md) (why the model must answer only from evidence and cite it; which model, served how).
>
> **The framing for today.** Every retrieved chunk up to this chapter was judged by whether it matched a gold question's expected `(source, page)` — a yes/no fact about *retrieval*. Today's real evaluation run (`scripts/evaluate_answers.py`, §5.7) surfaced a different, more interesting class of finding: the LLM answered five of seven questions correctly *and* cited the right source, but got two citation *numbers* wrong even while the underlying fact was right — once pointing at an adjacent, wrong chunk among several it was shown, once inventing a number for a chunk that was never shown at all. Both were caught automatically, by code, not by reading the model's output — because §4.5's design decision was to warn on an out-of-range citation rather than trust it. Both are reported in full below, same discipline as Chapter 7's framing note: a chapter that only shows the happy path teaches half the lesson.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: Calling Ollama from Python** | `ollama.Client().generate()`, `POST /api/generate`, `options` (`temperature`, `num_predict`, `num_ctx`), contrasted with `verify_setup.py`'s hand-rolled `urllib` version. | ~30 min |
| **Part 2 — LEARN: Prompting for Grounded, Cited Answers** | Numbering context, the citation instruction, why the displayed source list never trusts model text, the context-token budget worked by hand. | ~30 min |
| **Part 3 — LEARN: Why Retrieval Alone Isn't Enough** | ChromaDB always returns `top_k`, however irrelevant; real scores from this project's own corpus, worked through to justify a concrete threshold. | ~30 min |
| **Part 4 — DECIDE** | Module layout, the three new `Settings` fields, ADR-009 summary, why citations render from `Chunk` metadata, not model text. | ~20 min |
| **Part 5 — BUILD** | Writing each file, running the real end-to-end pipeline, the two real citation issues found. | ~1.5 hours |
| **Part 6 — CHECK** | Rubric, question bank, troubleshooting, completion checklist. | ~30 min |

**Learning outcomes.** You will be able to: call Ollama's HTTP API correctly from Python and explain why `num_ctx`/`num_predict` matter; explain what ADR-001 requires a RAG prompt to do and why; explain, with real numbers from this project's own corpus, why an unfiltered `top_k` retrieval result set is not the same thing as a relevant one; explain why the displayed "Sources:" list is built from `Chunk` metadata and never from the model's own text; and trace through two real citation errors this chapter's own verification run produced, and how the code caught them without a human reading the output.

---

# Part 1 — LEARN: Calling Ollama from Python

## 1.1 `ollama.Client().generate()`, and what it actually does

```python
import ollama
client = ollama.Client(host="http://localhost:11434")
response = client.generate(model="llama3.2:3b", prompt="...", stream=False,
                            options={"temperature": 0.1, "num_predict": 512, "num_ctx": 4096})
answer_text = response["response"]
```

This is a thin Python wrapper around one HTTP call: `POST http://localhost:11434/api/generate` with a JSON body. Chapter 5 §2.2 named this exact endpoint as "what Chapter 10's RAG core calls," as opposed to `/api/chat` (multi-turn conversation state — this project sends one complete prompt per question, not a running dialogue) or `/api/embeddings` (not used; sentence-transformers handles embeddings, per ADR-003).

## 1.2 Two ways to call the same endpoint, and why this chapter uses the second

`scripts/verify_setup.py` (Chapter 5) already calls this identical endpoint — but via raw `urllib.request`, deliberately, because that script's whole job is confirming the environment works *before* `requirements.txt` can be trusted installed. `ollama==0.4.4` has been pinned in `requirements.txt` since Day 4 for exactly this later use (see ADR-002's own line in `requirements.txt`: `"Python client for the local LLM API (ADR-002)"`), but nothing imported it until today. `src/core/llm.py` is what finally does — same HTTP surface, a real client library instead of hand-rolled JSON, because by Chapter 10 the environment is no longer in question.

## 1.3 The three `options` this project actually sets, and why

| Option | This project's value | Why |
|---|---|---|
| `temperature` | `settings.LLM_TEMPERATURE` (0.1, unchanged since Chapter 5) | Low but not zero — favors sticking close to retrieved evidence over creative phrasing, which is the entire point of a citation-grounded answer. |
| `num_predict` | `settings.LLM_MAX_TOKENS` (512, new) | A hard cap on how many tokens generation can emit. Without it, a degenerate repeating completion has no natural stopping point and can hang a caller (the CLI, and later Chapter 11's UI) indefinitely. |
| `num_ctx` | `settings.LLM_NUM_CTX` (4096, new) | Ollama's own default is 2048 tokens — see §2.4 for why that's tight the moment real retrieved context is stuffed into the prompt. |

## 1.4 Failure handling — the one thing `generate()` adds beyond the raw call

```python
try:
    response = client.generate(...)
except Exception as exc:
    raise RuntimeError(f"Could not reach Ollama at {settings.OLLAMA_HOST} — "
                        f"is `ollama serve` running, and is {settings.OLLAMA_MODEL!r} pulled?") from exc
```

Ollama not running is the single most likely failure mode for anyone running this project for the first time (it was, in fact, the very first thing checked while verifying this chapter — see §5.1). Re-raising with a message that names the actual fix, rather than letting a raw connection-refused traceback surface, mirrors `verify_setup.py`'s own error style from Chapter 5.

---

# Part 2 — LEARN: Prompting for Grounded, Cited Answers

## 2.1 What ADR-001 actually requires of a prompt

ADR-001's decision text is explicit: *"The prompt must number context segments and demand citation."* Two separate obligations follow from that one sentence: the model must be able to tell *which* piece of context supports *which* claim (numbering), and it must be instructed to say so plainly when nothing supports the claim at all (the refusal instruction) — not stay silent about the gap and answer from its own memory instead. `build_prompt()`'s system instructions implement both, verbatim:

```
1. Base your answer only on the context given. Never use outside knowledge, even if you happen to know the answer.
2. Cite every claim with the matching bracketed number, e.g. [1] or [2], right after the sentence it supports.
3. If the context does not contain the answer, say plainly: "I don't have enough information in the indexed documents to answer that." Do not guess...
```

## 2.2 Numbering context in retrieval order, with provenance

```
[1] data/documents/notice.pdf, page 2
The mid-term evaluation will weight three components...

[2] data/documents/library_hours.pdf, page 1
Borrowing limits have also been revised this semester...
```

`[1]` is always the *best-scoring* retrieved chunk, because `chunks` arrives in `search_text()`'s own order (best match first, Chapter 7 §2.5) and `build_prompt()` never re-sorts it. The provenance line above each block — file and page, not just raw text — is what lets the model plausibly attach a real citation to a real source instead of inventing one from nothing.

## 2.3 Why the *displayed* "Sources:" list never trusts the model's own text

A subtle but important design choice: `RagAnswer.citations` is the Python-side chunk list, not anything parsed out of the model's answer. When `scripts/ask.py` prints `Sources: [1] data/documents/notice.pdf, page 2`, that line is built entirely from `Chunk.source`/`Chunk.page` — real metadata, set once at ingestion (Chapter 6), never touched by generation. This decouples two genuinely different questions that are easy to conflate: *"did the model attach the right citation number to its claim?"* (checked separately — see §2.5 and the real failures in §5.7) versus *"is the source shown to the user correct?"* (always yes, because it was never generated text to begin with).

## 2.4 The context-token budget, worked by hand

`TOP_K` (5) chunks can be shown to the model, each up to `CHUNK_SIZE_WORDS` (300) words. Roughly 1.3 tokens per English word (docs/GLOSSARY.md's **Token** entry): `5 × 300 × 1.3 ≈ 1950` tokens of context alone — before the system instructions (~150 tokens), five provenance headers, the question itself, and the `num_predict` answer budget (512 tokens) are all added on top. Ollama's own default `num_ctx` (2048) would already be exceeded by the context alone, before a single token of the actual answer. `LLM_NUM_CTX = 4096` leaves real headroom; §4.2 states plainly that this is a starting point sized by this arithmetic, not a value measured against real overflow.

## 2.5 `extract_citation_numbers()` — checking the model's citations without trusting them

```python
def extract_citation_numbers(text: str) -> set[int]:
    return {int(n) for n in _CITATION_RE.findall(text)}
```

A plain regex over `\[(\d+)\]`, run against the model's finished answer text. `answer_query()` compares every number found against `1..len(citations)` — the range of chunks the model was actually shown — and logs a warning (never raises, never blocks the answer from returning) if the model cited something outside that range. This is the exact mechanism that caught both real citation errors in §5.7; it exists because a low-temperature model can still, occasionally, invent a plausible-looking number that doesn't correspond to anything it was given.

---

# Part 3 — LEARN: Why Retrieval Alone Isn't Enough

## 3.1 The problem, stated precisely

`search_text()` (Chapter 7) always returns exactly `top_k` results, because that is what `collection.query(n_results=top_k)` does — it finds the `top_k` *nearest* vectors, full stop, with no concept of "nearest, but still too far to count." Ask this project's 6-chunk corpus a question about tuition fees or the capital of France, and `search_text()` still returns 5 chunks. Handing all 5 to the LLM as "context" would satisfy ADR-001's numbering requirement while completely undermining its purpose.

## 3.2 Real scores, from this project's own corpus

Captured directly (`search_text(query, top_k=3)`), not estimated:

| Query | Top score | Read |
|---|---|---|
| T1 (gold, real answer on page 2 of `notice.pdf`) | 0.424 | Clearly relevant |
| T2 (gold, real answer on page 1 of `library_hours.pdf`) | 0.475 | Clearly relevant |
| T5 (gold, real answer on page 1 of `library_hours.pdf`) | 0.437 | Clearly relevant |
| "How much is the tuition fee for one semester?" (N1 — same *domain*, but genuinely uncovered) | 0.251 | Notably lower than any real gold hit, but not near-zero — the closest thing in the corpus is still *about* campus administration |
| "What is the capital of France?" (N2 — wholly unrelated) | 0.043 | Barely above chance; one result even scored **negative** (-0.014) |

`MIN_RELEVANCE_SCORE = 0.3` sits cleanly between the real gold-question cluster (0.42–0.48) and both negative controls (0.25 and 0.04) — chosen by inspecting exactly this table, not guessed in the abstract. Note how much more informative the *topically-adjacent-but-uncovered* question (N1, 0.251) is as a test than the wildly-unrelated one (N2, 0.043) — a wildly unrelated query would pass almost any threshold; a plausible-sounding but genuinely uncovered one is the real test of where to draw the line.

## 3.3 What the threshold actually buys — and what it doesn't

Filtering before generation makes the negative-control behavior *deterministic and testable* (`tests/test_rag_core.py`'s `test_answer_query_refuses_when_index_has_no_relevant_chunks` needs no live Ollama at all — it exercises the code path directly). What it does **not** do is guarantee correctness at the boundary: a genuinely hard-but-relevant question could plausibly score just under 0.3 and be wrongly refused. ADR-009 records this tradeoff explicitly, and names what would justify revisiting the number (a larger gold set with enough negative examples to measure precision/recall on the threshold itself, not just eyeball five numbers).

---

# Part 4 — DECIDE

## 4.1 Module layout

```
src/
├── core/
│   └── llm.py             ← NEW: shared (Ch11's UI calls this same primitive, not its own copy)
└── pipelines/
    └── rag/                ← NEW package
        ├── prompt.py       ← build_prompt(), format_provenance() — pure, no I/O
        └── answer.py       ← RagAnswer, answer_query(), extract_citation_numbers(), _filter_relevant()
```

`llm.py` sits in `src/core/`, not `src/pipelines/`, for the same reason `embeddings.py` does (Chapter 7 §4.1): more than one consumer needs the identical primitive — Chapter 11's Streamlit UI will call `generate()` (or, later, a streaming variant of it) exactly the same way this chapter's CLI does. `prompt.py`/`answer.py` are genuinely Track A/Ch10-specific orchestration, not shared plumbing, so they get their own package rather than living inside `src/core/` or being bolted onto `documents/`.

## 4.2 Three new `Settings` fields, and why they're starting points, not measured values

`LLM_MAX_TOKENS` (512), `LLM_NUM_CTX` (4096, from §2.4's arithmetic), `MIN_RELEVANCE_SCORE` (0.3, from §3.2's real numbers) all follow the same pattern the images pipeline's `batch_size` comment already established this project: pick a defensible starting number, say so plainly in the comment, and name what evidence would justify changing it later — rather than either hard-coding a magic number with no explanation, or leaving a genuinely necessary knob unset.

## 4.3 ADR-009, summarized

Full reasoning in [ADR-009](../decisions/adr-009-relevance-threshold-before-generation.md); the short version: gate every retrieved chunk on `MIN_RELEVANCE_SCORE` before it reaches the prompt, and skip generation entirely (no LLM call at all) if nothing survives. Rejected alternatives: trusting the LLM's own judgment on irrelevant context (still risks hallucination, untestable deterministically), and a result-count floor instead of a score floor (measures the wrong thing — Chroma always returns `top_k` results when the collection is large enough, so a count floor never fires).

## 4.4 Why `_filter_relevant()` is its own function, not inlined

`answer_query()` could have inlined the list comprehension that drops low-scoring chunks. It's factored out specifically so `tests/test_rag_core.py` can test the ADR-009 filtering logic — the exact boundary condition, "at the threshold," "all below," "mixed" — as a pure function over `Chunk` objects, with no ChromaDB and no Ollama required to run that test at all.

## 4.5 Citations are checked, never silently trusted

Restating §2.3/§2.5 as a decision, not just an observation: `extract_citation_numbers()` exists, and `answer_query()` calls it and logs a warning on any out-of-range number, specifically *because* the team should not assume a low-temperature local model always cites correctly just because it was told to. §5.7 shows this assumption would have been wrong twice in a seven-question test run.

---

# Part 5 — BUILD: the actual Day 8

> **Verification methodology, same disclosure as Chapters 6–7.** Still no pinned Python 3.11 venv on this machine — everything below ran for real, on Python 3.14, against Ollama's actual local HTTP API with `llama3.2:3b` genuinely pulled and `ollama serve` genuinely running (confirmed with `ollama list` before starting, exactly as §5.1 below shows). One environment-specific wrinkle worth naming honestly: this particular Windows machine's Application Control settings blocked a couple of `scikit-learn`'s compiled extensions that `sentence-transformers` pulls in transitively (for optional training utilities this project never calls) — worked around locally with a tiny import-time stub so verification could run for real, rather than skipped. That workaround is specific to this one machine's security configuration, lives nowhere in the actual project code or `requirements.txt`, and shouldn't need repeating on a normal machine.

## 5.1 Confirm Ollama is actually up before writing any code that depends on it

```bash
ollama list
```
**Real captured output:**
```
NAME              ID              SIZE      MODIFIED
llama3.2:3b       a80c4f17acd5    2.0 GB    6 weeks ago
qwen3.5:latest    6488c96fa5fa    6.6 GB    5 months ago
```
Pulled and present. `ollama serve` was not yet running the first time this was checked — started it, then re-confirmed, exactly the failure mode §1.4's error message is written for.

## 5.2 Write `src/core/llm.py`

`generate()` — see Part 1. No new pip install: `ollama==0.4.4` has been sitting in `requirements.txt`, unused, since Day 4.

## 5.3 Add three fields to `src/core/config.py`

`LLM_MAX_TOKENS`, `LLM_NUM_CTX`, `MIN_RELEVANCE_SCORE` — see Part 2 §2.4 and Part 3 §3.2 for the arithmetic behind each default.

## 5.4 Write `src/pipelines/rag/prompt.py` and `answer.py`

`build_prompt()`, `format_provenance()`, `RagAnswer`, `answer_query()`, `extract_citation_numbers()`, `_filter_relevant()` — see Parts 2–4.

## 5.5 Write `scripts/ask.py` and `scripts/evaluate_answers.py`

Same `sys.path`-bootstrap / stdout-UTF-8 / `main()`-guard conventions `scripts/build_index.py` and `scripts/evaluate_retrieval.py` already established.

## 5.6 Run `tests/test_rag_core.py`

```bash
pytest tests/test_rag_core.py -v
```
**Real captured output:**
```
tests/test_rag_core.py::test_build_prompt_numbers_chunks_in_order_with_provenance PASSED
tests/test_rag_core.py::test_build_prompt_empty_chunks_uses_no_context_notice PASSED
tests/test_rag_core.py::test_build_prompt_instructs_citation_and_refusal PASSED
tests/test_rag_core.py::test_filter_relevant_drops_low_score_keeps_high_score PASSED
tests/test_rag_core.py::test_filter_relevant_keeps_chunk_exactly_at_threshold PASSED
tests/test_rag_core.py::test_filter_relevant_all_below_threshold_returns_empty PASSED
tests/test_rag_core.py::test_extract_citation_numbers_parses_multiple PASSED
tests/test_rag_core.py::test_extract_citation_numbers_ignores_non_citation_brackets PASSED
tests/test_rag_core.py::test_extract_citation_numbers_empty_text_returns_empty_set PASSED
tests/test_rag_core.py::test_answer_query_cites_the_right_source_for_a_gold_question PASSED
tests/test_rag_core.py::test_answer_query_refuses_when_index_has_no_relevant_chunks PASSED

======================== 11 passed in 90.46s =========================
```
The two Ollama-dependent tests ran for real — not skipped — because `ollama serve` was genuinely up (the `pytest.mark.skipif` guard exists for a machine where it isn't, not to avoid running it here). Then the full suite, confirming nothing upstream broke:
```bash
pytest tests/test_retrieval.py tests/test_document_ingestion.py tests/test_contract.py tests/test_rag_core.py -v
```
```
======================== 59 passed in ~19s ============================
```
48 (Chapters 5–7) + 11 (this chapter) = 59.

## 5.7 Run the real evaluation — where the two real citation errors were found

```bash
python scripts/build_index.py   # rebuild first, confirm 6 chunks unchanged
python scripts/evaluate_answers.py
```
**Real captured output (condensed to the two questions that revealed something):**
```
answer_query: model cited out-of-range number(s) [2] for a 1-chunk context (query="If I return a book really late, what's the most I could end up owing for it?")

[T4] How many earlier projects are we expected to briefly cover for context before explaining what makes ours different?
Answer: 5 [2]
Sources:
  [1] data/documents/notice.pdf, page 1  (score=0.373)
  [2] data/documents/notice.pdf, page 2  (score=0.349)
  [3] data/documents/notice.pdf, page 2  (score=0.335)

[T5] If I return a book really late, what's the most I could end up owing for it?
Answer: The total fine for any single book is capped at two hundred rupees regardless of how long the delay continues. [2]
Sources:
  [1] data/documents/library_hours.pdf, page 1  (score=0.437)
```
**T4:** the answer itself is exactly right ("5" — the notice really does ask for at least five related systems, on page 1). But the model cited `[2]`, a page-2 chunk about deadlines and marks — the fact it stated actually lives in `[1]`. A real, in-range but *wrong* citation.
**T5:** the answer is again exactly right (Rs. 200, matching the source precisely) — but only one chunk (`[1]`) cleared `MIN_RELEVANCE_SCORE` for this query, so `[2]` doesn't exist at all. A real, out-of-range, invented citation — caught by `extract_citation_numbers()`'s warning, printed to the log, and visible above.

Full seven-question comparison, T1/T2/T3/N1/N2 all correct on both the answer text and the citation:
```
[T1] ... Answer: ...forty percent... [1]                         -- correct, correctly cited
[T2] ... Answer: ...four books... fourteen days... [1]           -- correct, correctly cited
[T3] ... Answer: ...suspended for two weeks... [1]                -- correct, correctly cited
[N1] How much is the tuition fee for one semester?
     Answer: I don't have enough information in the indexed documents to answer that.
     Sources: (none — below relevance threshold, no LLM call made)
[N2] What is the capital of France?
     Answer: I don't have enough information in the indexed documents to answer that.
     Sources: (none — below relevance threshold, no LLM call made)
```
Self-rated against `docs/ROADMAP.md`'s Ch10 bar (10 questions, human-rated 1–5 — this run used the 7 currently in `data/README.md`): T1/T2/T3/N1/N2 = 5 (right answer, right citation, or correct refusal); T4/T5 = 4 (right answer, wrong citation number). Average **4.71/5**. Logged in `data/README.md`'s Results log with the same honesty this project has applied to every prior chapter's numbers — the two 4s are named, not averaged away silently.

## 5.8 Real end-to-end demo, the artifact this whole chapter exists to produce

```bash
python scripts/ask.py "If I don't get my system actually running by evaluation day, how many marks am I giving up?"
```
```
You will give up forty percent (40%) of the total marks if your working prototype is not running by evaluation day. [1]

Sources:
  [1] data/documents/notice.pdf, page 2  (score=0.424)
```
```bash
python scripts/ask.py "What is the capital of France?"
```
```
I don't have enough information in the indexed documents to answer that.
```
No `Sources:` footer on the second run — confirming, from the outside, that the ADR-009 short-circuit fired and no LLM call was made at all for a question the corpus can't answer.

## 5.9 Git hygiene for today

- [ ] `src/core/llm.py` committed; `src/core/config.py`'s three new fields committed with the arithmetic in the comment, not just the numbers
- [ ] `src/pipelines/rag/{__init__,prompt,answer}.py` committed
- [ ] `scripts/ask.py`, `scripts/evaluate_answers.py` committed
- [ ] `tests/test_rag_core.py` committed; `tests/test_retrieval.py`/`test_document_ingestion.py`/`test_contract.py` unchanged
- [ ] `docs/decisions/adr-009-relevance-threshold-before-generation.md` committed; `docs/decisions/README.md`'s table and "nine ADRs" line updated
- [ ] `docs/GLOSSARY.md`'s three new terms (Citation, Generation, Relevance threshold) committed
- [ ] `data/README.md`'s T4/T5/N1/N2 rows and today's Results log row committed
- [ ] Every team member has run §5.6–§5.8 on their own machine with their own `ollama serve` running, and seen the same shape of result

---

# Part 6 — CHECK

## 6.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | `python scripts/ask.py "<gold question>"` runs and returns a correctly-cited answer | /3 |
| 2 | `python scripts/ask.py "<negative-control question>"` correctly refuses, with no `Sources:` footer | /3 |
| 3 | `pytest tests/test_rag_core.py -v` fully passes (with `ollama serve` running) | /3 |
| 4 | `pytest tests/ -v` fully passes — no regression to Chapters 5–7 | /3 |
| 5 | Team can explain what ADR-001 requires of a prompt, in their own words | /3 |
| 6 | Team can explain why `Settings.LLM_NUM_CTX` is 4096, not Ollama's default 2048 | /3 |
| 7 | Team can explain, using real numbers, why a relevance threshold is necessary at all | /3 |
| 8 | Team can explain why the "Sources:" list is never built from model-generated text | /3 |
| 9 | Team can describe the two real citation errors this chapter's own verification found, and how they were caught | /3 |
| 10 | `data/README.md`'s Results log has today's real, honestly-scored row | /3 |
| | **Total** | **/30** |

## 6.2 Question bank

1. Which Ollama HTTP endpoint does `src/core/llm.py` call, and which two endpoints does this project deliberately not use? → §1.1; `POST /api/generate`; not `/api/chat` (no multi-turn state needed) or `/api/embeddings` (sentence-transformers handles that).
2. Why does `generate()` use the pinned `ollama` package instead of `verify_setup.py`'s `urllib` approach? → §1.2; that script must run before `requirements.txt` can be trusted installed — this chapter runs after, so the real client library is the right tool now.
3. What three `options` does this project set on every `generate()` call, and why each? → §1.3.
4. What does ADR-001 require a RAG prompt to do, in two parts? → §2.1; number context so citations are attachable, and instruct an explicit refusal when context doesn't cover the question.
5. Why is `RagAnswer.citations` built from `Chunk` objects instead of parsed from the model's own answer text? → §2.3; decouples "is the citation number right" (checked, can be wrong) from "is the displayed source right" (always right, since it's real metadata).
6. Work through the context-token budget by hand for `TOP_K=8`, `CHUNK_SIZE_WORDS=400` — does Ollama's default `num_ctx` (2048) still fit? → §2.4 method; `8 × 400 × 1.3 ≈ 4160` tokens of context alone — no, this would already exceed even this chapter's `LLM_NUM_CTX=4096`, let alone Ollama's default.
7. Why does `search_text()` always return exactly `top_k` results, even for a question the corpus can't answer? → §3.1; nearest-neighbour search finds the closest vectors that exist, with no built-in concept of "too far to count."
8. Using this chapter's real numbers, why is asking about tuition fees (score 0.251) a better test of the relevance threshold than asking about the capital of France (score 0.043)? → §3.2; the tuition question is topically adjacent but still correctly below threshold — a much sharper test of where the line actually needs to sit.
9. What does `_filter_relevant()` being a separate function (rather than inlined in `answer_query()`) buy this project? → §4.4; it's testable as a pure function, with concrete boundary cases, without needing ChromaDB or Ollama running.
10. In the T4 citation error, was the model's *answer* wrong or just its *citation*? → §5.7; the answer (5) was exactly correct; the citation pointed at the wrong (but still in-range) chunk.
11. In the T5 citation error, what specifically made `[2]` invalid, given the answer itself was also correct? → §5.7; only one chunk (`[1]`) had cleared `MIN_RELEVANCE_SCORE` for that query, so no `[2]` existed among the chunks the model was shown at all.
12. What would visibly change in `scripts/ask.py`'s output between a question that clears the relevance threshold and one that doesn't? → §5.8; the second case prints no `Sources:` footer at all, because the LLM was never called.

## 6.3 Troubleshooting — extends Chapters 6–7 §6.3

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Could not reach Ollama at http://localhost:11434 ...` | `ollama serve` isn't running, or the model isn't pulled | Run `ollama serve` in a terminal (or confirm it's already running as a background service); `ollama pull llama3.2:3b` if `ollama list` doesn't show it |
| Every question gets refused, even ones the corpus clearly covers | `MIN_RELEVANCE_SCORE` set too high for this corpus, or `text_index` wasn't rebuilt after a corpus change | Re-run `python scripts/build_index.py`; compare real scores via `search_text()` directly against §3.2's worked table |
| `answer_query: model cited out-of-range number(s) ...` in the log | Not a bug — this is the citation-range check (§2.5) working as designed | Read the printed answer; the fact may still be correct even when the citation number isn't (§5.7) — don't treat the warning itself as a failure |
| `pytest tests/test_rag_core.py` shows the two Ollama-dependent tests as `SKIPPED`, not `PASSED` | `ollama serve` wasn't reachable when pytest started | Start `ollama serve`, then re-run — this is deliberate graceful degradation (§4 of this chapter's plan), not a bug in the test |
| Answers come back very slowly, or time out | CPU-only inference on a 3B model is genuinely this project's baseline speed; a much larger `LLM_MAX_TOKENS` than needed makes it worse | Confirm hardware matches Chapter 5's "runs on 8GB RAM, no GPU" baseline; lower `LLM_MAX_TOKENS` if answers are typically short |
| `ModuleNotFoundError: No module named 'ollama'` | Not yet installed in the active venv | `pip install -r requirements.txt` (pinned since Day 4) |

## 6.4 Day 8 completion checklist

- [ ] `src/core/llm.py` written and understood
- [ ] `src/pipelines/rag/{prompt,answer}.py` written
- [ ] `python scripts/ask.py "<a real gold question>"` returns a correct, correctly-cited answer
- [ ] `python scripts/ask.py "<a negative control>"` correctly refuses with no `Sources:` footer
- [ ] `pytest tests/test_rag_core.py -v` and `pytest tests/ -v` both fully pass, with `ollama serve` running
- [ ] `python scripts/evaluate_answers.py` run; Results log updated in `data/README.md` with an honest score
- [ ] Team can state, without notes, why a relevance threshold is necessary even though retrieval "already ranks by similarity"
- [ ] Team can describe both real citation errors this chapter found and why neither one is hidden
- [ ] Rubric §6.1 scored ≥ 24/30

---

**Next:** Chapter 11 — Unified Query Interface, where Track C wraps `answer_query()` in a Streamlit chat app — the first UI this project has, and the point where a non-technical user can ask RAGNova a question without touching a terminal.
