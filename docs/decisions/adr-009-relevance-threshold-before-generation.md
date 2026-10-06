# ADR-009: Gate retrieved chunks on a relevance-score floor before generation

## Status
Accepted — Day 8

## Context

ADR-001 commits this project to answering only from retrieved evidence, and to the LLM saying so plainly when the evidence doesn't cover a question — that is the entire mechanism by which RAG avoids hallucination (Objective O4, `data/README.md`'s negative-control rows). But ChromaDB's `.query()` (used by `search_text()`, Chapter 7) always returns exactly `top_k` nearest neighbours, regardless of how dissimilar every one of them actually is to the query. Ask this project's 6-chunk corpus something it has no answer for — tuition fees, the capital of France — and `search_text()` still hands back 5 chunks, because "closest of a bad lot" is still an answer to "find me the 5 nearest vectors." Handing all 5 to the LLM as if they were relevant context directly invites exactly the hallucination ADR-001 exists to prevent, and makes a negative-control gold question (`data/README.md` N1/N2) untestable in any deterministic way — the outcome would depend entirely on the LLM's own, unverified judgment about whether the context "looks relevant."

## Decision

Before a retrieved chunk is allowed into the generation prompt, its cosine-similarity `score` must clear `settings.MIN_RELEVANCE_SCORE` (0.3, a starting point — see Consequences). If no chunk clears it, generation is skipped entirely: `answer_query()` returns a fixed "I don't have enough information in the indexed documents to answer that" response without ever calling the LLM. This is implemented as a pure filter function (`src/pipelines/rag/answer.py`'s `_filter_relevant()`), independently testable without ChromaDB or Ollama running.

## Alternatives considered

1. **No threshold — trust the LLM's own judgment.** Send whatever `search_text()` returns and rely entirely on the prompt's instruction ("if the context doesn't contain the answer, say so") to produce an honest refusal. Rejected: this still stuffs 5 irrelevant chunks into context, which is exactly the setup that invites a model to connect dots that aren't there — and it makes negative-control tests non-deterministic, dependent on a specific model's specific judgment on a specific day rather than on this project's own retrieval math.
2. **Always call the LLM, word the instruction more strongly.** A stronger prompt ("do NOT guess, ONLY use exact matches") without any code-level filter. Rejected for the same core reason as (1): the filtering responsibility would live entirely inside an opaque model call this project can't unit test, instead of inside a plain Python function it can. It also spends a ~1-2 second local generation call on a question that was never going to be answerable.
3. **A hard result-count floor instead of a score floor** (e.g. "only proceed if `search_text` returns at least N results"). Rejected: `search_text` always returns `top_k` results whenever the collection has at least that many chunks (Chroma doesn't refuse to return distant neighbours), so a count-based floor would never actually fire — it measures the wrong thing.

## Consequences

**Positive**
+ Makes the negative-control gold rows (`data/README.md` N1/N2) genuinely testable and deterministic — `tests/test_rag_core.py`'s `test_answer_query_refuses_when_index_has_no_relevant_chunks` exercises this exact path without needing Ollama running at all.
+ Saves a wasted LLM call (and its latency) whenever retrieval already knows nothing relevant exists.
+ The filtering logic is a plain, three-line function over `Chunk.score` — inspectable and testable independently of any specific model's behaviour.

**Negative**
− `MIN_RELEVANCE_SCORE = 0.3` is a starting point picked by inspecting real scores from this project's own 6-chunk starter corpus (Chapter 10 §3.2), not a literature-derived or exhaustively-tuned value — a genuinely hard-but-relevant question could score just under 0.3 and be wrongly refused. Revisit as below.
− A single global threshold doesn't account for a query that's inherently vague (many mediocre-scoring but individually-somewhat-relevant chunks) versus one that's precise (one high-scoring chunk) — both are judged by the same cutoff today.

**Implications for other components**

- When Chapter 8/9's image/audio collections are eventually wired into retrieval (ADR-007's rank-based merge), each modality's scores must be filtered the same way *before* the cross-collection merge, not after — otherwise irrelevant image/audio results could still slip into a merged top-K purely by rank position even while scoring below any sensible relevance floor for their own collection.

## Revisit if

The gold set (`data/README.md`) grows enough negative and true-positive examples to actually measure precision/recall of the threshold itself, rather than eyeballing single examples — at that point, tune `MIN_RELEVANCE_SCORE` (or replace a single global constant with a per-query-type one) against real data instead of the current starting guess.

---

*Chapter 10 §3.2 and §5.4 work through the real scores that motivated 0.3 — a genuinely relevant chunk in this corpus scores comfortably in the 0.35–0.5 range, real off-topic queries score well under that; this ADR doesn't repeat that arithmetic, only records the decision it produced.*

## Measurement update (2026-10-05): the gold set grew, so the threshold could finally be measured

This is what "Revisit if" asked for. The corpus is now 19 documents (623 chunks) plus 8 audio transcripts, the gold set has 25 text questions and 6 negative controls (`python scripts/measure_relevance.py`), and the question is whether any score threshold separates answers from noise.

**It does not, and a smarter rule does not either.** Labelling each of the top 5 results as expected (E), noise (N: clearly the wrong material) or neutral (n: not the expected chunk but plausibly fine, e.g. a synthetic audio transcript repeating a PDF's facts), the weakest correct chunk (T4, 0.366) scores below the strongest noise chunk (T1's, 0.383).

| Rule on the top 5 | Questions whose expected chunk survives | Noise chunks kept, per question | Negatives refused by the rule alone |
|---|---|---|---|
| floor 0.30 (this ADR) | 25 / 25 | 1.04 | 4 / 6 |
| floor 0.35 | 25 / 25 | 0.60 | 5 / 6 |
| floor 0.40 | 20 / 25 | 0.40 | 6 / 6 |
| floor 0.50 | 15 / 25 | 0.28 | 6 / 6 |
| floor 0.30 + within 0.03 of the top score | 18 / 25 | 0.08 | 4 / 6 |
| floor 0.30 + within 0.10 of the top score | 24 / 25 | 0.80 | 4 / 6 |

Raising the floor trades correct answers for fewer noise chunks at about the same rate; "within X of the top" is worse, because a legitimate duplicate (the audio clip that repeats the notice) often outscores the expected chunk. Two negatives (the Python-sort question at 0.362 and the semester-fee question at 0.315) already clear 0.30, so the floor alone refuses only 4 of 6.

**End to end, through the real model**, the picture is gentler than the retrieval table suggests (`llama3.2:3b`, the production index of documents plus audio, 25 positives and 6 negatives, one run per variant, so differences of one question are within the model's own noise):

| Variant | Positives answered | Negatives refused |
|---|---|---|
| current: floor 0.30, original prompt, `TOP_K = 5` | 23 / 25 | 6 / 6 |
| floor 0.35 | 22 / 25 | 6 / 6 |
| prompt that says "ignore unrelated passages" | 22 / 25 | 6 / 6 |
| floor 0.35 and that prompt | 23 / 25 | 6 / 6 |
| `TOP_K = 3` | 22 / 25 | 6 / 6 (the expected chunk fell out of context for T14) |

No variant beats the current settings, so **this ADR's decision stands unchanged**: floor 0.30, `TOP_K = 5`, original prompt. All 6 negatives are refused end to end, but note that for two of them (the Python-sort question and the semester-fee question) it is the *model's* refusal, not the floor's; ADR-009's deterministic guarantee holds only for questions whose every chunk scores under 0.3.

**Two failure patterns the measurement exposed, neither a gating problem:**
1. `T14` and `T20` (a results table in the DPR paper, Table 1 of the OpenCLIP paper) are refused by every variant although the right chunk is in context. *(Corrected below: the first diagnosis of this, "PDF table text extracts as a run of numbers without its headers", was a hypothesis, tested and found wrong.)*
2. On an index of *documents only* (what `tests/test_rag_core.py` builds), T1's context is the correct chunk plus four irrelevant paper chunks and the model refused 3 of 3 runs, while on the production index it answers. Context composition changes a 3B model's behaviour unpredictably; `tests/test_rag_core.py::test_answer_query_cites_the_right_source_for_a_gold_question` is marked as an expected failure for that reason, with this section as the reason.

The step beyond these knobs, if retrieval precision becomes the limit, is a cross-encoder reranker (considered and deferred in ADR-011).

### Follow-up (2026-10-05): the T14 / T20 refusals, diagnosed properly

The first explanation above ("table text extracts without headers, so fix the table extraction") was tested before any code was written, and it does not hold:

- **PyMuPDF's `find_tables()` is not usable on this corpus.** It reported 176 "tables" over 259 pages in the 13 PDFs, mostly false positives (token boxes and letter grids in the BERT and *Algorithms* figures, empty plot grids in the OpenCLIP paper). DPR's target table came out fragmented (whole rows merged into single cells), OpenCLIP's Table 1 was not detected at all, and the lineless `strategy="text"` returned the entire page as one 77-row table with words split mid-token. Replacing page text with such output would have corrupted figure pages and still not fixed T20. Not adopted.
- **T14 is a distractor problem, not a parsing one.** Given the expected chunk **alone**, the model answers (0 of 3 refusals). Given it with the other four retrieved chunks (three from the same paper, with other numbers) it refuses 3 of 3 on the GPU, even with the right chunk moved to position 1. *(Refinement, same day: a CPU-only run of the same question over the same index **answered it correctly**, 78.4% against 59.1%, so the refusal is borderline and depends on the hardware's numerics, which is what sensitivity to distractors at the margin looks like. It also means one GPU run proves less than it seems to.)*
- **T20 is a capability limit of the 3B model.** It refuses 3 of 3 even with the expected chunk alone, and also when the table's rows are rebuilt from word positions, so the table's line structure is not the cause either. The chunk readably contains `Ours LAION-2B H/14 78.0 ...`; the model does not make the "which row is best" step.

So these two refusals are a limit of the generator on this question type, not a defect in ingestion, and no parsing change is justified. A larger model or a reranker plus a tighter prompt are the candidates if it matters; neither is in scope. `ROADMAP`, the results log and the report appendix are corrected to match.

