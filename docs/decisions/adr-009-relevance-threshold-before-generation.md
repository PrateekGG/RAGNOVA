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
