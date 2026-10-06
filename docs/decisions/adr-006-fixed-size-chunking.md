# ADR-006: Use fixed-size overlapping chunks, sized by ablation rather than literature

## Status
Accepted — Day 4

## Context

Document text (and audio transcripts, per ADR-005) must be split into segments before
embedding, because the embedding model truncates input beyond roughly 256 word-piece
tokens (Ch1 §1.5.4). The chunking strategy directly determines the shape of the `text`
field every downstream component consumes (Ch4 §4.2's interface-contract requirements),
so Track A needs this fixed before Chapter 6 is written.

## Decision

Split text into fixed-size chunks of approximately 300 words with 50 words of overlap
between consecutive chunks. The exact chunk size is treated as a **tunable parameter**,
to be determined empirically by ablation (Ch3 §3.6.7: chunk size 150 / 300 / 600 words),
not adopted as a fixed literature-backed value.

## Alternatives considered

1. **Semantic or recursive chunking** — splitting on structural or topical boundaries
   (paragraph breaks, heading changes, or LLM-identified semantic shifts) rather than a
   fixed word count. This is plausibly a better approach in principle: it avoids splitting
   a sentence or idea across a chunk boundary and produces chunks with more internally
   consistent meaning. **Rejected for this schedule, not on technical merit** — it requires
   either a layout-aware parser or an additional LLM call per document, more moving parts
   than a 9-day sprint with a beginner team can safely absorb this early, and it is
   explicitly recorded as future work.
2. **No overlap between chunks.** Rejected: a sentence spanning exactly the boundary between
   two zero-overlap chunks would appear intact in neither, silently losing retrievable
   content. The 50-word overlap exists specifically to guarantee every sentence appears
   whole in at least one chunk.
3. **One chunk per document.** Rejected outright: violates the embedding model's ~256-token
   limit (Ch1 §1.5.4) for any document longer than roughly 200 words — the model would
   silently truncate the rest, and most of a real document would never be indexed at all.
4. **Adopting a chunk size directly from a published paper or library default**, without
   validating it against this specific corpus. Rejected as intellectually dishonest: unlike
   ADR-001 (RAG vs. fine-tuning) or ADR-003 (two collections), where strong empirical
   evidence exists and is cited, **passage-granularity research is comparatively thin**
   relative to its practical importance (Ch3 §3.3.2) — general guidance here is largely
   empirical folklore from library documentation rather than peer-reviewed findings. Rather
   than cite a number as though it were settled science, this decision is explicitly
   deferred to the team's own ablation in Chapter 12.

## Consequences

**Positive**
+ Simple, fast, and requires no additional model calls or layout parsing — appropriate for
  the schedule.
+ The overlap guarantees no sentence is silently lost at a chunk boundary.
+ Because the parameter is explicitly *not* asserted from literature, the ablation in
  Chapter 12 becomes a genuine, defensible experiment rather than a formality — this is one
  of the few places in the project where the team's own measurement, not a citation, is the
  evidence.

**Negative**
− A chunk may still span two unrelated topics if they happen to fall within the same
  300-word window ("semantic smearing," Ch1 §1.5.4), producing an averaged embedding
  vector that matches neither topic well.
− The "right" chunk size is corpus-dependent and this project's finding will not
  necessarily generalise to a different corpus — a threat-to-validity point that should be
  stated explicitly (Ch3 §3.7).

**Implications for other components**
- Fixes the granularity of what a citation resolves to — a citation points to a ~300-word
  passage, not a single sentence and not a whole document.
- The Chapter 12 ablation (chunk size 150/300/600) is the mechanism that converts this from
  a placeholder default into a validated decision — it should not be skipped as "already
  decided."

## Revisit if

The Chapter 12 ablation shows a strong, consistent preference for a different chunk size
across the gold-standard question set, or if the project scope is extended to include
structurally complex documents (tables, multi-column layouts) where fixed-size splitting
performs poorly enough to justify the added complexity of semantic chunking.

---

*This ADR intentionally does not cite chunking-specific literature as decisive evidence —
see Chapter 3 §3.3.2 for why, and Chapter 3A for the sources that do exist (RAPTOR, late
chunking, LumberChunker), which describe hierarchical and embedding-aware alternatives
rather than settling the fixed-size question this ADR addresses.*

## Measurement update (2026-10-05): the 150 / 300 / 600 ablation

`scripts/run_ablations.py`, throwaway index per variant from the real documents, the 8 audio chunks copied from the real index, overlap held at one sixth of the size so size is the only variable, 25 gold text questions plus 6 out-of-corpus questions through the real model (`llama3.2:3b`, one run each). Transcript: `data/eval/ablations_2026-10-05.txt`.

| Size / overlap | Text chunks | Recall@5 | MRR | Answered (of 25) | Expected chunk reached the prompt | Largest prompt |
|---|---|---|---|---|---|---|
| 150 / 25 | 1199 | 0.92 | 0.69 | 18 | 23 / 25 | ~1,180 tokens |
| **300 / 50 (shipped)** | **631** | **1.00** | **0.81** | **22** | **25 / 25** | ~2,170 tokens |
| 600 / 100 | 335 | 0.96 | 0.83 | 23 (22 with the window raised to 8192) | 24 / 25 | ~4,150 tokens: **2 of 25 prompts overflow the 4096-token window** |

**The default stands.** 150 words is worse on every column (smaller chunks lose the surrounding sentences that make a fact answerable). 600 words has the highest MRR, but by 0.02, which is one question, and it answered no more questions at the correct window; its prompts also reach the model's context limit, where Ollama truncates silently. 300 has the best Recall@5 and the expected chunk reached the prompt every time. Caveats: one run of a stochastic model per variant (a one-question difference is noise), "answered" means it did not refuse, not that it was right, and n = 25. The choice of 300 was a starting point (this ADR's own words); it is now a measured one.

**A hazard this exposed:** a bigger chunk size (or TOP_K) can push a prompt past `LLM_NUM_CTX` with no error. `src/pipelines/rag/answer.py` now logs a warning when a prompt comes within 90% of the window.

