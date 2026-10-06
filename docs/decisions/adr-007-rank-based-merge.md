# ADR-007: Merge cross-modal results by rank rather than by raw similarity score

## Status
Accepted — Day 4

## Context

A single query must return results drawn from two independent ChromaDB collections
(ADR-003/ADR-004): `text_index` (384-d, cosine similarities typically 0.2–0.9) and
`image_index` (512-d CLIP embeddings). Both Track A's retrieval code and the query
interface depend on this decision, since it determines what a single "top-5 results" list
actually contains — it must be fixed before Chapter 10 is written.

The deciding constraint is empirical, not a design preference: contrastively-trained
multimodal models exhibit a documented **modality gap** — image and text embeddings occupy
distinct, non-overlapping regions of the shared space, and text-image similarity scores for
genuinely correct matches are systematically lower than text-text similarity scores for
correct matches [ModGap]. This is the same finding already cited in ADR-003 to justify
maintaining two separate collections; this ADR addresses the resulting merge problem.

## Decision

Merge the two collections' result lists by **rank**, not by raw similarity score — either
by simple rank-interleaving (alternate top results from each list) or by Reciprocal Rank
Fusion, `score = Σ 1/(k + rank_i)` with k ≈ 60 [RRF], whichever the Chapter 10 implementation
finds simpler to integrate correctly.

## Alternatives considered

1. **Concatenate both result lists and sort by raw cosine similarity.** Rejected as the
   default failure mode this ADR exists to prevent: because text-image scores are
   systematically lower than text-text scores even for correct matches [ModGap], sorting by
   raw score would rank every image below every text result regardless of actual relevance.
   Objective O3 (cross-modal retrieval) would silently fail — images would simply never
   appear in results — while every individual component (the embedder, the database, the
   retrieval code) reports success. This is precisely the "worst class of bug" flagged in
   Chapter 3A's modality-gap deep dive: nothing errors, and the failure is invisible without
   deliberately testing for it.
2. **Per-modality z-score normalisation before merging** (rescale each collection's scores
   to the same mean and standard deviation, then sort the combined, normalised list).
   Considered a reasonable alternative to rank-based merging, and noted for the Chapter 12
   ablation as a second candidate — but requires enough same-query samples per modality to
   estimate a stable distribution, which is a nontrivial requirement for a demo-scale corpus
   with potentially few images per query topic. Rank-based merging makes no distributional
   assumption and works with the very small top-K result sets this project actually returns.
3. **Route the query to only one collection based on a heuristic** (e.g., detect whether the
   query "sounds like" an image request and only search `image_index`). Rejected: brittle,
   requires an additional classification step, and defeats the actual requirement — Chapter
   1's problem statement asks for text queries to surface *both* relevant documents *and*
   relevant images simultaneously (the "email screenshot" example), not to choose one
   modality per query.
4. **Train a learned re-ranker** to combine both signals. Rejected as out of scope: requires
   labelled training data this project does not have, and is exactly the kind of "Advanced
   RAG" technique (Ch3A Theme 7, Gao et al.'s taxonomy) explicitly deferred to future work.

## Consequences

**Positive**
+ Directly addresses a documented, citable failure mode rather than an assumed one —
  converts what could look like an arbitrary implementation choice into a design decision
  grounded in evidence, matching this project's own methodological standard (Ch3 §3.3).
+ Makes no distributional assumption about either collection's score range, so it is robust
  even with very few results in one modality.
+ Directly ablatable (Ch3 §3.6.7): rank-based vs. raw-score merging can be tested against
  the cross-modal gold-set queries (`data/README.md`) to produce the project's own empirical
  confirmation that the modality gap is real *in this corpus*, not just in the cited paper —
  identified in Chapter 3 as the single most valuable experiment available to this project.

**Negative**
− Rank-based fusion discards the *magnitude* of similarity difference — a text result that
  is an overwhelmingly better match than the second-best result is treated identically to
  one that barely edges it out, which score-based merging would have preserved.
− The interface must never display a raw text similarity score and a raw image similarity
  score side by side as though they were on one comparable scale (a direct consequence
  already flagged in ADR-003) — this constrains what Chapter 11's citation UI is allowed to
  show.

**Implications for other components**
- Chapter 10's retrieval function returns a single merged, ranked list — Track C's UI code
  (Ch4 §4.1) consumes one ordered result list, not two separate ones to reconcile itself.
- The Chapter 12 ablation (rank-based vs. score-based merge) is not optional polish — it is
  the mechanism that validates this ADR's premise against RAGNova's actual corpus.

## Revisit if

The Chapter 12 ablation shows rank-based merging performing worse than per-modality score
normalisation on the cross-modal gold-set queries, in which case option 2 above should be
implemented instead.

---

*Citations: [ModGap] Liang, Zhang, Kwon, Yeung and Zou, "Mind the Gap: Understanding the
Modality Gap in Multi-modal Contrastive Representation Learning," NeurIPS 2022 — the same
source cited in ADR-003. [RRF] Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009. **Verify both
before use.***

## Measurement update (2026-10-05): rank-based against score-based merging, measured

The 16 cross-modal gold rows on the real index, merged by reciprocal rank fusion (k = 60, shipped) and by raw score (the rejected alternative), `scripts/run_ablations.py`:

| Merge policy | All 16 | text-to-image | image-to-document | audio-topic |
|---|---|---|---|---|
| **Rank-based RRF (shipped)** | **14 / 16** | **6 / 8** | 4 / 4 | 4 / 4 |
| Raw-score merge | 11 / 16 | 3 / 8 | 4 / 4 | 4 / 4 |

Score merging loses three correct images outright (I4, I5, I7 fall out of the top 5) and drops two others a rank or two (I1 2 -> 4, I6 2 -> 3). The mechanism is the modality gap this ADR was written about: a text chunk's cosine (about 0.3 to 0.65) outscores a correct image's CLIP score (about 0.2 to 0.35), so every text chunk sorts above every image. The decision is no longer an argument from the literature; on this corpus it is worth 3 of 8 text-to-image questions. I3 misses under both policies: its image is CLIP's third choice, which is a ranking limit of the image model, not of the merge. n = 16; one row is 0.0625.

### Same measurement on all 22 cross-modal rows (later the same day)

Six text-to-image rows (I9-I14) were added after the table above (see ADR-011's second update); the merge ablation was re-run on all 22 (`data/eval/ablations_merge_22rows_2026-10-05.txt`):

| Merge policy | All 22 | text-to-image | image-to-document | audio-topic |
|---|---|---|---|---|
| **Rank-based RRF (shipped)** | **18 / 22** | **10 / 14** | 4 / 4 | 4 / 4 |
| Raw-score merge | 14 / 22 | 6 / 14 | 4 / 4 | 4 / 4 |

The case for rank merging got stronger with the larger sample, not weaker: it is now worth 4 of 14 text-to-image questions (raw score loses I4, I5, I7 and I12 outright and pushes I1, I6 and I9 down a rank or two). **The four rows RRF misses (I3, I8, I10, I14) are missed by the score merge too**, so none of them is a merge-policy problem: I3 is CLIP ranking its image third, I8 and I14 are text-free photos the ADR-011 gate drops, and I10 is an image CLIP never ranks in its top 5. n = 22; one row is 0.045.
