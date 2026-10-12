# ADR-003: Maintain two vector collections rather than one

## Status
Accepted — Day 3

## Context

The system must answer a single query against content of four different kinds: document
text, speech transcripts, text extracted from images by OCR, and the visual content of
images themselves.

Textual content is embedded with `all-MiniLM-L6-v2`, producing **384-dimensional** vectors.
Image content is embedded with OpenCLIP ViT-B/32, producing **512-dimensional** vectors in
a different space, trained under a different objective.

Two hard constraints bound the choice:

- Vectors of different dimensionality cannot occupy one index; the similarity operation is
  undefined between them.
- CLIP's text encoder truncates input at **77 tokens** (~50 words), because it was trained
  on captions. Document segments of ~300 words cannot be represented by it.

## Decision

Maintain two separate ChromaDB collections — `text_index` (384-d) and `image_index` (512-d)
— queried independently, with results merged by rank (see ADR-007).

## Alternatives considered

1. **One collection, one model for everything.** Rejected: no available model handles both
   document-length text and image content well. Using CLIP for everything fails on the
   77-token limit, discarding most of every document segment. Using a text model for
   everything makes images unsearchable, defeating Objective O3.

2. **One collection, projecting CLIP vectors into the text model's space** (or vice versa).
   Rejected: requires training a projection layer on paired data, which needs data we do
   not have and validation we cannot perform within the schedule. It would also introduce
   an unvalidated learned component into a system whose value depends on the retrieved
   evidence being trustworthy.

3. **Two collections, merged by raw similarity score.** Rejected on the evidence of the
   documented **modality gap**: image and text embeddings occupy separate regions of CLIP
   space, so text-image similarities are systematically lower in magnitude than text-text
   similarities *even for perfect matches* [13]. Sorting a combined list by raw score
   would rank images last regardless of relevance. This is the reason ADR-007 exists.

4. **Caption every image with a vision-language model, then index captions as text.**
   Rejected: adds a generative model to the ingestion path, and a caption is a lossy
   summary — it discards visual detail a later query might target. Retained as a possible
   *supplementary* signal in future work, alongside rather than instead of CLIP embeddings.

## Consequences

**Positive**
+ Each modality is embedded by the model best suited to it.
+ Adding a modality later means adding a collection, not redesigning the index.
+ Collections can be rebuilt independently — re-indexing images does not touch documents.
+ Metadata filtering can be applied per collection.

**Negative**
− Two indexes to build, persist, and keep consistent.
− The merge policy becomes a tunable parameter requiring its own justification and ablation.
− Absolute similarity scores are **not comparable across collections**. The interface must
  therefore never display a text score and an image score side by side as though they were
  on one scale — doing so would be actively misleading to the user.

**Implications for other components**
- Ingestion must route each segment to the correct collection (Ch 6, 8).
- Retrieval must query both and merge (Ch 10, ADR-007).
- The shared chunk schema must carry a `modality` field so results can be traced back.

## Revisit if

- A single encoder emerges that handles both document-length text and images competently
  in one space at acceptable size for local execution.
- The corpus grows to a scale where maintaining two HNSW indexes becomes a memory problem
  (not expected below ~10⁶ segments).

**Note (2026-10-07):** a third collection, `image_text_index`, now exists (ADR-014): MiniLM vectors of each image's OCR text under the same ids as `image_index`. It does not change this decision: CLIP vectors and text vectors still never share a collection, and the third holds *text* vectors for *images*, keyed so the two image searches can be fused. It is written always and read only when `IMAGE_TEXT_SEARCH` is on (off by default).

---

*Citation [13] refers to Liang et al., "Mind the Gap: Understanding the Modality Gap in
Multi-modal Contrastive Representation Learning," NeurIPS 2022. **Verify before use.***
