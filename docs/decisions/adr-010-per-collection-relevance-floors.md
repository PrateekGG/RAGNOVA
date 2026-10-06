# ADR-010: One relevance floor per collection, not one shared number

## Status
Accepted — Day 12. Extends ADR-009; the image floor's value is provisional (see Revisit if).
**Update 2026-10-05:** the measurement below showed no CLIP floor separates correct from irrelevant images. [ADR-011](adr-011-image-corroboration-gate.md) adds a corroboration gate on top of this floor; the per-collection floors themselves stand.

## Context

ADR-009 gates every retrieved chunk on `MIN_RELEVANCE_SCORE = 0.3` before it reaches the prompt. That number was chosen by inspecting real **MiniLM text-to-text** scores (Chapter 10 §3.2: gold hits 0.42–0.48, negative controls 0.25 and 0.04).

Chapter 12 adds `image_index` to retrieval. Its scores come from **CLIP text-to-image** similarity, and ADR-003/ADR-007 already record the modality gap: correct text-image matches score systematically lower than correct text-text matches. Applying 0.3 to image scores would reproduce, one step earlier, the exact failure ADR-007 exists to prevent: images silently never reach the answer, while every component reports success.

## Decision

Each collection gets its own floor, applied to its own results before the rank merge (ADR-007):

- `text_index`: `settings.MIN_RELEVANCE_SCORE` (0.3, unchanged).
- `image_index`: `settings.MIN_IMAGE_RELEVANCE_SCORE` (0.2, **provisional**).

Implemented as `filter_by_floor(chunks, floor)` in `src/pipelines/rag/retrieve.py`, a pure function; `answer.py`'s `_filter_relevant()` now delegates to it with the text floor.

## Alternatives considered

1. **One shared floor (0.3).** Rejected: hides images, per Context.
2. **One shared floor, lowered to suit images (0.2).** Rejected: lets text chunks at 0.2–0.3 into the prompt, which Chapter 10 §3.2 measured as the negative-control zone (N1 scored 0.251). Fixes images by re-breaking ADR-009.
3. **Normalise scores per collection (z-scores), then one floor.** Deferred with ADR-007's option 2: needs enough same-query samples per collection to estimate a distribution, which a demo corpus with 12 images doesn't give.
4. **No floor for images.** Rejected: `image_index.query()` always returns `top_k` images too, so every question would carry five images of "closest of a bad lot" into the prompt.

## Consequences

**Positive**
+ Images can reach the answer at all.
+ Text behaviour, and every Chapter 10 test, is unchanged.
+ Each floor can be tuned from its own collection's measurements.

**Negative**
− Two numbers to justify instead of one.
− 0.2 is not yet measured on this corpus. Until it is, an image-heavy answer's precision is unknown.

## Revisit if

Once real CLIP scores for the I1–I3 gold questions and the N1–N2 negative controls exist (a table like Chapter 10 §3.2's, against `image_index`), set `MIN_IMAGE_RELEVANCE_SCORE` to a value between the two clusters and record the table here. If the clusters overlap, that is evidence for option 3.

## Measurement update (2026-10-05)

The "Revisit if" table, measured on the real corpus (15 images, real CLIP `ViT-B-32`). Each score is the top-1 `search_images(query_text=question)` result; the negatives are `data/README.md`'s N1/N2 plus the two controls Chapter 10 used.

| Question | Role | Image score |
|---|---|---|
| I1 Wi-Fi screen | positive | 0.341 |
| I2 architecture diagram | positive | 0.236 |
| I3 seminar poster | positive | 0.251 (expected image ranks 3rd; the top image scores 0.306) |
| I4 lab door sign | positive | 0.300 |
| N1 hostel mess menu | negative | 0.214 |
| N2 tuition refund | negative | 0.228 |
| Ch10 N1 tuition fee | negative | 0.237 |
| Ch10 N2 capital of France | negative | 0.134 |

**The clusters overlap.** I2's correct image (0.236) scores below the Ch10 tuition-fee control's top image (0.237), so no single floor separates in-corpus from out-of-corpus questions. This is the case the "Revisit if" section named as evidence for option 3.

**It has a visible consequence today.** The text side still refuses: N1 and N2 top out at 0.224 and 0.239, under the 0.3 floor. With `include_images=True`, though, `retrieve()` returns 2 irrelevant images for N1 and 3 for N2 (e.g. `notice_library_fines.png`, 0.214), because they clear 0.2, so `answer_query(include_images=True)` would send them to the LLM instead of refusing. With images off, all of these are refused.

The sample is tiny (4 positives, 4 negatives), so this is a direction rather than a verdict. `MIN_IMAGE_RELEVANCE_SCORE` is left at 0.2; what to do instead (per-collection score normalisation, or a margin-based gate) is still open.
