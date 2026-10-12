# ADR-011: Keep an image only if CLIP and its OCR text agree (or CLIP alone is confident)

## Status
Accepted — 2026-10-05. Extends ADR-010 (per-collection floors) and answers its "Revisit if" section. The two new numbers are provisional and rest on a small sample (see Consequences).

## Context

ADR-010 gave `image_index` its own relevance floor (`MIN_IMAGE_RELEVANCE_SCORE = 0.2`) and asked that it be tuned once real CLIP scores existed for the I-row and negative-control questions, adding: *"If the clusters overlap, that is evidence for option 3 [score normalisation]."* On 2026-10-05, on the real corpus (now 25 images), the clusters overlap: the 8 correct images score 0.216–0.346 under CLIP, while the top image for the 6 questions the corpus cannot answer scores 0.134–0.291. No single floor works:

| Rule (applied to the top-5 images per question) | Correct image kept | Irrelevant images kept, per question | Negative controls refused |
|---|---|---|---|
| CLIP ≥ 0.20 (ADR-010, before) | 8 / 8 | 3.50 | **1 / 6** |
| CLIP ≥ 0.25 | 6 / 8 | 1.62 | 4 / 6 |
| CLIP ≥ 0.30 | 3 / 8 | 0.38 | 6 / 6 |

Consequence in the product (measured on the earlier 15-image corpus, ADR-010): with `include_images=True`, the hostel-menu and tuition-refund questions (README N1/N2) pulled 2–3 irrelevant images into the prompt instead of being refused, because those images cleared 0.2. On the 25-image corpus the nearest images are different (a photo of programmers' notes at CLIP 0.272 for the hostel question, a code screenshot at 0.291 for the Python-sort question) but equally irrelevant and equally above 0.2. "Irrelevant images kept" is an upper bound (any image other than the expected one counts); the refusal column is exact.

`image_index` stores each image's OCR text, and OCR text is something MiniLM can score against the question: a second, independent signal that costs one embedding. For the correct images the question-to-OCR similarity ("agreement") was 0.295–0.580 (I8, a photo with no text, has none); for the top image of each negative control it was 0.070–0.292.

## Decision

`retrieve()` passes text → image results through `filter_images()` (`src/pipelines/rag/retrieve.py`). An image is kept only if it clears `MIN_IMAGE_RELEVANCE_SCORE` (0.2) **and** at least one of:

- CLIP alone reaches `IMAGE_CONFIDENT_SCORE` (0.30), or
- the MiniLM cosine between the question and the image's OCR text reaches `MIN_IMAGE_TEXT_AGREEMENT` (0.30).

Image → image results (an uploaded picture) skip the agreement test: there is no question text to corroborate with, and their scores are far higher. A blank question gives a weak image nothing to be corroborated by, so only CLIP-confident images survive it. Both new numbers live in `src/core/config.py` and can be set by environment variable.

Measured on the same data (`python scripts/measure_relevance.py`):

| Rule | Correct image kept | Irrelevant kept per question | Negatives refused |
|---|---|---|---|
| **Chosen:** CLIP ≥ 0.2 and (agree ≥ 0.30 or CLIP ≥ 0.30) | 7 / 8 | 1.38 | **6 / 6** |
| agree ≥ 0.25 or CLIP ≥ 0.30 | 7 / 8 | 1.88 | 4 / 6 |
| agree ≥ 0.30 only (no CLIP shortcut) | 6 / 8 | 1.38 | 6 / 6 |
| agree ≥ 0.35 or CLIP ≥ 0.30 | 5 / 8 | 0.88 | 6 / 6 |

End to end through the real model (`llama3.2:3b`, 25 positives + 6 negatives, `include_images=True`): all 6 negatives are refused, and N1/N2 now reach the model with zero chunks, so no LLM call is made. Cross-modal Recall@5 (`scripts/evaluate_cross_modal.py`) moved from 15/16 to 14/16 (I8 lost, see below; I2 improved from rank 3 to rank 1 because irrelevant images no longer sit above it).

## Alternatives considered

1. **Re-tune the single CLIP floor.** Rejected by the first table: raising it to separate the negatives (≥ 0.30) loses 5 of 8 correct images.
2. **Per-collection score normalisation (z-scores), then one floor** (ADR-010's option 3). Not needed, and still not feasible: it needs a score distribution per collection, which 25 images do not give.
3. **A margin gate** ("keep results within X of the top"). Rejected for the text side by the same measurement (a legitimate duplicate often sits above the expected chunk, so margins relative to the top discard correct answers), and it cannot refuse a question whose *every* image is wrong.
4. **A cross-encoder reranker.** The standard fix, and probably the right one if retrieval precision becomes the limiting factor. Not adopted now: it adds a model download and a new dependency, and the measured problem is solved without it. Recorded as the next step if this gate proves too brittle.

## Consequences

**Positive**
+ Out-of-corpus questions are refused with images enabled (6/6), deterministically and without a model call.
+ Uses signals already stored (OCR text) and one cheap embedding, only for images CLIP did not already convince.
+ Gate logic is a pure function, tested at its boundaries (`tests/test_integration.py`).

**Negative / limits, stated plainly**
− **Text-free photos are lost unless CLIP alone reaches 0.30.** I8 ("a photo of a free Wi-Fi hotspot sign", CLIP 0.216, no OCR text) now misses. A question that can only be answered by what a picture *looks like* is exactly where this rule is weakest.
− **The margins are thin.** I2's correct image passes at agreement 0.301 against the 0.30 threshold; the top image of the tuition-refund question reaches 0.292. The sample is 8 positives and 6 negatives; both numbers should be re-measured when the gold set grows.
− Images that legitimately pass can still hurt generation: with `include_images=True` the 3B model answered 20 of 25 text questions, against 23 of 25 without images (a single run each, so partly noise). Whoever wires the UI should treat "include images" as an opt-in toggle, not a silent default.
− I3 (the seminar-poster query) still misses: CLIP ranks that poster third, and the rank-based merge (ADR-007) interleaves it below text chunks that clear the 0.3 text floor. That is a ranking limit, not a gating one.

## Revisit if

The gold set or corpus grows enough to re-measure the table above with more than a handful of positives and negatives; or a reranker is adopted, which would make this gate redundant.

## Measurement update (2026-10-05, later): the sample doubled, the thresholds hold, and the misses are accounted for

This ADR's weakest point was its sample: 8 positives and 6 negatives, with the correct image for I2 passing at 0.301 against a 0.30 threshold. Six more text-to-image rows (I9-I14) and four more out-of-corpus questions (N5-N8) were written and committed **before** being measured (commit `102f350`), then run (`scripts/measure_relevance.py`, `data/eval/relevance_measurement_2026-10-05.txt`). Now 14 positives and 10 negatives (the 8 in the gold set, N1-N8, plus the two older controls `LEGACY_NEGATIVES` that the measurement scripts still include).

| Rule | Correct image kept | Out-of-corpus refused |
|---|---|---|
| CLIP >= 0.20 alone (ADR-010) | 13 / 14 | 3 / 10 |
| **Shipped: CLIP >= 0.2 and (agree >= 0.30 or CLIP >= 0.30)** | **11 / 14** | **9 / 10** |
| agree >= 0.35 or CLIP >= 0.30 | 9 / 14 | 10 / 10 |
| CLIP >= 0.30 alone | 5 / 14 | 10 / 10 |

**The thresholds are unchanged, and the larger sample did not contradict them**: the shipped rule is still the best trade-off on offer (the alternative that refuses 10 / 10 costs 2 more correct images), and the new positives with text agree comfortably (I9 0.437, I11 0.479, I12 0.563). What the new rows add:

- **The three drops are accounted for.** I8 and I14 are text-free photos (CLIP 0.216 and 0.267, no OCR to corroborate): the cost this ADR named, now 2 of 14. **I10 is not the gate's doing at all**: CLIP does not rank the 403 screenshot in its top 5, so the gate never sees it.
- **A "text-free photos pass at a lower CLIP score" exception would not work.** I14's photo has no OCR text, so the gate has only CLIP to go on, and CLIP scores it 0.267. The nearest image to the hostel-menu question, a photograph of programmers' notes whose OCR text does not match the question (agreement 0.094), scores **0.272**, higher. Any CLIP-only exception that admits I14 would admit that photo too.
- **One negative leaks**, "Who won the Turing Award in 2018?": the lab-door-sign photo passes at agreement **0.305**, its OCR mentioning an AI laboratory. That sits beside I2's correct image at 0.301: no value of the threshold separates those two, so it was left alone rather than tuned to one question. It did no harm: three end-to-end runs all refused correctly.
- **Honest bound:** on 24 questions the gate keeps 79% of correct images and refuses 90% of out-of-corpus questions, against 93% and 30% for the floor alone. The margins are still thin; they are now thin on a sample nearly twice the size. Cross-modal Recall@5 (`scripts/evaluate_cross_modal.py`, `data/eval/cross_modal_2026-10-05.txt`): text-to-image **10 / 14 (0.71)**, all 22 rows **18 / 22 (0.82)**.

## Measurement update (2026-10-07, third): near-miss questions, and the gate on 31 images

Six more images were added (held-out evidence for ADR-014) and the gate was run on a fresh index of 31 images with thresholds unchanged. On the older rows nothing moved: **11 / 14 correct images kept, 9 / 10 out-of-corpus questions refused**, text-to-image Recall@5 10 / 14. On 14 new positives it keeps **14 / 14**. On 8 new negatives it refuses **1 / 8**, and that is the finding: six of them are near-misses (a question about the shuttle bus asking for a fare the timetable does not state), and a relevance gate correctly sees that the shuttle timetable is *about* the question. The gate measures topic, not whether the image contains the answer, so near-miss questions pass it by design. The effective defence is the model: end to end all 8 were refused (16 / 16 across every negative in the gold set). The thresholds were not changed. See ADR-014 for the full table and the OCR-text search built beside this gate.
