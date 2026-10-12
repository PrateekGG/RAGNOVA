# ADR-014: Find images by the text inside them as well as by CLIP: built, measured, not switched on

## Status
Accepted as an **opt-in capability, not enabled by default**: 2026-10-07. The code ships switched off because the adoption rule, fixed before anything was measured, selected no variant. Refines ADR-003 (a third, derived collection), ADR-007 (rank fusion, here applied within images) and ADR-011 (the gate).

## Context

Images were found by CLIP alone. The text read from them (OCR) was used only to corroborate a weak CLIP match (ADR-011) and to build the prompt, never to *find* an image. So an image whose text answers the question was findable only if CLIP happened to rank it. Three observed cases: gold row I3 (the seminar poster is CLIP's third choice and falls to sixth place after the merge), gold row I10 (the 403 screenshot is not in CLIP's top five at all), and a canteen notice uploaded through the page on 2026-10-07 that ranked 6th at CLIP 0.143 for "When is breakfast served in the canteen?" while "Find the notice with canteen timings" ranked it 1st at 0.309. CLIP is better at what a picture looks like than at what a notice says.

## Decision

**Build** a second image search and keep it **off by default**:

1. `image_text_index`, a third ChromaDB collection holding a MiniLM vector of each image's OCR text under the *same chunk id* as its `image_index` entry (`src/pipelines/images/index.py::index_ocr_text`; written whenever an image is indexed, through `replace_source_chunks`, so a re-index whose text changed or vanished cannot leave old text searchable; pruned with the others by `build_index.py`). 29 of the 31 images have text; the two text-free photos correctly have none. That is 29 short texts through MiniLM, a negligible cost per build, and it means switching the channel on later needs no rebuild.
2. `search_image_text()` (the question against those vectors; its score is ADR-011's "agreement", never compared with a CLIP score) and `gate_image_channels()` (pure): the two rankings fused by rank with the existing `rrf_merge`, an image kept if it passes ADR-011's CLIP rule unchanged or, if `IMAGE_OCR_ONLY_MIN_AGREEMENT` is set, the text search returned it with at least that agreement (optionally also sharing a content word). With no text hits and no text rule it is exactly `filter_images()`; tested at both boundaries.
3. Settings `IMAGE_TEXT_SEARCH` (default off), `IMAGE_OCR_ONLY_MIN_AGREEMENT` (unset), `IMAGE_OCR_ONLY_NEEDS_SHARED_WORD` (off). Image-to-image search never uses the channel.

**To use it:** `IMAGE_TEXT_SEARCH=1` (re-ranking only) or `IMAGE_TEXT_SEARCH=1` plus `IMAGE_OCR_ONLY_MIN_AGREEMENT=0.45` (also admits strong text matches), in `.env`.

## What was measured, and the rule that was fixed first

The evidence set was written and committed before any retrieval was run on it (commit `4ddd5d9`): 14 new text-to-image rows (I15-I28; 8 on existing images, 6 on six new synthetic images) and 8 new negatives (N9-N16; six are near-misses that ask for a fact the image deliberately omits, such as a bus fare or a race result), tagged `heldout-2026-10-07`. The variants, their order of simplicity and the rule were committed in `scripts/evaluate_image_channels.py` (commit `f5285a2`) before it was run. **The rule:** only held-out rows may choose a variant; a variant qualifies only if it loses nothing held-out, gains at least 2 in kept + refused, and does no harm on the older rows (which set the shipped gate, so they may veto but never recommend). A scratch index was built fresh from the committed data (631 text chunks, 31 images, 29 text vectors; text Recall@5 1.00 and MRR 0.81, unchanged). Transcript: `data/eval/image_channels_2026-10-07.txt`.

| Variant | held-out kept /14 | held-out Recall@5 /14 | held-out refused /8 | old rows kept /14 | old rows Recall@5 /14 | old negatives refused /10 |
|---|---|---|---|---|---|---|
| **G0, shipped gate (CLIP only)** | 14 | 14 | 1 | 11 | 10 | 9 |
| F0, + text search, fused by rank | 14 | 14 | 1 | 11 | 11 | 9 |
| F1@0.55 / F1@0.50, + admit strong text matches | 14 | 14 | 1 | 11 | 11 | 9 |
| F1@0.45 | 14 | 14 | 1 | **12** | **12** | 9 |
| F2@0.55 / 0.50 / 0.45, + shared-word veto | 14 | 14 | 1 | 11 | 11 | 9 |

**No variant qualifies**: the shipped gate scores 15 held-out (kept + refused) and a variant needed 17. The rule says *not adopted*, and that is what shipped. Reading the table honestly:

- **The held-out positives could not discriminate.** CLIP alone already finds and keeps all 14 new correct images, including those on images it had never seen. There was no room to win. The text search finds 26 of the 28 text-to-image rows in its top five (it misses I8 and I14, the text-free photos) and CLIP finds 27 (it misses I10): the two searches are complementary, together covering all 28.
- **The held-out negatives are not separable by a relevance gate.** The shipped gate refuses only 1 of the 8 (N16); the other seven leak the image they are topically about (N9 the shuttle timetable, N10 the sports-day notice, and so on; N15, a far-miss about course credits, leaks two images: the ID card and the lab door sign). That is what a relevance gate does: it can say an image is *about* the question's topic, not that it *contains the answer*. The variants change none of these.
- **What the channel changes is on the older rows only:** I3 is recovered by every variant (rank 4, by fusion alone), and I10 by F1@0.45 (rank 2), with no negative changing status. The shared-word veto (F2) costs I10 again: that question is a paraphrase ("a page telling me I am not allowed to open something") of an image that says "Access Denied - 403 Forbidden", with no content word in common.

**End to end, through the real model** (`scripts/evaluate_answers.py --images --image-rows`, 69 questions, one run each of a stochastic 3B model, so about one question of noise; transcripts `data/eval/answers_2026-10-07_images-default.txt` and `..._images-channel-on.txt`):

| | shipped (channel off) | channel on, F1@0.45 |
|---|---|---|
| Negatives refused (all 16) | **16 / 16** (held-out 8 / 8) | **16 / 16** (held-out 8 / 8) |
| Expected image among the cited sources, 28 image questions | 24 / 28 (I3, I8, I10, I14 missing) | 26 / 28 (I8, I14 missing) |
| Image questions answered rather than refused | 21 / 28 | 22 / 28 |
| Text rows answered | 22 / 25 | 23 / 25 (T11 flipped: model noise) |

The seven held-out negatives that leak an image through the gate (N9-N15) are all refused by the model, so the second line of defence is doing the work the gate cannot. "Answered" understates the image rows: three of the shipped pass's refusals (I2, I6, I11) did cite the right image; they are "find me the diagram" requests, not questions, and a text-only model declines to answer them even though the citation is the answer.

## Alternatives considered

1. **Switch on F1@0.45 (or F0) now, on the strength of I3 and I10.** Rejected by the rule fixed beforehand: the two rows it recovers come from the set that tuned the gate, and the held-out set showed no gain. F0 deserves a note: it only reorders images the shipped gate already keeps, so it cannot add a leak; it is a candidate for a conscious later decision, not an automatic one.
2. **Put the OCR text into `text_index`.** Rejected: it mixes modalities in one collection (ADR-003), would apply the text floor (0.30) to image text where ADR-010 says images need their own, and would return an image as if it were a document passage.
3. **Store a second vector per image in `image_index`.** ChromaDB holds one embedding per id, and CLIP's text tower truncates at 77 tokens (ADR-003's reason for two collections).
4. **Keep CLIP only.** The status quo, and what ships. It fails I10 and, as the corpus grows, any question whose answer lies in the text of one of many look-alike notices, which CLIP's picture-level embedding does not tell apart.
5. **A shared-word veto (F2).** Tested, and it lost I10; it does not help the leaks that matter, which share words with their image by construction.
6. **A cross-encoder re-ranker over OCR text.** Not tried; a new model for a question this sample cannot settle.

## Consequences

**Positive**
+ The capability exists, is tested (25 tests, 11 mutations caught) and is one setting away, with no rebuild.
+ The measurement is reusable: evidence set, pre-registered script and rule are in the repository.
+ It found that the image gate measures topic, not answerability, and that the model's refusal is what handles near-miss questions (16 / 16 end to end).

**Negative / limits**
- The held-out set was at its ceiling for positives and its negatives are unseparable at retrieval level, so it could not select a variant. The positive evidence for the channel (I3, I10, +2 images cited end to end, no negative answered) is real but comes from rows the rule does not let decide.
- A third collection to build, prune and explain.
- The six new images are synthetic, in the same style as the other 17 text images: a result that holds on them may not hold on real photographed notices.

## Revisit if

The corpus gains many look-alike notices (two timetables, several lab rules), where CLIP cannot tell them apart and the text is the only difference. The right experiment then is a new held-out set of questions CLIP gets wrong, chosen without reference to the channel, with the rule written down first. Or if the owner decides F0 (reordering only) is worth enabling on the I3 evidence.
