# Chapter 12 — Integration, Testing & Human Feedback (Days 12–13)

> **Deliverables today:** the missing write paths from Chapters 8–9 into ChromaDB (`src/pipelines/images/index.py`, `src/pipelines/audio/index.py`); image search (`src/pipelines/images/search.py`); one retrieval call that searches both collections and merges them by rank (`src/pipelines/rag/retrieve.py`, ADR-007); a separate relevance floor for images ([ADR-010](../decisions/adr-010-per-collection-relevance-floors.md)); `scripts/build_index.py` indexing all three folders; `tests/test_integration.py`; and the human-feedback loop (`src/ui/feedback.py`, `scripts/summarize_feedback.py`, [`docs/feedback-log.md`](../feedback-log.md)).
>
> **Prerequisites:** [Chapter 10](ch10-rag-core-retrieval-and-generation.md) (`answer_query()`, ADR-009), [Chapter 11](ch11-unified-query-interface.md), [ADR-003](../decisions/adr-003-two-vector-collections.md) and [ADR-007](../decisions/adr-007-rank-based-merge.md).
>
> **The framing for today.** Chapter 4 put integration on Day 12 on purpose: it's the day the three tracks' outputs meet for real, and the day "it works on my laptop" stops being enough. Two real integration bugs were found today, both at a seam no single track's tests could see (§5.3). Neither crashed. Both would have shipped silently.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: What "integration" means here** | The three seams, and why each track's own tests can't see them. | ~15 min |
| **Part 2 — LEARN: Merging two collections by rank** | Reciprocal Rank Fusion, worked by hand. | ~20 min |
| **Part 3 — LEARN: One relevance floor per collection** | Why reusing 0.3 for images would hide every image. | ~15 min |
| **Part 4 — LEARN: Testing seams without the models** | What the fakes fake, and what they deliberately don't. | ~15 min |
| **Part 5 — BUILD** | Files, the two bugs found, test output. | ~1.5 hours |
| **Part 6 — LEARN + BUILD: The human-feedback loop** | Rating form → log → summary → what we changed. | ~30 min |
| **Part 7 — CHECK** | Rubric, questions, what's still open. | ~20 min |

---

# Part 1 — LEARN: What "integration" means here

Three tracks built three things against one frozen contract (`src/core/schemas.py`). Each track tested its own output. What nobody could test alone is the **seams**:

| Seam | Producer | Consumer | Question nobody asked until today |
|---|---|---|---|
| Image → ChromaDB | Track B's `ImageIngestionPipeline` | `vector_store.add_chunks()` | Does anything actually *write* images into `image_index`? (No: Chapter 8 stored them in an in-memory `ChunkStore` that vanishes on exit.) |
| Audio → ChromaDB | Track B's `AudioIngestor` | `add_chunks()` into `text_index` | Same question, same answer, for audio. |
| Two collections → one answer | `text_index` + `image_index` | `answer_query()` | How does one question get one ranked list out of two collections whose scores aren't comparable? |

Chapter 10 said this plainly: "image_index/audio have no write path into ChromaDB yet". Today builds those paths.

---

# Part 2 — LEARN: Merging two collections by rank

## 2.1 The trap (ADR-007, restated simply)

Text-to-text scores for a good match: around 0.4–0.5 in this corpus (Ch10 §3.2). CLIP text-to-image scores for a good match: much lower, around 0.2–0.3, because of the **modality gap**. Sort everything by score, and every image loses to every text chunk, even a mediocre one. Nothing errors. Images just never show up.

## 2.2 Reciprocal Rank Fusion, by hand

Throw the scores away. Keep only each item's position in its own list. Each item earns `1 / (k + rank)` from every list it's in, with `k = 60` (`settings.RRF_K`, the value from Cormack et al.'s paper).

| Item | Text rank | Image rank | RRF score |
|---|---|---|---|
| notice.pdf p2 | 1 | – | 1/61 = 0.01639 |
| midterm_schedule.png | – | 1 | 1/61 = 0.01639 |
| notice.pdf p1 | 2 | – | 1/62 = 0.01613 |
| synopsis_portal.png | – | 2 | 1/62 = 0.01613 |

Sorted: text #1, image #1, text #2, image #2. With two lists and no overlap, RRF is plain interleaving, which ADR-007 names as acceptable. An item in *both* lists gets both shares and jumps ahead (tested in `test_rrf_merge_rewards_a_chunk_found_in_both_lists`).

## 2.3 What `rrf_merge()` keeps and doesn't

It returns chunks with their **original** `.score`, not the fused number. The fused number only decides order. Displaying it would invite the same cross-scale comparison ADR-003 forbids, so it never leaves the function.

---

# Part 3 — LEARN: One relevance floor per collection

ADR-009 gates every chunk on `MIN_RELEVANCE_SCORE = 0.3` before it reaches the prompt. That number was measured on **text** scores. Apply it to CLIP scores and a correct image at 0.25 gets dropped. Same bug as §2.1, one step earlier.

So images get their own floor, `MIN_IMAGE_RELEVANCE_SCORE = 0.2` ([ADR-010](../decisions/adr-010-per-collection-relevance-floors.md)). Said honestly: 0.2 is **not measured** on this project's corpus yet. It's a starting point from CLIP's commonly reported range, to be replaced by a table like Ch10 §3.2's once the I1–I3 gold questions run against real CLIP (§7.3).

`test_retrieve_uses_a_separate_floor_for_images` pins the behaviour: a text chunk and an image both scoring 0.25 → the text one is dropped, the image one kept.

---

# Part 4 — LEARN: Testing seams without the models

`tests/test_integration.py` replaces every model with a small fake:

| Real thing | Fake | Why this fake is honest |
|---|---|---|
| MiniLM (384-d) | Bag-of-words hashed into 384 dims, unit length | Shares words → high score, no shared words → 0. Enough to make "which chunk comes first" deterministic. |
| OpenCLIP (512-d) | Red image → axis 0, blue → axis 1; text "red" → axis 0 | Makes "text finds the right image" and "image finds the right image" checkable exactly. |
| Whisper / `AudioIngestor` | Returns two canned transcript chunks | The seam under test is "chunks → ChromaDB", not speech recognition. |
| Ollama | Records the prompt, returns fixed text | The seam under test is "what context reaches the model". |

ChromaDB is **real** in every test (in a `tmp_path`, never the real `chroma_db/`), because the database *is* the seam.

What the fakes can't tell you: whether MiniLM ranks paraphrases well, whether CLIP matches a screenshot to its topic, whether Whisper hears the clip right, whether the LLM answers correctly. Those need real models, which is what §6's human testing and `scripts/evaluate_answers.py` are for.

---

# Part 5 — BUILD

> **Verification methodology.** Same as Chapter 11: a cloud container with no route to Hugging Face or Ollama, so real ChromaDB + real parsers + the fakes above. Existing Chapter 5–10 suites were re-run too: `test_contract.py`, `test_document_ingestion.py` and the pure parts of `test_rag_core.py` pass (45 passed, 1 skipped); the one `test_rag_core.py` test that needs MiniLM's weights fails here for exactly that reason and was not a regression.

## 5.1 Image write path: `index_image_files()` / `index_images_directory()`

Runs Chapter 8's `ImageIngestionPipeline.ingest_batch(..., store=False)`, then hands the chunks and their CLIP vectors to the same `add_chunks()` documents use, into `image_index`.

## 5.2 Audio write path: `index_audio_file()` / `index_audio_directory()`

Runs Chapter 9's `AudioIngestor.process_file()`, validates every chunk against the contract, normalizes the transcript text, embeds with MiniLM, upserts into `text_index`. Whisper is created and closed per file (Chapter 1 §1.9.2's lazy loading).

## 5.3 The two real bugs found at the seams

**Bug 1 — image citations carried one laptop's folder path.** `ImageIngestionPipeline` stores `Path(...).resolve()`, e.g. `/home/alice/RAGNova/data/images/wifi.png`. Fine inside Track B. At the seam it becomes a citation, and the moment Bob indexes the same file the "same" image has a different source. This is the exact bug Chapter 7 already fixed for documents (`_relative_to_cwd()`), reappearing in a different track. Fixed in `_to_index_chunk()`; `test_images_are_indexed_with_portable_sources_and_found_by_text` asserts `data/images/red.png`, not `/tmp/...`.

**Bug 2 — every audio chunk broke the contract.** `AudioIngestor._build_chunk()` looks up `settings.DEFAULT_TEXT_EMBEDDING_MODEL`, then `settings.EMBEDDING_MODEL`. Neither exists (the real name is `TEXT_EMBEDDING_MODEL`), so every audio chunk got `embedding_model=None`. Nothing crashed: `validate_chunk()` was simply never called on audio output. Caveman version: audio pipe say "chunk ready", chunk missing its name tag, nobody check name tag, database would take it anyway. Found by calling `validate_chunk()` at the seam. The fix to `AudioIngestor` itself is in [PR #5](https://github.com/Elitex07/RAGNova/pull/5) (Track B's file, Track B's PR); what this chapter adds is the guard: `index_audio_file()` refuses any chunk that fails the contract, with the reason, instead of indexing it (`test_audio_chunks_that_break_the_contract_are_refused`).

## 5.4 One retrieval call: `retrieve()`

```python
retrieve(query, include_images=False)   # exactly Chapter 10's behaviour
retrieve(query, include_images=True)    # + image_index by CLIP text, merged by rank
retrieve(query, include_images=True, query_image=img)   # image_index by CLIP image
```

`answer_query()` and `stream_answer()` both call it, with `include_images` defaulting to `False`, so `scripts/ask.py`, `evaluate_answers.py` and every Chapter 10 test behave exactly as before. Images in the prompt are labelled `(image)` and show their OCR text; an image with none says so, rather than leaving the model to invent a description.

## 5.5 `scripts/build_index.py` indexes all three folders

Documents and audio into `text_index`, images into `image_index`. CLIP and Whisper are only loaded if their folder has files, so a documents-only checkout stays fast.

## 5.6 Run the tests

```bash
pytest tests/test_integration.py -v
```
```
tests/test_integration.py::test_rrf_merge_interleaves_two_lists_by_rank_not_score PASSED
tests/test_integration.py::test_rrf_merge_rewards_a_chunk_found_in_both_lists PASSED
tests/test_integration.py::test_retrieve_uses_a_separate_floor_for_images PASSED
tests/test_integration.py::test_answer_query_with_images_sends_merged_context_to_the_llm PASSED
tests/test_integration.py::test_images_are_indexed_with_portable_sources_and_found_by_text PASSED
tests/test_integration.py::test_image_search_by_image PASSED
tests/test_integration.py::test_audio_is_indexed_into_text_index_and_cited_by_timestamp PASSED
tests/test_integration.py::test_audio_chunks_that_break_the_contract_are_refused PASSED
...
============================== 28 passed in 5.61s ==============================
```

---

# Part 6 — The human-feedback loop

## 6.1 What ROADMAP.md asked for

"3–5 outside users try the app with a feedback form", and a feedback log with "every piece of feedback recorded + what we changed". Chapter 3 §3.5.3 adds one rule: report outside testers separately from the team, because rating your own system is not evidence.

## 6.2 The pieces

1. **Form** (in the page, once PR #4's scaffold is wired): 1–5 stars, optional comment, optional tester name. Same 1–5 scale as Chapter 10's answer check, so the numbers are comparable.
2. **Log**: `record_feedback()` appends one JSON line per rating to `feedback/feedback.jsonl` (`settings.FEEDBACK_LOG_PATH`). JSON Lines because appending never rewrites earlier rows, and `git diff` shows exactly which ratings were added. A corrupt line is skipped, not fatal.
3. **Summary**: `python scripts/summarize_feedback.py` prints count, average, histogram, the same split per tester, and every answer rated 2 or lower in full, since those point at something to fix.
4. **Record**: [`docs/feedback-log.md`](../feedback-log.md), one row per finding: what the tester saw, what we changed, and the commit.

## 6.3 Running a session (per tester, ~15 minutes)

1. Build the index, start `ollama serve`, open the app.
2. Tester types their name. Give them the task card from `docs/feedback-log.md` (five questions of their own about the corpus, one voice question, one image question).
3. They rate every answer. Don't explain the system first; watch where they get stuck and note it.
4. Afterwards: `python scripts/summarize_feedback.py`, then add rows to `docs/feedback-log.md`.

---

# Part 7 — CHECK

## 7.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | `pytest tests/test_integration.py -v` passes | /3 |
| 2 | `python scripts/build_index.py` indexes documents, audio and images on a real machine | /3 |
| 3 | Team can compute an RRF merge of two short lists by hand | /3 |
| 4 | Team can explain why images need their own relevance floor | /3 |
| 5 | Team can describe both seam bugs, and why neither track's own tests caught them | /3 |
| 6 | At least 3 outside testers rated answers; summary pasted into `docs/feedback-log.md` | /3 |
| 7 | At least one change made because of feedback, recorded with its commit | /3 |
| | **Total** | **/21** |

## 7.2 Question bank

1. Why would sorting text and image results by score hide every image? → §2.1; modality gap.
2. RRF with k=60: an item ranked 1st in one list and 3rd in another. Its score? → 1/61 + 1/63 ≈ 0.0323.
3. Why does `rrf_merge()` return the original score, not the fused one? → §2.3.
4. A text chunk and an image both score 0.25. Which reaches the prompt, and why? → §3; the image only (0.25 ≥ 0.2, but < 0.3).
5. What exactly is faked in the integration tests, and what is real? → §4.
6. Why did the audio `embedding_model=None` bug never crash anything? → §5.3; nothing called `validate_chunk()` on audio output.

## 7.3 Still open after today (named, not hidden)

- **Measure `MIN_IMAGE_RELEVANCE_SCORE`** with real CLIP against the I1–I3 gold questions (arrive with PR #4's corpus), the same way Ch10 §3.2 measured 0.3.
- **Cross-modal Recall@5** in `data/README.md`'s results log: needs real CLIP + the corpus.
- **The ablations** Chapter 3 committed to (chunk size 150/300/600, rank vs. score merge): the code now supports both collections, so the merge ablation is runnable once real image scores exist.
- **The offline demonstration** from Chapter 1 §1.9.3: run the full app with the network disabled and record it.
- **Wire PR #4's scaffold** to `stream_answer()` and the feedback form (Chapter 11 §5.6).

---

**Next:** Chapter 13 — Mid-Term Report.
