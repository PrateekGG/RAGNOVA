# Chapter 11 — Unified Query Interface: the engine behind the chat screen (Days 10–11)

> **Deliverables today:** everything the Streamlit page needs that is *not* drawing: a streaming answer path (`generate_stream()` in `src/core/llm.py`, `stream_answer()` in `src/pipelines/rag/answer.py`); a per-citation display model (`src/ui/citations.py`); the upload, voice and image-query plumbing (`src/ui/backend.py`, `src/pipelines/audio/transcribe.py`, `index_document_file()`); and the feedback log the page writes ratings into (`src/ui/feedback.py`, used by Chapter 12).
>
> **Prerequisites:** [Chapter 10](ch10-rag-core-retrieval-and-generation.md) (`answer_query()`, `format_provenance()`, the citation-range check) and [ADR-008](../decisions/adr-008-docx-pagination-via-explicit-breaks.md) (why a Word page number needs a caveat).
>
> **What this chapter does not do, on purpose.** The page itself, `src/app.py`, is Member 3's Task 4 scaffold in [PR #4](https://github.com/Elitex07/RAGNova/pull/4): layout, chat box, a mocked answer, and a `# TODO: to wire the real call here` marker. That PR owns the screen. This chapter builds what the TODO will call, so that once PR #4 lands, wiring it is the "small diff later" the scaffold was designed for (§5.6 shows that diff). Audio *ingestion* fixes belong to [PR #5](https://github.com/Elitex07/RAGNova/pull/5) for the same reason.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: Streaming** | Why a 10-second answer feels broken without it, and what changes in the code. | ~15 min |
| **Part 2 — LEARN: What a citation looks like on screen** | The three rules the display must follow, each from an earlier decision. | ~20 min |
| **Part 3 — LEARN: One box, four kinds of input** | Typed text, voice, a query image, and uploads — how each becomes a question or an indexed file. | ~20 min |
| **Part 4 — DECIDE** | Why the page's logic lives outside the page, and the module layout. | ~10 min |
| **Part 5 — BUILD** | Files written, tests run, and the wiring diff for PR #4's scaffold. | ~1 hour |
| **Part 6 — CHECK** | Rubric, questions, troubleshooting. | ~20 min |

---

# Part 1 — LEARN: Streaming

## 1.1 The problem, in caveman terms

Model on laptop CPU slow. Full answer take 10 seconds. Screen show nothing for 10 seconds. User think app broken, click again. Now two questions queued, twice as slow.

Streaming fixes the *feeling*, not the speed: the total time is identical, but words appear as the model produces them, so the user sees it working from the first second.

## 1.2 What changes in the code

Ollama's `/api/generate` already supports it: `stream=True` returns a sequence of small JSON pieces instead of one big one. `generate_stream()` is `generate()` with that one flag flipped, yielding each piece's `"response"` text:

```python
for part in client.generate(model=..., prompt=prompt, stream=True, options={...}):
    yield part["response"]
```

Same `options` (temperature, `num_predict`, `num_ctx`), same "is `ollama serve` running?" error message, so nothing Chapter 10 decided is bypassed.

## 1.3 Why `stream_answer()` returns two things

```python
citations, pieces = stream_answer(question)
```

Retrieval runs first and finishes before a single token is generated, so the page can have the source list ready immediately while `pieces` is still streaming. Two consequences worth knowing:

- **ADR-009's short-circuit still holds.** If nothing clears the relevance floor, `citations` is `[]` and `pieces` yields the fixed "not enough information" sentence once. Ollama is never called.
- **The citation-range check has to wait.** Chapter 10 §2.5 checks `[n]` numbers in the *finished* answer. A stream isn't finished until the page has read it all, so the check was pulled out into `check_citations(answer_text, citations, query)` for the page to call at the end. It returns the out-of-range numbers, so the page can show a warning instead of only logging one.

---

# Part 2 — LEARN: What a citation looks like on screen

`citation_views(chunks)` turns the answer's chunk list into one `CitationView` per `[n]`: title, modality label, excerpt, original file path, whether the file exists, an optional caveat, and an audio start time. Pure data, no Streamlit. Three rules are baked in, and none of them are new: each one comes from a decision already on file.

| Rule | Where it comes from | What the code does |
|---|---|---|
| The label comes from metadata, never from the model | Ch10 §2.3 | Built from `format_provenance(chunk)`, with the path shortened to the file name. |
| A Word page number is shown with a caveat, a PDF one isn't | ADR-008 | `modality == "docx"` sets `note = DOCX_PAGE_NOTE` ("counted from manual page breaks only, so it can be off"). |
| No similarity score is ever shown | ADR-003, ADR-007 | Text and image scores are on different scales; putting 0.44 next to 0.23 invites a comparison that means nothing. The list order is the only ranking shown. |

Two more small behaviours: an image with no OCR text shows "(No readable text in this image.)" instead of an empty box, and an audio citation carries `audio_start_s` so the page can open a player at the cited second (`st.audio(path, start_time=...)`).

---

# Part 3 — LEARN: One box, four kinds of input

The problem statement's first pillar asks for one interface that takes text, files, images and voice. Each becomes one of two things: **a question**, or **a file added to the library**.

| Input | Becomes | Function |
|---|---|---|
| Typed text | a question | used as-is |
| Voice (mic or a clip) | a question | `transcribe_audio_bytes()` → `transcribe_query()` (Whisper) |
| A query image | a question (its OCR text) **plus** an image to search image_index with | `ocr_image()`; the image goes to `stream_answer(query_image=...)` |
| An uploaded PDF / DOCX / image / audio | a new indexed file | `save_upload()` → `index_file()` |

## 3.1 Voice: why not reuse `AudioIngestor`

`AudioIngestor` (Chapter 9) cuts long recordings into 300-word, overlapping, timestamped chunks for the index. A spoken question is one sentence that just needs to become a string. `transcribe_query()` is ten lines: load Whisper, transcribe, join the segments, drop the model. Dropping it matters: Chapter 1 §1.9.2's RAM budget only works if Whisper is loaded *when needed* rather than held next to the embedding model and the LLM all the time.

## 3.2 An image with no words

A user can attach a photo and type nothing. If the photo also has no readable text, there is no text to search with. Embedding an empty string and searching `text_index` with it would return five arbitrary chunks, so `retrieve()` skips the text search when the question is blank, and the prompt gets a stand-in question (`IMAGE_ONLY_QUESTION`) instead of `Question: ` followed by nothing.

## 3.3 Uploads: two real safety details

- **The filename.** A browser hands over whatever the file was called, including `../../` or characters Windows can't store. That name becomes both a path on disk and a citation label, so `safe_filename()` keeps only letters, digits, `.`, `-` and `_`: `"../../etc/My File (1).PDF"` → `"My_File_1.pdf"`.
- **Indexing one file, not the folder.** `index_documents_directory()` re-embeds every document each time. Chapter 11 split out `index_document_file()` so an upload costs one file's worth of embedding. The folder version now just calls it in a loop.

---

# Part 4 — DECIDE

## 4.1 Why the page's logic lives outside the page

Two reasons, one practical and one from Chapter 4.

1. **Testing.** A Streamlit script runs top to bottom on every click; testing it needs Streamlit's own harness and fakes for every model. Plain functions run under plain pytest in milliseconds. So everything that *can* be a plain function is one, and Chapter 12's integration tests cover them without Streamlit.
2. **Track C calls, it doesn't re-implement.** Ch4 §4.1's produces/consumes table says the interface consumes Track A/B's entry points. Every function in `backend.py` is a thin wrapper over one of theirs (`index_document_file`, `index_image_files`, `index_audio_file`, `transcribe_query`, `TesseractOCREngine`).

## 4.2 Module layout

```
src/
├── core/llm.py                 ← + generate_stream()
├── pipelines/
│   ├── documents/index.py      ← + index_document_file() (one file)
│   ├── audio/transcribe.py     ← NEW: transcribe_query() for spoken questions
│   └── rag/answer.py           ← + stream_answer(), check_citations(), IMAGE_ONLY_QUESTION
└── ui/                         ← NEW package: the page's non-drawing half
    ├── backend.py              ← uploads, indexing, voice/OCR, index counts, Ollama check
    ├── citations.py            ← CitationView, citation_views()
    └── feedback.py             ← FeedbackEntry, record/load/summarize (Chapter 12)
```

`src/app.py` (the page) is not in this list: it arrives with PR #4.

---

# Part 5 — BUILD

> **Verification methodology, stated plainly.** This chapter was built in a cloud container that cannot download model weights (Hugging Face and Ollama were both unreachable from it). So everything below was verified with the real ChromaDB and the real document parsers, and with small, honest fakes standing in for MiniLM, CLIP, Whisper and Ollama (see `tests/test_integration.py`'s docstring for exactly what each fake does). That proves the wiring. It does not prove answer quality: that needs the team's own machine with `ollama serve` running, per §6.4.

## 5.1 `generate_stream()` and `stream_answer()` — Part 1.
## 5.2 `citation_views()` — Part 2.
## 5.3 `backend.py`, `transcribe_query()`, `index_document_file()` — Part 3.
## 5.4 `feedback.py` — used by Chapter 12; the page's rating form calls `record_feedback()`.

## 5.5 Run the tests

```bash
pytest tests/test_integration.py -v
```
**Real captured output (condensed):**
```
tests/test_integration.py::test_stream_answer_short_circuits_without_calling_the_llm PASSED
tests/test_integration.py::test_stream_answer_streams_the_llm_output PASSED
tests/test_integration.py::test_image_only_question_gets_a_real_question_in_the_prompt PASSED
tests/test_integration.py::test_check_citations_reports_out_of_range_numbers PASSED
tests/test_integration.py::test_kind_of_and_safe_filename PASSED
tests/test_integration.py::test_uploaded_pdf_becomes_searchable PASSED
tests/test_integration.py::test_citation_views_follow_the_adrs PASSED
tests/test_integration.py::test_feedback_round_trip_and_summary PASSED
...
============================== 28 passed in 5.61s ==============================
```

## 5.6 Wiring PR #4's scaffold (after it merges)

The scaffold's `process_query()` returns `(answer, citations)` with hand-written citation dicts. The real version is roughly this, replacing the mock body:

```python
from src.pipelines.rag import stream_answer
from src.pipelines.rag.answer import check_citations
from src.ui.citations import citation_views

citations, pieces = stream_answer(user_query)
answer = st.write_stream(pieces)                 # words appear as they're generated
bad = check_citations(answer, citations, user_query)
for view in citation_views(citations):           # same fields the scaffold's citation-box shows
    ...  # view.modality_label, view.title, view.excerpt, view.note, view.file_path
```

Two things to carry over from Part 2 when doing it: escape the excerpt with `html.escape()` before putting it inside the scaffold's `unsafe_allow_html` citation box (it's text from whatever file someone uploaded), and show `view.note` under Word citations.

---

# Part 6 — CHECK

## 6.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | `pytest tests/test_integration.py -v` fully passes | /3 |
| 2 | Team can explain why streaming doesn't make answers faster, and why it's still worth doing | /3 |
| 3 | Team can explain why `check_citations()` can't run inside `stream_answer()` | /3 |
| 4 | Team can name the decision behind each of the three citation display rules | /3 |
| 5 | Team can explain why a spoken question doesn't go through `AudioIngestor` | /3 |
| 6 | Team can say what `safe_filename()` protects against, with an example | /3 |
| 7 | PR #4's `process_query()` wired to `stream_answer()` + `citation_views()` and run against a real index | /3 |
| | **Total** | **/21** |

## 6.2 Question bank

1. A full answer takes 10 s with or without streaming. What does streaming change? → §1.1; when the user first sees progress, not the total time.
2. What does `stream_answer()` return for a question nothing in the index is relevant to, and is Ollama called? → §1.3; `([], iter([NOT_ENOUGH_INFO]))`, and no.
3. Why is no similarity score shown next to a citation? → §2, ADR-003/007; text and image scores are on different scales.
4. Why does a Word citation get a caveat a PDF citation doesn't? → ADR-008; DOCX files don't store page numbers.
5. A user attaches a photo with no text and types nothing. What does `retrieve()` search? → §3.2; image_index only, text search skipped.
6. Why was `index_document_file()` split out of `index_documents_directory()`? → §3.3; an upload should embed one file, not the whole folder.

## 6.3 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Voice question fails with "Couldn't transcribe the audio" | Whisper's weights not downloaded yet, or the clip is empty | Run once with internet so faster-whisper can fetch `WHISPER_MODEL_SIZE`; record a longer clip |
| Image query ignores the text in the picture | Tesseract binary not installed (`ocr_image()` returns "" rather than failing) | Install Tesseract (Chapter 5 §5.3), or set `TESSERACT_CMD` |
| Uploaded file indexed but citation shows a long `/home/...` path | File saved outside the project root | Launch Streamlit from the project root; `save_upload()` writes under `data/` relative to it |

## 6.4 Day 10–11 completion checklist

- [ ] `pytest tests/test_integration.py -v` passes on each member's machine
- [ ] With `ollama serve` running and a real index built, `stream_answer()` streams a cited answer
- [ ] PR #4 merged, and its `process_query()` wired as §5.6 shows
- [ ] A voice question and an image question both answered once through the page
- [ ] Rubric §6.1 scored ≥ 17/21

---

**Next:** [Chapter 12 — Integration, Testing & Human Feedback](ch12-integration-testing-and-feedback.md): images and audio finally reach ChromaDB, one question searches both collections, and outside testers start rating answers.
