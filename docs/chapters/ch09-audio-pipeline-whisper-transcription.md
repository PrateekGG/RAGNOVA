# Chapter 9 — Audio Pipeline: Whisper Transcription (Day 10)

> **Deliverables today:** a complete audio ingestion pipeline (`src/pipelines/audio/`): Whisper transcription with VAD (voice activity detection), sliding-window chunking over timestamped Whisper segments, transcript chunks written into `text_index` (not a third collection — ADR-005's transcription-over-native-audio decision), portable source paths with timestamp ranges for citations, and `tests/test_audio_pipeline.py`.
>
> **Prerequisites:** [Chapter 7](ch07-embeddings-and-vector-database.md) (the `text_index` collection and `add_chunks()` primitive this chapter reuses for audio transcripts) and [ADR-005](../decisions/adr-005-transcription-over-clap.md) (why transcription, not native audio embeddings like CLAP).
>
> **The framing for today.** Audio is the third modality RAGNova handles, but unlike images (which got their own `image_index` collection in Chapter 8), audio transcripts go into the *existing* `text_index` alongside documents — because ADR-005 chose transcription over native audio embeddings. A transcript chunk is just text with a timestamp range (`start_s`, `end_s`) instead of a page number, embedded by the same MiniLM model (Chapter 7) as PDFs. This shared embedding space is what lets one text question retrieve a PDF paragraph, a DOCX section, and a spoken sentence in the same search.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: Whisper in Practice** | Loading faster-whisper, VAD (voice activity detection), language detection, word-level timestamps. | ~30 min |
| **Part 2 — LEARN: Sliding-Window Chunking** | Target chunk size in words (not seconds), overlap for continuity, splitting mid-segment. | ~30 min |
| **Part 3 — LEARN: Why Audio Goes Into text_index** | ADR-005's decision, embedding transcripts with MiniLM, timestamp-based citations. | ~20 min |
| **Part 4 — DECIDE** | Module layout, binding to `settings.TEXT_EMBEDDING_MODEL`, chunk target size, retry strategy. | ~20 min |
| **Part 5 — BUILD** | Writing each file, transcribing real audio, indexing into `text_index`, citations with timestamps. | ~2 hours |
| **Part 6 — CHECK** | Rubric, question bank, troubleshooting, completion checklist. | ~30 min |

**Learning outcomes.** You will be able to: call faster-whisper to transcribe audio with VAD; explain why VAD matters for chunking; explain sliding-window chunking over timestamped segments; explain why audio transcripts are embedded with MiniLM and written into `text_index`, not a separate collection; and trace through the timestamp-based citation format (`data/audio/talk.wav, 12s–20s`).

---

# Part 1 — LEARN: Whisper in Practice

## 1.1 Whisper vs. faster-whisper

OpenAI's original Whisper implementation is accurate but slow on CPU. `faster-whisper` is a CTranslate2-based reimplementation that's 4x faster with the same accuracy, using quantized models (int8) — crucial for Chapter 5's CPU-only, 8GB RAM constraint. Model sizes range from `tiny` (39M params, fast but less accurate) to `large` (1550M params, very accurate but slow). This project defaults to `base` (74M params) — a balance for CPU inference.

## 1.2 Loading and calling faster-whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.mp3", vad_filter=True)

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
```

`vad_filter=True` enables voice activity detection — the model only processes audio segments that actually contain speech, skipping silence and background noise. Without it, Whisper still returns segments for silent sections (often as empty strings or hallucinated filler), which would create useless chunks.

## 1.3 Language detection

```python
segments, info = model.transcribe("audio.mp3", vad_filter=True)
print(f"Detected language: {info.language} (probability={info.language_probability:.2f})")
```

Whisper auto-detects the spoken language from the first few seconds of audio. For this project (English-language corpus), this is a logging convenience, not a feature — but it's available if the corpus ever expands to non-English audio.

## 1.4 Word-level timestamps (available, but not used here)

```python
segments, info = model.transcribe("audio.mp3", word_timestamps=True)
for segment in segments:
    for word in segment.words:
        print(f"{word.word} [{word.start:.2f}s - {word.end:.2f}s]")
```

`word_timestamps=True` gives per-word timing. RAGNova's `AudioIngestor` does **not** turn it on: it works from segment-level timestamps only, and when it has to split a segment to hit the chunk target it *estimates* the split time by proportion (see §2.4). Word-level timing would make those splits exact, at the cost of slower transcription, which makes it a reasonable future improvement rather than something the current code relies on.

---

# Part 2 — LEARN: Sliding-Window Chunking

## 2.1 Why chunk audio at all?

A 10-minute recording transcribed in one piece would produce a single, multi-paragraph chunk. Embedding that entire block loses granularity — a question about one specific fact in minute 3 would retrieve the entire 10-minute chunk, and the LLM (Chapter 10) would have to find the relevant sentence buried in irrelevant context. Chunking by fixed time intervals (e.g., every 30 seconds) would split mid-sentence. Chunking by Whisper's segments (which are sentence-like) is better but still variable — some segments are 5 words, others 50.

## 2.2 Target chunk size in words, not seconds

Audio chunks target `settings.CHUNK_SIZE_WORDS` (300) words, the very same setting Chapter 6's document chunker uses. (`ingestion.py` also looks for an optional `AUDIO_CHUNK_TARGET_WORDS` via `getattr`, but no such field exists on `Settings`, so in practice audio simply inherits the document setting.) The ingestion pipeline buffers Whisper segments until the accumulated word count reaches 300, then emits a chunk and starts a new buffer. This produces semantically-coherent chunks (not split mid-sentence unless absolutely necessary) of roughly uniform size (better retrieval ranking than wildly variable chunk lengths).

## 2.3 Overlap for context continuity

`settings.CHUNK_OVERLAP_WORDS` (50) — the last 50 words of chunk N are repeated as the first 50 words of chunk N+1. Same rationale as Chapter 6's document overlap: a fact mentioned once, right at a chunk boundary, appears in two chunks' context rather than being unretrievable if it's on the "wrong" side of the split.

## 2.4 Splitting mid-segment when necessary

If a single Whisper segment contains 400 words and the current buffer already has 250 words, adding the entire segment would overshoot the 300-word target by 350 words. The chunker splits the segment at the 50-word mark (estimating the split time from the split point's position within the segment's text, since there is no per-word timing, §1.4), emits the first 50 words in the current chunk, and carries the remaining 350 words into the next chunk's buffer. This is `_split_segment()` in `src/pipelines/audio/ingestion.py` — the exact logic that makes "target 300 words" a real target, not just an average.

---

# Part 3 — LEARN: Why Audio Goes Into text_index

## 3.1 ADR-005: transcription over native audio embeddings

CLAP (Contrastive Language-Audio Pretraining) is an audio counterpart to OpenCLIP — it embeds audio clips and text into a shared space, enabling text-to-audio search without transcription. ADR-005 rejected it for two reasons:

1. **Transcripts are more useful than audio clips for LLM context.** Chapter 10's RAG prompt needs *text* to stuff into the context window, not an audio file. CLAP would require transcription *anyway* to produce readable citations.
2. **Transcripts are searchable by the same text embedding model as documents.** MiniLM (Chapter 7) embeds transcript text, so a text query retrieves documents and audio together, naturally — no separate collection, no modality-gap merge complexity beyond what Chapter 8's `image_index` already introduced.

## 3.2 Embedding transcripts with MiniLM

```python
from src.core.embeddings import embed_texts

chunks = [...]  # audio chunks with .text = transcript
embeddings = embed_texts([c.text for c in chunks])
```

The same `embed_texts()` function Chapter 7 built for documents. Audio transcripts are just text — the same 384-d MiniLM vectors, the same cosine similarity scoring, the same `text_index` collection.

## 3.3 Timestamp-based citations

A document chunk cites `(source, page)` — e.g., `data/documents/notice.pdf, page 2`. An audio chunk cites `(source, start_s, end_s)` — e.g., `data/audio/talk.wav, 12s–20s`. Chapter 10's `format_provenance()` already handles both modalities:

```python
# simplified from src/pipelines/rag/prompt.py (the real function also handles images)
if chunk.modality == "audio":
    return f"{chunk.source}, {chunk.start_s:.0f}s–{chunk.end_s:.0f}s"
return f"{chunk.source}, page {chunk.page}"  # documents
```

The user sees "listen starting at 12 seconds" — a directly actionable citation, same principle as "page 2."

---

# Part 4 — DECIDE

## 4.1 Module layout

```
src/pipelines/audio/
├── __init__.py
├── ingestion.py       ← AudioIngestor: transcribe + chunk + timestamp
├── transcribe.py      ← transcribe_query(): turn a user's voice query into text
├── index.py           ← index_audio_file(), index_audio_directory()
```

Kept under `pipelines/audio/` (not `core/`) for the same reason images are under `pipelines/images/` — this is Track B's domain code. Only the ChromaDB write primitive (`add_chunks()`) and text embedding (`embed_texts()`) are shared with other tracks.

## 4.2 Binding to `settings.TEXT_EMBEDDING_MODEL`

`AudioIngestor` stores `settings.TEXT_EMBEDDING_MODEL` (or `settings.DEFAULT_TEXT_EMBEDDING_MODEL`, fallback) in every chunk's `embedding_model` field — the same centralized config binding Chapter 8 established for CLIP. One source of truth: changing the embedding model in `.env` changes it for documents, audio, and any future text-based modality.

## 4.3 Target chunk size — 300 words, matching documents

Audio has no chunk-size setting of its own. It reads `settings.CHUNK_SIZE_WORDS = 300` (Chapter 6), so documents and transcripts always chunk the same way. Same reasoning: too small (e.g., 50 words) and retrieval is noisy (many short, low-context chunks per query); too large (e.g., 1000 words) and retrieval is coarse (one question about a 2-second fact retrieves 5 minutes of transcript). 300 is the starting point Chapter 6 already justified — revisit if audio-specific evaluation (gold questions about spoken content) shows a different optimum.

## 4.4 Retry strategy with exponential backoff

`AudioIngestor._transcribe_with_retry()` wraps `model.transcribe()` in a 3-attempt loop with exponential backoff: after the first failure it waits 1 s, after the second 2 s, and a third failure is re-raised to the caller. Whisper transcription on CPU can occasionally fail on specific audio files, and retrying after a brief pause succeeds more often than giving up immediately.

One subtlety worth knowing: `transcribe()` is lazy, so a failure can happen *after* some segments were already handed to the chunker, and a retry starts again from the beginning of the file. The loop therefore remembers the start time of the last segment it yielded (`last_yielded_start`) and skips anything at or before it; otherwise the repeated segments would be duplicated into the transcript. This was a real bug in an earlier version of this file (PR #5), and `test_real_audio_ingestor_does_not_duplicate_segments_when_a_retry_restarts_transcription` now guards it. Retries are currently silent; a `logger.warning` per failed attempt would be a worthwhile small improvement.

## 4.5 The transcript is part of the corpus: a cache keyed by the audio's bytes

**The problem, measured.** A retrieval number has to mean the same thing on every machine that reports it. Transcripts are an input to that number: they are indexed in `text_index` next to the documents (ADR-005), and a transcript word changes a chunk's embedding. On 2026-10-05 the same eight clips produced different transcripts in different places ("Overdue fines" in one build, "Overdoophines" in another), and the text MRR on the same 25 questions moved with them (0.75 against 0.81), because three questions had an audio chunk sitting just above the right PDF chunk. The pinned requirements environment and the newer-library one also transcribed the clips slightly differently (a different `faster-whisper` build), so two correct installs would report different results.

**What was and was not found.** Transcribing each clip five times with a fresh model gave one distinct output per clip, under the library defaults and under three stricter settings (`temperature=0`, `condition_on_previous_text=False`, `cpu_threads=1`); doing it inside the real ingestor after loading the embedding model, quietly and with two runs competing for the CPU, also gave identical text, and three consecutive `build_index.py` runs stored byte-identical transcripts. The Whisper weights had a single snapshot and the environment had not changed. So within one environment Whisper *was* reproducible whenever it was tested, and the one divergent early build is **unexplained and not reproducible** (an attempt to test full CPU saturation was abandoned as too slow to finish usefully). What *was* reproducible is the difference between environments. That settles where the fix belongs: not a decoding flag, which would fix nothing that was observable, but making the transcript an input that is stored rather than recomputed.

**The mechanism** (`src/pipelines/audio/transcript_cache.py`). The first time a clip is ingested, its segments are written to `data/transcripts/<sha256 of the audio>.<model size>.json` (text, rounded timestamps, the Whisper version and date, for provenance). Every later ingest, on any machine, reads that file instead of running Whisper. The key is the content hash, so editing or replacing a clip re-transcribes it, and the model size is part of the key. A corrupt file is ignored and rewritten; the write is atomic (temp file, then rename); a failed write warns but never retries the transcription; an abandoned run writes nothing. The files are committed with the corpus. Whisper is also loaded lazily now, so a rebuild whose transcripts are all cached never loads it at all (a full index build dropped from about 82 s to 48 s, and the pinned and unpinned environments then built **byte-identical audio chunks**).

**A defect found on the way.** Every earlier index cited audio end times that were *later than the audio*: for 5 of the 8 clips the last segment ended 0.1 to 1.8 s past the end of the file (a 33.26 s clip cited as ending at 33.54 s; one early chunk ended at 80.63 s on a 72.58 s clip). Segment ends are now clamped to the audio's duration, and a test pins it.

**What it does not do, stated plainly.** The *first* transcription of a new file still depends on the machine, and a poor first transcript persists until its cache file is deleted. Spoken *queries* (`transcribe_query`) are one-off and deliberately not cached.

---

# Part 5 — BUILD: the actual Day 10

## 5.1 Write `src/pipelines/audio/ingestion.py`

`AudioIngestor.process_file()` — see Parts 1–2. The full sliding-window chunking logic: buffering segments, counting words, splitting mid-segment when necessary, trimming overlap, building `Chunk` objects with timestamps.

Key details:
- `_validate_file()`: checks file exists, extension is supported (`.mp3`, `.wav`, `.m4a`, etc.), size ≤ 2GB
- `_transcribe_with_retry()`: 3 attempts with backoff
- `_create_chunks_from_segments()`: the sliding-window implementation
- `_build_chunk()`: constructs a `Chunk` with `modality="audio"`, `start_s=`, `end_s=`, `embedding_model=settings.TEXT_EMBEDDING_MODEL`

## 5.2 Write `src/pipelines/audio/transcribe.py`

`transcribe_query()` — a thin wrapper over `model.transcribe()` for Chapter 11's voice-query feature (a user records a question instead of typing it). Joins Whisper segments into a single string, stripping silence and empty segments.

## 5.3 Write `src/pipelines/audio/index.py`

`index_audio_file()` — see Part 3.2. Calls `AudioIngestor.process_file()`, validates every chunk with `validate_chunk()` (Chapter 5's contract), normalizes whitespace with `normalize_text()`, embeds with `embed_texts()`, and writes to `text_index` via `add_chunks()`.

## 5.4 Test the pipeline with real audio

`data/audio/` ships four short spoken clips (see `data/README.md`). Index one into a throwaway Chroma directory and search for what it says:

```bash
python -c "
import tempfile
from pathlib import Path
from src.core.vector_store import get_client
from src.pipelines.audio.index import index_audio_file
from src.pipelines.documents.search import search_text
from src.pipelines.rag.prompt import format_provenance

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    client = get_client(persist_dir=Path(tmp) / 'chroma')
    count = index_audio_file('data/audio/hod_project_announcement.wav', client=client)
    print(f'Indexed {count} chunk(s) from audio')

    for hit in search_text('how many marks does the prototype carry', top_k=3, client=client):
        print(f'{format_provenance(hit)}  score={hit.score:.3f}')
        print('   ', hit.text[:90].replace(chr(10), ' '), '...')
"
```

**Real captured output** (Windows, Python 3.13, real faster-whisper `base`; the first run also downloads the model):
```
Indexed 1 chunk(s) from audio
data/audio/hod_project_announcement.wav, 0s–33s  score=0.288
    Good morning final year students. This is an important announcement regarding your B.Tex C ...
```

The whole 32-second clip became a single chunk (it is far below the 300-word target), so its citation is the clip's full time range. Notice the transcript says "B.Tex C" where the clip's script says "B.Tech CSE-AIML": Whisper's `base` model makes small slips, which is one reason retrieval quality is measured rather than assumed.

Audio chunks are searchable alongside documents, using the same text query.

## 5.5 Run `tests/test_audio_pipeline.py`

Moved out of `test_integration.py` (its old Part 5), plus three new tests that run the **real** `AudioIngestor` with only Whisper faked. The older tests replace the whole ingestor with a `FakeIngestor`, so they never execute `ingestion.py` itself; the new ones do, and each fails if one of the bugs this file went through during review comes back (`embedding_model=None`, an unsupported `metadata=` argument to `Chunk`, chunk ids that change with Whisper's timestamp jitter, segments duplicated by a retry).

```bash
pytest tests/test_audio_pipeline.py -v
```

**Real captured output:**
```
tests/test_audio_pipeline.py::test_audio_is_indexed_into_text_index_and_cited_by_timestamp PASSED [ 16%]
tests/test_audio_pipeline.py::test_audio_chunks_that_break_the_contract_are_refused PASSED [ 33%]
tests/test_audio_pipeline.py::test_transcribe_query_joins_segments PASSED [ 50%]
tests/test_audio_pipeline.py::test_real_audio_ingestor_chunks_satisfy_the_contract PASSED [ 66%]
tests/test_audio_pipeline.py::test_real_audio_chunk_ids_ignore_timestamp_jitter_and_differ_per_file PASSED [ 83%]
tests/test_audio_pipeline.py::test_real_audio_ingestor_does_not_duplicate_segments_when_a_retry_restarts_transcription PASSED [100%]

======================== 6 passed, 1 warning in 12.47s =========================
```

## 5.6 Git hygiene for today

- [ ] `src/pipelines/audio/{ingestion,transcribe,index}.py` committed
- [ ] `tests/test_audio_pipeline.py` committed; audio tests removed from `test_integration.py`
- [ ] `docs/chapters/ch09-audio-pipeline-whisper-transcription.md` committed
- [ ] Every team member has confirmed `pytest tests/test_audio_pipeline.py` passes

---

# Part 6 — CHECK

## 6.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | `AudioIngestor.process_file()` transcribes real audio into chunks with timestamps | /3 |
| 2 | Audio chunks have `embedding_model = settings.TEXT_EMBEDDING_MODEL` | /3 |
| 3 | `index_audio_file()` writes audio chunks into `text_index` (not a separate collection) | /3 |
| 4 | `search_text()` retrieves audio chunks alongside document chunks for the same query | /3 |
| 5 | `pytest tests/test_audio_pipeline.py -v` fully passes | /3 |
| 6 | Audio chunks cite `(source, start_s, end_s)` instead of `(source, page)` | /3 |
| 7 | Team can explain why audio goes into `text_index`, not a third collection | /3 |
| 8 | Team can explain why VAD (`vad_filter=True`) matters for chunking | /3 |
| 9 | Team can explain why chunk size is measured in words, not seconds | /3 |
| 10 | Team can explain the retry-with-backoff strategy | /3 |
| | **Total** | **/30** |

## 6.2 Question bank

1. Why use `faster-whisper` instead of OpenAI's original Whisper? → §1.1; 4x faster on CPU via CTranslate2 quantization, same accuracy.
2. What does `vad_filter=True` do, and why does it matter? → §1.2; enables voice activity detection, skips silence and background noise, prevents empty/hallucinated chunks.
3. Why chunk audio at all instead of embedding the entire transcript? → §2.1; a 10-minute transcript in one chunk loses granularity — every query retrieves the whole thing, burying the relevant sentence.
4. Why is chunk size measured in words (300) instead of seconds (e.g., 30s)? → §2.2; spoken word rate varies — 30s of fast speech could be 150 words, 30s of slow speech 75 words. Word count is more uniform.
5. What does `CHUNK_OVERLAP_WORDS = 50` prevent (for audio, as for documents)? → §2.3; a fact mentioned once at a chunk boundary wouldn't be retrievable if it's split across two chunks with no overlap.
6. What is `_split_segment()` and why does it exist? → §2.4; splits a long Whisper segment mid-way (at a word boundary) so the target 300-word chunk size is a real target, not just an average.
7. Why did ADR-005 choose transcription over CLAP (native audio embeddings)? → §3.1; LLMs need text for context (not audio clips), and transcripts are searchable by the same MiniLM model as documents.
8. How are audio transcripts embedded? → §3.2; the same `embed_texts()` / MiniLM model as documents — they're just text with timestamps.
9. What's the citation format for an audio chunk? → §3.3; `data/audio/talk.wav, 12s–20s` (source + timestamp range, not page number).
10. Why does `AudioIngestor._transcribe_with_retry()` exist? → §4.4; Whisper on CPU occasionally times out on specific audio formats; retrying with exponential backoff succeeds more often than raising immediately.

## 6.3 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'faster_whisper'` | `faster-whisper` not installed | `pip install -r requirements.txt` |
| Transcription is very slow (>1 minute for a 2-minute audio file) | Using a large model (`large-v2`) or `device="cpu"` with no optimization | Confirm `WhisperModel("base", device="cpu", compute_type="int8")` — `base` is the default, `large` is much slower |
| Empty audio chunks (no text) even though the audio has speech | VAD too aggressive, or audio quality too poor | Try `vad_filter=False` temporarily to confirm; if that works, adjust `vad_parameters` (lower `min_silence_duration_ms`) |
| Audio chunks have `embedding_model=None`, contract validation fails | `AudioIngestor._build_chunk()` isn't resolving the model name from `Settings`. The real field is `TEXT_EMBEDDING_MODEL`; an earlier version of this file looked up two names that don't exist (`DEFAULT_TEXT_EMBEDDING_MODEL`, `EMBEDDING_MODEL`) and got `None` on every chunk | Confirm `_build_chunk()` reads `settings.TEXT_EMBEDDING_MODEL` first. `test_real_audio_ingestor_chunks_satisfy_the_contract` exists to catch exactly this |
| Whisper detects the wrong language | Audio has background music or heavy accents | Check `info.language` and `info.language_probability` — if probability <0.8, the audio may be unclear. Pre-process to remove background noise, or force language: `model.transcribe(..., language="en")` |
| `FileNotFoundError` when indexing audio files | Audio file path incorrect or not in a supported format | Confirm path exists and extension is in `SUPPORTED_EXTENSIONS` (`.mp3`, `.wav`, `.m4a`, etc.) |

## 6.4 Day 10 completion checklist

- [ ] `faster-whisper` installed and `AudioIngestor.process_file()` transcribes real audio
- [ ] At least 1 audio file indexed into `text_index`, timestamps confirmed in citations
- [ ] `search_text()` retrieves audio chunks alongside document chunks
- [ ] `pytest tests/test_audio_pipeline.py -v` fully passes
- [ ] Team can explain why audio transcripts go into `text_index`, not a separate collection
- [ ] Team can explain the sliding-window chunking logic
- [ ] Rubric §6.1 scored ≥ 24/30

---

**Next:** Chapter 10 (already completed) — RAG Core: Retrieval + Generation + Citations, built on `text_index`, which as of this chapter holds both document chunks and audio transcripts. Chapter 12 later adds `image_index` to retrieval and merges the lists by rank (ADR-007).
