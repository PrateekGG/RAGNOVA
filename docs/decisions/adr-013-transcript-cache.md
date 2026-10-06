# ADR-013: Cache each audio file's transcript, keyed by the audio's bytes

## Status
Accepted — 2026-10-05. Refines ADR-005 (transcription over native audio embeddings): transcripts stay the indexed form of audio; this decides where they come from on a rebuild.

## Context

Audio transcripts are indexed in `text_index` beside the documents (ADR-005), so a transcript word changes a chunk's embedding and can reorder retrieval results. Whisper is not reproducible across *environments*: the pinned requirements environment and the newer-library one transcribed the same eight clips slightly differently, and one early index build differed from every later one on all eight clips ("Overdue fines" against "Overdoophines"). That moved text MRR on the same 25 questions and the same corpus from 0.81 to 0.75, because three questions had an audio chunk just above the right PDF chunk. A result that depends on which machine built the index cannot go in a report.

What was tested before choosing: four decoding settings (library defaults; `temperature=0`; plus `condition_on_previous_text=False`; plus `cpu_threads=1`) x 8 clips x 5 fresh-model runs gave **one distinct transcript per clip in every setting, defaults included**; the real ingestor in a build-like process, quiet and contended, was identical; three consecutive real builds stored byte-identical audio chunks; the weights had one snapshot. So within one environment Whisper was reproducible in every test, the single divergent build is unexplained and not reproducible, and the difference that *is* reproducible is between environments.

## Decision

`AudioIngestor` stores each file's transcript (`data/transcripts/<sha256 of the audio>.<model size>.json`: text, timestamps rounded to 2 decimals, Whisper version, date) the first time it transcribes the file, and serves every later ingest from that file, on any machine, without running Whisper (`src/pipelines/audio/transcript_cache.py`, `WHISPER_TRANSCRIPT_CACHE_DIR`). Whisper is loaded lazily, so an all-cached rebuild never loads it. The transcripts are committed with the corpus. Segment ends are clamped to the audio's duration (every earlier index cited ends past the end of 5 of 8 clips, by 0.1-1.8 s).

Verified: a build in the pinned Python 3.13 environment and a build in the main environment produce byte-identical audio chunks (8/8); a full rebuild dropped from about 82 s to 48 s.

## Alternatives considered

1. **Harden Whisper's decoding** (`temperature=0`, no conditioning on previous text, fixed threads). Rejected as the fix: in every test the unhardened default was already deterministic, so there was no observed nondeterminism for these settings to remove, and they cannot make two library versions agree.
2. **Pin the Whisper stack more tightly.** Helps one team, not an examiner on another machine, and the weights and kernels still differ by hardware.
3. **Commit the built ChromaDB index instead.** Binary, tied to one ChromaDB version (ADR-012), large, and it hides which transcript a chunk came from. A text transcript is reviewable and diffable.
4. **Hand-edited transcripts.** Possible with this mechanism (edit the JSON), but not the default: the cache records what Whisper actually produced.

## Consequences

**Positive**
+ Audio chunks, and therefore text retrieval numbers, are identical on every machine and in both environments.
+ Rebuilds are faster and no longer need Whisper for known files; an upload of an already-seen clip is instant.
+ Transcripts are inspectable, diffable evidence for the report.

**Negative / limits**
− **The first transcription of a new file still depends on the machine**, and a poor first transcript persists until its cache file is deleted.
− One more folder of committed data (eight small JSON files today).
− If a reproducible nondeterminism inside one environment is ever found, this does not explain it; it only makes the outcome stable.
− Spoken *queries* are one-off and deliberately not cached.

## Revisit if

A new Whisper model size becomes the default (the key includes the size, so old entries just stop being used), or the divergent early build is ever reproduced and its cause found.
