# RAGNova — Roadmap: Chapters × 14-Day Timeline

This document is the master plan. It breaks the project into **13 chapters**, maps each to the official timeline, and defines how the team splits work.

---

## The big picture (read this first)

Building a multimodal RAG system sounds intimidating, but it decomposes into five simple questions:

1. **How do we get content out of files?** → *Ingestion* (parse PDFs, DOCX, run OCR on images, transcribe audio)
2. **How do we make content searchable by meaning, not keywords?** → *Embeddings + Vector Database*
3. **How do we search across different media types with one query?** → *Cross-modal embeddings (CLIP) + unified retrieval*
4. **How do we turn retrieved chunks into a cited answer?** → *Local LLM + prompt engineering + citation tracking*
5. **How does a human use it?** → *Chat UI with file upload, mic input, citation viewer*

Every chapter below serves one of these five questions.

```
 Files (PDF/DOCX/IMG/AUDIO)
        │  Chapter 6–9: Ingestion pipelines
        ▼
 Text chunks + image embeddings + transcripts
        │  Chapter 7: Embedding models
        ▼
 Vector Database (ChromaDB) ── stores meaning as numbers
        │  Chapter 10: Retrieval
        ▼
 Top-K relevant chunks ──► Local LLM (Ollama) ──► Answer + [1][2] citations
        │  Chapter 11: UI
        ▼
 Chat interface (Streamlit)
```

---

## Chapter list

### Phase 0 — Prerequisites (Day 0 / Day 1 evening)

| Ch | Title | Day | Deliverable |
|----|-------|-----|-------------|
| 0 | Getting Started From Absolute Zero | 0 | Working computer: VS Code + Python 3.11 + terminal literacy + **Python primer** + maths refresher |

> **Chapter 0 assumes nothing** — not VS Code, not a terminal, not what a neural network is, not `Σ` notation, not a single line of Python. §A.6 is a complete primer in the exact subset of Python this project uses, ending with three exercises that are real functions from Chapters 6, 7 and 10. Skip it only if you already program in Python and know what a virtual environment is. It is *not* the same as Chapter 5: Chapter 0 asks "can my computer run Python at all?", Chapter 5 asks "can it run this project's AI models?"

### Phase A — Ideation & Planning (Days 1–4)

| Ch | Title | Day | Deliverable |
|----|-------|-----|-------------|
| 1 | Objective & Problem Identification | 1 | Problem statement analysis, objectives, scope document |
| 2 | Synopsis & Presentation | 2 | 2–3 page synopsis + PPT #1 |
| 3 | Literature Review & Methodology | 3 | Survey of existing systems + our chosen methodology<br>*(+ [3A] annotated bibliography, [3B] paper-reading & citations)* |
| 4 | Timeline, Team & Modular Work Split | 4 | PPT #2, computed critical-path schedule, module ownership table<br>*(+ [4A] scheduling & teamwork sources)* |

### Phase B — Implementation (Days 5–13)

| Ch | Title | Days | What we build |
|----|-------|------|---------------|
| 5 | Environment Setup & Offline LLM | 5 | Python env, Ollama + local model running, project skeleton, **real `src/core/schemas.py` + passing contract test**<br>*(+ [5A] systems references)* |
| 6 | Document Ingestion (PDF/DOCX) | 6 | Parsers, text extraction, chunking strategy |
| 7 | Embeddings & the Vector Database | 7 | sentence-transformers, ChromaDB, first semantic search |
| 8 | Image Pipeline (CLIP + OCR) | 6–7 | Text→image and image→text search |
| 9 | Audio Pipeline (Whisper) | 8–9 | Speech-to-text, transcript chunking with timestamps |
| 10 | RAG Core: Retrieval + Generation + Citations | 8–9 | The brain — query → retrieve → generate → cite |
| 11 | Unified Query Interface (UI) | 10–11 | Streamlit chat app: text/file/image/audio/mic input |
| 12 | Integration, Testing & Human Feedback | 12–13 | End-to-end tests, feedback forms, bug-fix cycle |

> **Days above are the corrected, computed schedule** — see [Chapter 4 §2.3](chapters/ch04-timeline-and-team-split.md) for the full critical-path calculation. The original Day-1 estimates (Ch10 at Day 10–11, Ch11 at Day 11–12) contained an unexamined two-day gap for Track A that a naive Gantt reading didn't surface; closing it moves Ch8–Ch11 two days earlier without changing the Day-14 deadline. Ch12–Ch13 are unchanged — the day of margin recovered is deliberately banked as a buffer before integration, not spent (Ch4 §2.4).

### Phase C — Reporting (Day 14)

| Ch | Title | Day | Deliverable |
|----|-------|-----|-------------|
| 13 | Mid-Term Report | 14 | Formal report: everything documented in Phases A–B |

---

## The offline tech stack (decided in Ch 3, justified there)

All tools below are **free, open-source, and run fully offline** after a one-time download:

| Layer | Tool | Why |
|---|---|---|
| Local LLM | **Ollama** running Llama 3.2 (3B) or similar | Easiest way to run an LLM offline; one command install |
| Text embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Small (80 MB), fast on CPU, excellent quality |
| Cross-modal embeddings | **OpenCLIP** | Maps text AND images into the *same* vector space → cross-modal search |
| Speech-to-text | **faster-whisper** | OpenAI Whisper, optimized; runs on CPU |
| Vector database | **ChromaDB** | Embedded (no server), Python-native, beginner-friendly |
| PDF parsing | **PyMuPDF** | Fast, handles messy PDFs |
| DOCX parsing | **python-docx** | Standard library for Word files |
| OCR (text in images) | **Tesseract** (via pytesseract) | Extracts text from screenshots |
| UI | **Streamlit** | Chat UI in pure Python — no HTML/JS needed |

> **Hardware reality check:** everything above runs on a laptop with 8 GB RAM (16 GB comfortable). No GPU required — CPU is enough for a demo-scale corpus.

---

## Team split (3 members — finalized in Chapter 4)

The architecture splits naturally into three parallel tracks that meet at the vector database:

| Track | Owner | Chapters | Modules |
|---|---|---|---|
| **A: Text pipeline + RAG core** | Member 1 | 6, 7, 10 | Doc parsing, chunking, text embeddings, retrieval, LLM, citations |
| **B: Vision + Audio pipelines** | Member 2 | 8, 9 | CLIP image search, OCR, Whisper transcription |
| **C: UI + Integration + Docs** | Member 3 | 11, 12 | Streamlit app, wiring pipelines together, testing, feedback collection |

Everyone does Chapters 1–5 and 13 **together** — shared foundation, shared report.

With 2 members: merge Track C into A and B (each owns half the UI).

**Interface contract:** tracks stay independent because everything talks through ChromaDB and one shared Python module (`src/core/schemas.py`) defining what a "chunk" looks like. Requirements, the module ownership (produces/consumes) table, and a strawman schema sketch are specified in [Chapter 4 §4](chapters/ch04-timeline-and-team-split.md); the real, frozen schema, plus its passing contract test (`tests/test_contract.py`), was implemented in [Chapter 5](chapters/ch05-environment-setup-and-offline-llm.md) §3 and is live in the repo today.

---

## Human testing / feedback loops (built into the plan)

| When | What |
|---|---|
| End of Ch 7 | First demo to friends: "does semantic search feel better than Ctrl+F?" |
| End of Ch 10 | Answer-quality check: 10 test questions, humans rate answers 1–5 |
| Ch 12 | Structured feedback: 3–5 outside users try the app with a feedback form |
| Ongoing | `docs/feedback-log.md` — every piece of feedback recorded + what we changed |

---

## Progress tracker

- [ ] Ch 0 — Getting Started From Absolute Zero *(each member ticks their own)*
- [x] Ch 1 — Objective & Problem Identification
- [ ] Ch 2 — Synopsis & PPT (docs written; team deliverables pending)
- [ ] Ch 3 — Literature Review & Methodology (docs written; team deliverables pending)
- [x] Ch 4 — Timeline & Team Split (docs written; team deliverables pending)
- [x] Ch 5 — Environment Setup & Offline LLM (docs + real, tested code written — schemas.py, config.py, contract test, verify_setup.py; team still needs to install Ollama/deps on their own machines)
- [x] Ch 6 — Document Ingestion (docs + real, tested code written — src/pipelines/documents/ [pdf_parser, docx_parser, chunker, ingest], src/core/text_normalize.py, scripts/generate_sample_corpus.py + the 3-file synthetic starter corpus in data/documents/, tests/test_document_ingestion.py passing against the real generated files, ADR-008 written; team's full target corpus in data/README.md is still open)
- [x] Ch 7 — Embeddings & Vector DB (docs + real, tested code written — src/core/embeddings.py,
      src/core/vector_store.py, src/pipelines/documents/{index,search}.py,
      scripts/{build_index,evaluate_retrieval}.py; real ChromaDB text_index built from the Ch6
      corpus (6 chunks), first real semantic search, Recall@5=1.00/MRR=1.00 on 3 gold questions
      (see Ch7 §3.3 for why that's promising but not yet strong evidence); two real bugs found
      and fixed during verification — see Ch7's framing note and §2.3/§5.3)
- [x] Ch 8 — Image Pipeline (docs + real, tested code written — src/pipelines/images/{models,ocr,embedding,
      ingest,index,search,cli}.py; Tesseract OCR + OpenCLIP ViT-B/32 into `image_index`, text->image and
      image->image `search_images()`, portable relative sources, `ImageIngestionConfig` bound to
      `settings.CLIP_MODEL`, 15-image corpus in data/images/, tests/test_image_pipeline.py (5 tests), teaching
      doc ch08. The image CLI now indexes into `image_index` and refuses a CLIP model that differs from the
      settings (PR #8))
- [x] Ch 9 — Audio Pipeline (docs + real, tested code written — src/pipelines/audio/{ingestion,index,
      transcribe}.py; faster-whisper `base` transcription with VAD, word-count sliding-window chunks carrying
      timestamp ranges, written into `text_index` (ADR-005), voice-query `transcribe_query()`, 4 spoken clips
      in data/audio/, tests/test_audio_pipeline.py (16 tests, 13 of them run the real `AudioIngestor` with only
      Whisper faked; transcripts are cached by the audio's sha256 so every machine indexes the same text), teaching doc ch09. The former limitation (storage only upserted, so a file that now yielded fewer chunks
      left its old trailing chunks behind) was closed on 2026-10-05 in the storage layer for all three modalities:
      `replace_source_chunks()` + `prune_missing_sources()` in src/core/vector_store.py)
- [x] Ch 10 — RAG Core (docs + real, tested code written — src/core/llm.py, src/pipelines/rag/{prompt,answer}.py,
      scripts/{ask,evaluate_answers}.py, tests/test_rag_core.py, ADR-009; real end-to-end answers with
      citations against the Ch6/7 text_index via real Ollama (llama3.2:3b) — see Ch10 §5 for captured
      output. Deliberately text-only for now: image_index/audio have no write path into ChromaDB yet
      (Ch8/9), so ADR-007's cross-modal merge has nothing to merge with — the retrieval interface is
      shaped so wiring in a second collection later is additive, not a redesign. Answer-quality check run
      against 7 gold questions (T1-T5, N1-N2), self-rated 4.71/5 average, then re-run on the grown corpus
      with 29 questions (self-rated 4.21/5; transcripts in data/eval/) — see data/README.md's Results log)
- [ ] Ch 11 — UI (docs + real, tested code written for the non-drawing half — streaming answers
      (`generate_stream()`, `stream_answer()`), per-citation display rules (`src/ui/citations.py`),
      uploads / voice / image-query plumbing (`src/ui/backend.py`, `transcribe_query()`,
      `index_document_file()`). The page itself, `src/app.py` (PR #4, merged), is still the scaffold:
      `process_query()` returns canned keyword-matched answers with fake citations. Still open: wire it to the
      real backend as Ch11 §5.6 shows)
- [ ] Ch 12 — Integration & Feedback (docs + real, tested code written — image and audio write paths into
      ChromaDB, `search_images()`, rank-merged `retrieve()` across both collections (ADR-007), per-collection
      relevance floors (ADR-010) plus an OCR-corroboration gate for images (ADR-011), `build_index.py` indexing
      all three folders, `tests/test_integration.py` (32 passing, with fakes for the models), feedback log +
      `summarize_feedback.py` + `docs/feedback-log.md`. Two real seam bugs found (Ch12 §5.3). The corpus grew
      from 3 synthetic files to 19 documents / 25 images / 8 audio clips with 30 openly-licensed files downloaded
      (data/SOURCES.md, Git LFS); one gold set (data/gold_set.json: 25 text, 22 cross-modal, 8 negatives).
      Measured on the grown corpus (2026-10-05, data/README.md's Results log): text Recall@5 = 1.00 / MRR =
      0.81 (reproducible: audio transcripts are cached by sha256, data/transcripts/; two earlier builds with
      different transcripts gave 0.75), cross-modal Recall@5 = 0.82 (18/22; text-to-image 10/14; the four misses I3, I8, I10, I14 recorded, not tuned).
      Image floor settled by the ADR-011 gate, re-measured on 14 positives / 10 negatives (11/14 kept, 9/10 refused,
      thresholds held); the text floor was
      measured and deliberately left unchanged (ADR-009 update). The pinned requirements.txt was verified in a
      fresh Python 3.13 environment, which exposed and fixed three defects (unpinned `av` broke real
      transcription; chromadb 0.5.23's HNSW search missed the best chunk for ~1 in 3 questions, ADR-012;
      telemetry); performance measured (data/eval/performance_2026-10-05.txt). Real measured data is dumped
      into reports/methodology-draft.md's appendix. Ablations done (chunk size, top-K, merge policy: shipped values hold,
      ADR-006/007 amended). Offline verification done without the UI (RAGNOVA_OFFLINE switch;
      scripts/verify_offline.py). PDF table extraction was measured and rejected (T14/T20 refusals are generator limits, not
      parsing defects: ADR-009 follow-up). Still open: outside-tester sessions, the real network-off demo through
      the UI, wiring the UI, a Python 3.11 check — Ch12 §7.3). CPU-only measured: end-to-end latency misses the
      15 s target (median 29 s; 14 s with TOP_K=3), answer quality unchanged)
- [ ] Ch 13 — Mid-Term Report
