# Architecture Decision Records (ADRs)

An **ADR** is a short document recording one significant design decision: what was decided, what the alternatives were, why the alternatives were rejected, and what the decision costs you.

## Why we keep these

Three reasons, in ascending order of importance:

1. **Memory.** In week three nobody remembers why chunk size is 300. The ADR says.
2. **Onboarding.** A new reader understands the system by reading decisions, not by reverse-engineering code.
3. **Marks.** A methodology section that says *"we use ChromaDB"* is a parts list. One that says *"we use ChromaDB rather than FAISS because embedded operation removes a deployment dependency and metadata filtering is required for citation provenance, while FAISS's scale advantage is irrelevant below 10⁵ vectors"* is methodology. **ADRs are that second thing, written once and reused in the report.**

The teaching material is in [Chapter 3 §3.3](../chapters/ch03-literature-review-and-methodology.md).

## The rule that makes an ADR worth writing

> **An ADR without rejected alternatives is not an ADR — it is a note.**

The alternatives are the entire value. They prove the decision was *chosen* rather than defaulted into. Where an alternative was rejected for a reason found in the literature, **cite it** — that single line converts an implementation preference into a defensible design decision.

## Status values

| Status | Meaning |
|---|---|
| **Proposed** | Under discussion, not yet acted on |
| **Accepted** | Decided; implementation follows this |
| **Superseded by ADR-NNN** | Replaced later — **keep the original file**; the change of mind is part of the record and shows the design evolved for reasons |
| **Deprecated** | No longer applies, not replaced |

Never delete an ADR. A superseded decision plus its replacement tells a better story than a tidy folder.

## The ADRs this project needs

001 and 003 written on Day 3; 002, 004, 005, 006 and 007 written on Day 4 during [Chapter 4](../chapters/ch04-timeline-and-team-split.md)'s planning session; those seven are architecture decisions made before any pipeline code exists, referenced throughout the report. 008 came later, on Day 6, because it answers a question Day 4's planning session had no way to anticipate: once real DOCX parsing code existed, it turned out the file format itself doesn't store what a "page" is — that's not a design preference to weigh in a planning meeting, it's a fact about the format you only run into by trying to write the parser. 009 came on Day 8, for the same kind of reason: only once real generation code existed did it become obvious that ChromaDB always returns *something* for a query, however irrelevant, and that a negative-control question needs its own gate to be testable at all. 010 came on Day 12, at integration: the moment image results joined text results, ADR-009's single number turned out to be a *text* number.

| ADR | Decision | Interesting rejected alternative | Cites literature |
|---|---|---|---|
| [001](adr-001-rag-over-finetuning.md) | RAG rather than fine-tuning | Fine-tuning — no citations possible, per-file retraining, GPU requirement | ✅ |
| [002](adr-002-local-llm-over-cloud.md) | Local quantized LLM rather than a cloud API | Cloud API — violates Objective O6 | ✅ |
| [003](adr-003-two-vector-collections.md) | Two vector collections | One collection — CLIP truncates text at 77 tokens | ✅ |
| [004](adr-004-chromadb-over-faiss.md) | ChromaDB rather than FAISS or Qdrant | FAISS — no metadata layer, no persistence, scale advantage irrelevant here | ✅ |
| [005](adr-005-transcription-over-clap.md) | Transcription rather than native audio embeddings | CLAP — weaker on speech semantics, loses timestamps | ✅ |
| [006](adr-006-fixed-size-chunking.md) | Fixed-size overlapping chunks | Semantic/recursive chunking — better, more complex; future work. Explicitly the "thin literature, ablate instead" ADR (Ch3 §3.3.2) | |
| [007](adr-007-rank-based-merge.md) | Rank-based cross-modal merging | Score-based merging — defeated by the modality gap | ✅ |
| [008](adr-008-docx-pagination-via-explicit-breaks.md) | DOCX pages tracked from explicit breaks only | `w:lastRenderedPageBreak` — looks like a solution, absent from our own generated files and non-reproducible even in real ones | |
| [009](adr-009-relevance-threshold-before-generation.md) | Gate chunks on a relevance-score floor before generation | No threshold, trust the LLM's own judgment — makes negative controls non-deterministic and still risks hallucination on stuffed-in irrelevant context | |
| [010](adr-010-per-collection-relevance-floors.md) | A separate relevance floor per collection | One shared 0.3 — silently filters out every image (modality gap) | |
| [011](adr-011-image-corroboration-gate.md) | Keep a weak CLIP image only if its OCR text corroborates it | CLIP floor alone: refuses 3 of 10 out-of-corpus questions | ✅ |
| [012](adr-012-hnsw-search-effort.md) | Higher HNSW search effort (ef 200, construction 400, M 32) | ChromaDB defaults: missed the best chunk for about 1 question in 3 on the pinned version | ✅ |
| [013](adr-013-transcript-cache.md) | Cache each audio file's transcript by the audio's bytes | Hardening Whisper's decoding: it was already deterministic within an environment | ✅ |
| [014](adr-014-image-text-channel.md) | Find images by their text too: built, measured, **not enabled** | Enabling it on the older rows that tuned the gate | ✅ |
| [015](adr-015-cpu-only-latency.md) | CPU-only latency: one shared Ollama client on `127.0.0.1` for everyone, `TOP_K=3` as an opt-in CPU profile | Lowering the default `TOP_K`, or trimming each chunk (it returned bare citations) | ✅ |
| [016](adr-016-answer-from-picked-files-and-attached-images.md) | Answer from the files you pick (no relevance floor inside a scope), and an attached image is context the model can read | Lowering the global relevance floor; gluing the picture's text onto the search; removing look-alike image search | ✅ |
| [017](adr-017-name-lookups-lexical-admit.md) | Name and identifier lookups: a lexical admit and a who/what-is prompt rule, built and measured, **not enabled** | Switching them on by default (one near-miss negative answered wrongly); lowering the floors | ✅ |
| [018](adr-018-names-typos-follow-ups-fuller-answers.md) | Names, typos, follow-ups and fuller answers switched on: ADR-017 plus a name-phrase rule, a spelling fix, follow-up resolution, a detail rule, cited-only sources. Adopted: V8 26/28 against 18/28, not vetoed on the older rows | An LLM rewrite of follow-ups (a second model call); lowering the floors | ✅ |

**All ten are now written.**

## Template

Copy [`adr-template.md`](adr-template.md) to `adr-NNN-short-slug.md` and fill it in. Keep each under one page — an ADR that needs two pages is usually two decisions.
