# METHODOLOGY — DRAFT

> **⚠️ Rewrite every sentence in your own words.** You will be questioned on all of it. Guidance is in [Chapter 3 Part 3](../docs/chapters/ch03-literature-review-and-methodology.md). `⟨FILL⟩` marks a team-specific detail.
>
> Citation numbers `[n]` refer to the reference list in [`synopsis-draft.md`](synopsis-draft.md), extended with Chapter 3 Part 2's sources. **Renumber in order of first appearance** once the final text is assembled.

---

## 1. Research Approach

This project follows a **design-science methodology**: a software artefact is constructed to address an identified deficiency, and evaluated against functional and performance criteria defined in advance and derived from the literature. It is therefore constructive rather than hypothesis-testing; the claim under evaluation is that the artefact satisfies its stated objectives, not that a general proposition about the world holds.

Development follows an **incremental, modular process**. Three ingestion and retrieval pipelines are developed in parallel against a shared data schema frozen before implementation begins, integrated at a scheduled point rather than at the end of the schedule, and refined through two rounds of human evaluation. Two decisions make parallel development viable:

1. **Schema freeze (Day 5).** A single definition of an indexed segment — its identifier, source, modality, location and text — is fixed before any pipeline is written. Every pipeline emits this structure, so components remain substitutable and integration does not require renegotiation.
2. **Integration on Day 12, not Day 13.** A critical-path analysis of the nine-day implementation phase (Chapter 4 §2.3) shows a fully critical, zero-float schedule: the text-and-retrieval track and the vision-and-audio track each require exactly four days of work and converge simultaneously, leaving no naturally occurring slack anywhere in the plan. Integration is therefore scheduled to begin on Day 12 rather than at the theoretical earliest possible start, reserving one day as a deliberately inserted project buffer — in the sense of Goldratt's Critical Chain Method [Goldratt1997] — positioned immediately before the point at which all three tracks must converge. This directly addresses risk R6 (integration crunch — three tracks do not fit together) from the project's risk register: scheduling integration for Day 13 instead, as a literal "finish everything, then integrate" reading of the plan would suggest, would consume this buffer before it exists, leaving no recovery time between discovering an integration failure and the Day 14 report deadline.

## 2. System Architecture

⟨INSERT FIGURE 1 — architecture diagram⟩

The system separates an **indexing phase**, executed once per file, from a **query phase**, executed once per question. This separation is what makes interactive latency achievable: all model-heavy processing of documents occurs ahead of time, leaving only query embedding, nearest-neighbour lookup and generation on the interactive path.

⟨FILL: 3–4 sentences walking through the diagram.⟩

## 3. Ingestion Methodology

### 3.1 Document ingestion
PDF files are parsed with PyMuPDF and DOCX files with python-docx, retaining page numbers for provenance. Extracted text is normalised (whitespace collapsed, control characters removed) before segmentation.

### 3.2 Segmentation
Text is divided into fixed-size overlapping segments of approximately 300 words with 50 words of overlap.

**Justification.** The embedding model accepts a maximum of 256 word-piece tokens; input beyond this is silently truncated, so segments must fit within that bound. Overlap ensures that a sentence spanning a segment boundary appears intact in at least one segment, preventing loss of retrievable content at boundaries. Segment size is treated as a **tunable parameter and validated by ablation** at 150, 300 and 600 words (§8.4) rather than asserted.

**Alternative considered.** Semantic or recursive segmentation, which splits on structural boundaries rather than fixed counts, is expected to produce more topically coherent segments. It is deferred as future work on grounds of implementation complexity within the available schedule. ⟨FILL: adjust if you implement it.⟩

### 3.3 Audio ingestion
Audio is transcribed using faster-whisper, an optimised implementation of Whisper [16], which emits text segments annotated with start and end times. Transcripts are segmented identically to documents, with timestamps carried into segment metadata so that a citation can reference a position within a recording.

**Alternative considered and rejected.** Native audio embeddings via contrastive audio-language pretraining [17] were considered. They were rejected because the corpus consists of speech, whose information content is linguistic rather than acoustic; transcription both matches the content type and preserves the temporal information required for citation. ⟨FILL: cite CLAP correctly.⟩

### 3.4 Image ingestion
Images are processed along two complementary paths. A CLIP vision encoder [11], [12] produces a semantic embedding representing image content. In parallel, Tesseract OCR [18] extracts any literal text present, which is indexed as ordinary text.

**Justification for both paths.** The two capture different information: the visual embedding supports queries describing what an image *depicts*, while OCR supports queries containing text *within* the image, such as an identifier or a name. Neither subsumes the other.

### 3.5 Provenance metadata
Every segment carries, from creation, a record comprising segment identifier, source path, modality, and location (page number for documents, start and end times for audio). This metadata is the mechanism by which citation is possible; it is created at ingestion and never discarded. ⟨FILL: reference your schema definition.⟩

## 4. Indexing Methodology

Two vector collections are maintained in ChromaDB:

| Collection | Model | Dimensions | Contents |
|---|---|---|---|
| `text_index` | `all-MiniLM-L6-v2` [4] | 384 | Document segments, transcript segments, OCR text |
| `image_index` | OpenCLIP ViT-B/32 [11], [12] | 512 | Image embeddings |

**Justification for separation.** The two models produce vectors of different dimensionality in unrelated spaces, so a single collection is not merely inadvisable but ill-defined. Furthermore, CLIP's text encoder truncates at 77 tokens, making it unsuitable for document-length passages. See ADR-003.

**Approximate nearest-neighbour search.** ChromaDB indexes vectors using hierarchical navigable small-world graphs [7], giving approximately logarithmic search complexity. At the scale of this project (about 650 vectors: 631 text segments, of which 623 come from documents and 8 from audio transcripts, plus 25 image embeddings *[data: data/README.md, 2026-10-05]*) exact search would also be tractable; the approximate index is adopted for architectural correctness and scalability rather than present necessity. **This is stated explicitly rather than implied**, since claiming a performance benefit not observed at our scale would be unsupported.

## 5. Retrieval Methodology

A textual query is embedded by both models and used to search both collections. An uploaded image is embedded by the CLIP vision encoder to retrieve similar images, while OCR text extracted from it queries the text collection. A spoken query is transcribed by the same Whisper model and thereafter treated as text.

**Cross-modal result merging.** Text-image similarities produced by CLIP are systematically lower in magnitude than text-text similarities, a documented property of contrastively-trained multimodal encoders known as the modality gap [13]. Merging result lists by raw similarity score would therefore rank images below text results irrespective of relevance. Results are consequently merged by **rank** rather than score, following rank-fusion practice [9]. See ADR-007. *[Data: the policy is reciprocal rank fusion with k = 60 (ADR-007), applied after each collection is gated on its own relevance floor, 0.3 for text (ADR-009) and 0.2 for images (ADR-010), and, for text-to-image results, after an OCR-corroboration gate (ADR-011).]*

**Top-K selection.** The K highest-ranked segments are passed to generation, with K = 5 by default. K is validated by ablation at 3, 5 and 10 (§8.4). Larger K supplies more context but degrades attention to mid-prompt content [23] and increases latency.

## 6. Generation Methodology

Retrieved segments are assembled into a numbered context block and supplied to a 4-bit quantized Llama 3.2 model served locally by Ollama. Quantization reduces memory from approximately 6 GB to approximately 2 GB with limited quality loss [29], [30], which is what permits execution on the target hardware.

The prompt instructs the model to answer using only the supplied context, to annotate each claim with the bracketed index of its supporting segment, and to state explicitly when the context does not contain the answer. Sampling temperature is set low (0.1 *[data: `LLM_TEMPERATURE`, src/core/config.py]*) to favour faithfulness over variety.

**Citation validation.** Emitted citation markers are checked against the set of supplied segments; markers not corresponding to a supplied segment are removed before display. Each citation is rendered alongside the retrieved text so that a user can verify support directly. This reduces but does not eliminate unsupported generation [22]; the limitation is stated rather than concealed.

## 7. Data Methodology

**Corpus.** 52 files spanning PDF, DOCX, PNG/JPG and WAV/MP3 *[data: 19 documents (13 PDF, 6 DOCX), 25 images, 8 audio clips; 22 files are synthetic campus-themed material generated for Chapter 6 and 30 were downloaded under open licences, see data/SOURCES.md]*. Sampling is **purposive rather than random**: the corpus is constructed to include cross-modal pairs — an image and a document addressing the same subject, and an audio recording discussing a subject also present in a document — so that cross-modal retrieval can be exercised rather than assumed.

**Gold-standard question set.** 25 text queries, 16 cross-modal queries and 4 negative controls were authored **before implementation began**, each annotated with the segment expected to be retrieved; six more cross-modal queries and four more negative controls (I9-I14, N5-N8) were added later, and each was committed to the repository **before** it was measured. Authoring the evaluation set in advance prevents the system from being tuned, consciously or otherwise, to the questions used to evaluate it.

> **Data note (2026-10-05): the sentence above overstates one thing; check it before it goes into the .docx.** Only the first five text questions (T1-T5) predate implementation. T6-T13 were added on 2026-09-29, and T14-T25, I5-I8, M3-M4, A3-A4 and N3-N4 on 2026-10-05, after the pipelines existed. What is true, and verifiable from the repository history, is that **each row was written before it was measured**, so none was adjusted to flatter a result. The set is one file, `data/gold_set.json`.

**Disclosed limitation.** The gold set was authored by the same team that designed the system, which risks question phrasing that unconsciously favours the chosen architecture. Mitigation: external participants author additional questions during the evaluation phase, and results on the external subset are **reported separately** from results on the internal set. ⟨FILL: confirm this happens in Chapter 12.⟩

**Ethics and licensing.** All corpus files are owned by the team or freely redistributable; no confidential material is included in the submitted corpus. Local execution means no corpus content is transmitted to any external service at any stage.

## 8. Evaluation Methodology

Metrics, targets and protocol are fixed in advance of any results.

### 8.1 Retrieval

| Metric | Definition | Target |
|---|---|---|
| **Recall@5** | Proportion of queries for which a correct segment appears in the top 5 | ≥ 0.80 |
| **MRR** | Mean of the reciprocal rank of the first correct segment | ≥ 0.65 |
| **Cross-modal Recall@5** | Recall@5 restricted to text-to-image queries | ≥ 0.70 |

Both Recall@5 and MRR are reported because they measure different properties: whether a relevant segment was found at all, and how highly it was ranked. Definitions follow standard retrieval evaluation practice [10].

### 8.2 Generation

Automatic overlap metrics such as BLEU and ROUGE are **not** used: a correct answer expressed in different wording is penalised, while a fluent but unsupported answer may score well. Answers are instead rated by humans on four dimensions following RAG evaluation practice [31]:

| Dimension | Rater's question |
|---|---|
| Faithfulness | Is every claim supported by the cited segment? |
| Answer relevance | Does the answer address the question asked? |
| Context relevance | Were the retrieved segments actually useful? |
| Citation correctness | Do citation markers point to segments that support the claims? |

Target: mean faithfulness ≥ 4/5.

**Rating protocol.** ⟨FILL: N ≥ 3⟩ raters, of whom at least one is external to the team. A written rubric defining each point on the 1–5 scale is agreed before rating begins. When configurations are compared, raters are not told which configuration produced which answer. At least ⟨FILL: N⟩ items are rated by two raters, and inter-rater agreement is reported. Results are reported as mean **and range**, since identical means can conceal very different distributions.

### 8.3 System performance

Measured on an Intel Core i5-14600K desktop with 15.7 GiB RAM, Windows 11 Home (build 10.0.26300) and Python 3.13.14; the language model ran on an NVIDIA GeForce RTX 5060 Ti GPU, while embedding, OCR and transcription ran on the CPU *[data: data/eval/performance_2026-10-05.txt; **CPU-only inference, the project's stated target, has not been measured**]*. Reported: indexing throughput per modality, retrieval latency, end-to-end latency, and peak memory. **Median and worst case** are reported rather than best case.

| Measure | Target |
|---|---|
| Retrieval latency | < 1 s |
| End-to-end latency | < 15 s |
| PDF indexing | ≥ 1 page/s |
| Whisper WER on test clips | < 15% |

### 8.4 Ablation studies

| Ablation | Variants | Question answered |
|---|---|---|
| Segment size | 150 / 300 / 600 words | Was the segmentation parameter chosen or guessed? |
| Top-K | 3 / 5 / 10 | Does additional context help or dilute? [23] |
| Cross-modal merge policy | Rank-based vs score-based | Does the modality gap [13] affect *our* corpus measurably? |

### 8.5 Offline verification

After indexing completes, all network interfaces are disabled. The system must then ingest a new file, answer a text query, answer an image query, transcribe and answer a spoken query, and render citations, with no functional degradation. The result of this test is reported as a pass/fail criterion for Objective O6.

## 9. Threats to Validity

| Type | Threat | Mitigation |
|---|---|---|
| **Internal** | Observed retrieval quality may reflect corpus properties rather than model capability | Ablations (§8.4) with the corpus held fixed across comparisons |
| **External** | Results obtained on 52 English files on one hardware configuration may not generalise to larger, multilingual or noisier corpora | Scope limits stated explicitly; no extrapolation claimed |
| **Construct** | Recall@5 measures retrieval, not answer usefulness | Both retrieval metrics and human answer ratings reported |
| **Conclusion** | With 25 text questions, a single item shifts Recall@5 by 0.04 (1/25), and with 22 cross-modal questions by 0.045; small differences are not meaningful | Sample size reported beside every figure; no claims made on differences within one item's width |
| **Bias** | Gold set authored by the system's designers | External participants author additional questions; those results reported separately (§7) |

---

## Pre-submission checklist

- [ ] Every `⟨FILL⟩` replaced
- [ ] Every section rewritten in the team's own words
- [ ] Every design decision names the alternatives considered and the reason for rejection
- [ ] At least three decisions cite literature
- [ ] All seven ADRs written in [`docs/decisions/`](../docs/decisions/)
- [ ] Metrics and targets fixed **before** any results were collected
- [ ] Human rating protocol specifies rater count, external participation, written rubric, blinding, and agreement reporting
- [ ] Threats-to-validity section present and honest
- [ ] Citation numbers renumbered in order of first appearance
- [ ] Every citation verified against the actual source

---

## Appendix — Measured data, dumped 2026-10-05 (not yet written up)

Raw material for the real report. Every figure below is a measured result with its source; nothing here is an estimate. The caveats are part of the data, so keep them when the numbers are used. Figures are from the grown corpus unless stated.

### A. Corpus (`data/README.md`, `data/SOURCES.md`)

| Set | Documents | Images | Audio |
|---|---|---|---|
| Synthetic starter set (Chapter 6, generated) | 3 (2 PDF, 1 DOCX) | 15 | 4 text-to-speech clips |
| Downloaded study materials (open licences, each licence verified at its source) | 16 (11 PDF, 5 DOCX) | 10 | 4 clips cut from CC BY-SA Spoken Wikipedia readings |
| **Total** | **19 (13 PDF, 6 DOCX): 623 chunks** | **25** | **8: 8 transcript chunks** |

Licences used: CC BY 4.0 / 2.0, CC BY-SA 2.0 / 3.0 / 4.0, CC0, public domain. NonCommercial material was excluded (for example OpenStax's Python book, CC BY-NC-SA).

### B. Gold set (`data/gold_set.json`)

25 text questions (T1-T13 on the synthetic corpus, T14-T25 on the downloaded one), 22 cross-modal (14 text-to-image, 4 image-to-document, 4 audio-topic), 8 negative controls (I9-I14 and N5-N8 were committed before they were measured). Wording overlap between a question and its answer chunk: median 50%, range 27-88%, never zero (`scripts/check_gold_set.py`); the earlier claim of "no shared keywords" was corrected. Dense retrieval against a BM25 keyword baseline **was** run (see C2 below).

### C. Retrieval (methodology section 8.1; `scripts/evaluate_retrieval.py`, `scripts/evaluate_cross_modal.py`)

| Metric | Result | n | Target |
|---|---|---|---|
| Recall@5 (text) | 1.00 | 25 | >= 0.80 |
| MRR (text) | 0.81 (reproducible; two earlier builds with different audio transcripts gave 0.75) | 25 | >= 0.65 |
| Cross-modal Recall@5, text-to-image only | **0.71 (10/14)**; 0.75 (6/8) on the first 8 rows | 14 | >= 0.70 (met by one question) |
| Cross-modal Recall@5, all 22 (text-to-image, image-to-document, audio-topic) | **0.82 (18/22)**; 0.88 (14/16) before I9-I14 were added | 22 | |

Context: the same 13 starter questions scored MRR 0.81 before the corpus grew and 0.77 after (on the first build); Recall@5 stayed 1.00. **MRR depended on the transcripts, and now does not:** the 8 audio chunks sit in the same index as the documents, and different Whisper transcripts moved 3 of the 25 questions between rank 1 and rank 2 (0.75 against 0.81). Transcripts are now cached by the audio's sha256 and committed (`data/transcripts/`), so any machine builds the same index; a build in each of the two environments gave byte-identical audio chunks. Before the image gate (ADR-011) cross-modal was 15/16; the gate costs one text-free photo and refuses all 6 out-of-corpus questions instead of 1. With 25 questions one item moves Recall@5 by 0.04.

### C2. Dense versus keyword search (`scripts/evaluate_bm25_baseline.py`, `data/eval/bm25_baseline_2026-10-05.txt`)

| Ranker (same 631 chunks, same hit rule) | Recall@5 | MRR |
|---|---|---|
| Dense (MiniLM), all 25 questions | 1.00 | 0.81 |
| BM25, all 25 | 0.84 | 0.66 |
| Dense, 16 questions with low wording overlap (<= 50%) | 1.00 | 0.82 |
| BM25, same 16 | 0.75 | 0.55 |
| Dense, 9 questions with higher overlap (> 50%) | 1.00 | 0.80 |
| BM25, same 9 | 1.00 | 0.87 |

Dense wins where the question's wording differs from the document's, and keyword search is as good (slightly better in rank) where it repeats it. Head to head: dense higher on 10 questions, BM25 on 5, tied on 10. An informational dense-plus-BM25 hybrid scored 0.96 / 0.82, no better than dense alone. Limits: BM25 without stemming; n = 25 (one question = 0.04).

### D. Answer quality (methodology section 8.2; `data/eval/answers_2026-10-05_*.txt`)

- **Rated by Claude (an AI assistant), not by an independent human, and not on the four-dimension rubric of section 8.2.** One 1-5 score per answer combining correctness and citation. Average 4.21 / 5 over 29 questions (25 text, 4 negatives): 18 rated 5, 4 rated 4, 3 rated 3, 3 rated 2, 1 rated 1. Positives 4.08, negatives 5.00. Target of section 8.2 (mean faithfulness >= 4/5) is therefore not yet evidenced by human raters.
- Citation correctness (a section 8.2 dimension), checked mechanically by looking for the fact in each cited chunk: 14 of 19 checked citations supported, 5 not (T7, T8, T13, T16, and one of T19's two). The Sources list shown to the user is always correct because it is built from chunk metadata, not model text.
- Every out-of-corpus question was refused (4/4 negatives; 3 with no model call).
- Of the 25 text questions, 2 are refused even though the right chunk is retrieved: T14 (the model answers from the right chunk alone but is distracted by four other chunks) and T20 (it cannot answer even from the right chunk alone). Both were tested and are generator limits, not parsing defects; a table-extraction fix was measured and rejected (`find_tables()` gave 176 mostly false detections in 259 pages).
- With images included (`--images`): 16 correct, 2 muddled, 7 with no usable answer, against 18 / 3 / 4 text-only.
- Single runs of a stochastic 3B model at temperature 0.1: a one-question difference is within noise.

### E. Relevance gating and partial ablations (ADR-009, ADR-011; `scripts/measure_relevance.py`)

| Variant, end to end, 25 positives and 6 negatives | Positives answered | Negatives refused |
|---|---|---|
| floor 0.30, TOP_K 5 (current) | 23 / 25 | 6 / 6 |
| floor 0.35 | 22 / 25 | 6 / 6 |
| prompt that tells the model to ignore unrelated passages | 22 / 25 | 6 / 6 |
| floor 0.35 and that prompt | 23 / 25 | 6 / 6 |
| TOP_K 3 | 22 / 25 | 6 / 6 |

No single text relevance floor separates answers from noise (weakest correct chunk 0.366, strongest noise chunk 0.383). Image gate: CLIP >= 0.2 alone refused 1 of 6 out-of-corpus questions; with OCR corroboration it refused 6 of 6 and kept 7 of 8 correct images. **Re-measured on 14 positives and 10 negatives (thresholds unchanged):** CLIP >= 0.2 alone keeps 13 of 14 correct images and refuses 3 of 10; the shipped corroboration rule keeps 11 of 14 and refuses 9 of 10. The three lost are two text-free photos (the gate's named cost) and one image CLIP never ranks in its top 5; the one leak (a lab-door-sign photo for a Turing Award question) is harmless end to end, the model refused it 3 of 3 times. ### E2. Ablations (methodology section 8.4; `scripts/run_ablations.py`, `data/eval/ablations_2026-10-05.txt`)

| Segment size / overlap | Chunks | Recall@5 | MRR | Answered of 25 (end to end) |
|---|---|---|---|---|
| 150 / 25 | 1199 | 0.92 | 0.69 | 18 |
| **300 / 50 (shipped)** | **631** | **1.00** | **0.81** | **22** |
| 600 / 100 | 335 | 0.96 | 0.83 | 23 (22 at a larger window) |

| Top-K | Retrieval | Answered of 25 (end to end) |
|---|---|---|
| 3 | Recall@3 0.96 | 23 |
| **5 (shipped)** | **Recall@5 1.00** | **23** |
| 10 | Recall@10 1.00 | 25 at the shipped window (inflated: 2 prompts were silently truncated), 23 at a larger one |

| Cross-modal merge | All 16 (first run) | Text-to-image | All 22 (re-run) | Text-to-image |
|---|---|---|---|---|
| **Rank-based RRF, k = 60 (shipped)** | **14 / 16** | **6 / 8** | **18 / 22** | **10 / 14** |
| Raw-score merge | 11 / 16 | 3 / 8 | 14 / 22 | 6 / 14 |

Findings: the shipped chunk size and top-K are at or near the best on this corpus (so they were chosen, not guessed, and the ablation says so); 600-word chunks and K = 10 push prompts past the model's 4096-token window; rank-based merging is worth 3 of 8 text-to-image questions against score-based merging (4 of 14 on the later, larger set; the modality gap, measured), and the four cross-modal rows that miss under rank merging (I3, I8, I10, I14) miss under score merging too, so they are limits of CLIP's ranking and of the image gate, not of the merge. One run of a stochastic model per variant, "answered" means not refused, n = 25. No shipped default changed.

### F. System performance (methodology section 8.3; `data/eval/performance_2026-10-05.txt`)

| Measure | Result (median / worst) | Target |
|---|---|---|
| Text retrieval latency, warm, n = 25 | 38 ms / 40 ms (19 ms median in an earlier run; one machine, so expect a factor of two) | < 1 s |
| Retrieval with images (CLIP + gate + merge), n = 8 | 232 ms / 254 ms | < 1 s |
| End-to-end answer, warm, n = 13 | **GPU: 2.9 s / 3.4 s. CPU-only (`OLLAMA_NUM_GPU=0`): 29.2 s / 47.2 s, target NOT met.** With `TOP_K = 3` on CPU: 14.1 s median / 30.3 s worst (median under the target only) | < 15 s |
| PDF indexing | 15.9 pages/s | >= 1 page/s |
| Image indexing (CLIP + OCR) | 2.7 images/s | |
| Audio transcription (Whisper `base`, CPU, transcript cache bypassed) | 9.5x to 10.2x real time (two runs) | |
| Whisper word error rate, 4 synthetic clips | mean 9.3% to 10.7% depending on the environment, worst 20.8% in both | < 15% (mean met, worst case not) |

Caveats that belong with these numbers: the first end-to-end figure is a GPU figure, and **the project's stated target hardware, a CPU-only laptop, was then measured and misses the 15 s target by about 2x** (retrieval and indexing are unaffected, since embedding, OCR and transcription run on the CPU either way; answer quality on CPU matched the GPU: 23 vs 22 of 25 answered, 8/8 negatives refused on both); the word error rate is on clean synthetic speech (which flatters it) with no number normalisation (which penalises it, "forty percent" heard as "40%"); peak memory was not measured; indexing figures include first-use model loading.

### G. Environment and reproducibility

`requirements.txt` installed exactly as pinned in a fresh Python 3.13.14 environment (Windows 11): installs, `pip check` clean, 114 tests pass plus 1 expected failure, in both that environment and the newer-library one. Real-component verification found three defects the fake-model test suite could not see: an unpinned `av` breaking real transcription, `chromadb 0.5.23`'s approximate search missing the best chunk for about 1 in 3 questions at default settings (fixed, ADR-012), and ChromaDB telemetry (disabled). **Python 3.11, the documented target, was not tested.** `chroma-hnswlib` has no Python 3.13 wheel and compiled from source.

### H. Not done yet (so the report does not claim it)

- Independent human rating, any external rater, inter-rater agreement (section 8.2).
- External testers and the feedback loop (they need the wired UI, which is still a scaffold).
- The ablations of section 8.4 (segment size, TOP_K = 10, merge policy).
- The offline verification of section 8.5 *through the UI with the network adapter off*. **Done without the UI** (`scripts/verify_offline.py`, `data/eval/offline_2026-10-05.txt`): ingest a new file, text query, image query, spoken query and citations all pass with every non-loopback connection blocked. It found that, with no offline setting, loading the embedding model makes 31 Hugging Face Hub attempts and stalls the first question about 49 s; `RAGNOVA_OFFLINE=1` removes both (0 attempts, 0.6 s).
- Peak memory, and a Python 3.11 check. (CPU-only latency was measured: see F; it misses the target.)
