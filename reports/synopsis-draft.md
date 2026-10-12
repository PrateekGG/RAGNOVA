# PROJECT SYNOPSIS — DRAFT

> **⚠️ READ BEFORE USING THIS FILE**
>
> This is a **starting draft, not a submission.** Two reasons you must rewrite it in your own words:
>
> 1. Your department's format, margins, and section names may differ — ask your guide for the official template and match it.
> 2. **You will be questioned on every sentence in the viva.** Text you did not write is text you cannot defend. Work through line by line, rephrase into language you would naturally use, and delete anything you could not explain if interrupted mid-sentence.
>
> `⟨FILL⟩` marks a team-specific detail. Every one must be replaced before submission.
>
> Guidance for each section is in [Chapter 2 §1.3](../docs/chapters/ch02-synopsis-and-presentation.md). Target length **1,000–1,500 words / 2–3 pages**.

---

## Cover page

| Field | Value |
|---|---|
| **Project Title** | RAGNova: An Offline Multimodal RAG System for Unified Semantic Retrieval Across Documents, Images, and Speech |
| **Degree** | Bachelor of Technology in Computer Science & Engineering (Artificial Intelligence & Machine Learning) |
| **Institution** | ⟨FILL: college / university name⟩ |
| **Department** | ⟨FILL: department⟩ |
| **Academic Year / Semester** | ⟨FILL⟩ |
| **Team Members** | ⟨FILL: Name 1 — Roll No.⟩ · ⟨FILL: Name 2 — Roll No.⟩ · ⟨FILL: Name 3 — Roll No.⟩ |
| **Project Guide** | ⟨FILL: name and designation⟩ |
| **Date of Submission** | ⟨FILL⟩ |

---

## 1. Abstract

*(150–200 words. Write this LAST. Five moves: context → problem → approach → outcome → significance. No citations, no figure references, no undefined acronyms. See Ch 2 §1.5.)*

> Personal and organisational knowledge is increasingly distributed across heterogeneous formats — text documents, captured images and screenshots, and voice recordings. Existing retrieval tools address this poorly: they match keywords rather than meaning, they are siloed by modality so that a textual query cannot retrieve an image, and the capable semantic systems available today depend on cloud services, which is unacceptable for confidential material and unusable without connectivity. This work proposes RAGNova, a fully offline multimodal Retrieval-Augmented Generation system that ingests PDF, DOCX, image, and audio files into a unified semantic index. Documents and speech transcripts are embedded using a sentence-transformer model, while images are embedded using a contrastive vision-language model that places images and text in a shared vector space, enabling cross-modal retrieval in three directions: text-to-image, image-to-document, and audio-to-all. Retrieved passages are supplied to a locally executed large language model, which generates answers constrained to that evidence and annotated with numbered citations resolving to the source file, page, or timestamp. Retrieval quality is evaluated using Recall@5 and Mean Reciprocal Rank, and answer quality through human faithfulness ratings. All components execute on commodity hardware without network access.

**Word count:** ⟨FILL — aim 150–200⟩

---

## 2. Introduction

*(150–200 words. General → specific in four sentences. Do NOT write a history of AI. See Ch 2 §1.3 Section 3.)*

> The volume of unstructured information held by individuals and small organisations has grown faster than the tools available to search it. A student accumulates lecture recordings, annotated PDFs, and screenshots of announcements; a researcher holds papers, figures, and interview audio; a professional retains contracts, scanned receipts, and meeting recordings. This material is semantically related but technically fragmented, and the tools available to search it operate on filenames and exact keywords rather than meaning.
>
> Recent advances have made meaning-based retrieval practical. Dense vector embeddings represent text as points in a geometric space where proximity corresponds to semantic similarity, and contrastive vision-language pretraining has extended this principle so that images and text occupy a single shared space. In parallel, quantization techniques have reduced large language models to a size that executes on consumer hardware, and Retrieval-Augmented Generation has established a method for grounding model outputs in supplied evidence rather than parametric memory.
>
> These developments are individually mature but are rarely combined. In particular, they are rarely combined under an offline constraint, which is precisely the setting in which confidential material must be searched.

---

## 3. Problem Statement

*(100–150 words. Name a specific deficiency and its consequence — not a general area of interest. Compress Ch 1 §2.1.)*

> Existing retrieval systems fail heterogeneous personal and organisational corpora on three specific counts. First, they are predominantly **lexical**, matching character strings rather than meaning, and therefore fail on synonymy and paraphrase. Second, they are **siloed by modality**: images are searchable only by filename or manual tags, and speech is not searchable at all without prior manual transcription. Third, systems that do offer semantic and generative capability are **cloud-dependent**, imposing privacy exposure, recurring per-query cost, and a hard dependence on connectivity.
>
> The consequence is that a user who possesses the answer to their own question, within their own files, frequently cannot retrieve it — and cannot retrieve it at all when the relevant content is an image or a recording. No unified, offline system currently accepts a natural-language question and returns a synthesised, citation-backed answer drawn from documents, images, and audio alike.

---

## 4. Objectives

*(100–150 words. Numbered, verifiable, testable verbs. From Ch 1 §2.3.)*

The project aims to design, implement, and evaluate a system that will:

1. **Ingest and index** PDF, DOCX, image (PNG/JPG), and audio (WAV/MP3) files into a unified vector store, preserving provenance metadata for every indexed segment.
2. **Retrieve semantically**, matching queries by meaning rather than keyword, achieving Recall@5 ≥ 0.80 and Mean Reciprocal Rank ≥ 0.65 on a predefined evaluation set.
3. **Support cross-modal retrieval** in three directions — text-to-image, image-to-document, and audio-to-all — achieving cross-modal Recall@5 ≥ 0.70.
4. **Generate grounded answers** using a locally executed large language model constrained to retrieved context, annotated with numbered citations resolving to source file and location, with mean human faithfulness ≥ 4/5.
5. **Provide a unified interface** accepting typed queries, uploaded documents, drag-and-dropped images, attached audio, and spoken input, with expandable citations.
6. **Operate entirely offline** after initial setup, verified by a disconnected-network acceptance test.
7. **Document the system** to a standard permitting independent reproduction and extension.

---

## 5. Literature Review

*(150–200 words. Three to five sentences, ending by naming THE GAP. Full review is Chapter 3 — do not spend it here. See Ch 2 §1.3 Section 6.)*

> The transformer architecture [1] underpins the models used throughout this work. Sentence-BERT [2] adapted transformer encoders to produce sentence-level embeddings directly comparable by cosine similarity, making dense semantic retrieval practical, and Dense Passage Retrieval [3] demonstrated that such embeddings outperform lexical methods on open-domain question answering. Lewis et al. [4] introduced Retrieval-Augmented Generation, combining a dense retriever with a generative model to ground outputs in retrieved evidence and thereby reduce unsupported generation.
>
> Extension beyond text was enabled by CLIP [5], which trains image and text encoders jointly on large-scale caption data such that both modalities occupy a shared embedding space, permitting cross-modal retrieval without manual annotation. Whisper [6] provides robust multilingual speech recognition, allowing spoken content to be converted into the textual representation that existing retrieval methods already handle. Efficient similarity search at scale is addressed by hierarchical navigable small-world graphs [7].
>
> These components are individually mature, but published systems combine them only partially: multimodal RAG implementations are typically text-centric with images treated as an adjunct, cross-modal retrieval is rarely bidirectional, and almost all assume cloud-hosted embedding and generation services. **The gap addressed by this project is the integration of all four modalities under a strict offline constraint, with citation transparency as a first-class requirement rather than a presentation-layer addition.**

---

## 6. Proposed Methodology / System Architecture

*(250–350 words — the largest section. INSERT THE ARCHITECTURE DIAGRAM HERE. Be concrete: name the tools and the data flow. See Ch 2 §1.3 Section 7 and §2.3.)*

**⟨INSERT FIGURE 1: System Architecture — see Ch 2 §3.3⟩**

*Figure 1: RAGNova system architecture. Four ingestion pipelines converge on a unified vector store; queries are embedded, retrieved, and passed to a local language model which generates cited answers.*

The system operates in two phases: an offline **indexing** phase executed once per file, and an online **query** phase executed per question.

**Indexing.** Each file is dispatched by type to a dedicated pipeline. PDF and DOCX documents are parsed to plain text using PyMuPDF and python-docx respectively, retaining page numbers. Extracted text is segmented into overlapping chunks of approximately 300 words, with 50 words of overlap so that passages spanning a boundary remain intact in at least one chunk. Audio files are transcribed using faster-whisper, which emits text segments with start and end timestamps; transcripts are chunked identically to documents, retaining those timestamps. Images are processed along two complementary paths: a CLIP vision encoder produces a semantic embedding of the image content, while Tesseract OCR extracts any literal text present, which is indexed as ordinary text. Every resulting chunk is embedded and stored with provenance metadata comprising the source path, modality, and page number or timestamp.

**Storage.** Two ChromaDB collections are maintained. The first holds 384-dimensional sentence-transformer embeddings of all textual content, including transcripts and OCR output. The second holds 512-dimensional CLIP embeddings of images. Separation is required because the two models produce vectors in different spaces and dimensionalities, and CLIP's text encoder truncates inputs at 77 tokens, making it unsuitable for document-length passages.

**Query.** A textual query is embedded by both models and used to search both collections. An uploaded image is embedded by the CLIP vision encoder to retrieve visually and semantically similar images, while its OCR text queries the document index. A spoken query is transcribed by the same Whisper model and thereafter treated as text. Because CLIP text-image similarities are systematically lower in magnitude than text-text similarities, results from the two collections are merged by rank rather than by raw score.

**Generation.** The highest-ranked chunks are assembled into a numbered context block and supplied to a quantized Llama 3.2 model served locally by Ollama, with instructions to answer solely from the provided evidence, to cite each claim by its bracketed index, and to state explicitly when the evidence is insufficient. Generation temperature is kept low to favour faithfulness. Citation markers are validated against the supplied chunks before display, and each citation expands in the interface to reveal the source document, page or timestamp, and the exact retrieved text.

---

## 7. Tools & Technologies

*(75–100 words. Group by layer. State the offline justification. See Ch 2 §1.3 Section 8.)*

| Layer | Technology | Rationale |
|---|---|---|
| Language model | Ollama serving Llama 3.2 (3B, 4-bit quantized) | Executes locally on CPU; ~2 GB memory footprint |
| Text embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-d) | Compact, CPU-efficient, strong retrieval quality |
| Cross-modal embeddings | OpenCLIP ViT-B/32 (512-d) | Shared image-text vector space enables cross-modal search |
| Speech recognition | faster-whisper | Offline transcription with segment-level timestamps |
| Vector store | ChromaDB | Embedded database; no server; metadata filtering |
| Document parsing | PyMuPDF, python-docx | Robust text and page extraction |
| OCR | Tesseract | Extracts literal text from screenshots |
| Interface | Streamlit | Multimodal input handling in pure Python |
| Language | Python 3.11 | Ecosystem support for all of the above |

All components are open-source and execute locally; no component requires network access after initial installation. *[Data, 2026-10-05: development and evaluation were performed on a desktop with an Intel Core i5-14600K, 16 GB RAM and Windows 11; the language model ran on an NVIDIA RTX 5060 Ti GPU, and embedding, OCR and transcription ran on the CPU. CPU-only operation, the stated target, WAS measured (2026-10-05, re-measured 2026-10-07 after an engine update and a connection fix): retrieval, indexing and answer quality are unaffected; end-to-end answer latency on a CPU-only desktop processor is a median of 15.8 s against the 15 s target at the defaults and 9.2 s with the documented `TOP_K=3` profile. A laptop has not been measured, so keep the "commodity hardware" claim qualified accordingly.]*

---

## 8. Expected Outcomes

*(100–150 words. Include numeric evaluation targets. See Ch 2 §1.3 Section 9.)*

> The project will deliver a working desktop application providing a single conversational interface over a heterogeneous corpus of 50–200 files. The system will accept typed, uploaded, image, and spoken input; retrieve semantically across all indexed modalities; and produce answers annotated with citations that expand to reveal the originating document page, transcript segment, or image metadata.
>
> Performance will be evaluated against a gold-standard question set constructed before implementation. Retrieval targets are Recall@5 ≥ 0.80 and Mean Reciprocal Rank ≥ 0.65 for textual queries, and Recall@5 ≥ 0.70 for cross-modal queries. Answer quality will be assessed by human raters on faithfulness, relevance, and citation correctness, targeting a mean of at least 4 out of 5. Offline operation will be verified by executing the complete query workflow with all network interfaces disabled. Accompanying deliverables comprise full technical documentation and a recorded user-feedback log.

---

## 9. Timeline

*(50–75 words plus the table. Names in the ownership column. See Ch 2 §1.3 Section 10.)*

| Days | Phase | Activities | Owner |
|---|---|---|---|
| 1 | Ideation | Problem identification, objectives, scope | All |
| 2 | Proposal | Synopsis and presentation | All |
| 3 | Research | Literature review, methodology | All |
| 4 | Planning | Timeline, module decomposition, ownership | All |
| 5 | Setup | Environment, local model deployment, shared schema | All |
| 6–7 | Core text pipeline | Document parsing, chunking, embeddings, vector store | ⟨FILL: Member 1⟩ |
| 8–9 | Vision pipeline | CLIP image indexing, OCR, cross-modal retrieval | ⟨FILL: Member 2⟩ |
| 9–10 | Audio pipeline | Whisper transcription, timestamped chunking | ⟨FILL: Member 2⟩ |
| 10–11 | RAG core | Retrieval, prompt construction, generation, citations | ⟨FILL: Member 1⟩ |
| 11–12 | Interface | Unified multimodal query interface | ⟨FILL: Member 3⟩ |
| 12–13 | Integration & testing | End-to-end testing, evaluation, user feedback | All |
| 14 | Reporting | Mid-term report | All |

Modules are decoupled through a shared chunk schema fixed on Day 5, permitting the three pipelines to be developed in parallel and integrated on Day 12.

---

## 10. References

*(IEEE format, numbered in order of first citation. 5–10 entries for a synopsis.)*

> **⚠️ VERIFY EVERY ENTRY against the actual paper before submitting.** Venue names, volumes, and years must be exact — an incorrect citation is worse than a missing one. Confirm each on the publisher's site, arXiv, or Google Scholar. **Read at least the abstract and conclusion of everything you cite**; a panel may ask what a paper says.

```
[1] A. Vaswani et al., "Attention Is All You Need," in Proc. Advances in Neural
    Information Processing Systems (NeurIPS), 2017.

[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
    BERT-Networks," in Proc. Conf. Empirical Methods in Natural Language Processing
    (EMNLP-IJCNLP), 2019.

[3] V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question
    Answering," in Proc. Conf. Empirical Methods in Natural Language Processing
    (EMNLP), 2020.

[4] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks," in Proc. Advances in Neural Information Processing Systems (NeurIPS),
    2020.

[5] A. Radford et al., "Learning Transferable Visual Models From Natural Language
    Supervision," in Proc. Int. Conf. Machine Learning (ICML), 2021.

[6] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision,"
    in Proc. Int. Conf. Machine Learning (ICML), 2023.

[7] Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest
    Neighbor Search Using Hierarchical Navigable Small World Graphs," IEEE Trans.
    Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824–836, 2020.

[8] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts,"
    Transactions of the Association for Computational Linguistics, vol. 12, 2024.

[9] Chroma, "Chroma Documentation." [Online]. Available: https://docs.trychroma.com
    [Accessed: ⟨FILL: date⟩].

[10] Ollama, "Ollama Documentation." [Online]. Available: https://ollama.com
     [Accessed: ⟨FILL: date⟩].
```

---

## Pre-submission checklist

- [ ] Every `⟨FILL⟩` replaced
- [ ] Entire document rewritten in the team's own words
- [ ] Word count 1,000–1,500 (excluding cover page and references)
- [ ] Abstract written last; contains all five moves; no citations or undefined acronyms
- [ ] Figure 1 inserted, arrows labelled, readable at print size
- [ ] Every reference verified against the actual source, and read
- [ ] Tense consistent throughout (future *or* present for proposed work — pick one)
- [ ] No buzzwords any team member cannot define
- [ ] One person has done a single editing pass for consistent voice
- [ ] Formatted to the department's official template
- [ ] Guide's required file format confirmed (Word or PDF — ask, do not assume)
