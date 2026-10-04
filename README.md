# RAGNova

**An Offline Multimodal Retrieval-Augmented Generation (RAG) System**

> B.Tech CSE-AIML Project — built by a team of 2–3 students, learning every layer from first principles.

---

## What is this project?

RAGNova is a system that lets you **ask questions in plain language** and get **answers grounded in your own files** — PDFs, Word documents, images, and voice recordings — all running **completely offline** (no internet, no cloud APIs, no data leaving your machine).

Think of it as "ChatGPT over your own documents," except:

1. It runs on your laptop with a local LLM (no OpenAI/Anthropic API calls).
2. It understands more than text — you can search *images* with text, search *documents* with an image, or find files related to something *spoken in an audio clip*.
3. Every answer comes with **numbered citations** you can click to open the original source.

## The three pillars (from the problem statement)

| Pillar | What it means |
|---|---|
| **Unified Query Interface** | One chat/search box. Type a question, upload a DOCX/PDF, drag-and-drop an image, attach audio, or speak your query. |
| **Semantic & Cross-Modal Search** | Text→image ("email screenshot" finds the image), image→text (upload a screenshot, find related docs), audio→everything (play a clip, find related files). |
| **Citation Transparency** | Every answer cites sources `[1][2]`. Expand a citation to open the document, see the transcript segment, or inspect image metadata. |

## Repository map

```
RAGNova/
├── README.md              ← you are here
├── docs/
│   ├── ROADMAP.md         ← chapter breakdown + 14-day timeline + team split
│   ├── GLOSSARY.md        ← every jargon term, explained simply
│   ├── decisions/         ← Architecture Decision Records (ADR-001 … 010)
│   ├── feedback-log.md    ← human feedback + what we changed (Ch12)
│   └── chapters/          ← one teaching doc per chapter (read in order)
├── src/
│   ├── core/              ← SHARED — schemas.py, config.py (Ch4 §1.3, Ch5 §3)
│   ├── pipelines/         ← per-track code, from Chapter 6 onward (rag/ = Ch10 + Ch12 cross-modal retrieval)
│   └── ui/                ← Ch11: the chat page's non-drawing half (uploads, citations, feedback)
├── tests/                 ← test_contract.py — the interface contract, enforced (Ch4 §1.6)
├── scripts/               ← verify_setup.py and other one-off setup scripts
├── data/                  ← sample corpus + gold-standard eval set (documents/images/audio)
├── requirements.txt       ← pinned dependencies (Ch5 §4.2)
├── .env.example           ← committed config template — copy to .env (not committed)
└── reports/               ← synopsis, PPTs, methodology drafts, mid-term report
```

**Try it now:** with `.venv` activated (Ch5 §5.1), run `pytest tests/test_contract.py -v` — this passes today, before a single ingestion pipeline exists, because it tests the *shape* of data every track promises to produce, not any pipeline's internals.

## How to use this repo as a learner

**Never installed VS Code? Never opened a terminal?** Start at [Chapter 0 — Getting Started From Absolute Zero](docs/chapters/ch00-getting-started-from-zero.md). It assumes nothing, takes ~90 minutes, and gets your machine ready plus decodes the maths notation used later.

**Tip:** these docs are Markdown. Open one in VS Code and press `Ctrl + Shift + V` to read it properly formatted instead of as raw text.

1. Start with [docs/ROADMAP.md](docs/ROADMAP.md) — the full plan.
2. Read chapters in order in [docs/chapters/](docs/chapters/). Each chapter has:
   - **Learn** — the concept, explained from zero.
   - **Decide** — the design choices we face, and why we pick what we pick.
   - **Build** — hands-on implementation steps.
   - **Check** — how to verify it works (and what "working" looks like).
3. Unknown word? Check [docs/GLOSSARY.md](docs/GLOSSARY.md).

## Team

Track assignments finalized in [Chapter 4](docs/chapters/ch04-timeline-and-team-split.md) — names below are placeholders; each track is one person's module, chosen to minimise coupling between tracks (Ch4 §1.2), not divided by rough workload.

| Member | Track | Owns |
|---|---|---|
| ⟨Member 1⟩ | A — Text pipeline + RAG core | Chapters 6, 7, 10: parsing, chunking, text embeddings, retrieval, generation, citations |
| ⟨Member 2⟩ | B — Vision + audio pipelines | Chapters 8, 9: CLIP image search, OCR, Whisper transcription |
| ⟨Member 3⟩ | C — Interface + integration | Chapters 11, 12: Streamlit app, wiring pipelines together, testing |

*2-member fallback:* Track C dissolves; its work splits between Tracks A and B (Ch4 §3.6).
