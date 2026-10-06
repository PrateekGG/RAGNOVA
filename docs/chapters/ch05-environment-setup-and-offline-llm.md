# Chapter 5 — Environment Setup & Offline LLM (Day 5)

> **Deliverables today:** a working, verified Python environment; Ollama installed, running, and serving a real local model over its API; the **real** `src/core/schemas.py` implementing Chapter 4's strawman; the **real** contract test from Chapter 4 §1.6/§4.4, passing; a pinned `requirements.txt`; a shared config module; and a project skeleton every track builds inside starting tomorrow.
>
> **Prerequisites:** [Chapter 0](ch00-getting-started-from-zero.md) (Python, terminal, venv literacy) and [Chapter 4](ch04-timeline-and-team-split.md) §4 (the interface-contract requirements this chapter implements).
>
> **Companion file:** [Chapter 5A — Systems References](ch05a-systems-references.md) — the specifications and documentation behind the packaging, model-format, and API material in Part 1–2.
>
> **The framing for today.** Every chapter before this one produced documents. This one produces code — the first code in the repository, and specifically the *only* code every track is required to depend on. Get today wrong and the mistake compounds for nine days; get it right and the rest of the project rests on ground that doesn't move. This chapter is written so that everything it claims about the code is checked — every code block below was actually run against a real Python interpreter and a real (or genuinely absent) Ollama instance while this chapter was written, and the outputs shown are the outputs that were actually produced, not their expected shape.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: Packaging & Environments** | What pip's resolver actually does; wheels vs. source; why pinning matters more for ML than for most software; Windows-specific install failure modes. | ~50 min |
| **Part 2 — LEARN: How Ollama Actually Works** | GGUF format and memory-mapping; Ollama's architecture and API surface; the generation parameters that matter; a worked KV-cache memory calculation. | ~50 min |
| **Part 3 — LEARN: From Strawman to Real Contract** | Why a dataclass, not Pydantic; a field-by-field walkthrough of the real schema; why the contract test is designed to reject bad input, not just accept good input. | ~40 min |
| **Part 4 — DECIDE** | The exact Python version, the pinning policy, the config strategy, the directory layout, the exact models — all with reasons. | ~25 min |
| **Part 5 — BUILD** | The actual Day-5 setup, in order, with real verified output at every step. | ~3 hours |
| **Part 6 — CHECK** | Rubric, question bank, an ML-specific troubleshooting table, completion checklist. | ~35 min |

**Learning outcomes.** You will be able to: explain what a Python wheel is and why it avoids compiling code on your machine; explain why ML projects pin dependency versions more strictly than most software; explain what GGUF and memory-mapping buy a locally-run LLM; call Ollama's HTTP API directly and explain what each generation parameter controls; compute an approximate KV-cache memory footprint by hand; explain the dataclass-over-Pydantic trade-off and defend it; read and extend `validate_chunk()`; explain why a contract test must contain tests that are *supposed* to fail; and diagnose the specific installation failures this project's dependencies are prone to on Windows.

---

# Part 1 — LEARN: Packaging & Environments, Properly

Chapter 0 §A.3–A.4 got you a working `python` and the concept of a virtual environment. This section goes one level deeper, into exactly the mechanics that will matter the moment `pip install -r requirements.txt` does something confusing — which, with an 8-package ML stack across three different laptops, it eventually will.

## 1.1 What `pip install` actually does

When you run `pip install sentence-transformers`, four things happen, in order, and knowing them turns a cryptic failure into a diagnosable one.

**1. Dependency resolution.** `sentence-transformers` itself depends on other packages (`torch`, `transformers`, `numpy`, ...), each of which may depend on others, each with version constraints (`torch>=1.11.0`, say). Pip's resolver — since pip 20.3, a genuine **backtracking** algorithm — must find one version of every package in this graph that simultaneously satisfies every constraint. This is provably a hard combinatorial problem in general; in practice it usually resolves in seconds, but when it can't, pip's error message ("ResolutionImpossible") lists the conflicting constraints, and reading that list, not rerunning the same command hoping it works, is the correct next step.

**2. Fetching.** For each resolved package, pip downloads a distribution — usually a **wheel** (§1.2 below) — from PyPI (or a configured mirror).

**3. Building** (only if no compatible wheel exists for your platform). Pip falls back to downloading the source distribution and compiling it locally. This is where installs fail on a fresh Windows machine that lacks a C compiler — see §1.4.

**4. Installing.** The package's files are copied into your environment's `site-packages` directory — a folder that is, itself, the entire mechanism behind "activating a virtual environment": activation prepends that venv's `site-packages` to Python's module search path (`sys.path`), so `import torch` finds *this* environment's torch, not any other one installed elsewhere on the machine. This is worth stating explicitly because it demystifies venvs completely: **a virtual environment is not a sandbox in any deep OS sense — it is a folder of packages and a modified search-path.** That is the entire mechanism, and it's why deleting a venv folder and recreating it is always safe: nothing outside that folder depends on it.

## 1.2 Wheels vs. source distributions

A **wheel** (`.whl`) is a pre-built, ready-to-copy package — for a pure-Python library, just the `.py` files in the right layout; for a library with compiled components (torch, numpy), a wheel contains **already-compiled binary code** for one specific combination of Python version, operating system, and CPU architecture, encoded directly in its filename:

```
torch-2.5.1-cp311-cp311-win_amd64.whl
      │      │     │        └── platform: 64-bit Windows
      │      │     └── ABI tag (matches the Python build)
      │      └── cp311 = built for CPython 3.11
      └── package version
```

When pip finds a wheel matching your exact Python version and platform, installation is just decompressing files — seconds, no compiler needed. **This is precisely why Chapter 0 insisted on Python 3.11 specifically**: PyTorch and its ecosystem publish wheels for a defined set of Python versions, typically lagging 3–6 months behind a new Python release. Running the newest Python available is not "more correct" for this project — it is a real, avoidable failure mode.

> **This is not hypothetical.** While writing this chapter, the machine used to verify its code was running Python 3.14.5 — released well after this project's ML dependencies had wheels for it. `pip show pytest` worked (pytest is pure Python, platform-independent), but a project depending on `torch` on that exact interpreter would be at real risk of pip falling back to a source build, or failing outright with no compatible distribution found. This is precisely the failure mode this section warns about, encountered while writing the warning.

If no matching wheel exists, pip falls back to the source distribution (`.tar.gz`) and attempts to **build** it locally — which for a library with compiled C/C++/Rust extensions requires a working compiler toolchain most Windows machines do not have installed by default. The error in this case is long, mentions `Microsoft Visual C++`, and is intimidating; the actual fix (§6.3's troubleshooting table) is almost always "use a Python version with published wheels" rather than "install a C++ compiler," because the latter is a much bigger, riskier fix for a 9-day-schedule team to attempt.

## 1.3 Why ML dependencies deserve stricter pinning than most software

Ordinary software has relatively stable APIs; ML libraries — sentence-transformers, transformers, chromadb — are under fast, active development, and **breaking changes between minor versions are common, not exceptional.** A function's default behaviour, a model's expected input shape, or even a class's constructor signature can change between releases that look adjacent (`3.2.0` → `3.3.0`).

This is the concrete reason `requirements.txt` (Part 5 §5.4) pins **exact** versions (`sentence-transformers==3.3.1`) rather than minimum versions (`sentence-transformers>=3.3.1`). The difference matters specifically because of Chapter 3 §3.8's reproducibility requirement: a Recall@5 number measured in Chapter 12 must still be reproducible when the report is read weeks later, on a different machine, by an examiner running `pip install -r requirements.txt` fresh. An unpinned dependency makes that promise unverifiable — if the numbers don't match, is it the code, the data, or a library that silently changed behaviour underneath you? Pinning removes one entire axis of uncertainty.

## 1.4 Windows-specific failure modes worth knowing about before they happen

| Symptom | Cause | Fix |
|---|---|---|
| `error: Microsoft Visual C++ 14.0 or greater is required` | No compatible wheel found; pip is trying to build a compiled package from source | First check you're on Python 3.11 (§1.2) — this alone fixes most cases. If it persists, install "Build Tools for Visual Studio" (a genuine fix, but a large download — try the version fix first) |
| `FileNotFoundError` / path-too-long errors deep in a package's own files | Windows historically limits paths to 260 characters; ML packages with deeply nested module trees can exceed it | Chapter 0 §A.3 already asked you to enable "Disable path length limit" during the Python install — if you skipped it, re-run the installer's Modify option and enable it now |
| `pip install` appears to hang for minutes with no output | Not hanging — `torch`'s CPU wheel is several hundred MB; slow connections make this look frozen | Wait; check Task Manager's network usage if genuinely unsure. Never `Ctrl+C` mid-download and retry immediately — let it finish or fail |
| Two team members get different results from "the same" code | One member's `pip install` resolved slightly different transitive dependency versions because `requirements.txt` had a loose constraint somewhere | This is exactly what §1.3 pins against — audit `requirements.txt` for any `>=` that should be `==` |
| `ollama` command not found after installing | Installer didn't add Ollama to PATH, or the terminal was open before installing (same class of bug as Chapter 0 §A.3's Python PATH issue) | Close and reopen your terminal; if still missing, reinstall and confirm the installer completed |

---

# Part 2 — LEARN: How Ollama Actually Works

Chapter 1 §1.1.5 and §1.9 explained *why* a quantized model fits in laptop RAM. This section explains the actual machinery that serves it to your code.

## 2.1 GGUF: the file format, and why it enables something specific

Ollama (via its `llama.cpp`-derived backend) stores model weights in **GGUF** (GPT-Generated Unified Format) — a binary file containing a metadata header (architecture, quantization scheme, tokenizer data) followed by the model's tensors laid out contiguously on disk.

The property that matters most for us is **memory-mapping (`mmap`)**. Rather than reading the entire model file into RAM before inference can begin, the operating system maps the file's contents directly into the process's virtual address space and loads pages from disk **lazily, on first access**. Two direct, practical consequences:

1. **"Loading" a 2 GB model file does not necessarily mean 2 GB is immediately read from disk.** The OS's page cache means portions of the file already read (by a previous run, or by another process) are served from RAM without a disk read at all — which is part of why a second run of the same model is typically faster to start than the first.
2. **The very first inference request after loading is often measurably slower** than subsequent ones, because the specific tensor pages needed for that computation are being paged in from disk for the first time. If your own testing shows a slow first response and fast subsequent ones, this is why — it is not a bug, and Chapter 12's latency measurements should account for it (report a "warm" latency, and separately note the cold-start cost, rather than let one confusingly named number hide both).

## 2.2 Ollama's architecture and API surface

Ollama runs as a **background service** (a long-lived process, started by `ollama serve` or launched automatically by the desktop app) that listens on `http://localhost:11434`. This is the concrete mechanism behind Chapter 1 §B.3's "local API is still an API" point — your Python code and the model-serving process are two separate programs on the same machine, talking over ordinary HTTP, the same protocol used for internet requests, just never leaving `localhost`.

The endpoints this project actually uses:

| Endpoint | Purpose |
|---|---|
| `GET /api/tags` | List models currently pulled and available locally |
| `POST /api/pull` | Download a model (what `ollama pull llama3.2:3b` does under the hood) |
| `POST /api/generate` | One-shot text generation from a prompt — **what Chapter 10's RAG core calls** |
| `POST /api/chat` | Multi-turn conversation with role-tagged messages — an alternative to `/api/generate` worth knowing exists, not used by default here since our prompts are single-shot (retrieved context + question, not an ongoing conversation) |
| `POST /api/embeddings` | Generate an embedding vector from a model that supports it — **not what we use for text embeddings** (Chapter 1 §1.5 already committed to sentence-transformers for that); listed here so you know it exists and don't accidentally reach for it |

**"`ollama pull`" is closer to a container-image pull than a file download.** A model is described by a manifest listing several content-addressed layers (the weights, a template, license text, parameters) — Ollama downloads only the layers it doesn't already have, and layers shared between model variants (say, two different quantizations of the same base model) are deduplicated on disk. This is why pulling a second, related model is often faster than the first.

## 2.3 Generation parameters that matter to this project

| Parameter | What it controls | Our setting, and why |
|---|---|---|
| `temperature` | Sharpness of the next-token probability distribution (Ch1 §1.1.3) | `0.1` (near-deterministic) — faithfulness over variety in a citation-grounded system |
| `num_ctx` | The context window size, in tokens, the model is configured to use | Set deliberately (Part 4 §4.5) — larger costs more KV-cache memory (§2.5 below) for headroom we may not need, given Chapter 1 §1.3's "few good chunks" argument |
| `num_predict` | Maximum tokens to generate before stopping | Bounded, so a malformed prompt can't cause an unbounded generation that blocks the interface |
| `top_p`, `top_k` | Alternative/additional ways to restrict sampling to the most probable tokens | Left at Ollama's defaults; redundant with a near-zero temperature, not worth tuning for this project |

## 2.4 Two ways to talk to Ollama — and why our code uses the second one

`ollama run llama3.2:3b` drops you into an interactive terminal REPL — excellent for manually sanity-checking a model's behaviour, useless for a program that needs to send one prompt, get one structured response, and continue. Every line of code this project writes talks to the **HTTP API** instead (§2.2's table), exactly the way `scripts/verify_setup.py` (Part 5 §5.5) does — because a RAG pipeline is a program calling a service, not a human typing into a terminal.

## 2.5 Worked example — estimating KV-cache memory

Chapter 1 §1.9.2 stated the KV cache "adds a few hundred MB to a couple of GB" without deriving the number. Here is the derivation, so the claim is checkable rather than asserted.

The KV cache stores, for every token already processed, the attention **key** and **value** vectors at every layer — recomputing them from scratch for every new token would make generation far slower than it already is. Its size follows directly from the model's architecture:

```
KV cache bytes = 2 (K and V) × num_layers × num_kv_heads × head_dim × context_length × bytes_per_element
```

Using **approximate, representative** architecture figures for a 3B-parameter Llama-family model with grouped-query attention (verify the exact figures for your specific model against its published config before quoting them as fact — the arithmetic technique, not these specific numbers, is the point of this worked example):

```
num_layers      ≈ 28
num_kv_heads    ≈ 8      (grouped-query attention uses fewer KV heads than
                          query heads — this is a deliberate efficiency
                          technique in modern architectures, precisely to
                          shrink this cache)
head_dim        ≈ 128
bytes_per_elem  = 2       (FP16 — the common default for the KV cache even
                          when weights themselves are 4-bit quantized;
                          llama.cpp can also quantize the KV cache itself,
                          which would roughly halve this estimate)

Per-token cost = 2 × 28 × 8 × 128 × 2 bytes
              = 114,688 bytes
              ≈ 112 KB per token

At context_length = 4096 tokens:  112 KB × 4096 ≈ 448 MB
At context_length = 8192 tokens:  112 KB × 8192 ≈ 896 MB
```

**Three consequences worth internalising, not just the number itself:**

1. **KV-cache memory scales linearly with context length**, which is a direct, quantitative reason Chapter 1 §1.3's "retrieve few, good chunks" argument matters operationally, not just for lost-in-the-middle accuracy (Ch3A Theme 7) — a bigger `num_ctx` is not free even when the model handles long context well.
2. **The estimate above is deliberately approximate** — it is meant to teach the calculation, not to be quoted as this project's exact figure. Before this number goes in a report, verify the real architecture parameters for the exact model and quantization actually pulled (Part 5 §5.2), and state in the report that the figure is measured or estimated, honestly, whichever it is.
3. This is a genuine example of the "worked example, then verify against your own configuration" discipline this whole curriculum has used since Chapter 1's cosine-similarity calculation — the arithmetic technique generalises, the specific numbers don't automatically.

---

# Part 3 — LEARN: From Strawman to Real Contract

## 3.1 Why a dataclass, and not Pydantic — the trade-off, properly stated

Chapter 4 §4.3 sketched a strawman using `@dataclass`. Today that becomes real, and the choice deserves a real justification rather than "because Chapter 0 already covered it."

**Pydantic** is the library most production RAG systems actually use for exactly this purpose. It gives you, largely for free: automatic runtime type validation (assigning a string to an `int` field raises immediately), automatic parsing from JSON, and automatic JSON-schema export — genuinely powerful, and worth knowing exists, because you will meet it in almost any real-world Python project of this kind.

**We chose `@dataclass` plus an explicit `validate_chunk()` function instead, for three specific reasons, not because dataclasses are "simpler" in the abstract:**

1. **Zero additional dependency.** `dataclasses` is in the Python standard library. One fewer package to pin, install, and have go wrong across three laptops (§1.3).
2. **The validation logic is visible, not hidden inside a library.** `validate_chunk()` (§3.3 below) is fifty lines of ordinary Python any team member can read start to finish. Pydantic's validation happens inside the library's own machinery — powerful, but opaque to a beginner team on Day 5.
3. **It matches what Chapter 0 §A.6.16 already taught.** No new syntax to learn today, on top of everything else Day 5 asks of the team.

**The honest cost, stated plainly:** a plain dataclass does *not* validate anything on its own — you can construct a `Chunk` with `page="not a number"` and Python will not complain until something later tries to use it as an integer. This is exactly why `validate_chunk()` exists as a *separate, explicit* function rather than being left implicit: the team is choosing to do by hand, visibly, what Pydantic would otherwise do automatically and invisibly. If the project were longer-lived or the team larger, this trade would likely flip — noted honestly as a "revisit if" condition, the same way every ADR states one.

## 3.2 Walking through the real schema

Open [`src/core/schemas.py`](../../src/core/schemas.py) alongside this section — the code below is the actual file, not a simplified version of it.

**Required fields** (`chunk_id`, `source`, `modality`, `text`, `embedding_model`) map directly onto Chapter 4 §4.2's requirements table — nothing here should be a surprise if you read that table carefully. **`Modality` is a `Literal["pdf", "docx", "image", "audio"]`**, not a bare `str`: this is a small but genuinely useful piece of type-checking, because most editors (including VS Code, per Chapter 0 §A.2) will flag `modality="pdff"` as a type error *while you're typing it*, before you ever run the code — a typo caught for free, for zero runtime cost.

**Modality-specific fields default to `None`.** This is deliberate: a `Chunk` for a PDF genuinely has no meaningful `start_s`, and Python's `None` says exactly that, rather than a magic sentinel value like `-1` or `""` that could be confused with real data. This is also precisely what `chunk.get("page")` from Chapter 0's dictionary discussion generalises into for a typed object — you handle "this field may not exist for this modality" the same way, just with `Optional[int]` in the type system instead of a runtime `.get()` call.

**`score` is separate from everything else, and the docstring says why**: it is the one field no ingestion pipeline ever sets — only Chapter 10's retrieval code, after the fact. Keeping it on the same class (rather than a second `RetrievedChunk` type wrapping a `Chunk`) is a real simplicity trade-off made for a 9-day schedule, stated as such in the code, so a future reader understands it was a choice, not an oversight.

**`to_chroma_record()` exists because of one genuinely easy-to-miss real API constraint**: ChromaDB's metadata dictionaries do not accept `None` as a value — passing one raises at runtime, not at write-time, so it is exactly the kind of failure that would otherwise surface for the first time during Chapter 7's first real indexing run. `to_chroma_record()` handles this once, centrally, so no individual track has to rediscover the constraint independently.

## 3.3 Walking through `validate_chunk()`

Reproduced in full in the source file; the design choices worth naming explicitly:

- **It returns a list of every problem found, not just the first one.** A malformed chunk with three problems reported as three lines saves a debugging round-trip compared to fixing one, rerunning, discovering the next.
- **It checks cross-contamination, not just presence.** The last two checks in the function — "a pdf chunk should not carry `start_s`" — exist specifically because Chapter 4 §1.7's counterfactual described exactly this failure mode: a field that's *technically* present and *technically* the right type, but semantically wrong for its modality, silently passing every naive check.
- **Modality-specific requirements are enforced only for the modality they apply to.** An image chunk is not required to have a `page` — the function checks `if chunk.modality in ("pdf", "docx")` before demanding one, so it never rejects a perfectly valid image record for a reason that only makes sense for documents.

## 3.4 Walking through the contract test — and why some tests are *supposed* to fail

Open [`tests/test_contract.py`](../../tests/test_contract.py). Three groups of tests, each doing a different job:

**Group 1 (fixture validity)** proves each track's *promised* output — the hand-written fixture — actually satisfies the schema. This is the direct implementation of Chapter 4 §4.4's three-fixture table.

**Group 2 (routing)** proves `default_collection()` sends each modality to the correct ChromaDB collection per ADR-003 — a one-line check, but exactly the kind of thing that's embarrassing to get backwards on Day 12.

**Group 3 is the one worth pausing on.** `test_rejects_missing_page_on_document`, `test_rejects_audio_with_end_before_start`, and two others **deliberately construct broken chunks and assert that `validate_chunk()` catches them.** This is not redundant with Group 1 — it answers a different, more important question. Group 1 asks *"does a good chunk pass?"* Group 3 asks *"does a bad chunk actually get caught?"* **A validation function that always returns `[]` — that never finds any problem, ever — would pass every test in Group 1 while being completely useless.** Group 3 exists so the contract test has real teeth, and it is a general testing principle worth remembering well beyond this project: a test suite that has never seen one of its own tests fail has not yet proven anything about the code it claims to protect.

**The fourth group (`test_round_trips_through_chromadb`) uses `pytest.importorskip`.** This is a standard pytest pattern for exactly Day 5's situation: `chromadb` may not be installed on every machine yet (Part 5 hasn't necessarily finished), and the test should **skip with a clear reason**, not fail, when its dependency is genuinely absent. This distinction — a skip is not a failure, and pretending it is would hide a real signal under a false one — is worth understanding for any test suite you write in the future, not just this one.

---

# Part 4 — DECIDE

## 4.1 Python version: 3.11, held firm

Chapter 0 §A.3 already chose this; today is where the reason becomes concrete rather than a rule to follow on faith. Per §1.2's wheel-availability argument, and per the very real Python-3.14 near-miss described in this chapter's own verification process, **every team member's machine must report `python --version` as `3.11.x` before continuing.** Run `python scripts/verify_setup.py` (Part 5 §5.5) to confirm.

> **What was actually tested on 2026-10-05, and what was not.** `requirements.txt` was installed exactly as pinned into a fresh Python **3.13.14** environment (Windows 11) and the full suite run in it. It works, which the "3.13 will not install" warning above did not predict, but it confirmed the warning's reasoning in two ways. (1) `chroma-hnswlib 0.7.6` has no prebuilt 3.13 wheel, so pip **compiled it from source**; that succeeded only because C++ build tools were present, and a machine without them stops at the `Microsoft Visual C++ 14.0 or greater is required` error in the troubleshooting table below. (2) A pin was missing: `av`, which `faster-whisper` needs but does not constrain, resolved to 19.x and broke real transcription, a failure the test suite could not see because it fakes Whisper. It is pinned now. A third finding was not about the Python version at all: the pinned `chromadb 0.5.23` returned a different top-5 from exact search for about one question in three at its default settings, fixed in `src/core/vector_store.py` and guarded by a test. **Python 3.11 itself was not tested.** The recommendation stands: 3.11 avoids the compiler requirement, 3.13 works if the build tools are installed.

## 4.2 Pinning policy

Exact pins (`==`) for everything except `torch` (deliberately left to resolve automatically as a transitive dependency — §1.4's table explains why a manual pin here is a common source of cross-platform breakage). Documented in `requirements.txt`'s own comments, per Chapter 3 §3.8's reproducibility standard: every pin should be checkable and re-derivable, not a mystery number.

## 4.3 Configuration strategy: environment variables via `.env`, not a YAML file, not hardcoding

Three options existed. **Hardcoding values directly in each track's code** was rejected immediately — it is precisely the "same value defined independently in three files, drifts silently" failure mode Chapter 4's whole framework exists to prevent. **A YAML or TOML config file** was considered and is a reasonable alternative; rejected in favour of `.env` mainly because `.env` cleanly separates **machine-specific settings** (which model tag is actually pulled, local paths) from the **committed defaults** (`.env.example`), using `.gitignore` to enforce the separation automatically rather than relying on every team member remembering not to commit secrets or local paths. `python-dotenv` is a two-line, zero-configuration way to load it. See `src/core/config.py`.

## 4.4 Directory layout

```
RAGNova/
├── src/
│   ├── core/            ← shared, joint-owned (Ch4 §1.3): schemas.py, config.py
│   └── pipelines/        ← per-track code lives here from Chapter 6 onward
├── tests/                ← the contract test lives here; each track adds its own tests alongside
├── scripts/               ← one-off / setup scripts (verify_setup.py)
├── data/                  ← the corpus (Ch1 §3.3)
├── docs/                  ← everything you've been reading
├── reports/               ← synopsis, PPTs, methodology drafts
├── requirements.txt
├── .env.example           ← committed
├── .env                   ← NOT committed (.gitignore) — machine-specific
└── .gitignore
```

`src/core/` is deliberately the *only* directory every track is expected to modify jointly, matching Chapter 4 §1.3's information-hiding boundary precisely: everything under `src/pipelines/` is one track's business; everything in `src/core/` is a shared contract, changed only per the note at the bottom of `schemas.py` itself.

## 4.5 The exact models, restated with their tags

| Component | Exact identifier | Source |
|---|---|---|
| LLM | `llama3.2:3b` (Ollama tag; resolves to the instruct-tuned, 4-bit quantized build) | ADR-002 |
| Text embeddings | `all-MiniLM-L6-v2` | Ch1 §1.5, Ch3A Theme 3 |
| Cross-modal | `ViT-B-32`, pretrained tag `laion2b_s34b_b79k` (OpenCLIP) | ADR-003 |
| Speech | Whisper `base` (via faster-whisper) | ADR-005, Ch1 §1.7.6 |

**Confirm the Ollama tag resolves to an instruct model, not a base model**, per Chapter 1 §1.1.4's warning — `ollama show llama3.2:3b` prints the model's template and parameters; an instruct model's template will visibly wrap your prompt in role-formatting tags, a base model's will not.

---

# Part 5 — BUILD: the actual Day 5

## 5.1 Create and activate the environment

```bash
cd path/to/RAGNova
python -m venv .venv
```

**Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate.ps1
```
If this fails with an execution-policy error, see Chapter 0 §A.4's fix (`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`).

**Mac/Linux:**
```bash
source .venv/bin/activate
```

Confirm activation worked: your prompt should now show `(.venv)` at the start.

## 5.2 Install Ollama and pull the model

Download from [ollama.com/download](https://ollama.com/download) if not already done (Chapter 1 §3.4 asked the team to start this on Day 1 night). Then:

```bash
ollama pull llama3.2:3b
```

**~2 GB download — this is exactly the download Chapter 1 flagged as the most likely cause of a schedule delay if left to the last minute.** If your team is only starting this now, do it first and read ahead while it downloads.

Confirm the model is present and correctly tagged:

```bash
ollama list
```

## 5.3 Install Tesseract (a separate binary, not just a Python package)

`pytesseract` (in `requirements.txt`) is a thin Python wrapper — it calls out to an actual Tesseract OCR **executable** that must be installed separately, on the OS itself, not via pip. This genuinely trips people up, because the pip install *succeeds* while the actual functionality is entirely absent until this step is done.

- **Windows:** download the installer from the [Tesseract at UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page, run it, and note the install path (commonly `C:\Program Files\Tesseract-OCR`). Add that folder to your system PATH, the same PATH concept Chapter 0 §A.3 covered for Python.
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

Verify from a **fresh terminal** (PATH changes need a new terminal, same rule as Chapter 0 §A.3):
```bash
tesseract --version
```

## 5.4 Install the Python dependencies

```bash
pip install -r requirements.txt
```

This is a large install (torch alone is several hundred MB) — expect several minutes. If it fails, consult §1.4's table before retrying blindly.

## 5.5 Run the setup verification script

```bash
python scripts/verify_setup.py
```

**Here is the genuine output this script produced on the machine used to write this chapter**, captured honestly rather than idealised — at the point of writing, this machine had Ollama's CLI installed but the service was not running and no model had been pulled yet, which is exactly the "fresh clone, nothing done yet" state most team members will start from:

```
RAGNova Day 5 setup check
========================================
[1/4] Python version: 3.14.5
      NOTE: this project targets Python 3.11 (Chapter 0 §A.3). You're running 3.14 —
      likely fine for this script, but if a later `pip install` fails with a
      build error, this is the first thing to check.
[2/4] Checking Ollama is running at http://localhost:11434 ...
      FAILED — could not reach Ollama: [WinError 10061] No connection could be
      made because the target machine actively refused it
      Fix: open a terminal and run `ollama serve`, or on
      Windows/Mac just launch the Ollama app — it runs as a
      background service once started. Then re-run this script.
[3/4] SKIPPED — Ollama isn't reachable yet (see check 2 above).
[4/4] Checking sentence-transformers can load the embedding model ...
      SKIPPED — sentence-transformers not installed yet.
      Fix: pip install -r requirements.txt

========================================
0/4 checks passed.
Not all checks passed yet — that's normal on a fresh clone. Fix the failures
above in order; later checks often depend on earlier ones.
```

**This is shown deliberately, not hidden.** The script's job is to fail clearly and helpfully on an unfinished setup, and this is what that looks like — every failure names its cause and its fix, and the script exits with a non-zero status (`1`) so it can also be used as a real pass/fail gate, not just a status printout. After completing §5.1–5.4 on your own machine, running this script should instead report `4/4 checks passed. Environment ready.` — if it doesn't, work through the failures in the order printed; check 3 depends on check 2 succeeding first, by design (§4 of the script's own logic).

## 5.6 Run the contract test

```bash
pytest tests/test_contract.py -v
```

**The real, captured output**, run against the actual `src/core/schemas.py` in this repository as this chapter was written:

```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: X:\RAGNova
collecting ... collected 10 items

tests/test_contract.py::test_text_fixture_is_valid PASSED                [ 10%]
tests/test_contract.py::test_image_fixture_is_valid PASSED               [ 20%]
tests/test_contract.py::test_audio_fixture_is_valid PASSED               [ 30%]
tests/test_contract.py::test_text_and_audio_route_to_text_collection PASSED [ 40%]
tests/test_contract.py::test_image_routes_to_image_collection PASSED     [ 50%]
tests/test_contract.py::test_rejects_missing_page_on_document PASSED     [ 60%]
tests/test_contract.py::test_rejects_audio_with_end_before_start PASSED  [ 70%]
tests/test_contract.py::test_rejects_wrong_modality_string PASSED        [ 80%]
tests/test_contract.py::test_rejects_cross_contaminated_fields PASSED    [ 90%]
tests/test_contract.py::test_round_trips_through_chromadb SKIPPED (c...) [100%]

======================== 9 passed, 1 skipped in 0.02s =========================
```

**Nine passed, one skipped.** The skip is `test_round_trips_through_chromadb`, exactly because `chromadb` was not yet installed in the environment used for this specific check (§3.4's `importorskip` pattern working as designed) — once your team completes §5.4, re-running this command should show `10 passed` with no skips. **This is the actual, checkable state of the interface contract on the day it was written — not a claim about what the code should do, a report of what it does.**

## 5.7 A manual sanity check of the LLM itself

Once §5.2 is done and `ollama serve` is running (or the desktop app is open), confirm generation end-to-end:

```bash
ollama run llama3.2:3b "Reply with exactly the word: OK"
```

If it replies `OK` (or close to it — small models don't always follow instructions perfectly, which is itself worth noting for Chapter 10), the model is genuinely working. For the programmatic check every pipeline will actually rely on, re-run `python scripts/verify_setup.py` — its checks 2 and 3 exercise the exact HTTP API path (§2.4) the rest of the project uses, not the interactive REPL.

## 5.8 Git hygiene for today

- [ ] `.gitignore` committed (already created — covers `.venv/`, `__pycache__/`, `.env`, `chroma_db/`)
- [ ] `.env.example` committed; **`.env` itself never committed**
- [ ] `requirements.txt` committed
- [ ] `src/core/schemas.py`, `src/core/config.py`, `tests/test_contract.py`, `scripts/verify_setup.py` committed
- [ ] Every team member has independently run §5.5 and §5.6 on their **own** machine and confirmed the same results — this is the actual, practiced version of Chapter 4 §4.4's fixture requirement, not a formality

---

# Part 6 — CHECK

## 6.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | Every team member's `python --version` reports 3.11.x | /3 |
| 2 | `pip install -r requirements.txt` completes on every machine | /3 |
| 3 | Ollama installed, running, model pulled and confirmed instruct-tuned | /3 |
| 4 | `verify_setup.py` reports 4/4 on every machine | /3 |
| 5 | `pytest tests/test_contract.py` reports all tests passing (10/10, once chromadb is installed) | /3 |
| 6 | Team can explain the dataclass-vs-Pydantic trade-off without notes | /3 |
| 7 | Team can explain why Group 3's tests are *supposed* to fail when given broken input | /3 |
| 8 | Team can explain what a wheel is and why Python 3.11 specifically matters | /3 |
| 9 | Team can explain mmap's effect on cold-start vs. warm latency | /3 |
| 10 | `.env` is git-ignored; `.env.example` is committed | /3 |
| | **Total** | **/30** |

## 6.2 Question bank

1. What does pip's resolver actually do, and why can it fail even when every individual package is installable? → §1.1; backtracking search for a mutually satisfying version set.
2. What is a wheel, and why does Python version matter for getting one? → §1.2; pre-built binary distribution, published per Python-version/platform combination.
3. Why pin exact versions instead of minimums for this project specifically? → §1.3; ML libraries change behaviour between close-looking releases, and Chapter 3's reproducibility standard depends on it.
4. What does memory-mapping buy a locally-served LLM, concretely? → §2.1; lazy, on-demand loading from disk, page-cache reuse across runs, and the reason first-inference latency differs from later requests.
5. Name three endpoints Ollama's API exposes and what each does. → §2.2's table.
6. Why does this project's code use the HTTP API rather than `ollama run`? → §2.4; a program needs a structured request/response, not an interactive terminal.
7. Derive, roughly, the KV-cache memory cost at an 8k context window. → §2.5's worked calculation, ~900 MB at the stated approximate architecture figures.
8. Why dataclass instead of Pydantic? → §3.1's three reasons: no extra dependency, visible validation logic, matches what Chapter 0 already taught — with the honest cost named too.
9. Why does `validate_chunk()` return a list instead of raising on the first problem? → §3.3; shows every issue at once.
10. What specific ChromaDB API constraint does `to_chroma_record()` handle? → §3.2; metadata values cannot be `None`.
11. What would it mean if the contract test suite only ever had passing tests, forever? → §3.4; it would prove nothing — a validator that rejects nothing would still pass every "good chunk" test.
12. What does `pytest.importorskip` do, and why is a skip different from a failure? → §3.4; skips when a dependency is genuinely absent, without falsely reporting a code defect.
13. Why `.env` rather than hardcoding values in each track's files? → §4.3; separates machine-specific settings from shared defaults, enforced by `.gitignore` rather than memory.
14. Which directory is every track allowed to edit jointly, and which is exclusively their own? → §4.4; `src/core/` joint, `src/pipelines/<track>/` exclusive.

## 6.3 Troubleshooting — ML-stack-specific (extends Chapter 0 §E)

| Symptom | Cause | Fix |
|---|---|---|
| `Microsoft Visual C++ 14.0 or greater is required` | No wheel for your Python version; pip fell back to source build | Confirm Python 3.11 first (§1.2); this alone resolves most cases |
| `ollama: command not found` after install | PATH not updated, or terminal opened before install | Fresh terminal; reinstall if still missing |
| `verify_setup.py` check 2 fails even though you ran `ollama pull` | The Ollama **service** isn't running — pulling a model doesn't start the server | Run `ollama serve` or open the desktop app; a pulled model with no running service is still unreachable |
| `ollama run` works but the API check fails | Different symptom, same root cause as above, or a firewall blocking `localhost:11434` | Check firewall rules for local loopback traffic — rare, but happens on locked-down corporate/college machines |
| `pytesseract.pytesseract.TesseractNotFoundError` | The Python wrapper is installed but the actual Tesseract binary isn't on PATH | Complete §5.3 fully, including the fresh-terminal PATH check |
| `chromadb` install hangs or fails oddly | Occasionally has its own compiled-dependency chain | Confirm Python 3.11; as a last resort, check ChromaDB's own release notes for known Windows issues at the pinned version |
| Contract test import error: `ModuleNotFoundError: No module named 'src'` | Running `pytest` from the wrong directory, or the venv isn't activated | Run from the project root (`X:\RAGNova`), confirm `(.venv)` is in your prompt |
| `verify_setup.py`'s Ollama-generation check times out | The model is genuinely slow on your hardware, or `num_predict`/context defaults are large | Confirm you pulled the 3B tag, not a larger one by mistake; try `ollama run llama3.2:3b` directly to isolate whether it's the script or the model itself |
| Output text shows `?` or garbled characters where an em-dash or `§` should be | Windows console defaulting to a legacy codepage instead of UTF-8 — a real issue found and fixed in this project's own `verify_setup.py` | Add `sys.stdout.reconfigure(encoding="utf-8")` near the top of any script that prints such characters, as `verify_setup.py` now does |

## 6.4 Day 5 completion checklist

- [ ] Every member's Python confirmed at 3.11.x
- [ ] Virtual environment created and activated (visible `(.venv)` prompt)
- [ ] Ollama installed, service running, `llama3.2:3b` pulled and confirmed instruct-tuned
- [ ] Tesseract binary installed and on PATH, verified from a fresh terminal
- [ ] `pip install -r requirements.txt` completed without unresolved errors
- [ ] `python scripts/verify_setup.py` reports 4/4 on every team member's machine
- [ ] `pytest tests/test_contract.py -v` reports all tests passing, no unexpected skips
- [ ] `.env` created locally from `.env.example`, confirmed **not** tracked by git
- [ ] Team can state the dataclass-vs-Pydantic trade-off and the "tests that should fail" principle without notes
- [ ] Rubric §6.1 scored ≥ 24/30

---

**Next:** Chapter 6 — Document Ingestion (Day 6), where Track A writes the first pipeline that actually produces a `Chunk` from a real PDF, and runs it through the contract test built today for the first time against real, not hand-written, data.
