# RAGNova Validation, Errors & Warnings Report

**Generated:** 2026-09-26  
**Environment:** Python 3.11.9 (Windows)  
**Test Framework:** Pytest 8.3.4  
**Local LLM:** Ollama (`llama3.2:3b` @ `http://localhost:11434`)  
**Resolution Status:** All Identified Errors Resolved & Verified (100% Pass)

---

## Executive Summary

| Scope | Status | Details |
| :--- | :--- | :--- |
| **All Test Suites (`pytest`)** | **92 / 92 PASSED** | All unit, contract, inventory, ingestion, RAG core, and retrieval tests succeeded. |
| **Setup & Dependencies (`scripts/verify_setup.py`)** | **4 / 4 PASSED** | Python 3.11 OK, Ollama running & responsive (0.3s), MiniLM embeddings OK. |
| **Corpus Data Assets** | **VALIDATED** | Starter PDFs/DOCX present, 12 images present (OCR & CLIP compliant), 4 audio WAVs (16kHz mono). |
| **Syntactic & Compilation Check** | **0 ERRORS** | `python -m compileall src scripts tests` completed cleanly. |
| **Root Test Discovery** | **RESOLVED** | Configured `pytest.ini` (`testpaths = tests`, `pythonpath = .`) and guarded `smoke_test.py`. |
| **Isolated Test Execution** | **RESOLVED** | Patched `tests/test_corpus_inventory.py` with `PROJECT_ROOT` path insertion. |

---

## Resolved Issues

### 1. Root Pytest Discovery Crash (`INTERNALERROR: SystemExit: 0`) & Collection Side-Effects — RESOLVED ✅
- **Location:** [`src/pipelines/images/smoke_test.py`](../src/pipelines/images/smoke_test.py) & [`pytest.ini`](../pytest.ini)
- **Problem:** Running default `pytest` from repo root collected `smoke_test.py` as a test module, executing module-level code ending in `sys.exit(0)` and crashing pytest collection. Furthermore, importing during collection triggered OCR and CLIP model execution.
- **Resolution Applied:**
  1. Added [`pytest.ini`](../pytest.ini) explicitly scoping `testpaths = tests` and `pythonpath = .`.
  2. Added `__test__ = False` to [`src/pipelines/images/smoke_test.py`](../src/pipelines/images/smoke_test.py).
  3. Encapsulated all check logic, OCR/model execution, and CLI exit reporting inside a `main()` function guarded by `if __name__ == "__main__": sys.exit(main())`. Importing the module now produces zero side-effects during test collection.

---

### 2. Isolated Test Execution (`ModuleNotFoundError: No module named 'src'`) — RESOLVED ✅
- **Location:** [`tests/test_corpus_inventory.py:20`](../tests/test_corpus_inventory.py#L20-L21)
- **Problem:** Running `pytest tests/test_corpus_inventory.py` individually failed because `PROJECT_ROOT` was defined but not inserted into `sys.path`.
- **Resolution Applied:** Added `if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))`. Standalone execution now passes 5/5 tests in 0.48s.

---

## Active Warnings Summary (Non-Breaking)

### 1. ChromaDB Pydantic V2 Deprecation (180 occurrences)
- **Source:** `chromadb/types.py:144`
- **Message:**
  ```text
  PydanticDeprecatedSince211: Accessing the 'model_fields' attribute on the instance is deprecated. Instead, you should access this attribute from the model class. Deprecated in Pydantic V2.11 to be removed in V3.0.
  ```
- **Nature:** Upstream deprecation notice inside ChromaDB 0.5.23 when running under Pydantic 2.13+. Non-blocking and does not affect vector indexing or search correctness.

### 2. PyMuPDF / Swig C-Extension Deprecation Warnings
- **Source:** `<frozen importlib._bootstrap>:241` via PyMuPDF (fitz)
- **Message:**
  ```text
  DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
  DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
  DeprecationWarning: builtin type swigvarlink has no __module__ attribute
  ```
- **Nature:** C-extension SWIG wrapper warning on Python 3.11+. Purely informational; PDF parsing and text chunking execute without error.

### 3. ChromaDB / PostHog Telemetry Argument Error
- **Source:** Standalone scripts initializing ChromaDB without disabled telemetry (`scripts/evaluate_retrieval.py`)
- **Message:**
  ```text
  Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
  ```
- **Nature:** Chroma's background anonymized telemetry fails to dispatch due to a PostHog library API change. Can be silenced by setting `ANONYMIZED_TELEMETRY=False` in environment config.

---

## Final Verification Output

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.4, pluggy-1.6.0
rootdir: C:\Users\prate\RAGGNOVA\RAGNOVA
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collected 92 items

tests\test_contract.py ..........                                        [ 10%]
tests\test_corpus_inventory.py .....                                     [ 16%]
tests\test_document_ingestion.py ..........................              [ 44%]
tests\test_integration.py ............................                   [ 75%]
tests\test_rag_core.py ...........                                       [ 86%]
tests\test_retrieval.py ............                                     [100%]

====================== 92 passed, 185 warnings in 29.43s ======================
```
