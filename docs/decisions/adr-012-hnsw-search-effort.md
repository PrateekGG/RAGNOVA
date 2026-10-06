# ADR-012: Raise ChromaDB's HNSW search effort so approximate search matches exact search

## Status
Accepted — 2026-10-05. Amends ADR-004's use of ChromaDB (same database, different collection settings).

## Context

ChromaDB finds nearest neighbours with HNSW, an *approximate* index: it trades a little accuracy for speed by exploring only a bounded set of candidates (`search_ef`, default 10) in a graph built with `construction_ef` (default 100) and `M` (default 16). Those defaults are tuned for millions of vectors.

Every retrieval number measured before 2026-10-05 came from an environment with `chromadb 1.5.9`. The environment `requirements.txt` tells students to install pins `chromadb 0.5.23`. Verifying the pins in a fresh Python 3.13 environment exposed a difference the embeddings did not cause (the query vector and the brute-force top 5 were identical to six digits in both environments): on the same 623-chunk text index, **0.5.23's top 5 differed from exact search for 9 or 10 of 29 gold questions, and its top 1 was wrong once or twice, varying from run to run** (HNSW insertion order is nondeterministic). A concrete case: the right chunk for T1 (`notice.pdf` page 2, exact rank 1, cosine 0.424) was absent from the top 8. `chromadb 1.5.9` matched exact search on all 29.

| Collection settings, `chromadb 0.5.23`, 623 chunks, 29 questions | Top 1 wrong | Top 5 differs from exact |
|---|---|---|
| defaults (before) | 1–2 | 9–10 |
| `search_ef` 100 | 0 | 1 |
| `search_ef` 100, `construction_ef` 200 | 0 | 1 |
| **`search_ef` 200, `construction_ef` 400, `M` 32 (chosen)** | **0** | **0** |

## Decision

Both collections are created with `hnsw:search_ef = 200`, `hnsw:construction_ef = 400`, `hnsw:M = 32` in addition to `hnsw:space = cosine` (`src/core/vector_store.py`). `chromadb 1.5.9` accepts the same settings and stays exact. A test (`tests/test_retrieval.py`) builds the real index and requires the approximate top 5 to equal a brute-force top 5 for every gold question and negative control; it fails (8 of 29 differ) with the settings removed, and it runs in both ChromaDB versions.

## Alternatives considered

1. **Leave the defaults.** Rejected: silently missing the best chunk in a third of queries is a retrieval bug that no downstream threshold or prompt can recover from.
2. **Move the pin to `chromadb 1.x`.** Exact at the defaults, and it is what all the earlier measurements used. Rejected for now: a major-version jump (a rewritten storage engine) changes the on-disk format and the whole install path for a bug that one collection setting removes. Worth revisiting if ChromaDB 0.5.x stops being installable.
3. **Brute-force exact search** (fetch the stored vectors and rank them in NumPy). Exact and simple at this size. Rejected: it abandons the database's search entirely and does not scale past a few tens of thousands of chunks.
4. **Only raise `search_ef`.** Fixed top 1 but left 1 of 29 top-5 sets different. Cheap, but the build-time parameters cost almost nothing at this size.

## Consequences

**Positive**
+ Approximate search equals exact search on the whole gold set, in both ChromaDB versions, so retrieval numbers no longer depend on which version a machine installed.
+ Memory and build time rise slightly (`M` 32 stores more links per vector); at a few thousand chunks this is negligible.

**Negative / limits**
− **Applies only when a collection is created.** An index built before this change keeps the old graph: delete `chroma_db/` and run `scripts/build_index.py`.
− Measured on one corpus of 623 text chunks. A much larger corpus may need more effort; the new test will say so if retrieval degrades.
− `search_ef` and the other build parameters are not a theoretical guarantee of exactness; the test is the check.

## Revisit if

The corpus grows by an order of magnitude, or the pin moves to ChromaDB 1.x (then re-run the test and the measurement above).
