"""
Ask RAGNova a question end-to-end: retrieve -> filter -> generate -> cite.

Run with:  python scripts/ask.py "your question here"
(after scripts/build_index.py has built the real index, and `ollama serve`
is running with settings.OLLAMA_MODEL pulled)

Everything before Chapter 10 produced chunks, vectors, or ranked search
results — this is the first script in the project that answers a question.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same fix as scripts/evaluate_retrieval.py, same reason: an unconfigured
# Windows console can't render an em-dash in a "not enough information"
# message or a citation's own text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipelines.rag import answer_query
from src.pipelines.rag.prompt import format_provenance


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"', file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = answer_query(query)

    print(result.answer)
    if result.citations:
        print("\nSources:")
        for i, chunk in enumerate(result.citations, start=1):
            print(f"  [{i}] {format_provenance(chunk)}  (score={chunk.score:.3f})")


if __name__ == "__main__":
    main()
