"""
BM25 (src/core/bm25.py) pinned against numbers worked out by hand, so the
keyword baseline the dense retriever is compared with is itself trustworthy.
Pure Python: no models, no ChromaDB.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.bm25 import BM25, tokenize

K1, B = 1.5, 0.75

# Three tiny documents, already tokenised.
D1 = ["cat", "sat"]
D2 = ["dog", "sat"]
D3 = ["cat", "cat", "dog", "bird"]
DOCS = [D1, D2, D3]
AVG = (2 + 2 + 4) / 3          # average length = 8/3


def _idf(n_containing: int, n_docs: int = 3) -> float:
    return math.log(1 + (n_docs - n_containing + 0.5) / (n_containing + 0.5))


def _term(idf: float, f: int, length: int) -> float:
    return idf * f * (K1 + 1) / (f + K1 * (1 - B + B * length / AVG))


def test_tokenize_lowercases_splits_on_punctuation_and_drops_stopwords():
    assert tokenize("The Cat, sat on THE mat!") == ["cat", "sat", "mat"]
    assert tokenize("") == []


def test_scores_match_hand_computed_okapi_bm25():
    bm25 = BM25(DOCS, k1=K1, b=B)
    got = bm25.scores(["cat"])
    # "cat" appears in D1 (once) and D3 (twice): 2 of 3 documents.
    expected = [_term(_idf(2), 1, 2), 0.0, _term(_idf(2), 2, 4)]
    assert got == pytest.approx(expected)


def test_a_rarer_term_scores_higher_than_a_common_one():
    bm25 = BM25(DOCS)
    # "bird" is in 1 document, "sat" in 2: for equal term frequency the rarer term wins.
    assert bm25.scores(["bird"])[2] > bm25.scores(["sat"])[0]


def test_repeating_a_term_saturates_instead_of_growing_linearly():
    bm25 = BM25([["x"] * 1 + ["pad"], ["x"] * 4 + ["pad"], ["x"] * 16 + ["pad"], ["other"]])
    s = bm25.scores(["x"])
    assert s[0] < s[1] < s[2]
    assert (s[2] - s[1]) < (s[1] - s[0]) * 4      # sixteen is nowhere near 16x one


def test_duplicate_query_terms_do_not_double_count_and_unknown_terms_add_nothing():
    bm25 = BM25(DOCS)
    assert bm25.scores(["cat", "cat"]) == bm25.scores(["cat"])
    assert bm25.scores(["cat", "zzz"]) == bm25.scores(["cat"])
    assert bm25.scores(["zzz"]) == [0.0, 0.0, 0.0]


def test_top_orders_best_first_and_breaks_ties_by_document_order():
    bm25 = BM25(DOCS)
    assert bm25.top(["cat"], 3)[0] == 2            # D3 has "cat" twice
    # Both D1 and D2 contain "sat" once and have equal length: tie -> earlier document first.
    assert bm25.top(["sat"], 2) == [0, 1]
    assert bm25.top(["cat"], 1) == [2]
