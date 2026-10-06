"""
A small Okapi BM25 keyword ranker, used as the baseline that dense (MiniLM)
retrieval has to beat.

Why it exists: the project's central claim is that *semantic* search finds
what keyword search ("Ctrl+F") misses. A claim like that needs a keyword
baseline to be measured against; without one it is an assertion. BM25 is the
standard strong keyword baseline in retrieval research, and writing the
~30 lines here (rather than adding a dependency) keeps it auditable: the
test pins the arithmetic against hand-computed numbers.

For a query q and a document d, BM25 sums over the query's distinct terms t:

    idf(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avg|d|))

where f(t, d) is how often t occurs in d, |d| is d's length in tokens, and
idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)) with N documents and n(t) of
them containing t. (The "1 +" inside the log keeps idf non-negative, as
Lucene does, so a term in more than half the documents can never subtract
from a score.) k1 = 1.5 controls how fast repeated occurrences saturate and
b = 0.75 how strongly long documents are penalised: the usual defaults.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Function words add noise to a bag-of-words ranker, and any real keyword
# search drops them, so the baseline does too. Kept short and plain.
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may me my no not of on or our should so some such than that the their them then there these they this to
was we were what when where which who whom why will with would you your
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric words, minus stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """BM25 over a fixed list of already-tokenised documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._term_counts = [Counter(doc) for doc in documents]
        self._lengths = [len(doc) for doc in documents]
        n = len(documents)
        self._avg_length = (sum(self._lengths) / n) if n else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        self._idf = {
            term: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """One BM25 score per document, in document order. A term that no
        document contains contributes nothing."""
        terms = [t for t in set(query_tokens) if t in self._idf]
        out = []
        for counts, length in zip(self._term_counts, self._lengths):
            norm = self.k1 * (1.0 - self.b + self.b * length / self._avg_length) if self._avg_length else self.k1
            total = 0.0
            for term in terms:
                f = counts.get(term, 0)
                if f:
                    total += self._idf[term] * f * (self.k1 + 1.0) / (f + norm)
            out.append(total)
        return out

    def top(self, query_tokens: list[str], k: int) -> list[int]:
        """Indices of the k best documents, best first. Ties keep document
        order (Python's sort is stable), so results are reproducible."""
        scores = self.scores(query_tokens)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
