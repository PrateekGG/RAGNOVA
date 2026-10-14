"""
Name and identifier lookups: admit a chunk because the question's distinctive words are literally in it.

Why (2026-10-11): "who is alex morgan?" was refused on the live page although the corpus holds Alex Morgan's
ID card. The card is indexed with its text ("STUDENT ID Alex Morgan | Roll No: 23AIML042 ..."), but

  * CLIP cannot match a name (text-to-image score 0.151 against a 0.2 floor), and
  * MiniLM scores a two-word name against a whole ID-card's text at 0.16 against a 0.30 bar (ADR-011),

so the image gate drops it. Dense search is built for meaning, and a name has none to match. A name is
exactly what an exact-word lookup is good at, so this adds one, as a second way in beside the floors.

THE RULE (fixed before it was measured; scripts/evaluate_entity_questions.py)
  * The question's CONTENT words are its words minus question words, filler and anything shorter than 3
    characters (digits are kept: "302").
  * A content word is DISTINCTIVE when it occurs in 1 to `max_df` indexed FILES (not chunks), UNKNOWN when it occurs in none.
  * Mode "distinctive": a chunk is admitted when it contains EVERY distinctive word, provided the distinctive
    words are at least MIN_COVERAGE (60%) of distinctive + unknown. Mode "all": it must contain every KNOWN
    content word (distinctive or common), with known words at least MIN_COVERAGE of known + unknown; and in both
    modes there must be at least one distinctive word, so a question made only of common words looks nothing up.
    "alex morgan" admits the card; "alex smith" does not (smith is unknown: 1 of 2 = 50%); "alex morgan's blood
    group" does not (no chunk holds alex, morgan AND blood). The two modes differ when a name's first word is
    common: "alex" is in 8 indexed files (paper authors), so in "distinctive" mode only "morgan" has to match.
  * At most LIMIT chunks are admitted, shortest first (the more concentrated chunk), and the caller puts them
    first in their modality's list.

  * NAME PHRASES (added 2026-10-11, the fix ADR-017 named): a title or an initial in the question must sit next to
    the same word in the chunk. "Who is Dr. S. Sharma?" needs "dr s" and "s sharma" side by side; the lab sign says
    "Dr. S. Rao" and "Mr. V. Sharma", so it is NOT admitted (before this, "sharma" alone admitted it and the model
    answered with Dr. S. Rao). "Dr. S. Rao" and "Mr. V. Sharma" still are. An initial is one letter followed by a
    dot, or one letter right after a title, so the pronoun "I" never counts.

TYPOS (spelling_fix, 2026-10-11): "tell me lab rul no. 2" was refused while "tell me lab rules" was answered. "rul" is
in no indexed file, so dense search gets a damaged question and nothing clears the floors. spelling_fix() swaps a word
the corpus does not know for a COMMON corpus word it is one keystroke away from (or the start of). It is used for the
similarity searches only; the lexical admit above still sees the question as typed, so an unknown NAME still vetoes.

FULL NAMES (2026-10-11): with nemotron-3-nano:4b, "Who is Alex Smith?" (E15, a negative) was answered: a chunk held
"alex" and "smith" in two different names and the model glued them into one. Two or more capitalised words in a row in
the question ("Alex Smith") are a name the same way a title is, so they must sit side by side in the chunk too.

What it does NOT do: it does not judge whether the chunk ANSWERS the question. An admitted chunk is context,
and the model's own refusal rule still decides, as for every image the gate lets through (ADR-011).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.schemas import Chunk
from src.pipelines.rag.retrieve import _GENERIC_WORDS

MIN_COVERAGE = 0.6
LIMIT = 3

# Words that open or pad a question and say nothing about its subject, beyond retrieve._GENERIC_WORDS.
_FILLER = frozenset("""
explain describe mean means meaning know known called named name tell more details detail information info
whats hows
""".split())

_POSSESSIVE = re.compile(r"['’]s\b")
# A word, or an identifier that keeps its inner punctuation: lib-884920, vpn.ragnova.edu, final_synopsis_team12.pdf
_TOKEN = re.compile(r"[a-z0-9](?:[a-z0-9._\-]*[a-z0-9])?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(_POSSESSIVE.sub("", text.lower()))


def content_tokens(query: str) -> list[str]:
    """The question's content words, in order, without repeats."""
    seen: dict[str, None] = {}
    for token in tokenize(query):
        if token in _GENERIC_WORDS or token in _FILLER:
            continue
        if len(token) < 3 and not any(ch.isdigit() for ch in token):
            continue
        seen.setdefault(token)
    return list(seen)


_TITLES = frozenset({"dr", "mr", "mrs", "ms", "miss", "prof"})
_WORD_DOT = re.compile(r"([a-z0-9]+)(\.?)")


def _plain_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _POSSESSIVE.sub("", text.lower()))


def _spaced_words(text: str) -> str:
    return f" {' '.join(_plain_words(text))} "


def name_phrase_pairs(query: str) -> list[tuple[str, str]]:
    """The word pairs inside the question's name phrases: "Dr. S. Sharma" -> [("dr", "s"), ("s", "sharma")]."""
    return [pair for phrase in name_phrases(query) for pair in zip(phrase, phrase[1:])]


def name_phrases(query: str) -> list[tuple[str, ...]]:
    """The names a title, initials or capitals pin down, each as the words that must stand together in a chunk:
    "Dr. S. Sharma" -> [("dr", "s", "sharma")], "Mr Sharma" -> [("mr", "sharma")], "Alex Smith" -> [("alex", "smith")].
    Checked as whole phrases, not pair by pair: "Dr. S. Rao and Mr. S. Sharma" holds both pairs of "Dr. S. Sharma"
    but not the name. [] when the question has no title, no dotted initial and no capitalised full name."""
    phrases: list[tuple[str, ...]] = []
    words = [(m.group(1), bool(m.group(2))) for m in _WORD_DOT.finditer(_POSSESSIVE.sub("", query.lower()))]
    i = 0
    while i < len(words):
        word, dotted = words[i]
        starts = word in _TITLES or (len(word) == 1 and word.isalpha() and dotted)
        if not starts:
            i += 1
            continue
        phrase = [word]
        j = i + 1
        while j < len(words):
            nxt, nxt_dotted = words[j]
            if len(nxt) == 1 and nxt.isalpha() and (nxt_dotted or phrase[-1] in _TITLES):
                phrase.append(nxt)          # another initial
                j += 1
                continue
            if nxt.isalpha() and len(nxt) >= 2 and nxt not in _TITLES:
                phrase.append(nxt)          # the name the title / initials belong to
                j += 1
            break
        if len(phrase) >= 2:
            phrases.append(tuple(phrase))
        i = j if j > i + 1 else i + 1
    return phrases + [p for p in _capitalised_names(query) if not any(_contains(q, p) for q in phrases)]


def _contains(phrase: tuple[str, ...], part: tuple[str, ...]) -> bool:
    return f" {' '.join(part)} " in f" {' '.join(phrase)} "


_NAME_WORD = re.compile(r"[A-Z][a-z]+")
_EDGE_PUNCT = ".,;:!?\"'()[]“”‘’"


def _capitalised_names(query: str) -> list[tuple[str, ...]]:
    """Runs of two or more capitalised words, as typed: "Who is Alex Smith?" -> [("alex", "smith")]. A word with other
    punctuation inside it, a common word ("Who", "Tell", "About") or anything else ends a run, so "What is
    Fusion-in-Decoder?" and an all-lowercase question give []."""
    names: list[tuple[str, ...]] = []
    run: list[str] = []
    for raw in query.split() + [""]:
        bare = _POSSESSIVE.sub("", raw)
        word = bare.strip(_EDGE_PUNCT)
        lower = word.lower()
        if _NAME_WORD.fullmatch(word) and lower not in _GENERIC_WORDS and lower not in _FILLER and lower not in _TITLES:
            run.append(lower)
            if bare.rstrip(_EDGE_PUNCT) == bare:
                continue        # nothing between this word and the next but a space: the name may go on
        if len(run) >= 2:
            names.append(tuple(run))
        run = []
    return names


@dataclass(frozen=True)
class Entry:
    chunk_id: str
    source: str
    collection: str          # "text" or "image": which ChromaDB collection holds the chunk
    tokens: frozenset[str]
    words: str = ""          # the chunk's words in order, space-separated with a space at each end, for name phrases


class LexicalIndex:
    """Which words occur in which chunks and in how many files. Built from the index's own documents."""

    def __init__(self, rows):
        """`rows`: (chunk_id, source, collection, text) for every chunk."""
        self.entries: list[Entry] = []
        self._files: dict[str, set[str]] = {}
        for chunk_id, source, collection, text in rows:
            tokens = frozenset(tokenize(text))
            self.entries.append(Entry(chunk_id, source, collection, tokens, _spaced_words(text)))
            for token in tokens:
                self._files.setdefault(token, set()).add(source)

    def df(self, token: str) -> int:
        """In how many indexed FILES the word occurs."""
        return len(self._files.get(token, ()))

    def vocabulary(self):
        return self._files.keys()


def admit(query: str, index: LexicalIndex, max_df: int = 2, min_coverage: float = MIN_COVERAGE, limit: int = LIMIT,
          mode: str = "distinctive") -> list[Entry]:
    """The chunks the rule above admits for `query`, best first. [] when there is nothing distinctive to look up."""
    if mode not in ("distinctive", "all"):
        raise ValueError(f"unknown lexical mode {mode!r}: use 'distinctive' or 'all'")
    tokens = content_tokens(query)
    distinctive = [t for t in tokens if 1 <= index.df(t) <= max_df]
    known = [t for t in tokens if index.df(t) >= 1]
    unknown = [t for t in tokens if index.df(t) == 0]
    if not distinctive:
        return []
    required = distinctive if mode == "distinctive" else known
    if len(required) / (len(required) + len(unknown)) < min_coverage:
        return []
    phrases = [f" {' '.join(p)} " for p in name_phrases(query)]
    matches = [e for e in index.entries if all(t in e.tokens for t in required) and all(p in e.words for p in phrases)]
    matches.sort(key=lambda e: len(e.tokens))
    return matches[:limit]


# ---------------------------------------------------------------------------
# Typos
# ---------------------------------------------------------------------------

SPELLING_MIN_DF = 2       # only ever correct TO a word at least this many files share: a common word, never a name


def _one_edit_apart(a: str, b: str) -> bool:
    """True when b is a with one letter changed, added or removed."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def correct_word(word: str, index: LexicalIndex) -> str | None:
    """The common corpus word `word` was most likely meant to be, or None. Only alphabetic words of 3+ letters that
    no indexed file contains are considered. A candidate is a word that starts with `word` and is at most 2 letters
    longer ("rul" -> "rule"), or, for a word of 4+ letters, one keystroke away ("lirbary" is not; "libary" is).
    The candidate in the most files wins; ties go to the shorter word, then alphabetical order."""
    if len(word) < 3 or not word.isalpha() or index.df(word):
        return None
    candidates = []
    for known in index.vocabulary():
        if not known.isalpha() or index.df(known) < SPELLING_MIN_DF:
            continue
        if (known.startswith(word) and len(known) - len(word) <= 2) or (len(word) >= 4 and _one_edit_apart(word, known)):
            candidates.append(known)
    if not candidates:
        return None
    return min(candidates, key=lambda k: (-index.df(k), len(k), k))


def spelling_fix(query: str, index: LexicalIndex) -> str:
    """`query` with every word correct_word() can fix replaced; everything else (case, punctuation) kept."""
    def swap(match: re.Match) -> str:
        fixed = correct_word(match.group(0).lower(), index)
        return fixed if fixed else match.group(0)
    return re.sub(r"[A-Za-z]+", swap, query)


# ---------------------------------------------------------------------------
# Against the real index
# ---------------------------------------------------------------------------

_cache: dict[tuple, LexicalIndex] = {}


def _index_for(client) -> LexicalIndex:
    """The lexical index of this ChromaDB, rebuilt when the number of chunks in either collection changes (an upload,
    a rebuild). Re-indexing a changed file to the SAME number of chunks is not noticed until the process restarts."""
    from src.core.vector_store import get_image_collection, get_text_collection

    text_collection, image_collection = get_text_collection(client), get_image_collection(client)
    # A new client object is made per call when none is passed (lexical_hits and fix_spelling each make one), so the
    # key is the database's path, not the object: keyed by id() the index was rebuilt twice for every question.
    where = getattr(client, "_identifier", "") or id(client)
    key = (where, text_collection.count(), image_collection.count())
    if key not in _cache:
        _cache.clear()
        rows = []
        for collection, name in ((text_collection, "text"), (image_collection, "image")):
            if collection.count():
                got = collection.get(include=["documents", "metadatas"])
                rows += [(i, m["source"], name, d) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
        _cache[key] = LexicalIndex(rows)
    return _cache[key]


def lexical_hits(query: str, client=None, max_df: int = 2, mode: str = "distinctive") -> tuple[list[Chunk], list[Chunk]]:
    """(text chunks, image chunks) the rule admits for `query`, each with no score: they were not ranked by
    similarity, they matched words."""
    from dataclasses import replace

    from src.core.vector_store import get_client, get_image_collection, get_text_collection
    from src.pipelines.documents.search import _chunk_from_result

    client = client or get_client()
    admitted = admit(query, _index_for(client), max_df=max_df, mode=mode)
    text_chunks: list[Chunk] = []
    image_chunks: list[Chunk] = []
    for collection, name, bucket in ((get_text_collection(client), "text", text_chunks), (get_image_collection(client), "image", image_chunks)):
        ids = [e.chunk_id for e in admitted if e.collection == name]
        if not ids:
            continue
        got = collection.get(ids=ids, include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        for chunk_id in ids:                      # keep the rule's order
            text, metadata = by_id[chunk_id]
            bucket.append(replace(_chunk_from_result(chunk_id, text, metadata, 1.0), score=None))
    return text_chunks, image_chunks


def fix_spelling(query: str, client=None) -> str:
    """spelling_fix() against the real index. Returns `query` unchanged when the index is empty."""
    from src.core.vector_store import get_client

    return spelling_fix(query, _index_for(client or get_client()))
