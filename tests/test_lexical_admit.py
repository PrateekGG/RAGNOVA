"""
Name and identifier lookups (src/pipelines/rag/lexical.py) and the who/what-is prompt rule (2026-10-11).

"Who is Alex Morgan?" was refused on the live page although the corpus holds Alex Morgan's ID card: dense search has no
meaning to match in a name (CLIP 0.151 against a floor of 0.2, MiniLM agreement 0.16 against 0.30), and with the card in
scope the shipped prompt made the 3B model refuse "who is X" anyway. Both additions are OFF by default; what these
tests pin is the rule itself and that off means off: no call, no change to a single result or prompt byte.

No models, no ChromaDB: the index is built from a handful of invented chunks.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import Chunk
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.lexical import LexicalIndex, admit, content_tokens, tokenize
from src.pipelines.rag.prompt import build_prompt

ROWS = [
    # chunk_id, source, collection, text
    ("card", "data/images/card.png", "image", "RAGNova Institute STUDENT ID Alex Morgan | Roll No: 23AIML042 Library Barcode: *LIB-884920*"),
    ("paper_a", "data/documents/a.pdf", "text", "Alex Graves proposed a method. Kaufmann published it."),
    ("paper_d", "data/documents/d.pdf", "text", "Morgan Stanley published a report on markets."),
    ("paper_b", "data/documents/b.pdf", "text", "Alex Krizhevsky trained a network on images."),
    ("paper_c", "data/documents/c.pdf", "text", "Alex Wang evaluated language understanding."),
    ("blood", "data/images/blood.png", "image", "Blood Donation Camp Date: 20 October Health Centre"),
    ("lab1", "data/images/lab.png", "image", "Research Laboratory Room 302 Faculty In-Charge: Dr. S. Rao Lab Superintendent: Mr. V. Sharma"),
    ("lab2", "data/images/plaque.png", "image", "Directory Floor 2: AI Research Lab (Room 302)"),
    ("lib1", "data/documents/library.pdf", "text", "The library is open from 8 to 10. Library fines are 2 rupees. Library desk on floor 1."),
    ("lib2", "data/documents/library.pdf", "text", "Library borrowing: four books. Library renewals."),
    ("vpn", "data/images/vpn.png", "image", "Gateway: vpn.ragnova.edu:443 Tunnel Status: Connected"),
]


@pytest.fixture(scope="module")
def index():
    return LexicalIndex(ROWS)


def _ids(entries):
    return [e.chunk_id for e in entries]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def test_identifiers_keep_their_inner_punctuation_and_possessives_are_dropped():
    assert tokenize("What is vpn.ragnova.edu? LIB-884920, final_synopsis_team12.pdf.") == \
        ["what", "is", "vpn.ragnova.edu", "lib-884920", "final_synopsis_team12.pdf"]
    assert tokenize("Alex Morgan's roll number") == ["alex", "morgan", "roll", "number"]
    assert tokenize("Morgan’s") == ["morgan"]                      # a typographic apostrophe too


def test_content_tokens_drop_question_words_filler_and_short_words_but_keep_digits_and_repeat_nothing():
    assert content_tokens("Who is Dr. S. Rao?") == ["rao"]              # "dr" and "s" are too short to identify anything
    assert content_tokens("Tell me about Room 302 and room 302") == ["room", "302"]
    assert content_tokens("What does this mean?") == []


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

def test_a_words_document_frequency_counts_files_not_chunks(index):
    assert index.df("library") == 2            # three chunks hold it (two in library.pdf, one on the card) but only two FILES do
    assert index.df("alex") == 4               # four different files
    assert index.df("nothing-like-this") == 0


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_a_name_admits_the_card_even_when_its_first_word_is_common(index):
    """alex is in 4 files (too common at max_df=2), so only morgan has to match; the card holds it."""
    assert "card" in _ids(admit("who is alex morgan?", index, max_df=2))


def test_mode_all_needs_every_known_word_so_only_the_card_matches_alex_morgan(index):
    assert _ids(admit("who is alex morgan?", index, mode="all")) == ["card"]
    assert set(_ids(admit("who is alex morgan?", index, mode="distinctive"))) == {"card", "paper_d"}     # only morgan has to match


def test_an_identifier_and_a_lone_distinctive_word_are_looked_up(index):
    assert _ids(admit("What is vpn.ragnova.edu?", index)) == ["vpn"]
    assert _ids(admit("What is LIB-884920?", index)) == ["card"]
    assert _ids(admit("Who is Dr. S. Rao?", index)) == ["lab1"]


def test_a_number_that_two_files_share_admits_both(index):
    assert set(_ids(admit("What is Room 302?", index, max_df=2))) == {"lab1", "lab2"}


@pytest.mark.parametrize("question", [
    "Who is Alex Smith?",             # smith is unknown: 1 distinctive of 2 content words, below 60%
    "Who is Morgan Freeman?",         # freeman unknown
    "Who is Jordan Blake?",           # nothing known at all
    "What is Room 999?",              # 999 unknown
    "What is vpn.example.org?",       # unknown identifier
])
def test_a_question_with_an_unknown_word_is_not_looked_up(index, question):
    assert admit(question, index) == []
    assert admit(question, index, mode="all") == []


def test_a_question_asking_for_a_fact_no_chunk_holds_admits_nothing(index):
    """alex, morgan and blood are each known; no chunk holds them together."""
    assert admit("What is Alex Morgan's blood group?", index) == []
    assert admit("What is Alex Morgan's blood group?", index, mode="all") == []


def test_a_question_of_only_common_words_looks_nothing_up(index):
    assert admit("What is the library?", index, max_df=1) == []          # library is in 2 files: common at max_df=1
    assert admit("What is the library?", index, max_df=2) != []          # the same word is distinctive at max_df=2


def test_the_coverage_bar_is_adjustable_and_the_default_is_sixty_percent(index):
    assert admit("who is alex smith?", index, max_df=4, min_coverage=0.4) != []      # 1 of 2 = 50% clears 40%
    assert admit("who is alex smith?", index, max_df=4) == []                         # but not 60%


def test_at_most_limit_chunks_are_admitted_shortest_first(index):
    got = admit("library", index, max_df=2, limit=1)
    assert _ids(got) == ["lib2"]                                         # the shorter of the two library chunks
    assert len(admit("library", index, max_df=2, limit=2)) == 2


def test_an_unknown_mode_is_refused_not_silently_treated_as_the_default(index):
    with pytest.raises(ValueError, match="unknown lexical mode"):
        admit("who is alex morgan?", index, mode="fuzzy")


# ---------------------------------------------------------------------------
# retrieve(): wired in, and off means off
# ---------------------------------------------------------------------------

def _chunk(chunk_id, source="data/documents/x.pdf", modality="pdf", score=0.5):
    return Chunk(chunk_id=chunk_id, source=source, modality=modality, text="t", embedding_model="m",
                 page=1 if modality == "pdf" else None, score=score)


@pytest.fixture
def wired(monkeypatch):
    """retrieve() with every search replaced; records the calls to the lexical search."""
    calls = []
    dense_text = [_chunk("d1"), _chunk("d2")]
    dense_image = [_chunk("i_dense", "data/images/z.png", "image", 0.6)]

    def lexical_search(query, client=None, max_df=2, mode="distinctive"):
        calls.append((query, max_df, mode))
        return [_chunk("lex_text", score=None)], [_chunk("lex_image", "data/images/card.png", "image", None)]

    monkeypatch.setattr(retrieve_module, "search_text", lambda q, top_k=None, client=None, sources=None: list(dense_text))
    monkeypatch.setattr(retrieve_module, "filter_by_floor", lambda chunks, floor: list(chunks))

    def image_search(query_text=None, query_image=None, top_k=None, client=None, sources=None):
        return list(dense_image)

    def set_lexical(on, mode="distinctive", max_df=2):
        monkeypatch.setattr(retrieve_module, "settings", dataclasses.replace(
            retrieve_module.settings, LEXICAL_ADMIT=on, LEXICAL_MODE=mode, LEXICAL_MAX_DF=max_df))

    return calls, lexical_search, image_search, set_lexical


def test_with_the_setting_off_the_lexical_search_is_never_called_and_nothing_changes(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(False)
    found = retrieve_module.retrieve("who is alex morgan?", include_images=True, image_search=image_search,
                                     image_agreement=lambda q, c: 1.0, lexical_search=lexical_search)
    assert calls == []
    assert {c.chunk_id for c in found} == {"d1", "d2", "i_dense"}


def test_with_the_setting_on_lexical_chunks_come_first_in_their_modality_and_a_chunk_found_both_ways_appears_once(wired, monkeypatch):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)

    def lexical_with_overlap(query, client=None, max_df=2, mode="distinctive"):
        return [_chunk("d2", score=None)], []                           # d2 is also a dense hit

    found = retrieve_module.retrieve("anything", lexical_search=lexical_with_overlap)
    assert [c.chunk_id for c in found] == ["d2", "d1"]


def test_a_lexical_chunk_is_returned_even_when_nothing_cleared_the_floors(wired, monkeypatch):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)
    monkeypatch.setattr(retrieve_module, "filter_by_floor", lambda chunks, floor: [])     # the floor drops every dense hit
    found = retrieve_module.retrieve("who is alex morgan?", include_images=True,
                                     image_search=lambda **kw: [], image_agreement=lambda q, c: 0.0,
                                     lexical_search=lexical_search)
    assert {c.chunk_id for c in found} == {"lex_text", "lex_image"}


def test_the_settings_reach_the_lexical_search(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True, mode="all", max_df=3)
    retrieve_module.retrieve("who is alex morgan?", lexical_search=lexical_search)
    assert calls == [("who is alex morgan?", 3, "all")]


def test_a_text_only_question_keeps_lexical_text_chunks_and_drops_lexical_images(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)
    found = retrieve_module.retrieve("who is alex morgan?", include_images=False, lexical_search=lexical_search)
    assert [c.chunk_id for c in found][0] == "lex_text"
    assert not any(c.modality == "image" for c in found)


def test_a_picked_scope_or_a_blank_question_never_uses_the_lexical_search(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)
    retrieve_module.retrieve("who is alex morgan?", sources=["data/images/card.png"], lexical_search=lexical_search)
    retrieve_module.retrieve("   ", include_images=True, image_search=image_search, lexical_search=lexical_search)
    assert calls == []


def test_top_k_still_bounds_the_result(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)
    found = retrieve_module.retrieve("q", top_k=2, include_images=True, image_search=image_search,
                                     image_agreement=lambda q, c: 1.0, lexical_search=lexical_search)
    assert len(found) == 2


# ---------------------------------------------------------------------------
# The prompt rule
# ---------------------------------------------------------------------------

def _chunks():
    return [Chunk(chunk_id="a", source="data/images/card.png", modality="image", text="STUDENT ID Alex Morgan", embedding_model="m")]


def test_the_default_prompt_is_byte_for_byte_what_it_was():
    assert build_prompt("who is alex morgan?", _chunks()) == build_prompt("who is alex morgan?", _chunks(), who_what_rule=False)
    assert "who or what someone or something is" not in build_prompt("q", _chunks())


def test_the_rule_adds_the_who_what_instruction_and_keeps_the_refusal_sentence_and_the_citation_rule():
    prompt = build_prompt("who is alex morgan?", _chunks(), who_what_rule=True)
    assert "If the question asks who or what someone or something is, answer with what the context says about it" in prompt
    assert "does not contain the specific fact asked for" in prompt
    assert "I don't have enough information in the indexed documents to answer that." in prompt
    assert "Cite every claim with the matching bracketed number" in prompt
    assert prompt.endswith("Question: who is alex morgan?\n\nAnswer:")


def test_the_setting_reaches_the_prompt_the_model_is_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda query, **kw: _chunks())
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")
    for flag in (False, True):
        seen.clear()
        monkeypatch.setattr(answer_module, "settings", dataclasses.replace(answer_module.settings, PROMPT_WHO_WHAT_RULE=flag))
        answer_module.answer_query("who is alex morgan?")
        assert ("who or what someone or something is" in seen["prompt"]) is flag


def test_a_coverage_exactly_at_the_bar_is_admitted(index):
    assert admit("who is alex smith?", index, max_df=4, min_coverage=0.5) != []      # 1 of 2 = 0.5 clears a bar of 0.5


def test_top_k_bounds_a_text_only_question_too(wired):
    calls, lexical_search, image_search, set_lexical = wired
    set_lexical(True)
    found = retrieve_module.retrieve("q", top_k=1, include_images=False, lexical_search=lexical_search)
    assert [c.chunk_id for c in found] == ["lex_text"]


def test_a_short_number_is_kept_because_it_can_be_the_whole_identifier():
    assert content_tokens("Who sits in room 42?") == ["sits", "room", "42"]
    assert content_tokens("What is an ID?") == []                      # a two-letter word without a digit is not
