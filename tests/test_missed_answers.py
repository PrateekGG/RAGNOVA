"""
Questions the live page refused or answered thinly on 2026-10-11 (ADR-018), and the rules that fix them:

  * "Who is Dr. S. Sharma?" must not admit Mr. V. Sharma's sign (the near-miss that kept ADR-017 off), while
    "Dr. S. Rao" and "Mr. V. Sharma" still are admitted: the name-phrase rule in lexical.py.
  * "tell me lab rul no. 2": the typo is fixed for the similarity searches (lexical.spelling_fix).
  * "tell me more" after an answer: followup.py, and the earlier exchange in the prompt.
  * one-line answers to "tell me in brief" and to a bare name: the detail rule in prompt.py.
  * only the sources the answer cites are listed (citations.cited_views).

No models, no ChromaDB.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.schemas import Chunk
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.answer import is_refusal
from src.pipelines.rag.followup import is_follow_up, resolve
from src.pipelines.rag.lexical import LexicalIndex, admit, correct_word, name_phrase_pairs, name_phrases, spelling_fix
from src.pipelines.rag.prompt import build_prompt
from src.ui.citations import citation_views, cited_views

ROWS = [
    ("lab1", "data/images/lab.png", "image",
     "Research Laboratory Room 302 Faculty In-Charge: Dr. S. Rao (Associate Professor) Lab Superintendent: Mr. V. Sharma"),
    ("rules", "data/images/rules.png", "image",
     "AIML Computing Laboratory Acceptable Use Rules 1. Each account is for one student. 2. Credential sharing is forbidden."),
    ("policy", "data/documents/it.docx", "text", "IT rules: the lab account rules apply to every student. Library hours."),
    ("hours", "data/documents/library.pdf", "text", "Library rules and hours for every student."),
    ("resume", "data/documents/resume.pdf", "text", "Projects: Raiden, a Discord bot. Engineered Riffy, a Lavalink client."),
]


@pytest.fixture(scope="module")
def index():
    return LexicalIndex(ROWS)


# ---------------------------------------------------------------------------
# Names with titles and initials
# ---------------------------------------------------------------------------

def test_a_title_and_initials_become_the_word_pairs_they_pin_down():
    assert name_phrase_pairs("Who is Dr. S. Sharma?") == [("dr", "s"), ("s", "sharma")]
    assert name_phrase_pairs("who is mr sharma") == [("mr", "sharma")]
    assert name_phrase_pairs("Who is Dr S Rao") == [("dr", "s"), ("s", "rao")]       # an initial after a title, no dot
    assert name_phrase_pairs("Who is Alex Morgan?") == [("alex", "morgan")]           # a capitalised full name
    assert name_phrase_pairs("who is alex morgan") == []                              # typed lowercase: nothing pinned
    assert name_phrase_pairs("Who is Alex, Smith?") == []
    assert name_phrase_pairs("What is Fusion-in-Decoder?") == []
    assert name_phrase_pairs("Can I see the lab rules?") == []                       # the pronoun I is not an initial


def test_the_near_miss_that_kept_the_lexical_admit_off_is_not_admitted(index):
    assert admit("Who is Dr. S. Sharma?", index) == []
    assert admit("Who is Dr. Sharma?", index) == []
    assert admit("Who is Mr. S. Rao?", index) == []


def test_a_name_is_matched_as_a_whole_phrase_not_pair_by_pair():
    # Greptile on PR #11: "Dr. S. Rao and Mr. S. Sharma" holds ("dr", "s") and ("s", "sharma"), but not Dr. S. Sharma.
    staff = LexicalIndex([("staff", "data/images/staff.png", "image", "Faculty: Dr. S. Rao and Mr. S. Sharma.")])
    assert admit("Who is Dr. S. Sharma?", staff, max_df=4) == []
    assert [e.chunk_id for e in admit("Who is Mr. S. Sharma?", staff, max_df=4)] == ["staff"]
    assert name_phrases("Who is Dr. S. Sharma?") == [("dr", "s", "sharma")]


def test_the_people_the_sign_does_name_are_still_admitted(index):
    assert [e.chunk_id for e in admit("Who is Dr. S. Rao?", index)] == ["lab1"]
    assert [e.chunk_id for e in admit("Who is Mr. V. Sharma?", index)] == ["lab1"]
    assert [e.chunk_id for e in admit("who is sharma", index)] == ["lab1"]          # no title asked, none required


def test_a_full_name_split_across_two_people_is_not_admitted():
    # E15 with nemotron-3-nano:4b: a reference list held "alex" and "smith" in two names, and the model glued them.
    names = LexicalIndex([
        ("refs", "data/documents/paper.pdf", "text", "References. Alex Wang, Noah A. Smith and others. Dense retrieval."),
        ("card", "data/images/card.png", "image", "Student ID card. Name: Alex Morgan. Roll No: 23AIML042."),
    ])
    assert admit("Who is Alex Smith?", names) == []
    assert [e.chunk_id for e in admit("Who is Alex Morgan?", names)] == ["card"]
    assert [e.chunk_id for e in admit("who is alex smith", names)] == ["refs"]   # lowercase: as before ADR-018's rule


def test_a_bare_word_the_corpus_holds_once_is_looked_up(index):
    assert [e.chunk_id for e in admit("riffy", index)] == ["resume"]


# ---------------------------------------------------------------------------
# Typos
# ---------------------------------------------------------------------------

def test_a_cut_off_word_is_completed_to_the_common_word_it_starts(index):
    assert correct_word("rul", index) == "rules"
    assert spelling_fix("tell me lab rul no. 2", index) == "tell me lab rules no. 2"


def test_a_word_one_keystroke_from_a_common_word_is_fixed(index):
    assert correct_word("libary", index) == "library"
    assert correct_word("studnt", index) == "student"


def test_known_words_names_and_short_or_numeric_words_are_left_alone(index):
    assert correct_word("lab", index) is None          # known
    assert correct_word("smith", index) is None        # nothing common is one keystroke away
    assert correct_word("riffx", index) is None        # "riffy" is in one file only: never correct TO a name
    assert correct_word("no", index) is None           # too short
    assert spelling_fix("What is LIB-884920?", index) == "What is LIB-884920?"


def test_the_similarity_searches_get_the_fixed_question_and_the_lexical_admit_the_typed_one(monkeypatch):
    seen = {}
    monkeypatch.setattr(retrieve_module, "settings",
                        dataclasses.replace(retrieve_module.settings, QUERY_SPELLING_FIX=True, LEXICAL_ADMIT=True))
    monkeypatch.setattr(retrieve_module, "search_text", lambda q, **kw: seen.setdefault("dense", q) and [])

    def lexical(q, **kw):
        seen["lexical"] = q
        return [], []

    retrieve_module.retrieve("lab rul", lexical_search=lexical, spelling_fixer=lambda q, client=None: "lab rules")
    assert seen == {"lexical": "lab rul", "dense": "lab rules"}


def test_with_the_spelling_fix_off_the_fixer_is_never_called(monkeypatch):
    monkeypatch.setattr(retrieve_module, "search_text", lambda q, **kw: [])

    def must_not_run(q, client=None):
        raise AssertionError("called")

    retrieve_module.retrieve("lab rul", spelling_fixer=must_not_run)       # conftest switches it off


# ---------------------------------------------------------------------------
# Follow-ups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", ["tell me more", "Tell me more.", "why?", "more", "elaborate", "who made it?",
                                      "what about rule 3?", "and the second one?", "explain that"])
def test_questions_that_lean_on_the_previous_turn_are_follow_ups(question):
    assert is_follow_up(question)


@pytest.mark.parametrize("question", ["what is greedy algorithm tell me in brief", "riffy", "tell me lab rules",
                                      "who is alex morgan?", "why does dijkstra fail with negative edges?", ""])
def test_questions_that_stand_on_their_own_are_not(question):
    assert not is_follow_up(question)


def test_a_follow_up_is_searched_with_the_previous_question_and_its_own_words():
    assert resolve("tell me more", "who is raiden?") == "who is raiden?"
    assert resolve("what about rule 3?", "tell me lab rules") == "tell me lab rules rule 3"
    assert resolve("tell me lab rules", "who is raiden?") == "tell me lab rules"
    assert resolve("tell me more", None) == "tell me more"


def test_a_follow_up_with_a_new_number_or_name_replaces_the_old_one():
    # Greptile on PR #11: "what about rule 3?" after "lab rule 2" searched rules 2 AND 3.
    search = resolve("what about rule 3?", "tell me lab rule no. 2")
    assert "3" in search.split() and "2" not in search.split() and "lab" in search
    search = resolve("what about Mr. V. Sharma?", "Who is Dr. S. Rao?")
    assert "rao" not in search.lower() and "mr v sharma" in search
    assert resolve("what about rule 3?", "tell me lab rules") == "tell me lab rules rule 3"     # nothing to replace


def _chunk():
    return Chunk(chunk_id="r", source="data/documents/resume.pdf", modality="pdf", text="Raiden is a Discord bot.",
                 page=1, embedding_model="m", score=0.9)


def test_a_follow_up_searches_with_the_earlier_question_and_shows_the_exchange_to_the_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: seen.setdefault("search", q) and [_chunk()])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")

    answer_module.answer_query("tell me more", history=[("who is raiden?", "Raiden [1] is a Discord app.")])

    assert seen["search"] == "who is raiden?"
    assert "Earlier conversation:\nUser: who is raiden?\nAssistant: Raiden is a Discord app." in seen["prompt"]
    assert seen["prompt"].endswith("Question: tell me more\n\nAnswer:")


def test_a_question_that_stands_alone_ignores_the_history(monkeypatch):
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: seen.setdefault("search", q) and [_chunk()])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")

    answer_module.answer_query("tell me lab rules", history=[("who is raiden?", "Raiden is a Discord app.")])

    assert seen["search"] == "tell me lab rules"
    assert "Earlier conversation" not in seen["prompt"]


def test_a_question_with_an_attached_image_is_never_a_follow_up():
    assert answer_module.search_question("what is this?", [("who is raiden?", "x")], attachments=[_chunk()]) == "what is this?"
    assert answer_module.search_question("what is this?", [("who is raiden?", "x")]) == "who is raiden?"


def test_the_earlier_answer_is_shortened_and_loses_its_citation_numbers():
    long_answer = " ".join(["word"] * 200) + " [3]"
    prompt = build_prompt("more", [_chunk()], history=[("q", long_answer)])
    assert "[3]" not in prompt.split("Earlier conversation:")[1]
    assert "word ..." in prompt


# ---------------------------------------------------------------------------
# Fuller answers
# ---------------------------------------------------------------------------

def test_the_detail_rule_asks_for_a_paragraph_for_brief_and_treats_a_bare_term_as_tell_me_about_it():
    prompt = build_prompt("riffy", [_chunk()], detail_rule=True)
    assert "two to four sentences" in prompt and "only a name or a term" in prompt
    assert "reply with the refusal sentence of the last rule, word for word" in prompt
    assert "I don't have enough information in the indexed documents to answer that." in prompt
    assert "detail" not in build_prompt("riffy", [_chunk()])          # off: the prompt every earlier run used


def test_the_rules_are_numbered_in_order_with_the_refusal_last():
    prompt = build_prompt("q", [_chunk()], who_what_rule=True, detail_rule=True, history=[("a", "b")])
    numbers = [line.split(".")[0] for line in prompt.split("\n") if line[:2] in {f"{i}." for i in range(1, 9)}]
    assert numbers == ["1", "2", "3", "4", "5", "6"]
    assert "6. If the context does not mention" in prompt


def test_the_shipped_settings_switch_the_new_rules_on():
    from src.core.config import Settings
    shipped = Settings()
    assert shipped.LEXICAL_ADMIT and shipped.PROMPT_WHO_WHAT_RULE and shipped.PROMPT_DETAIL_RULE and shipped.QUERY_SPELLING_FIX


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def test_only_the_cited_sources_are_listed_and_they_keep_their_numbers():
    chunks = [_chunk(), dataclasses.replace(_chunk(), chunk_id="b", source="data/images/fair.png", modality="image")]
    views = citation_views(chunks)
    assert [v.number for v in cited_views(views, {2})] == [2]
    assert cited_views(views, {2})[0].title.startswith("[2] ")
    assert [v.number for v in cited_views(views, set())] == [1, 2]       # nothing cited: show them all
    assert [v.number for v in cited_views(views, {9})] == [1, 2]         # only out-of-range numbers: the same


# ---------------------------------------------------------------------------
# Whole-file questions under "Answer from"
# ---------------------------------------------------------------------------

def test_with_files_picked_the_model_is_told_what_it_and_this_file_mean():
    prompt = build_prompt("List the names, dates and numbers it mentions.", [_chunk()],
                          scope=["data/documents/it_onboarding.docx"])
    assert "The user chose to answer from these files only: it_onboarding.docx." in prompt
    assert "list what they mention is answered from that context" in prompt
    assert "chose to answer from" not in build_prompt("q", [_chunk()])


def test_the_picked_files_reach_the_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [_chunk()])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")
    answer_module.answer_query("Give me a short summary of this file.", sources=["data/documents/resume.pdf"])
    assert "answer from these files only: resume.pdf" in seen["prompt"]


def test_with_files_picked_it_means_the_files_so_only_a_wordless_question_is_a_follow_up():
    history = [("who is raiden?", "x")]
    assert answer_module.search_question("summarise it", history, sources=["a.pdf"]) == "summarise it"
    assert answer_module.search_question("tell me more", history, sources=["a.pdf"]) == "who is raiden?"
    assert answer_module.search_question("summarise it", history) == "who is raiden? summarise"


def _prompt_for(monkeypatch, question, sources):
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [_chunk()])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")
    answer_module.answer_query(question, sources=sources)
    return seen["prompt"]


@pytest.mark.parametrize("question", ["explain", "Explain.", "summarize", "tell me more", "brief overview"])
def test_a_question_with_no_subject_asks_about_the_picked_files(monkeypatch, question):
    # 2026-10-11: "explain" with algorithms_ch04_greedy.pdf picked was refused; "Give me a short summary" was not.
    prompt = _prompt_for(monkeypatch, question, ["data/documents/greedy.pdf"])
    assert f'The user wrote "{question}" about the files they chose' in prompt


@pytest.mark.parametrize("question, sources", [("explain greedy choice", ["a.pdf"]), ("explain", None)])
def test_a_question_with_a_subject_or_without_picked_files_is_asked_as_typed(monkeypatch, question, sources):
    assert "about the files they chose" not in _prompt_for(monkeypatch, question, sources)


def _attachment():
    return Chunk(chunk_id="attachment-1", source="attached image (shot.png)", modality="image",
                 text="Wi-Fi password: ragnova-guest-2026", embedding_model="none")


def test_a_follow_up_is_answered_from_the_earlier_turns_attachment(monkeypatch):
    # Greptile on PR #11: after "What does this image say?" about an unindexed screenshot, "tell me more" lost it.
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: seen.setdefault("prompt", prompt) and "ok [1]")
    history = [("What does this image say?", "It gives the guest Wi-Fi password [1].")]
    result = answer_module.answer_query("tell me more", history=history, earlier_attachments=[_attachment()])
    assert result.citations[0].chunk_id == "attachment-1" and "ragnova-guest-2026" in seen["prompt"]


def test_a_question_that_stands_on_its_own_does_not_get_the_earlier_attachment(monkeypatch):
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [])
    history = [("What does this image say?", "It gives the guest Wi-Fi password [1].")]
    result = answer_module.answer_query("What time does the library open?", history=history,
                                        earlier_attachments=[_attachment()])
    assert result.citations == [] and is_refusal(result.answer)


def test_a_follow_ups_context_is_trimmed_against_the_question_it_searched_with(monkeypatch):
    # Greptile on PR #11: with CONTEXT_WORDS_PER_CHUNK on, "tell me more" trimmed chunks against "tell me more".
    seen = {}
    monkeypatch.setattr(answer_module, "retrieve", lambda q, **kw: [_chunk()])
    monkeypatch.setattr(answer_module, "generate", lambda prompt: "ok [1]")
    monkeypatch.setattr(answer_module, "trim_chunks", lambda chunks, query, words: seen.setdefault("q", query) and chunks)
    answer_module.answer_query("tell me more", history=[("who is raiden?", "A Discord bot [1].")])
    assert seen["q"] == "who is raiden?"


def test_the_word_index_is_built_once_per_database_not_once_per_client(tmp_path):
    # Greptile on PR #11: lexical_hits() and fix_spelling() each made a client, and the cache was keyed by id(client).
    from src.core.vector_store import get_client
    from src.pipelines.rag.lexical import _index_for
    assert _index_for(get_client(tmp_path)) is _index_for(get_client(tmp_path))


# --- a refusal as small models actually type it (nemotron-3-nano:4b, 2026-10-11) ---

@pytest.mark.parametrize("text", [
    "I don't have enough information in the indexed documents to answer that.",
    "I don' have enough information in the indexed documents to answer that.",
    "I don' t have enough information in the indexed documents to answer that.",
    "I don’t have enough information in the indexed documents to answer that.",
    "  I do not have enough information in the indexed documents to answer that.",
])
def test_a_mangled_refusal_is_still_a_refusal(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "Alex Smith is an author of the paper [1].",
    "I do have enough information in the indexed documents to answer that: rule 2 forbids sharing [1].",
    "Room 302 is the AI Research Lab. I don't have enough information in the indexed documents to say more.",
    "",
])
def test_an_answer_is_not_a_refusal(text):
    assert not is_refusal(text)
