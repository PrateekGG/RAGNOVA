"""Prompt construction — the concrete implementation of ADR-001's mandate
that the model answer only from retrieved evidence and cite it.

Deliberately a pure function: no I/O, no imports of `src.core.llm` or
`src.core.vector_store`. Given the same `query` and `chunks`, it always
returns the same string — which is what makes it independently unit
testable without ChromaDB or Ollama running.
"""

from __future__ import annotations

import re

from src.core.schemas import Chunk

# The id prefix attachments.py gives an attached image. Kept here as a literal
# rather than imported: attachments.py is the one that may grow, this file must
# stay a pure function with nothing to import but the schema.
ATTACHMENT_ID_PREFIX = "attachment__"

_HEADER = (
    "You answer questions using ONLY the numbered context below, taken from "
    "the user's own indexed files. Follow these rules exactly:\n"
)
_RULE_ONLY_CONTEXT = (
    "Base your answer only on the context given. Never use outside "
    "knowledge, even if you happen to know the answer."
)
_RULE_CITE = (
    "Cite every claim with the matching bracketed number, e.g. [1] or "
    "[2], right after the sentence it supports."
)
# PROMPT_WHO_WHAT_RULE (ADR-017): for a "who / what is X" question the model says what the context states about X.
_RULE_WHO_WHAT = (
    "If the question asks who or what someone or something is, answer with "
    "what the context says about it (for a person, what the context states about "
    "them), and nothing more."
)
# PROMPT_DETAIL_RULE (2026-10-11): with only rules 1-3 the 3B model answered "what is greedy algorithm, tell me in
# brief" and "who is Raiden" with one bare sentence and no citation, though five sources were retrieved. Its last
# sentence was added after the first V8 run: asked for detail, the model refused N20 in its own words ("The context
# does not mention the number of layers ...") instead of the fixed sentence, so neither the page nor the evaluation
# could tell it was a refusal, and the page listed sources under it.
_RULE_DETAIL = (
    "Use the context fully: answer in complete sentences with the relevant details "
    "it gives (definitions, key points, steps, numbers, names). When the user asks "
    "for a brief answer, give a short paragraph of two to four sentences with the key "
    "points, not a single line. A question that is only a name or a term means: what "
    "does the context say about it? When the context does not hold the answer, do not "
    "explain what is missing: reply with the refusal sentence of the last rule, word for word."
)
# Only when there is an earlier exchange (follow-ups, 2026-10-11): "tell me more" means nothing on its own.
_RULE_HISTORY = (
    "The earlier conversation is shown only so you know what a follow-up such as "
    "\"tell me more\", \"it\" or \"they\" refers to. Take every fact from the "
    "numbered context, never from the earlier answers."
)
_REFUSAL = (
    '"I don\'t have enough information in the indexed documents to answer '
    'that." Do not guess, and do not cite a number if you are not using it.'
)
_RULE_REFUSE = "If the context does not contain the answer, say plainly: " + _REFUSAL
# With the who/what rule the refusal also covers "does not contain the specific fact asked for", so that the new
# rule cannot be read as permission to guess.
_RULE_REFUSE_STRICT = (
    "If the context does not mention what the question asks about, or does not "
    "contain the specific fact asked for, say plainly: " + _REFUSAL
)


def _instructions(who_what_rule: bool = False, detail_rule: bool = False, has_history: bool = False) -> str:
    """The numbered rules. With every option off this is, byte for byte, the prompt every earlier measurement used."""
    rules = [_RULE_ONLY_CONTEXT, _RULE_CITE]
    if who_what_rule:
        rules.append(_RULE_WHO_WHAT)
    if detail_rule:
        rules.append(_RULE_DETAIL)
    if has_history:
        rules.append(_RULE_HISTORY)
    rules.append(_RULE_REFUSE_STRICT if who_what_rule else _RULE_REFUSE)
    return _HEADER + "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))


_SYSTEM_INSTRUCTIONS = _instructions()
_SYSTEM_INSTRUCTIONS_WHO_WHAT = _instructions(who_what_rule=True)

_NO_CONTEXT_NOTICE = "(No relevant context was found for this question.)"

# How much of the previous answer the model is shown for a follow-up: enough to know the topic, not so much that
# it crowds out the context (the model reads about 150 tokens a second on a CPU, ADR-015).
HISTORY_ANSWER_WORDS = 80


def _history_section(history: list[tuple[str, str]] | None) -> str:
    """The earlier exchange(s), (question, answer) oldest first, without their [n] markers: those numbers belonged to
    another context and would point at the wrong block here."""
    if not history:
        return ""
    lines = []
    for question, answer in history:
        words = re.sub(r"\s*\[\d+\]", "", answer).split()
        short = " ".join(words[:HISTORY_ANSWER_WORDS]) + (" ..." if len(words) > HISTORY_ANSWER_WORDS else "")
        lines.append(f"User: {question}\nAssistant: {short}")
    return "Earlier conversation:\n" + "\n".join(lines) + "\n\n"


def format_provenance(chunk: Chunk) -> str:
    """One-line "where this came from" description for a chunk — used both
    to label a numbered context block in the prompt, and (by
    scripts/ask.py) to render the human-facing "Sources:" footer.

    Documents cite a page (ADR-008's honest-best-effort DOCX page still
    applies here — this function doesn't distinguish pdf from docx). Audio
    would cite a timestamp range; no audio chunk exists in the vector store
    yet (Ch9 isn't wired in), but the branch is written now so this
    function doesn't need to change the day it is.
    """
    if chunk.chunk_id.startswith(ATTACHMENT_ID_PREFIX):
        return chunk.source                      # already reads "attached image (name)"
    if chunk.modality in ("pdf", "docx") and chunk.page is not None:
        return f"{chunk.source}, page {chunk.page}"
    if chunk.modality == "audio" and chunk.start_s is not None and chunk.end_s is not None:
        return f"{chunk.source}, {chunk.start_s:.0f}s–{chunk.end_s:.0f}s"
    if chunk.modality == "image":
        return f"{chunk.source} (image)"
    return chunk.source


# What the model is shown for an image whose OCR found no text. The LLM
# reads text only, so it can't look at the picture — saying so plainly
# stops it from inventing a description, while still letting the image
# appear as a numbered, citable source ("see [3]").
_IMAGE_WITHOUT_TEXT = "(An image matching the question. No readable text was found in it.)"


# The same, for an image the user attached to the question (attachments.py).
_ATTACHMENT_WITHOUT_TEXT = (
    "(The user attached this image. No readable text was found in it, and you cannot "
    "look at pictures, so say that you cannot tell what it shows.)"
)


def _attachment_note(chunks: list[Chunk]) -> str:
    """When the user attached an image, which numbered block it is. Without this
    the context can hold several images (the attachment plus look-alikes retrieval
    found) and "this image" has no single referent: measured 2026-10-10, 3 of the 5
    "this image" questions about attached pictures were answered from a different
    image or refused, e.g. "What does this image say?" answered with another
    notice's text (data/eval/scoped_sources_2026-10-10_run2_*; ADR-016)."""
    numbers = [i for i, chunk in enumerate(chunks, start=1) if chunk.chunk_id.startswith(ATTACHMENT_ID_PREFIX)]
    if not numbers:
        return ""
    which = " and ".join(f"[{n}]" for n in numbers)
    return (
        f"The user attached an image to this question: {which}. When the question says \"this image\", "
        f"\"the image\" or \"the picture\", it means {which}, not any other image in the context.\n\n"
    )


def _context_text(chunk: Chunk) -> str:
    if chunk.chunk_id.startswith(ATTACHMENT_ID_PREFIX) and not chunk.text.strip():
        return _ATTACHMENT_WITHOUT_TEXT
    if chunk.modality == "image" and not chunk.text.strip():
        return _IMAGE_WITHOUT_TEXT
    if chunk.modality == "image":
        return f"Text read from the image (OCR): {chunk.text}"
    return chunk.text


def _scope_note(scope: list[str] | None) -> str:
    """When the user picked files under "Answer from", which files "it" / "this file" means. Without this the model
    never learns that a scope exists: on 2026-10-11, with it_onboarding.docx picked, "List the names, dates and numbers
    it mentions." was refused though the blocks held a two-week suspension and other numbers (ADR-018)."""
    if not scope:
        return ""
    names = ", ".join(name.replace("\\", "/").rsplit("/", 1)[-1] for name in scope)
    return (
        f"The user chose to answer from these files only: {names}. When the question says \"it\", \"this file\", "
        f"\"the document\" or \"these files\", it means them, and the numbered context is taken from them. A request "
        f"to summarise them, give their key points, or list what they mention is answered from that context.\n\n"
    )


def build_prompt(
    query: str,
    chunks: list[Chunk],
    who_what_rule: bool = False,
    detail_rule: bool = False,
    history: list[tuple[str, str]] | None = None,
    scope: list[str] | None = None,
) -> str:
    """Build the full prompt sent to the LLM: system instructions, numbered
    context blocks (best match first, matching `chunks`' own order), then
    the question.

    `chunks` is expected to already be filtered to ones worth showing the
    model (see settings.MIN_RELEVANCE_SCORE in answer.py) — this function
    itself applies no threshold, so an empty list here always means
    "nothing survived filtering," not "nothing was retrieved."
    """
    if chunks:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(f"[{i}] {format_provenance(chunk)}\n{_context_text(chunk)}")
        context_section = "\n\n".join(blocks)
    else:
        context_section = _NO_CONTEXT_NOTICE

    instructions = _instructions(who_what_rule, detail_rule, bool(history))
    return (
        f"{instructions}\n\n"
        f"Context:\n{context_section}\n\n"
        f"{_history_section(history)}"
        f"{_scope_note(scope)}"
        f"{_attachment_note(chunks)}"
        f"Question: {query}\n\n"
        f"Answer:"
    )
