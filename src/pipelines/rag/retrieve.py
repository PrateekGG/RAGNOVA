"""
Cross-modal retrieval: search text_index and image_index separately, gate
each on its own relevance floor, then merge the two lists by RANK
(ADR-007), never by raw score.

Chapter 10 searched text_index only (documents, and — once indexed —
audio transcripts, which live there too per ADR-005). This module is the
Chapter 12 piece that lets one question also surface images.
"""

from __future__ import annotations

import re
from dataclasses import replace

from PIL import Image

from src.core.config import settings
from src.core.embeddings import embed_text
from src.core.schemas import Chunk
from src.pipelines.documents.search import search_text


def rrf_merge(ranked_lists: list[list[Chunk]], k: int | None = None) -> list[Chunk]:
    """Reciprocal Rank Fusion (ADR-007): every chunk earns
    `1 / (k + rank)` from each list it appears in (rank starting at 1),
    and the merged list is sorted by that total, best first.

    Why rank, in one line: CLIP's text-to-image scores are lower than
    MiniLM's text-to-text scores *even for perfect matches* (the modality
    gap), so sorting by raw score would bury every image. Rank ignores
    the scale difference; the #1 image and the #1 document tie.

    Pure function, no I/O. Ties keep input order (Python's sort is
    stable), so with one text list and one image list the result reads
    text #1, image #1, text #2, image #2, ... — plain interleaving, which
    ADR-007 names as the simplest acceptable form of rank merging.

    Returned chunks keep their ORIGINAL per-collection `.score`: the fused
    number is only used for ordering, never displayed, because showing
    it next to a cosine score would invite exactly the cross-scale
    comparison ADR-003 forbids.
    """
    k = settings.RRF_K if k is None else k
    fused: dict[str, float] = {}
    first_seen: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)
    order = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [replace(first_seen[cid]) for cid in order]


def filter_by_floor(chunks: list[Chunk], floor: float) -> list[Chunk]:
    """Keep chunks whose score clears `floor` (ADR-009's gate, with the
    floor passed in so text and image can each use their own number)."""
    return [c for c in chunks if c.score is not None and c.score >= floor]


def _ocr_agreement(query: str, chunk: Chunk) -> float:
    """MiniLM cosine similarity between the question and the text read from
    an image (its OCR). Vectors are unit length, so a dot product is the
    cosine. Only ever called for an image that CLIP alone did not convince."""
    return sum(a * b for a, b in zip(embed_text(query), embed_text(chunk.text)))


def filter_images(chunks: list[Chunk], query: str, agreement=None) -> list[Chunk]:
    """ADR-011's gate for text -> image results.

    CLIP's score alone cannot tell "the right image" from "the least wrong
    image": measured on this corpus, correct images score 0.216-0.346 while
    the top image for a question the corpus cannot answer scores up to
    0.291 (ADR-010's measurement update, repeated in ADR-011). So an image
    must clear MIN_IMAGE_RELEVANCE_SCORE *and* either

      - reach IMAGE_CONFIDENT_SCORE on CLIP alone, or
      - have OCR text whose MiniLM similarity to the question reaches
        MIN_IMAGE_TEXT_AGREEMENT: two independent models agreeing.

    An image with no readable text (a photo) cannot be corroborated, so it
    is kept only when CLIP alone is confident; that is the price of the
    rule, and it is stated rather than hidden. `agreement(query, chunk)`
    defaults to the MiniLM cosine above; tests inject a fake so they need
    no model weights.
    """
    agreement = agreement or _ocr_agreement
    kept = []
    for chunk in filter_by_floor(chunks, settings.MIN_IMAGE_RELEVANCE_SCORE):
        if chunk.score >= settings.IMAGE_CONFIDENT_SCORE:
            kept.append(chunk)
        elif (
            query.strip()
            and chunk.text.strip()
            and agreement(query, chunk) >= settings.MIN_IMAGE_TEXT_AGREEMENT
        ):
            kept.append(chunk)
    return kept


# Words that say nothing about WHAT a question is about: question words,
# function words, and the generic names for kinds of picture ("show me the
# screenshot of ..."), which would otherwise "share a word" with any image whose
# text happens to contain one of them. Used only by shares_content_word().
_GENERIC_WORDS = frozenset("""
a an the of in on at to for and or but if is are was were be been am do does did has have had
i me my we our you your he she it its they them their this that these those there here
what which who whom whose when where why how can could would should will shall may might must
with from by as about into over under than then so not no any some all each every more most
show find tell give get see look want need please
picture pictures image images photo photos photograph screenshot screenshots poster notice
page window diagram
""".split())


def _content_words(text: str) -> set[str]:
    """Lower-cased alphanumeric words of 3+ letters, minus _GENERIC_WORDS, with a
    trailing plural "s" removed ("books" matches "book"). Deliberately crude: it
    is a veto against answers that share no word at all, not a ranking signal."""
    words = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if len(raw) < 3 or raw in _GENERIC_WORDS:
            continue
        words.add(raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw)
    return words


def shares_content_word(query: str, text: str) -> bool:
    """True when the question and the text have at least one content word in
    common (see _content_words)."""
    return bool(_content_words(query) & _content_words(text))


def gate_image_channels(
    clip_hits: list[Chunk],
    ocr_hits: list[Chunk],
    query: str,
    agreement=None,
    ocr_only_min_agreement: float | None = None,
    require_shared_word: bool = False,
) -> list[Chunk]:
    """ADR-014: the image gate over TWO searches, CLIP (`clip_hits`, scores are
    CLIP cosines) and the OCR-text search (`ocr_hits`, scores are the question's
    MiniLM cosine with the image's text). Returns the surviving images, best
    first, in the order of the two rankings fused by rank (rrf_merge, ADR-007).

    An image is kept when EITHER
      - it passes ADR-011's rule on its CLIP score (filter_images' logic,
        unchanged: floor, then confident-by-CLIP or corroborated by its text), OR
      - `ocr_only_min_agreement` is set, the OCR search found it and its text
        agrees with the question at least that much (and, with
        `require_shared_word`, shares a content word with it), whatever CLIP
        thought of it.

    With no OCR hits and `ocr_only_min_agreement=None` this is exactly
    filter_images(clip_hits, query): the equivalence is tested, and it is what
    makes the channel safe to switch off. `agreement(query, chunk)` defaults to
    the MiniLM cosine and is injected by tests, as in filter_images().
    """
    agreement = agreement or _ocr_agreement
    clip_score = {c.chunk_id: c.score for c in clip_hits}
    ocr_score = {c.chunk_id: c.score for c in ocr_hits}
    kept = []
    for chunk in rrf_merge([clip_hits, ocr_hits]):
        score = clip_score.get(chunk.chunk_id)
        passes_clip_rule = False
        if score is not None and score >= settings.MIN_IMAGE_RELEVANCE_SCORE:
            passes_clip_rule = score >= settings.IMAGE_CONFIDENT_SCORE or bool(
                query.strip()
                and chunk.text.strip()
                and agreement(query, chunk) >= settings.MIN_IMAGE_TEXT_AGREEMENT
            )
        passes_ocr_rule = (
            ocr_only_min_agreement is not None
            and ocr_score.get(chunk.chunk_id, -1.0) >= ocr_only_min_agreement
            and (not require_shared_word or shares_content_word(query, chunk.text))
        )
        if passes_clip_rule or passes_ocr_rule:
            kept.append(chunk)
    return kept


def _lexical_first(lexical: list[Chunk], ranked: list[Chunk]) -> list[Chunk]:
    """`lexical` chunks first, then `ranked` without any of them (a chunk found both ways appears once)."""
    if not lexical:
        return ranked
    taken = {c.chunk_id for c in lexical}
    return [*lexical, *(c for c in ranked if c.chunk_id not in taken)]


def retrieve(
    query: str,
    top_k: int | None = None,
    client=None,
    include_images: bool = False,
    query_image: Image.Image | None = None,
    image_search=None,
    image_agreement=None,
    image_text_search=None,
    sources: list[str] | None = None,
    lexical_search=None,
    spelling_fixer=None,
) -> list[Chunk]:
    """The one retrieval call the RAG core makes: relevant chunks from
    every enabled collection, merged into one ranked list of at most
    `top_k`.

    `sources` is the scope the user chose (the "answer from these files only"
    picker in the page). None, the default, is everything and behaves exactly as
    before: the floors and the image gate decide what is relevant. With a list,
    the search is limited to those files and NO floor or gate is applied, because
    the user has already said where the answer lives: a question like "who is
    this person?" scores far below the corpus-wide floor against a CV that is
    plainly the thing being asked about (measured 2026-10-10 on an uploaded CV:
    1 of 6 questions about it retrieved it). The model's own refusal rule is what
    still guards a question the scoped files cannot answer.

    - text_index is always searched with `query` and gated on
      MIN_RELEVANCE_SCORE (exactly Chapter 10's behaviour).
    - image_index is searched with CLIP whenever `include_images=True`
      *or* a `query_image` is given (passing an image always means
      "search images too," even if the caller forgot the flag) — by
      `query_image` if the user uploaded one (image -> image), else by
      the question text (text -> image, even if that text is blank —
      unlike text_index, an image search with nothing typed is still a
      meaningful "show me relevant images" action). Text -> image results
      go through filter_images() (ADR-011: a CLIP floor plus OCR
      corroboration); image -> image results, whose scores are far higher
      and which have no question text to corroborate with, are gated on
      MIN_IMAGE_RELEVANCE_SCORE alone.

    With include_images=False and query_image=None, this returns exactly
    what Chapter 10's answer_query() used to compute inline, so the CLI
    and every existing test behave identically.

    With settings.LEXICAL_ADMIT on (off by default) and no `sources` scope, chunks whose text holds every
    distinctive word of the question (lexical.py) are put FIRST in their modality's list, ahead of the
    similarity-ranked hits and whether or not those cleared the floors: a name has no meaning for dense
    search to match, and an exact word is the right tool for it. `lexical_search(query, client=, max_df=)`
    returns (text chunks, image chunks) and is injected by tests (settings.LEXICAL_MODE, LEXICAL_MAX_DF).

    With settings.QUERY_SPELLING_FIX on, the similarity searches and the image gate use the question with
    words the corpus does not know swapped for the common word they were most likely meant to be
    (lexical.spelling_fix); `spelling_fixer(query, client=)` is injected by tests.

    `image_search` defaults to search_images() and `image_agreement` to the
    MiniLM OCR-agreement scorer; tests inject fakes so they don't need
    CLIP's or MiniLM's weights.
    """
    top_k = top_k or settings.TOP_K
    # An empty question happens for real in the UI: someone uploads a
    # photo with no text in it and types nothing. Embedding "" and
    # searching text_index with it would return 5 arbitrary chunks, so
    # there is simply no text search in that case — only the image one.
    lexical_text: list[Chunk] = []
    lexical_image: list[Chunk] = []
    if settings.LEXICAL_ADMIT and sources is None and query.strip():
        if lexical_search is None:
            from src.pipelines.rag.lexical import lexical_hits as lexical_search
        lexical_text, lexical_image = lexical_search(
            query, client=client, max_df=settings.LEXICAL_MAX_DF, mode=settings.LEXICAL_MODE
        )

    # The similarity searches get the question with its typos fixed ("lab rul no. 2" -> "lab rule no. 2"); the
    # lexical admit above and the prompt keep it as typed (lexical.spelling_fix says why).
    if settings.QUERY_SPELLING_FIX and query.strip():
        if spelling_fixer is None:
            from src.pipelines.rag.lexical import fix_spelling as spelling_fixer
        query = spelling_fixer(query, client=client)

    text_hits = []
    if query.strip():
        if sources is None:
            text_hits = filter_by_floor(
                search_text(query, top_k=top_k, client=client), settings.MIN_RELEVANCE_SCORE
            )
        else:
            text_hits = search_text(query, top_k=top_k, client=client, sources=sources)

    # query_image implies image search even if the caller forgot
    # include_images=True — a passed-in image should never be silently
    # dropped. include_images=True with no query_image still means
    # "search images too, using the question text," even when that text
    # is blank (test_retrieve_skips_text_search_for_an_empty_question
    # deliberately locks this in: an explicit include_images=True with no
    # image still runs the image branch on whatever query text exists,
    # blank or not — unlike the text branch, which is worth skipping
    # outright on blank input, "search with nothing typed" is still a
    # meaningful action for images).
    search_images_too = include_images or query_image is not None
    if not search_images_too:
        return _lexical_first(lexical_text, text_hits)[:top_k]
    text_hits = _lexical_first(lexical_text, text_hits)

    if image_search is None:
        from src.pipelines.images.search import search_images as image_search

    if sources is not None:
        # The user's scope: rank within it, no gate (see the docstring).
        if query_image is not None:
            image_hits = image_search(query_image=query_image, top_k=top_k, client=client, sources=sources)
        elif query.strip():
            image_hits = image_search(query_text=query, top_k=top_k, client=client, sources=sources)
        else:
            image_hits = []
    elif query_image is not None:
        raw_image_hits = image_search(query_image=query_image, top_k=top_k, client=client)
        image_hits = filter_by_floor(raw_image_hits, settings.MIN_IMAGE_RELEVANCE_SCORE)
    else:
        raw_image_hits = image_search(query_text=query, top_k=top_k, client=client)
        if settings.IMAGE_TEXT_SEARCH:
            if image_text_search is None:
                from src.pipelines.images.search import search_image_text as image_text_search
            ocr_hits = image_text_search(query_text=query, top_k=top_k, client=client)
            image_hits = gate_image_channels(
                raw_image_hits,
                ocr_hits,
                query,
                image_agreement,
                settings.IMAGE_OCR_ONLY_MIN_AGREEMENT,
                settings.IMAGE_OCR_ONLY_NEEDS_SHARED_WORD,
            )
        else:
            image_hits = filter_images(raw_image_hits, query, image_agreement)

    image_hits = _lexical_first(lexical_image, image_hits)
    return rrf_merge([text_hits, image_hits])[:top_k]
