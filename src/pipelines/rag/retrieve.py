"""
Cross-modal retrieval: search text_index and image_index separately, gate
each on its own relevance floor, then merge the two lists by RANK
(ADR-007), never by raw score.

Chapter 10 searched text_index only (documents, and — once indexed —
audio transcripts, which live there too per ADR-005). This module is the
Chapter 12 piece that lets one question also surface images.
"""

from __future__ import annotations

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


def retrieve(
    query: str,
    top_k: int | None = None,
    client=None,
    include_images: bool = False,
    query_image: Image.Image | None = None,
    image_search=None,
    image_agreement=None,
) -> list[Chunk]:
    """The one retrieval call the RAG core makes: relevant chunks from
    every enabled collection, merged into one ranked list of at most
    `top_k`.

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

    `image_search` defaults to search_images() and `image_agreement` to the
    MiniLM OCR-agreement scorer; tests inject fakes so they don't need
    CLIP's or MiniLM's weights.
    """
    top_k = top_k or settings.TOP_K
    # An empty question happens for real in the UI: someone uploads a
    # photo with no text in it and types nothing. Embedding "" and
    # searching text_index with it would return 5 arbitrary chunks, so
    # there is simply no text search in that case — only the image one.
    text_hits = []
    if query.strip():
        text_hits = filter_by_floor(
            search_text(query, top_k=top_k, client=client), settings.MIN_RELEVANCE_SCORE
        )

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
        return text_hits

    if image_search is None:
        from src.pipelines.images.search import search_images as image_search

    if query_image is not None:
        raw_image_hits = image_search(query_image=query_image, top_k=top_k, client=client)
        image_hits = filter_by_floor(raw_image_hits, settings.MIN_IMAGE_RELEVANCE_SCORE)
    else:
        raw_image_hits = image_search(query_text=query, top_k=top_k, client=client)
        image_hits = filter_images(raw_image_hits, query, image_agreement)

    return rrf_merge([text_hits, image_hits])[:top_k]
