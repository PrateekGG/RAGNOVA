# ADR-016: Answer from the files you pick, and treat an attached image as a source

## Status
Accepted: 2026-10-10. Refines ADR-009 (the relevance floor stays the default; a user-chosen scope bypasses it) and ADR-011 (the image gate is likewise bypassed inside a scope). Changes what the page does with an attached image (Chapter 11).

## Context

Two reports from the first person to use the page on their own material, 2026-10-10:

1. **A document added with "Add to Corpus" could not be asked about.** A two-page CV was uploaded and indexed (the chunk counter rose by 2). Of six questions about it, one retrieved it: "who is *name*", "what is in the resume", "what are the skills in my resume", "what projects are listed on the resume" and "tell me about *the file's name*" retrieved nothing from it. Every question is gated on a corpus-wide relevance floor (`MIN_RELEVANCE_SCORE`, 0.3, ADR-009), and a broad question scores below it against a long, dense chunk. The floor is what refuses questions the corpus cannot answer (the negative controls), so it cannot simply be lowered.
2. **An attached image was never read.** Two screenshots that were not in the corpus were attached (Tesseract read 540 and 3,539 characters from them); both were refused. The picture was only a *search key*: CLIP looked for similar indexed images and its OCR text was glued onto the question, but the text was never shown to the model as something it could answer from. The model is text-only (`llama3.2:3b`), so what a picture can contribute is the text in it.

The same cause sits under both: the page is a **closed-world search over a fixed corpus**, while the person expected what a notebook tool does, where the sources you add are the scope of the conversation.

## Decision

1. **"Answer from" picker (scope).** `retrieve(..., sources=[...])` limits the search to the chosen files (a ChromaDB `$in` filter on `source`, in `search_text`, `search_images`, `search_image_text`). **Inside a scope there is no relevance floor and no image gate**: the user has said where the answer lives. The model's own refusal rule is what still guards a question the scoped files cannot answer. **With nothing picked (the default) the path is unchanged**, tested for equality, so every shipped threshold and every earlier evaluation stands.
2. **An upload that indexes becomes the scope**, with a banner ("Answering from X only") and a **Use everything** button. One that indexes to 0 chunks (a scanned PDF with no text layer; a picture with no text) is reported as a failure; the old page said "Indexed (0 chunks added)" in green.
3. **An attached image is context [1].** It is built by `image_attachment()`, placed first, labelled "attached image (name)", never filtered, bounded to 400 words (about 520 of the 4,096 context tokens), and cited like any chunk. When one is attached the prompt also says which block it is and that "this image" means it. The model is asked the question as typed; **the picture's text is not appended to the retrieval query** (see below). The page reports whether the picture had text, had none, or Tesseract is missing: three different facts the old page reported as one.
4. **A refusal lists no sources.** With a scope or an attachment chunks are retrieved before the model decides they do not hold the answer; "Sources" under a refusal reads as grounding.
5. One-click questions (summary, key points, names/dates/numbers) appear while a scope is set.

## What was measured

`scripts/check_scoped_sources.py` (functional check, not an evaluation: a handful of rows, real model, scratch copy of the index, a synthetic CV and synthetic notices, nothing personal). Expectations are in the script; the four runs are in `data/eval/scoped_sources_2026-10-10_*`.

| | result (final run) | what it shows |
|---|---|---|
| Scoped to the CV, 6 questions about it | 5/6 by the pre-declared keyword check; the sixth ("what is in this document?") answered "a resume or CV with education, skills, projects and contact information", a sensible answer the keyword list missed | the report 1 failure no longer happens |
| Scoped to the CV, 4 questions it cannot answer (capital of France, working-prototype marks, Wi-Fi name, 2018 World Cup) | 4/4 refused | taking the floor off did not make the model answer questions the file does not contain (n=4) |
| Same CV, **not** scoped, "who is this person?" | refused | why the scope exists (U3, predicted to fail, did) |
| Attached library notice, 5 questions | 5/5 | the report 2 failure no longer happens |
| Attached shuttle notice (an image the fix was not written against), 3 questions | 3/3 | |
| Attached text-free picture, "what does this show?" | refused, no invented description | |

**A failure found on the way and fixed.** The first runs answered "What does this image say?" from a *different* image already in the index (the circulation-desk notice) or refused: the context held several images and "this image" had no single referent: 3 of the 5 "this image / this picture" questions failed (A2 and A5 on the notice, A7 on the shuttle). Naming the attachment's block in the prompt fixed it (notice 5/5, shuttle 3/3). The fix was written after seeing A2, and A5-A9 (a second image the fix was not written against) were added before re-running, but the whole set is still eight attachment questions, so read it as "the failure is gone on these rows", not as a rate.

**Gluing the picture's text onto the retrieval query was removed.** Nine synthetic rows pass either way (9/9). On four questions about two real screenshots (the user's own, not committed): 0 useful answers with the glue, 2 without, and fewer unrelated context blocks (1-2 instead of 2-6). One run each of a stochastic model on four questions is weak evidence; it is recorded as the reason, not as proof (`data/eval/attachment_retrieval_glue_2026-10-10.txt`).

**Look-alike image search kept.** The page still also uses an attached picture as a search key (similar indexed images are retrieved). A pre-declared probe asked whether that hurts the answer about the attachment: complete answers 8/12 with it, 5/12 without; the rule (switch only on +3 for "without") says keep (`data/eval/attachment_lookalikes_2026-10-10.txt`). Repeats of one row often returned identical answers, so the effective sample is the four rows.

## Alternatives considered

- **Lower `MIN_RELEVANCE_SCORE`.** Rejected: it is the negative-control defence (ADR-009/010) and a global change would be tuned on the one CV that exposed it. A per-question scope leaves the evaluated default alone.
- **Auto-scope to the last upload permanently.** Rejected: surprising; it silently excludes the rest of the corpus. The banner and the button make the scope visible and undoable.
- **Add a vision-language model** so a photo can be described, not only read. Not done: a multi-gigabyte download, a new dependency and a new failure mode, and it needs the user's approval for the download. The page says plainly that it reads text in pictures and cannot describe a photo.
- **Delete uploaded files / a "remove source" button.** Not done: uploads land in `data/documents`, `data/images`, `data/audio` next to the shipped corpus, and removing from the index and the folder is a destructive action that needs its own design.

## Consequences

- A person can add a file and ask about it, which is what the first user tried; the evaluated corpus-wide behaviour is untouched unless a scope is chosen.
- **Inside a scope the only refusal guard is the model's.** Four off-topic questions were refused; that is a small sample on a 3B model, and a larger or a different model needs the check re-run.
- **Answers to "what does this image say?" are often only the headline** ("The image says LIBRARY NOTICE") although the dates are in context [1]; across the four "describe it" rows, 8 of 12 page-style answers were complete. The probe suggests this is the model's habit, not the retrieval (dropping the look-alike search did not help).
- A picture with no text, or a photo, cannot be answered about. A screenshot of the app itself contains the app's own refusal text; that was tested as a cause of one refusal and **rejected** (removing the sentence changed nothing). That screenshot is refused with or without it, and the attachment alone answers "what does this say" but not "what is this about".
- **Uploads still write into the shipped corpus folders.** A personal file added through the page (a CV with a phone number) sits untracked in `data/documents/` until someone deletes it, and `git add data/` would stage it.
- The pinned compatibility environment (Streamlit 1.41.1, ChromaDB 0.5.23) could not run the new tests on 2026-10-10: Windows Application Control blocks scikit-learn's native library there (`import sentence_transformers` fails with no repository code involved). Verified separately without the repository: the `$in` filter behaves identically on ChromaDB 0.5.23 and 1.5.9, and Streamlit 1.41.1 *raises* on a stored multiselect value that is not among the options (1.65 silently drops it), which is why the page removes stale picks before drawing the picker. The page test for that guard can only fail on 1.41.1, so it is **unverified** until the pinned environment runs.

## Revisit if

- The model changes (re-run `scripts/check_scoped_sources.py`, especially the four off-topic rows inside a scope).
- A second person tries it and the scope default (select the upload) surprises them.
- Uploads need removing, or a "personal" folder separate from the shipped corpus.
- A vision model is approved for download: then a photo can be described.
