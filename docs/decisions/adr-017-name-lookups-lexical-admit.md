# ADR-017: Name and identifier lookups: a lexical admit and a who/what-is prompt rule, built and measured, not switched on

## Status
Accepted as **opt-in capabilities, not enabled by default**: 2026-10-11. Both are off because the adoption rule, fixed in `scripts/evaluate_entity_questions.py` and committed before the run (`b4ff52d`), named no candidate: every variant that fixed more name questions also answered one near-miss negative it should have refused. Refines ADR-011 (the image gate that drops a name) and ADR-009 (the floor); the same shape of outcome as ADR-014.

## Context

"Who is Alex Morgan?" was refused on the live page although the corpus holds Alex Morgan's ID card (`photo_id_card_sample.png`, indexed with its text: "STUDENT ID Alex Morgan | Roll No: 23AIML042 ..."). Looking at it, there were two separate causes:

1. **Retrieval.** CLIP cannot match a name: the text-to-image score for the card is 0.151 against a 0.2 floor. MiniLM scores a two-word name against a whole ID card's text at 0.16 against the gate's 0.30 (ADR-011). Dense search is built for meaning, and a name has none to match. Questions that use the card's own words ("what is Alex Morgan's *student id*") get through, which is why this had not been seen.
2. **The model.** With the card as the *only* source in scope, the shipped prompt still made `llama3.2:3b` refuse "who is Alex Morgan?", "tell me about Alex Morgan" and "what do you know about Alex Morgan?" (3 of 5 answerable questions), while answering "what is Alex Morgan's roll number?" correctly.

It is not about this one card: on 13 name / identifier / product lookups the corpus mentions literally, **the shipped system answers 5**, and in most of the misses nothing relevant is retrieved at all.

## Decision

Build both fixes and keep them **off**:

1. `src/pipelines/rag/lexical.py`, `LEXICAL_ADMIT=1`. A chunk is admitted when every *distinctive* word of the question (one occurring in at most `LEXICAL_MAX_DF` indexed files) is literally in it (`LEXICAL_MODE=distinctive`), or every word the corpus knows (`LEXICAL_MODE=all`). Unknown words veto: "Alex Smith" is not looked up (smith is in no file). Admitted chunks (at most three, shortest first) go first in their modality's list and bypass the floor and the image gate; the model's refusal rule still decides, as for every image the gate lets through. Not used inside an "Answer from" scope, which already bypasses the floors (ADR-016).
2. `PROMPT_WHO_WHAT_RULE=1`. One more rule in the prompt: for "who or what is X", say what the context states about X, and nothing more; the refusal sentence now also covers "does not contain the specific fact asked for". The default prompt is byte-identical (tested).

## What was measured, and the rule that was fixed first

23 entity rows, committed before any run (`eebdb01`): 13 positives (3 documents, 10 images: Alex Morgan, "LIB-884920", "Dr. S. Rao", "Room 302", "vpn.ragnova.edu", ...) and 10 negatives (far-misses, and near-misses such as "Alex Smith", "Morgan Freeman", "Room 999", "Alex Morgan's blood group"). A positive is correct when the answer is not a refusal, contains a keyword, **and** cites an expected source; a negative when it is refused. One of the rows, E4 ("Who is Alex Morgan?"), is the question that failed and was written after seeing it; the other twenty-two were not tuned to a fix. Eight variants: retrieval {off, lexical distinctive max_df 2, max_df 4, lexical all-known-words} x prompt {shipped, with the rule}. **Rule: a candidate needs a score at least 3 above the shipped system's and no fewer negatives refused.** One run each of a stochastic 3B model (`data/eval/entity_stage1_2026-10-11.txt`).

| variant | positives correct /13 | expected source cited /13 | negatives refused /10 | score /23 |
|---|---|---|---|---|
| **V0 shipped** | 5 | 8 | **10** | 15 |
| V1 lexical (max_df 2), shipped prompt | 9 | | 9 | 18 |
| V2 lexical (max_df 4), shipped prompt | 10 | | 9 | 19 |
| V3 lexical (all words), shipped prompt | 9 | | 9 | 18 |
| V4 prompt rule only | 7 | | 10 | 17 |
| **V5 lexical (max_df 2) + prompt rule** | **11** | **13** | 9 | **20** |
| V6 lexical (max_df 4) + rule | 11 | | 9 | 20 |
| V7 lexical (all words) + rule | 11 | | 9 | 20 |

**No candidate**, because of one row: **E17, "Who is Dr. S. Sharma?"** There is no Dr. S. Sharma; the lab sign names *Dr. S. Rao* and, separately, *Mr. V. Sharma*. The title and initial ("Dr.", "S.") are too short to count as content words, so "sharma" alone matches Mr. V. Sharma's chunk, and the model answered with a person who is not the one asked about: "Dr. S. Rao (Associate Professor)" (V5), "Dr. S. Sharma is the Lab Superintendent" (V2). That is the failure the negatives condition exists to catch, and it is a confident wrong answer, not a harmless extra citation.

Reading the rows rather than the totals:

- **Retrieval is fixed by the lexical admit**: the expected source is cited for 13 of 13 in V5 (8 of 13 shipped), including the ID card, the lab sign and the VPN screenshot.
- **Neither piece is enough alone.** Lexical retrieval with the shipped prompt (V1-V3) retrieves the card and the model still refuses "who is Alex Morgan?". The rule alone (V4) cannot answer what was never retrieved.
- **Two positives still fail in V5, both the model's**: "Tell me about Alex Morgan." is refused although the card is cited, and "What is LIB-884920?" is answered with the bare echo "LIB-884920 [1]".
- **The older rows are not damaged** (informational: stage 1 named no winner, so the declared stage 2 was not required; `data/eval/entity_stage2_V5_informational_2026-10-11.txt`). V5 against V0 on T1-T33, N1-N20 and I1-I28: total 70 against 69, no older negative newly answered, T13 and I13 lost and three others gained.

## Alternatives considered

- **Lower the floors / gate thresholds.** Rejected: they are the negative-control defence (ADR-009/010/011), and any value that admits "alex morgan" at 0.16 admits almost everything.
- **Adopt V5 anyway.** Rejected by the rule, and the wrong answer it produces (naming a different person) is the kind a viva examiner asks for: "who is Dr. Sharma?".
- **Require the title and initials to match as a phrase** ("dr s sharma" contiguous), which would stop E17. Not done in this round: it needs its own rows, written first, with near-misses of the same kind (a surname shared by two people, a title that differs). It is the obvious next step and the reason the code is kept.
- **A larger model.** The two failures that remain (E5, E6) are the 3B model's reading of "tell me about" and of a bare identifier; not tested.

## Consequences

- Nothing changes by default. A user who wants name lookups to work can switch both on and accepts the near-miss risk above: in `.env`, `LEXICAL_ADMIT=1`, `LEXICAL_MODE=distinctive`, `PROMPT_WHO_WHAT_RULE=1` (variant V5).
- **Picking the card under "Answer from" does not fix it on its own.** Scoped retrieval has no floor, so the card is retrieved, but the shipped prompt still makes the model refuse "who is Alex Morgan?" (3 of 5 such questions in the probe that found this). With the shipped settings the phrasings that work are the ones that use the card's own words ("what is Alex Morgan's roll number?", "what is Alex Morgan's student id?").
- The lexical index is cached per process and rebuilt when either collection's chunk count changes; re-indexing a file to the same number of chunks is not noticed until a restart.
- The sample is small (13 + 10 rows, one run each), the keywords are a crude check (E6's bare echo is the example), and E4 was written after its failure.

## Revisit if

- A phrase-adjacency rule for titles and initials is written and measured on new near-miss rows: the leak is the only thing standing between V5 and adoption.
- The model changes: E5 and E6 are model failures and may not survive a larger one.
- A teammate adds a document or image where a name or identifier is the point (the rows are cheap to extend).
