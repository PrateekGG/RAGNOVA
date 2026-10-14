# ADR-018: Names, typos, follow-ups and fuller answers: switched on by default

## Status
**Accepted** (2026-10-11). V8 passed both stages of ADR-017's rule, unchanged, on the second run (below). Builds on ADR-017.

## Context

On 2026-10-11 the live page, with every default as shipped:

| question | what happened | cause |
|---|---|---|
| "who is alex morgan?" | refused | ADR-017: the image gate drops the ID card, and the prompt makes the 3B model refuse "who is X". Both fixes existed and were off because of one near-miss: "Who is Dr. S. Sharma?" was answered with Dr. S. Rao. |
| "riffy" (in an uploaded CV) | refused | the same: one word scores far under the floor; dense search has no meaning to match in a name. |
| "tell me lab rul no. 2" | refused; "tell me lab rules" answered | "rul" is in no file, so the question's vector is damaged and nothing clears the floor or the image gate. |
| "tell me more" after "Raiden is a Discord app" | refused | every question was searched on its own; "tell me more" has nothing to search for. |
| "what is greedy algorithm tell me in brief", "priyanshu jain" | one bare line, sometimes no citation | nothing in the prompt asks the model to use the context; "in brief" was read as one line. |
| "tell me about internship fair" | an algorithms chapter listed as source [1], uncited | the page listed every retrieved chunk, cited or not. |

## Decision

1. **Name-phrase rule** in `lexical.py`: a title or initial in the question must sit next to the same word in the chunk ("dr s", "s sharma"). Stops E17 (the sign says "Dr. S. Rao" and "Mr. V. Sharma"), keeps E7/E8. With it, `LEXICAL_ADMIT` and `PROMPT_WHO_WHAT_RULE` are **on by default** (ADR-017's V5).
2. **Spelling fix** (`QUERY_SPELLING_FIX=1`): a word no file contains is replaced, for the similarity searches only, by the common word (in 2+ files) it starts or is one keystroke from. The lexical admit keeps the typed words, so an unknown name still vetoes it.
3. **Follow-ups** (`followup.py`, no extra model call): a question with no content words, a continuation opener ("what about", "and"), or a pronoun with at most two content words is searched together with the previous turn's search question, and the prompt shows the previous exchange (answer cut to 80 words, citation numbers removed). Other questions are handled exactly as before.
4. **Detail rule** (`PROMPT_DETAIL_RULE=1`): answer in complete sentences with the details the context gives; "in brief" means two to four sentences; a bare name or term means "what does the context say about it".
5. **Cited-only sources**: the page lists only the sources the answer cites, with their numbers; all of them when it cites none.

## How to measure it (on the machine with Ollama and the index)

    python scripts/evaluate_entity_questions.py                    # stage 1: V8 against V0 on E1-E28
    python scripts/evaluate_entity_questions.py --regression V8    # stage 2: the older rows may veto it

The rule is ADR-017's, unchanged: a candidate needs a score at least 3 above V0 and no fewer negatives refused; stage 2 vetoes a total 2 or more below V0's, or any older negative newly answered.

## First measurement (2026-10-11, on the author's machine)

Stage 1: V8 27/28 against V0 18/28 (positives 14/15 against 5/15, negatives refused 13/13 both), the winner.
Stage 2: V8 total 74 against V0 67, no older positive lost, but **vetoed**: N20 ("How many layers does the T5 model
used as the generative reader have?") counted as answered. `--show N20` showed the answer was a refusal in the model's
own words ("The context does not mention the number of layers ..."), caused by the detail rule alone (V8 minus the
detail rule refused with the fixed sentence). The detail rule now ends with "reply with the refusal sentence of the
last rule, word for word". The adoption rule, the refusal check and the rows are unchanged; both stages are to be
re-run on this prompt.

## Second measurement (2026-10-11, after the refusal fix; the run that decides)

| | positives /15 | negatives refused /13 | score /28 |
|---|---|---|---|
| V0 (main) | 5 | 13 | 18 |
| V5 (ADR-017's best) | 12 | 13 | 25 |
| **V8** | **13** | **13** | **26** (+8, the winner) |

Stage 2 (older rows, may only veto): V0 total 67 (text 26, negatives 20, images 21), V8 73 (text 32, negatives 20,
images 21); no older negative newly answered, no older positive lost: **not vetoed**. V8 still misses E5 ("Tell me
about Alex Morgan.") and, in this run, E4 ("Who is Alex Morgan?", answered in the first run): the card is cited in
both, the 3B model refuses. One run each of a stochastic model; a one-question difference is noise.

## Addendum: full names and mangled refusals (2026-10-11, nemotron-3-nano:4b)

A V0/V8 run with `OLLAMA_MODEL=nemotron-3-nano:4b`, `LLM_THINK=1` scored V0 20/28, V8 23/28. Two findings:

- **Mangled refusals.** E18, E19 and E23 were refused as "I don' have ..." / "I don' t have ...", which the exact-prefix
  check scored as answers (and the app would have listed sources under and used as follow-up history).
  `answer.is_refusal()` now compares letters only; the app and the evaluation scripts share it. Rescored, V8 is 26/28.
  This changes how an answer is judged, not what is answered; the llama3.2:3b runs above typed the sentence exactly.
- **E15 "Who is Alex Smith?" answered.** The lexical admit let in a chunk holding "alex" and "smith" in two different
  names, and the model glued them. Two or more capitalised words in a row in the question are now a name phrase like
  a title: they must sit side by side in the chunk (`lexical._capitalised_name_pairs`). A question typed in lowercase
  pins nothing, as before.

Re-run with both changes (2026-10-11, `--variants V0,V8`):

| model | V0 pos/neg/score | V8 pos/neg/score | stage 2 (V0 vs V8) |
|---|---|---|---|
| llama3.2:3b | 5/15, 13/13, 18/28 | 13/15, 13/13, **26/28** (+8, winner) | 67 vs 73, **not vetoed** |
| nemotron-3-nano:4b (`LLM_THINK=1`) | 8/15, 13/13, 21/28 | 14/15, 13/13, **27/28** (+6, winner) | not run |

E15 is refused by both models under V8, and no row was lost to the full-name rule (E4/E5 still cite the ID card).
V8 stays adopted with both changes. The default model stays llama3.2:3b; switching it is a separate decision.

## Alternatives considered

- **Rewrite follow-ups with the model** into a standalone question: better on hard cases, but a second generation per follow-up, which is seconds on a CPU (ADR-015).
- **Lower the floors**: rejected again for ADR-017's reason.

## Consequences

- Every setting can be switched off in `.env`; with all four off and no history, the prompt is byte-for-byte the old one (tested).
- The spelling fix can turn an unknown word into a common one the user did not mean; it only changes what the similarity search sees, and the floors and the refusal rule still apply.
- Follow-up detection is a word rule; "why does it fail?" after an unrelated answer is glued to that answer's question.
