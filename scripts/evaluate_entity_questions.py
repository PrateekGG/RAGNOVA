"""
Does a lexical admit, a who/what-is prompt rule, or both, fix name lookups without breaking anything else?

Written 2026-10-11, after "who is alex morgan?" was refused on the live page although the corpus holds Alex Morgan's
ID card. Two causes were found: the image gate dropped the card (CLIP 0.151 against a 0.2 floor, MiniLM agreement 0.16
against 0.30), and with the card in scope the shipped prompt still made the 3B model refuse "who is X" (3 of 5 such
questions) while answering "what is X's roll number". Both fixes exist and are OFF by default:
src/pipelines/rag/lexical.py (LEXICAL_ADMIT) and the prompt rule (PROMPT_WHO_WHAT_RULE). Whether either is ever
switched on is decided here, by a rule written before anything was run.

DECLARED BEFORE RUNNING (committed before the first run)

  Rows. The entity rows of data/gold_set.json (section "entity"). E1-E23 (13 positives, 10 negatives) were committed
  before the first run; E24-E28 were added with ADR-018's name-phrase rule and typo fix, before V8 was run. E1-E23 were
  committed before any retrieval or model was run on them (E4 is the question that failed; it was written after seeing the
  failure, the rest were not). Each is asked through answer_query(include_images=True), as the page asks it.

  A positive is CORRECT when its answer is not a refusal, contains at least one of its answer_keywords
  (case-insensitive), AND at least one of its expected sources is among the numbered citations.
  A negative is CORRECT when its answer is a refusal.   SCORE = correct positives + correct negatives.

  Variants: retrieval in {off, L2 = lexical distinctive max_df 2, L4 = lexical distinctive max_df 4,
  LA = lexical all-known-words max_df 2} times prompt in {P0 = shipped, P1 = with the who/what-is rule}: eight.
  V0 = (off, P0) is what ships today.

  STAGE 1 (this script, no flags). A variant is a CANDIDATE when SCORE >= V0's + 3 AND its correct negatives are
  at least V0's. The WINNER is the candidate with the highest SCORE; ties go to the variant with fewer components
  (V0, then one of the two, then both), then L2 before LA before L4. No candidate: nothing is adopted.

  STAGE 2 (--regression VARIANT). The earlier rows may only VETO, never recommend: text T1-T33 answered
  substantively (not refused, not a bare "[n]"), negatives N1-N20 refused, text-to-image I1-I28 answered while citing
  the expected image (the evaluate_answers.py definitions), for V0 and for the winner. The winner is VETOED when its
  total is 2 or more below V0's, or when it answers any earlier negative that V0 refused. A vetoed winner is
  recorded as "tested, not adopted", like table extraction and the OCR-text image channel (ADR-014).

  A candidate that is adopted still changes a shipped default, so it gets an ADR and the .env.example line is
  changed from "opt-in" to "default". The model is stochastic; a difference of one question is noise.

    $env:CHROMA_PERSIST_DIR = "<scratch copy of chroma_db without any personal file>"
    python scripts/evaluate_entity_questions.py                    # stage 1
    python scripts/evaluate_entity_questions.py --regression V4    # stage 2 for the winner stage 1 named
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.config import settings as BASE_SETTINGS
from src.core.gold import load_gold_set
from src.pipelines.rag import answer as answer_module
from src.pipelines.rag import retrieve as retrieve_module
from src.pipelines.rag.answer import NOT_ENOUGH_INFO, answer_query

_spec = importlib.util.spec_from_file_location("evaluate_answers", PROJECT_ROOT / "scripts" / "evaluate_answers.py")
evaluate_answers = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_answers"] = evaluate_answers
_spec.loader.exec_module(evaluate_answers)

# name -> (LEXICAL_ADMIT, LEXICAL_MODE, LEXICAL_MAX_DF, PROMPT_WHO_WHAT_RULE)
RETRIEVAL = {"off": (False, "distinctive", 2), "L2": (True, "distinctive", 2), "L4": (True, "distinctive", 4), "LA": (True, "all", 2)}
VARIANTS = {}
for _i, (_r, _p) in enumerate([(r, p) for p in (False, True) for r in RETRIEVAL]):
    VARIANTS[f"V{_i}"] = (_r, _p)
# V0 = (off, P0)  V1 = (L2, P0)  V2 = (L4, P0)  V3 = (LA, P0)  V4 = (off, P1)  V5 = (L2, P1)  V6 = (L4, P1)  V7 = (LA, P1)
# V8 (ADR-018, added 2026-10-11 after the first run): V5 with the name-phrase rule now inside the lexical admit, plus
# the detail rule and the spelling fix: what ships since ADR-018. V0-V7 run with those two OFF so they reproduce the
# first run's settings (the name-phrase rule cannot be switched off: it is part of the admit).
VARIANTS["V8"] = ("L2", True)
ADR018 = {"V8"}

REQUIRED_GAIN = 3
REGRESSION_TOLERANCE = 2


# --show only: V8 with one of its four pieces switched off, to see which piece changes one row's answer.
ABLATIONS = {"V8-lexical": {"LEXICAL_ADMIT": False}, "V8-whowhat": {"PROMPT_WHO_WHAT_RULE": False},
             "V8-detail": {"PROMPT_DETAIL_RULE": False}, "V8-spelling": {"QUERY_SPELLING_FIX": False}}


def configure(variant: str) -> None:
    off = ABLATIONS.get(variant, {})
    variant = "V8" if off else variant
    retrieval, rule = VARIANTS[variant]
    on, mode, max_df = RETRIEVAL[retrieval]
    extras = variant in ADR018
    new = dataclasses.replace(BASE_SETTINGS, LEXICAL_ADMIT=on, LEXICAL_MODE=mode, LEXICAL_MAX_DF=max_df, PROMPT_WHO_WHAT_RULE=rule,
                              PROMPT_DETAIL_RULE=extras, QUERY_SPELLING_FIX=extras)
    new = dataclasses.replace(new, **off)
    retrieve_module.settings = new
    answer_module.settings = new


def describe(variant: str) -> str:
    retrieval, rule = VARIANTS[variant]
    if variant in ADR018:
        return f"{variant} (ADR-018: L2 + who/what + detail rule + spelling fix)"
    return f"{variant} (retrieval {retrieval}, prompt {'P1 who/what-is rule' if rule else 'P0 shipped'})"


def components(variant: str) -> int:
    retrieval, rule = VARIANTS[variant]
    return (retrieval != "off") + bool(rule) + (2 if variant in ADR018 else 0)


ASK_ATTEMPTS = 3          # a crashed Ollama runner (HTTP 500) is restarted by the server on the next request
ASK_RETRY_WAIT = 5        # seconds between attempts
ERRORS: list[str] = []


def ask(question: str, **kwargs):
    """answer_query, retried when the model server fails. A question that still fails gets an empty answer
    with no citations, which no rule counts as correct (not a refusal, no keyword, no source), so an error
    can only lower a variant's score; every error is listed at the end."""
    for attempt in range(1, ASK_ATTEMPTS + 1):
        try:
            return answer_query(question, **kwargs)
        except Exception as exc:  # ollama.ResponseError, connection errors, timeouts
            print(f"   ! attempt {attempt}/{ASK_ATTEMPTS} failed on {question!r}: {type(exc).__name__}: {str(exc)[:160]}")
            if attempt < ASK_ATTEMPTS:
                time.sleep(ASK_RETRY_WAIT)
    ERRORS.append(question)
    return SimpleNamespace(answer="", citations=[])


def refused(result) -> bool:
    return evaluate_answers._refused(result)


def judge(row: dict, result) -> bool:
    if row["kind"] == "negative":
        return refused(result)
    cited = any(c.source in row["expected_sources"] for c in result.citations)
    said = any(k in result.answer.lower() for k in row["answer_keywords"])
    return not refused(result) and said and cited


def run_entity(variant: str, verbose: bool) -> dict:
    configure(variant)
    rows = load_gold_set()["entity"]
    out = {"variant": variant, "ok": {}, "retrieved": {}}
    for row in rows:
        result = ask(row["question"], include_images=True)
        out["ok"][row["id"]] = judge(row, result)
        if row["kind"] == "positive":
            out["retrieved"][row["id"]] = any(c.source in row["expected_sources"] for c in result.citations)
        if verbose:
            print(f"[{'ok ' if out['ok'][row['id']] else 'BAD'}] {row['id']:4} {row['kind']:8} {row['question']!r:46} -> {result.answer.strip()[:150]!r}"
                  f"  cites {[c.source.split('/')[-1] for c in result.citations][:3]}")
    out["pos"] = sum(out["ok"][r["id"]] for r in rows if r["kind"] == "positive")
    out["neg"] = sum(out["ok"][r["id"]] for r in rows if r["kind"] == "negative")
    out["score"] = out["pos"] + out["neg"]
    out["found"] = sum(out["retrieved"].values())
    return out


def stage1(only: list[str] | None = None) -> int:
    gold = load_gold_set()["entity"]
    n_pos = sum(r["kind"] == "positive" for r in gold)
    n_neg = len(gold) - n_pos
    print(f"model {BASE_SETTINGS.OLLAMA_MODEL}, TOP_K {BASE_SETTINGS.TOP_K}; {n_pos} positives, {n_neg} negatives\n")
    results = {}
    for variant in only or VARIANTS:
        print(f"######## {describe(variant)}")
        results[variant] = run_entity(variant, verbose=True)
        r = results[variant]
        print(f"-> positives correct {r['pos']}/{n_pos} (expected source cited in {r['found']}/{n_pos}), negatives refused {r['neg']}/{n_neg}, SCORE {r['score']}/{len(gold)}\n")

    base = results["V0"]
    print("=" * 78)
    print(f"{'variant':44} {'pos':>5} {'neg':>5} {'score':>6}   vs V0")
    for variant, r in results.items():
        print(f"{describe(variant):44} {r['pos']:>2}/{n_pos} {r['neg']:>3}/{n_neg} {r['score']:>4}/{len(gold)}   {r['score'] - base['score']:+d}")
    candidates = [v for v, r in results.items() if v != "V0" and r["score"] >= base["score"] + REQUIRED_GAIN and r["neg"] >= base["neg"]]
    print(f"\ncandidates (SCORE >= V0 + {REQUIRED_GAIN} and negatives refused >= V0's {base['neg']}): {candidates or 'none'}")
    if not candidates:
        print("RESULT: no candidate. Nothing is adopted; both stay off.")
        return 0
    order = {"off": 0, "L2": 1, "LA": 2, "L4": 3}
    winner = sorted(candidates, key=lambda v: (-results[v]["score"], components(v), order[VARIANTS[v][0]]))[0]
    print(f"WINNER: {describe(winner)}. Run stage 2: python scripts/evaluate_entity_questions.py --regression {winner}")
    return 0


def regression_score(variant: str) -> dict:
    configure(variant)
    old_gold = evaluate_answers.GOLD_QUESTIONS
    positives = [q for q in old_gold if q["id"].startswith("T")]
    negatives = [q for q in old_gold if q["id"].startswith("N")]
    out = {"substantive": set(), "refused": set(), "images": set()}
    for q in positives:
        result = ask(q["question"])
        if not refused(result) and not evaluate_answers._bare(result):
            out["substantive"].add(q["id"])
    for q in negatives:
        if refused(ask(q["question"])):
            out["refused"].add(q["id"])
    for q in evaluate_answers.IMAGE_QUESTIONS:
        result = ask(q["question"], include_images=True)
        if not refused(result) and any(c.source in q["expected_images"] for c in result.citations):
            out["images"].add(q["id"])
    out["total"] = len(out["substantive"]) + len(out["refused"]) + len(out["images"])
    return out


def stage2(winner: str) -> int:
    print(f"model {BASE_SETTINGS.OLLAMA_MODEL}; older rows: T1-T33, N1-N20, I1-I28 (may only veto)\n")
    base, won = regression_score("V0"), regression_score(winner)
    for name, r in (("V0", base), (winner, won)):
        print(f"{name}: text substantive {len(r['substantive'])}, negatives refused {len(r['refused'])}, images answered and cited {len(r['images'])}, TOTAL {r['total']}")
    newly_answered = sorted(base["refused"] - won["refused"])
    lost = sorted((base["substantive"] - won["substantive"]) | (base["images"] - won["images"]))
    print(f"older negatives V0 refused that {winner} answers: {newly_answered or 'none'}")
    print(f"older positives V0 got that {winner} lost: {lost or 'none'}")
    vetoed = won["total"] <= base["total"] - REGRESSION_TOLERANCE or bool(newly_answered)
    print("RESULT:", f"VETOED. {winner} is tested, not adopted." if vetoed else f"not vetoed.")
    print("Only the variant that stage 1 named WINNER may be adopted (with an ADR and the .env.example line changed to a default). "
          "For any other variant this run is information, not a decision.")
    return 0


def show(row_id: str) -> int:
    """Information only, decides nothing: one row's full answer and sources under V0, V8 and V8 minus each piece."""
    rows = {q["id"]: q for q in evaluate_answers.GOLD_QUESTIONS}
    rows.update({q["id"]: q for q in evaluate_answers.IMAGE_QUESTIONS})
    rows.update({r["id"]: r for r in load_gold_set()["entity"]})
    if row_id not in rows:
        print(f"no row {row_id}")
        return 1
    question = rows[row_id]["question"]
    images = row_id.startswith(("I", "E"))
    print(f"{row_id}: {question!r} (include_images={images})\n")
    for variant in ("V0", "V8", *ABLATIONS):
        configure(variant)
        result = ask(question, include_images=images)
        print(f"#### {variant}: {'REFUSED' if refused(result) else 'ANSWERED'}")
        print(result.answer.strip())
        for n, c in enumerate(result.citations, start=1):
            print(f"   [{n}] {c.source.split('/')[-1]}" + (f" p{c.page}" if c.page else "") + f"  {c.text[:160]!r}")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--regression", metavar="VARIANT", help="stage 2: run the older rows for V0 and this variant")
    parser.add_argument("--show", metavar="ROW", help="information only: one row's answers under V0, V8 and V8 minus each piece")
    parser.add_argument("--variants", metavar="LIST",
                        help="stage 1 on these variants only, e.g. V0,V8 (V0 is always run: it is the baseline); "
                             "for trying another model, not for adopting a variant")
    args = parser.parse_args()
    if args.show:
        code = show(args.show)
    elif args.regression:
        if args.regression not in VARIANTS or args.regression == "V0":
            parser.error(f"VARIANT must be one of {[v for v in VARIANTS if v != 'V0']}")
        code = stage2(args.regression)
    else:
        only = None
        if args.variants:
            only = ["V0"] + [v.strip() for v in args.variants.split(",") if v.strip() and v.strip() != "V0"]
            unknown = [v for v in only if v not in VARIANTS]
            if unknown:
                parser.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
        code = stage1(only)
    if ERRORS:
        print(f"\n{len(ERRORS)} question(s) failed {ASK_ATTEMPTS} times on the model server and were scored wrong:")
        for q in ERRORS:
            print(f"  {q!r}")
    return code


if __name__ == "__main__":
    sys.exit(main())
