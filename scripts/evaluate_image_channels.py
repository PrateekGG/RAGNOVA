"""
Does finding images by their text (ADR-014) help, and which gate variant should
go with it? Measured on the gold set's text-to-image rows and negatives.

Run with:  python scripts/evaluate_image_channels.py
(after scripts/build_index.py; set CHROMA_PERSIST_DIR to measure a scratch index)

WRITTEN AND COMMITTED BEFORE IT WAS RUN. The variants, their order of
simplicity and the adoption rule below are fixed here, not chosen after seeing
numbers.

Two sets of rows, reported separately:
  - HELD-OUT: rows tagged `heldout-2026-10-07` in data/gold_set.json (positives
    I15-I28, negatives N9-N16). Written and committed before any retrieval was
    run on them, on images the design had not seen. Only these choose a variant.
  - REGRESSION: every older row (I1-I14, N1-N8 and the two legacy negatives).
    They were used to set the shipped gate, so they can only veto a variant
    that makes things worse, never recommend one.

Variants (all share the shipped CLIP floor and corroboration rule):
  G0        the shipped gate, CLIP search only (no OCR-text search)
  F0        + the OCR-text search, rankings fused by rank; nothing new admitted
  F1@T      F0 + admit an image the text search returned with agreement >= T,
            whatever CLIP thought, T in {0.55, 0.50, 0.45}
  F2@T      F1@T + the question must share a content word with the image's text

Order of simplicity, used to break ties (earlier = simpler):
  F0, F1@0.55, F1@0.50, F1@0.45, F2@0.55, F2@0.50, F2@0.45

Adoption rule. A variant V other than G0 QUALIFIES only if, with k = images kept
among positives, r = negatives refused (no image survives the gate), and
R = Recall@5 of the merged result:
  held-out:   k(V) >= k(G0), r(V) >= r(G0), R(V) >= R(G0), and
              (k + r)(V) >= (k + r)(G0) + 2          (a clear win, not one question)
  regression: k(V) >= k(G0) and r(V) >= r(G0)        (no harm to the old rows)
Of the qualifiers the one with the highest held-out k + r is adopted; ties go to
the simplest. If none qualifies the channel is NOT adopted, and the result is
recorded as "tested, not adopted" (ADR-014).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.config import settings
from src.core.gold import LEGACY_NEGATIVES, load_gold_set
from src.pipelines.documents.search import search_text
from src.pipelines.images.search import search_image_text, search_images
from src.pipelines.rag.retrieve import filter_by_floor, filter_images, gate_image_channels, rrf_merge

HELDOUT = "heldout-2026-10-07"
THRESHOLDS = (0.55, 0.50, 0.45)
VARIANTS: list[tuple[str, dict | None]] = [("G0", None), ("F0", {})]
VARIANTS += [(f"F1@{t:.2f}", {"ocr_only_min_agreement": t}) for t in THRESHOLDS]
VARIANTS += [(f"F2@{t:.2f}", {"ocr_only_min_agreement": t, "require_shared_word": True}) for t in THRESHOLDS]
SIMPLICITY = ["F0", "F1@0.55", "F1@0.50", "F1@0.45", "F2@0.55", "F2@0.50", "F2@0.45"]
CLEAR_WIN = 2


def collect(question: str) -> dict:
    """Everything the variants need for one question, computed once."""
    top_k = settings.TOP_K
    return {
        "clip": search_images(query_text=question, top_k=top_k),
        "ocr": search_image_text(question, top_k=top_k),
        "text": filter_by_floor(search_text(question, top_k=top_k), settings.MIN_RELEVANCE_SCORE),
    }


def images_for(variant: tuple[str, dict | None], hits: dict, question: str) -> list:
    name, kwargs = variant
    if kwargs is None:
        return filter_images(hits["clip"], question)
    return gate_image_channels(hits["clip"], hits["ocr"], question, **kwargs)


def merged(images: list, hits: dict) -> list:
    return rrf_merge([hits["text"], images])[: settings.TOP_K]


def main() -> None:
    gold = load_gold_set()
    positives = gold["text_to_image"]
    negatives = [dict(r, tier=r.get("tier")) for r in gold["negatives"]]
    negatives += [{"id": f"L{i + 1}", "question": q, "tier": None} for i, q in enumerate(LEGACY_NEGATIVES)]
    print(f"floors: image {settings.MIN_IMAGE_RELEVANCE_SCORE}, confident {settings.IMAGE_CONFIDENT_SCORE}, "
          f"agreement {settings.MIN_IMAGE_TEXT_AGREEMENT}, text {settings.MIN_RELEVANCE_SCORE}; top_k={settings.TOP_K}")
    print(f"positives: {sum(r.get('tier') == HELDOUT for r in positives)} held-out + "
          f"{sum(r.get('tier') != HELDOUT for r in positives)} regression;  "
          f"negatives: {sum(r['tier'] == HELDOUT for r in negatives)} held-out + "
          f"{sum(r['tier'] != HELDOUT for r in negatives)} regression\n")

    pos_hits = {r["id"]: collect(r["query"]) for r in positives}
    neg_hits = {r["id"]: collect(r["question"]) for r in negatives}

    # --- which search can find each correct image at all ---------------------------------
    print("=== Which search finds the correct image in its top 5 (before any gate) ===")
    print(f"{'row':5s} {'set':10s} {'CLIP':>5s} {'text':>5s}")
    for r in positives:
        in_clip = any(c.source in r["expected_sources"] for c in pos_hits[r["id"]]["clip"])
        in_ocr = any(c.source in r["expected_sources"] for c in pos_hits[r["id"]]["ocr"])
        print(f"{r['id']:5s} {'held-out' if r.get('tier') == HELDOUT else 'regression':10s} "
              f"{'yes' if in_clip else '-':>5s} {'yes' if in_ocr else '-':>5s}")

    # --- every variant ------------------------------------------------------------------
    results: dict[str, dict] = {}
    for variant in VARIANTS:
        name = variant[0]
        out = {"held": {"k": 0, "R": 0, "r": 0}, "reg": {"k": 0, "R": 0, "r": 0}, "rows": {}}
        for r in positives:
            hits = pos_hits[r["id"]]
            images = images_for(variant, hits, r["query"])
            kept = any(c.source in r["expected_sources"] for c in images)
            top = merged(images, hits)
            rank = next((i for i, c in enumerate(top, 1) if c.source in r["expected_sources"]), None)
            bucket = out["held" if r.get("tier") == HELDOUT else "reg"]
            bucket["k"] += kept
            bucket["R"] += rank is not None
            out["rows"][r["id"]] = (kept, rank)
        for r in negatives:
            hits = neg_hits[r["id"]]
            images = images_for(variant, hits, r["question"])
            refused = not images
            (out["held"] if r["tier"] == HELDOUT else out["reg"])["r"] += refused
            # a SET of surviving images: a pure reordering is not a difference
            out["rows"][r["id"]] = (not refused, sorted(c.source.split("/")[-1] for c in images))
        results[name] = out

    n = {"held_p": sum(r.get("tier") == HELDOUT for r in positives), "reg_p": sum(r.get("tier") != HELDOUT for r in positives),
         "held_n": sum(r["tier"] == HELDOUT for r in negatives), "reg_n": sum(r["tier"] != HELDOUT for r in negatives)}
    print("\n=== Variants ===")
    print(f"{'variant':9s} | held-out: kept/{n['held_p']}  R@5/{n['held_p']}  refused/{n['held_n']}  k+r | "
          f"regression: kept/{n['reg_p']}  R@5/{n['reg_p']}  refused/{n['reg_n']}")
    for name, out in results.items():
        h, g = out["held"], out["reg"]
        print(f"{name:9s} |           {h['k']:6d}  {h['R']:6d}  {h['r']:10d}  {h['k'] + h['r']:3d} | "
              f"            {g['k']:6d}  {g['R']:6d}  {g['r']:12d}")

    # --- rows that differ from the shipped gate -------------------------------------------
    print("\n=== Rows whose outcome differs from G0 (kept = correct image survives the gate; leak = an image survives) ===")
    base = results["G0"]["rows"]
    for name in [v[0] for v in VARIANTS[1:]]:
        diffs = []
        for rid, value in results[name]["rows"].items():
            if value != base[rid]:
                is_neg = rid.startswith(("N", "L"))
                if is_neg:
                    diffs.append(f"{rid} leak: {value[1]}" if value[0] else f"{rid} now refused")
                else:
                    diffs.append(f"{rid} kept={value[0]} rank={value[1]} (G0: kept={base[rid][0]} rank={base[rid][1]})")
        print(f"{name:9s} " + ("; ".join(diffs) if diffs else "(identical to G0)"))

    # --- the pre-registered rule --------------------------------------------------------
    g0h, g0g = results["G0"]["held"], results["G0"]["reg"]
    qualifiers = []
    for name in SIMPLICITY:
        h, g = results[name]["held"], results[name]["reg"]
        ok = (h["k"] >= g0h["k"] and h["r"] >= g0h["r"] and h["R"] >= g0h["R"]
              and (h["k"] + h["r"]) >= (g0h["k"] + g0h["r"]) + CLEAR_WIN
              and g["k"] >= g0g["k"] and g["r"] >= g0g["r"])
        if ok:
            qualifiers.append(name)
    print("\n=== Adoption rule (fixed before this was run) ===")
    print(f"G0 held-out k+r = {g0h['k'] + g0h['r']};  a variant needs k+r >= {g0h['k'] + g0h['r'] + CLEAR_WIN}, "
          f"no loss on kept/refused/R@5 held-out, and none on the regression rows")
    if qualifiers:
        best = max(qualifiers, key=lambda q: (results[q]["held"]["k"] + results[q]["held"]["r"], -SIMPLICITY.index(q)))
        print(f"qualifying variants: {qualifiers}\nRULE SAYS: adopt {best}")
    else:
        print("qualifying variants: none\nRULE SAYS: NOT adopted (tested, not adopted)")


if __name__ == "__main__":
    main()
