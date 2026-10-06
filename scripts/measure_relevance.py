"""
Measure how well relevance gating separates answers from noise, on the real
indexed corpus, for text AND images, and compare candidate gating rules on
the same data.

Run with:  python scripts/measure_relevance.py
(after scripts/build_index.py; set CHROMA_PERSIST_DIR to measure a scratch index)

Why this exists: ADR-009's text floor (0.3) and ADR-010's image floor (0.2)
were each picked by eye on a tiny corpus. On the grown corpus both leak: for
the question "if I don't get my system running by evaluation day, how many
marks do I lose?" the right chunk is rank 1 (0.424), but four irrelevant
paper chunks (0.318-0.383) also clear 0.3, and the 3B model then refuses to
answer. A threshold is only a good gate if relevant and irrelevant scores
are separable, so this measures that directly instead of guessing.

For every gold question it takes the top results and labels each chunk
  E  expected: the gold row's source, on one of its expected pages
  N  noise: clearly the wrong material
  n  neutral: not the expected chunk, but plausibly also fine, so it is
     counted neither as a hit nor as noise

and reports, per candidate rule:
  recall   questions whose expected chunk survives the rule
  noise    N chunks that survive, per question (lower is better)
  refused  negative controls where NOTHING survives (higher is better)

What counts as noise, stated so it can be argued with: the corpus has two
halves that share no topics (the synthetic campus starter set, whose audio
clips deliberately repeat the PDFs' facts, and the downloaded study
materials). For a starter-set question (T1-T13), a chunk from a downloaded
document is N and anything else is n (an audio transcript repeating the
answer is legitimate). For a study-materials question (T14-T25), a chunk
from a different file is N and another chunk of the expected file is n.
Image rows: any image other than the expected one is N (conservative).
Negative controls: every chunk is N. So noise is still an upper bound for
the study-materials half; compare rules to each other, not to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.config import settings
from src.core.embeddings import embed_text
from src.core.gold import LEGACY_NEGATIVES, load_gold_set
from src.pipelines.documents.search import search_text
from src.pipelines.images.search import search_images

TOP_K = 5

# Negative controls from earlier chapters (src/core/gold.py), so ADR-009/ADR-010's
# original numbers stay comparable.
EXTRA_NEGATIVES = LEGACY_NEGATIVES


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


NEW_PREFIXES = ("acl_", "arxiv_", "algorithms_", "zenodo_")   # the downloaded study materials


def _label(hit, row: dict) -> str:
    if hit.source == row["expected_source"] and hit.page in row["expected_pages"]:
        return "E"
    if Path(row["expected_source"]).name.startswith(NEW_PREFIXES):
        return "n" if hit.source == row["expected_source"] else "N"
    return "N" if Path(hit.source).name.startswith(NEW_PREFIXES) else "n"


def text_data(gold: dict) -> tuple[list[dict], list[dict]]:
    """Per positive question: its scored, labelled top-K. Per negative: its top-K."""
    positives = []
    for row in gold["text"]:
        hits = search_text(row["question"], top_k=TOP_K)
        positives.append({"id": row["id"], "hits": [(h.score, _label(h, row)) for h in hits]})
    negatives = []
    for q in [r["question"] for r in gold["negatives"]] + EXTRA_NEGATIVES:
        negatives.append({"q": q, "hits": [(h.score, "N") for h in search_text(q, top_k=TOP_K)]})
    return positives, negatives


def text_rule(floor: float, margin: float | None):
    def keep(hits: list[tuple[float, str]]) -> list[tuple[float, str]]:
        if not hits:
            return []
        top = hits[0][0]
        return [h for h in hits if h[0] >= floor and (margin is None or h[0] >= top - margin)]
    return keep


def report_text(positives, negatives) -> None:
    print("\n=== TEXT (text_index, MiniLM) ===")
    print("Top-5 per question, best first: E=expected, N=noise, n=neutral, with the cosine score:")
    for p in positives:
        print(f"  {p['id']:4s} " + "  ".join(f"{label}{score:.3f}" for score, label in p["hits"]))
    top_neg = [max(s for s, _ in n['hits']) for n in negatives]
    print(f"  negatives: top score per control = {[round(x, 3) for x in top_neg]}")

    print(f"\n{'rule':32s} {'recall':>8s} {'noise/q':>8s} {'qs w/ noise':>12s} {'refused':>10s}")
    rules = [("floor 0.30 (current)", 0.30, None)]
    rules += [(f"floor {f:.2f}", f, None) for f in (0.35, 0.40, 0.45, 0.50)]
    rules += [(f"floor 0.30 + within {m:.2f} of top", 0.30, m) for m in (0.03, 0.05, 0.08, 0.10)]
    rules += [(f"floor 0.40 + within {m:.2f} of top", 0.40, m) for m in (0.05, 0.10)]
    for name, floor, margin in rules:
        keep = text_rule(floor, margin)
        recall = sum(1 for p in positives if any(lab == "E" for _, lab in keep(p["hits"])))
        noise = [sum(1 for _, lab in keep(p["hits"]) if lab == "N") for p in positives]
        refused = sum(1 for n in negatives if not keep(n["hits"]))
        print(f"{name:32s} {recall:>5d}/{len(positives):<2d} {sum(noise) / len(noise):8.2f} "
              f"{sum(1 for x in noise if x):>9d}/{len(noise):<2d} {refused:>6d}/{len(negatives)}")


def image_data(gold: dict) -> tuple[list[dict], list[dict]]:
    def scored(q: str, expected: set[str] | None) -> list[tuple[float, bool, float | None]]:
        out = []
        qv = embed_text(q)
        for h in search_images(query_text=q, top_k=TOP_K):
            # `agree`: MiniLM similarity between the query and the image's own
            # OCR text. None when the image has no readable text at all.
            agree = _cos(qv, embed_text(h.text)) if h.text.strip() else None
            out.append((h.score, bool(expected) and h.source in expected, agree))
        return out

    positives = [{"id": r["id"], "hits": scored(r["query"], set(r["expected_sources"]))}
                 for r in gold["text_to_image"]]
    negatives = [{"q": q, "hits": scored(q, None)}
                 for q in [r["question"] for r in gold["negatives"]] + EXTRA_NEGATIVES]
    return positives, negatives


def image_rule(floor: float, agree_min: float | None, ceiling: float | None):
    def keep(hits):
        kept = []
        for clip, ok, agree in hits:
            if clip < floor:
                continue
            if agree_min is not None:
                corroborated = agree is not None and agree >= agree_min
                confident = ceiling is not None and clip >= ceiling
                if not (corroborated or confident):
                    continue
            kept.append((clip, ok, agree))
        return kept
    return keep


def report_image(positives, negatives) -> None:
    print("\n=== IMAGES (image_index, CLIP; 'agree' = MiniLM similarity of the query to the image's OCR text) ===")
    for p in positives:
        rel = [(c, a) for c, ok, a in p["hits"] if ok]
        noise = [c for c, ok, _ in p["hits"] if not ok]
        a_txt = "no OCR text" if not rel or rel[0][1] is None else f"agree={rel[0][1]:.3f}"
        print(f"  {p['id']:4s} expected clip={rel[0][0]:.3f} {a_txt}" if rel else f"  {p['id']:4s} expected NOT IN TOP {TOP_K}",
              f" best_noise clip={max(noise):.3f}" if noise else "")
    print("  negatives, top image per control: " + ", ".join(
        f"{max(c for c, _, _ in n['hits']):.3f}" for n in negatives))
    tops = [max(n["hits"], key=lambda t: t[0]) for n in negatives]
    print("  ...and that top image's 'agree': " + ", ".join(
        "-" if a is None else f"{a:.3f}" for _, _, a in tops))

    print(f"\n{'rule':40s} {'recall':>8s} {'noise/q':>8s} {'qs w/ noise':>12s} {'refused':>10s}")
    rules = [("clip >= 0.20 (current)", 0.20, None, None)]
    rules += [(f"clip >= {f:.2f}", f, None, None) for f in (0.25, 0.30)]
    for a in (0.20, 0.25, 0.30, 0.35):
        for c in (0.30, None):
            label = f"clip >= 0.20 & (agree >= {a:.2f}" + (f" or clip >= {c:.2f})" if c else ")")
            rules.append((label, 0.20, a, c))
    for name, floor, a, c in rules:
        keep = image_rule(floor, a, c)
        recall = sum(1 for p in positives if any(ok for _, ok, _ in keep(p["hits"])))
        noise = [sum(1 for _, ok, _ in keep(p["hits"]) if not ok) for p in positives]
        refused = sum(1 for n in negatives if not keep(n["hits"]))
        print(f"{name:40s} {recall:>5d}/{len(positives):<2d} {sum(noise) / len(noise):8.2f} "
              f"{sum(1 for x in noise if x):>9d}/{len(noise):<2d} {refused:>6d}/{len(negatives)}")


def main() -> None:
    gold = load_gold_set()
    print(f"floors in effect: text={settings.MIN_RELEVANCE_SCORE}  image={settings.MIN_IMAGE_RELEVANCE_SCORE}  top_k={TOP_K}")
    tp, tn = text_data(gold)
    report_text(tp, tn)
    ip, ineg = image_data(gold)
    report_image(ip, ineg)


if __name__ == "__main__":
    main()
