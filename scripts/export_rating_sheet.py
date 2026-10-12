"""
Prepare a BLIND rating sheet for an independent human rater, and read it back.

Why: every answer-quality number in this project so far was rated by the AI
assistant that helped build the system (4.21/5 on 29 answers, 2026-10-05), which
methodology section 9 names as a threat to validity, and the protocol of section
8.2 asks for raters outside the team. This makes that a 30-minute job for a person
instead of a setup task.

    # 1. Build the sheet from a saved run of scripts/evaluate_answers.py
    python scripts/export_rating_sheet.py export data/eval/answers_2026-10-07_images-default.txt sheet.csv key.csv

    # 2. Send sheet.csv (NOT key.csv) to the rater. They fill the four rating columns, 1-5.

    # 3. Read it back
    python scripts/export_rating_sheet.py import sheet.csv key.csv

The sheet shows only the question, the answer and the sources the system cited. It does
not show the expected source, whether the system refused, the gold row id, or the order
of the gold set: rows are shuffled with a fixed seed, and the mapping back to the gold
ids lives in key.csv, which the rater never sees. Rating is on the four dimensions of
methodology section 8.2 (faithfulness, answer relevance, citation correctness, and
context relevance), each 1-5.

What the import does NOT claim: the earlier 4.21/5 was ONE score combining correctness
and citation, and its per-answer values were not kept, so the human numbers can be
compared with it by average only, never answer by answer.
"""

from __future__ import annotations

import csv
import random
import re
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DIMENSIONS = ("faithfulness", "relevance", "context", "citation")
SHEET_COLUMNS = ["row", "question", "answer", "sources_cited", *DIMENSIONS, "comment"]
SEED = 20261007
RULER = "=" * 70

_BLOCK_START = re.compile(r"^\[([TNI]\d+)\] (.*)$")
_SOURCE_LINE = re.compile(r"^\s+\[\d+\] (.*?)\s+\(score=")


def parse_transcript(text: str) -> list[dict]:
    """The question blocks of a scripts/evaluate_answers.py transcript: id, question, answer
    (possibly several lines) and the cited sources' labels. The TALLIES block and the
    closing instructions are not question blocks and are skipped."""
    items = []
    for block in text.split(RULER):
        lines = [line for line in block.strip("\n").split("\n")]
        if not lines:
            continue
        match = _BLOCK_START.match(lines[0].strip())
        if not match:
            continue
        body = "\n".join(lines[1:])
        answer_part = body.split("\nAnswer: ", 1)
        if len(answer_part) < 2:
            continue
        answer, _, source_text = answer_part[1].partition("\nSources:")
        sources = [m.group(1) for line in source_text.split("\n") if (m := _SOURCE_LINE.match(line))]
        items.append({"id": match.group(1), "question": match.group(2).strip(),
                      "answer": answer.strip(), "sources": sources})
    return items


def build_sheet(items: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """(sheet rows, key rows). Rows are shuffled with a fixed seed and given opaque ids r01...;
    the key maps each back to its gold id. The sheet carries nothing that reveals the expected answer."""
    order = list(items)
    random.Random(seed).shuffle(order)
    sheet, key = [], []
    for number, item in enumerate(order, start=1):
        row_id = f"r{number:02d}"
        sheet.append({"row": row_id, "question": item["question"], "answer": item["answer"],
                      "sources_cited": " | ".join(item["sources"]) if item["sources"] else "(none)",
                      **{dimension: "" for dimension in DIMENSIONS}, "comment": ""})
        key.append({"row": row_id, "gold_id": item["id"]})
    return sheet, key


def read_ratings(sheet_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    """Validate a filled sheet and attach each row's gold id. Raises ValueError, naming the row and
    column, for a blank or non-1-to-5 rating, a row missing from the key, or a missing row."""
    gold_of = {row["row"]: row["gold_id"] for row in key_rows}
    seen, ratings = set(), []
    for row in sheet_rows:
        row_id = row["row"]
        if row_id not in gold_of:
            raise ValueError(f"row {row_id} is not in the key file")
        scores = {}
        for dimension in DIMENSIONS:
            raw = (row.get(dimension) or "").strip()
            if not raw:
                raise ValueError(f"row {row_id}: the {dimension!r} rating is blank")
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f"row {row_id}: {dimension!r} must be a whole number 1-5, got {raw!r}") from None
            if not 1 <= value <= 5:
                raise ValueError(f"row {row_id}: {dimension!r} must be 1-5, got {value}")
            scores[dimension] = value
        seen.add(row_id)
        ratings.append({"row": row_id, "gold_id": gold_of[row_id], "comment": row.get("comment", ""), **scores})
    missing = sorted(set(gold_of) - seen)
    if missing:
        raise ValueError(f"rows missing from the sheet: {missing}")
    return ratings


def summarize(ratings: list[dict]) -> dict:
    """Mean per dimension overall and by kind of row (T text questions, N out-of-corpus questions,
    I image questions), the overall mean of the four dimensions, and the answers rated 2 or lower."""
    def means(rows: list[dict]) -> dict:
        return {dimension: round(statistics.mean(r[dimension] for r in rows), 2) for dimension in DIMENSIONS}

    by_kind = {kind: [r for r in ratings if r["gold_id"].startswith(kind)] for kind in "TNI"}
    return {
        "n": len(ratings),
        "overall": means(ratings),
        "overall_mean": round(statistics.mean(r[d] for r in ratings for d in DIMENSIONS), 2),
        "by_kind": {kind: {"n": len(rows), **means(rows)} for kind, rows in by_kind.items() if rows},
        "low": sorted(r["gold_id"] for r in ratings if min(r[d] for d in DIMENSIONS) <= 2),
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:       # BOM so Excel reads the accents
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "export":
        items = parse_transcript(Path(argv[2]).read_text(encoding="utf-8"))
        if not items:
            print(f"no question blocks found in {argv[2]}")
            return 1
        sheet, key = build_sheet(items)
        _write_csv(Path(argv[3]), sheet, SHEET_COLUMNS)
        _write_csv(Path(argv[4]), key, ["row", "gold_id"])
        print(f"wrote {len(sheet)} rows to {argv[3]} (send this one) and the key to {argv[4]} (keep this one)")
        return 0
    if len(argv) == 4 and argv[1] == "import":
        try:
            ratings = read_ratings(_read_csv(Path(argv[2])), _read_csv(Path(argv[3])))
        except ValueError as problem:
            print(f"cannot read the sheet: {problem}")
            return 1
        report = summarize(ratings)
        print(f"{report['n']} answers rated by a human. Mean of the four dimensions: {report['overall_mean']}/5")
        print("by dimension:", report["overall"])
        for kind, label in (("T", "text questions"), ("N", "out-of-corpus questions"), ("I", "image questions")):
            if kind in report["by_kind"]:
                print(f"  {label}: {report['by_kind'][kind]}")
        print("answers with any rating of 2 or lower:", report["low"] or "none")
        print("Compare by AVERAGE only with the earlier 4.21/5 (one combined score, rated by the AI assistant, "
              "per-answer values not kept). Methodology 8.2's target: mean faithfulness >= 4/5.")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
