"""
Guards for the gold-standard evaluation set (data/gold_set.json).

Four hand-copied lists of these questions drifted before they became one
JSON file: tests/test_retrieval.py checked 3 of 13, scripts/evaluate_answers.py
held 5 of 13 plus the old negative controls. The scripts and tests now read
the JSON, so they cannot drift from each other; what is left is the
human-readable copy in data/README.md, which these tests keep honest, and a
check that every file a row points at really exists.

Pure file checks: no models, no ChromaDB, no network.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.gold import cross_modal_rows, load_gold_set

README = PROJECT_ROOT / "data" / "README.md"


@pytest.fixture(scope="module")
def gold():
    return load_gold_set()


def _readme_rows() -> dict[str, list[str]]:
    """Every gold row of data/README.md's tables: id -> its cells."""
    rows: dict[str, list[str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| [TIMAN]\d+ \|", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    return rows


def _path(cell: str) -> str:
    """'`documents/notice.pdf`' -> 'data/documents/notice.pdf'."""
    return "data/" + cell.strip().strip("`")


def test_gold_set_loads_and_meets_its_documented_targets(gold):
    # data/README.md's own targets: 20 text questions, 10 cross-modal queries.
    assert len(gold["text"]) >= 20
    assert len(cross_modal_rows(gold)) >= 10
    assert len(gold["negatives"]) >= 2


def test_every_expected_file_in_the_gold_set_exists(gold):
    missing = []
    for row in gold["text"]:
        missing += [row["expected_source"]]
    for row in cross_modal_rows(gold):
        missing += row["expected_sources"] + [row.get("query_image"), row.get("clip")]
    missing = [p for p in missing if p and not (PROJECT_ROOT / p).exists()]
    assert not missing, f"gold set points at files that do not exist: {sorted(set(missing))}"


def test_readme_tables_list_exactly_the_gold_set_rows(gold):
    in_json = {r["id"] for sec in ("text", "text_to_image", "image_to_doc", "audio_topic", "negatives")
               for r in gold[sec]}
    in_readme = set(_readme_rows())
    assert in_readme == in_json, (
        f"data/README.md and data/gold_set.json disagree. "
        f"Only in README: {sorted(in_readme - in_json)}; only in JSON: {sorted(in_json - in_readme)}"
    )


def test_readme_tables_carry_the_same_question_text_and_files_as_the_gold_set(gold):
    rows = _readme_rows()
    problems = []
    for r in gold["text"]:
        cells = rows[r["id"]]
        if cells[1] != r["question"]:
            problems.append(f"{r['id']}: question text differs")
        if _path(cells[2]) != r["expected_source"]:
            problems.append(f"{r['id']}: expected source differs ({cells[2]} vs {r['expected_source']})")
    for r in gold["text_to_image"]:
        cells = rows[r["id"]]
        if cells[1] != r["query"]:
            problems.append(f"{r['id']}: query text differs")
        # a cell may name several acceptable images ("`a.png` (or `b.png`)"): compare them all
        if sorted("data/" + path for path in re.findall(r"`([^`]+)`", cells[2])) != sorted(r["expected_sources"]):
            problems.append(f"{r['id']}: expected image(s) differ")
    for r in gold["image_to_doc"]:
        cells = rows[r["id"]]
        if _path(cells[1]) != r["query_image"]:
            problems.append(f"{r['id']}: query image differs")
    for r in gold["audio_topic"]:
        cells = rows[r["id"]]
        if _path(cells[1]) != r["clip"]:
            problems.append(f"{r['id']}: clip differs")
        if cells[2] != r["query"]:
            problems.append(f"{r['id']}: question text differs")
    for r in gold["negatives"]:
        if rows[r["id"]][1] != r["question"]:
            problems.append(f"{r['id']}: question text differs")
    assert not problems, "data/README.md drifted from data/gold_set.json:\n  " + "\n  ".join(problems)


def test_loader_rejects_a_duplicate_id_and_a_missing_key(tmp_path):
    import json

    good = load_gold_set()
    dup = {**good, "negatives": good["negatives"] + [dict(good["negatives"][0])]}
    (tmp_path / "dup.json").write_text(json.dumps(dup), encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        load_gold_set(tmp_path / "dup.json")

    broken = {**good, "text": [{"id": "T1", "question": "q"}] + good["text"][1:]}
    (tmp_path / "broken.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_gold_set(tmp_path / "broken.json")
