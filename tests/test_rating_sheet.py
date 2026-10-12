"""
The blind rating sheet for an independent human rater (scripts/export_rating_sheet.py).

What matters: the parser reads real evaluate_answers output; the sheet reveals
nothing about the expected answer, the gold id or the gold order; a half-filled or
out-of-range sheet is refused with the row and column named instead of silently
averaged; and the averages are the right averages.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("export_rating_sheet", PROJECT_ROOT / "scripts" / "export_rating_sheet.py")
sheets = importlib.util.module_from_spec(_spec)
sys.modules["export_rating_sheet"] = sheets
_spec.loader.exec_module(sheets)

RULER = "=" * 70
DASHES = "-" * 70

TRANSCRIPT = f"""Running 3 gold questions through answer_query(), text + images (include_images=True)...

{RULER}
[T1] How many marks is the prototype worth?
{DASHES}
Expected: data/documents/notice.pdf, page 2
{DASHES}
Answer: The prototype carries forty percent [1].
It is demonstrated on evaluation day.
Sources:
  [1] data/documents/notice.pdf, page 2  (score=0.424)
  [2] data/audio/hod_project_announcement.wav, 0s-33s  (score=0.412)

{RULER}
[N1] What is the hostel mess menu for Wednesday lunch?
{DASHES}
Expected: refusal (not covered by the corpus)
{DASHES}
Answer: I don't have enough information in the indexed documents to answer that.
Sources: (none - below relevance threshold, no LLM call made)

{RULER}
[I3] What poster shows the upcoming AI and Machine Learning seminar venue?
{DASHES}
Expected: data/images/notice_seminar_poster.png
{DASHES}
Answer: The seminar is in Auditorium B [1].
Sources:
  [1] data/images/notice_seminar_poster.png (image)  (score=0.331)

{RULER}
TALLIES (automatic; a person still judges whether each answer is right)
  text rows answered (not refused):        1/1
"""


def _items():
    return sheets.parse_transcript(TRANSCRIPT)


# ------------------------------------------------------------------ parsing

def test_the_parser_reads_ids_questions_multiline_answers_and_sources():
    items = _items()
    assert [i["id"] for i in items] == ["T1", "N1", "I3"]
    assert items[0]["question"] == "How many marks is the prototype worth?"
    assert items[0]["answer"] == "The prototype carries forty percent [1].\nIt is demonstrated on evaluation day."
    assert items[0]["sources"] == ["data/documents/notice.pdf, page 2", "data/audio/hod_project_announcement.wav, 0s-33s"]
    assert items[1]["sources"] == []
    assert items[2]["sources"] == ["data/images/notice_seminar_poster.png (image)"]


def test_the_tallies_block_is_not_a_question():
    assert all(i["id"] != "TALLIES" for i in _items()) and len(_items()) == 3


def test_the_parser_reads_a_real_saved_run():
    path = PROJECT_ROOT / "data" / "eval" / "answers_2026-10-07_images-default.txt"
    if not path.exists():
        pytest.skip("the saved run is not in this checkout")
    items = sheets.parse_transcript(path.read_text(encoding="utf-8"))
    ids = [i["id"] for i in items]
    assert len(items) == 69 and len(set(ids)) == 69
    assert all(i["question"] and i["answer"] for i in items)
    assert any(i["sources"] for i in items) and any(not i["sources"] for i in items)


# ------------------------------------------------------------------ the sheet

def test_the_sheet_is_shuffled_deterministically_and_keeps_every_row_once():
    items = _items() * 1
    big = [{"id": f"T{n}", "question": f"q{n}", "answer": f"a{n}", "sources": []} for n in range(1, 21)]
    sheet1, key1 = sheets.build_sheet(big)
    sheet2, key2 = sheets.build_sheet(big)
    assert key1 == key2                                                    # same seed, same order
    assert [k["gold_id"] for k in key1] != [i["id"] for i in big]          # it really is shuffled
    assert sorted(k["gold_id"] for k in key1) == sorted(i["id"] for i in big)
    assert sheets.build_sheet(big, seed=1)[1] != key1                      # another seed, another order
    assert [r["row"] for r in sheet1] == [f"r{n:02d}" for n in range(1, 21)]
    assert items


def test_the_sheet_reveals_neither_the_gold_id_nor_the_expected_answer():
    sheet, key = sheets.build_sheet(_items())
    cells = " ".join(str(v) for row in sheet for v in row.values())
    assert not re.search(r"\b[TNI]\d+\b", cells), "a gold id leaked into the sheet"
    assert "Expected" not in cells and "refusal (not covered" not in cells
    assert set(sheet[0]) == set(sheets.SHEET_COLUMNS)
    assert all(row[d] == "" for row in sheet for d in sheets.DIMENSIONS)


# ------------------------------------------------------------------ reading ratings back

def _filled(value="4"):
    sheet, key = sheets.build_sheet(_items())
    for row in sheet:
        for dimension in sheets.DIMENSIONS:
            row[dimension] = value
    return sheet, key


def test_a_complete_sheet_is_read_back_with_the_right_gold_ids():
    sheet, key = _filled()
    sheet[0]["faithfulness"] = "5"
    ratings = sheets.read_ratings(sheet, key)
    assert sorted(r["gold_id"] for r in ratings) == ["I3", "N1", "T1"]
    first = next(r for r in ratings if r["row"] == sheet[0]["row"])
    assert first["gold_id"] == next(k["gold_id"] for k in key if k["row"] == sheet[0]["row"])
    assert first["faithfulness"] == 5


@pytest.mark.parametrize("bad,fragment", [("", "blank"), ("6", "1-5"), ("0", "1-5"), ("abc", "whole number")])
def test_a_blank_or_out_of_range_rating_is_refused_with_the_row_and_column_named(bad, fragment):
    sheet, key = _filled()
    sheet[1]["citation"] = bad
    with pytest.raises(ValueError, match=rf"{sheet[1]['row']}.*citation|citation.*{sheet[1]['row']}") as problem:
        sheets.read_ratings(sheet, key)
    assert fragment in str(problem.value)


def test_a_row_the_key_does_not_know_and_a_row_missing_from_the_sheet_are_refused():
    sheet, key = _filled()
    with pytest.raises(ValueError, match="not in the key"):
        sheets.read_ratings(sheet + [{**sheet[0], "row": "r99"}], key)
    with pytest.raises(ValueError, match="missing"):
        sheets.read_ratings(sheet[:-1], key)


def test_the_summary_averages_by_dimension_by_kind_and_flags_low_ratings():
    ratings = [
        {"row": "r01", "gold_id": "T1", "comment": "", "faithfulness": 5, "relevance": 5, "context": 4, "citation": 5},
        {"row": "r02", "gold_id": "T2", "comment": "", "faithfulness": 3, "relevance": 3, "context": 2, "citation": 2},
        {"row": "r03", "gold_id": "N1", "comment": "", "faithfulness": 5, "relevance": 5, "context": 5, "citation": 5},
    ]
    report = sheets.summarize(ratings)
    assert report["n"] == 3
    assert report["overall"]["faithfulness"] == round((5 + 3 + 5) / 3, 2)
    assert report["by_kind"]["T"] == {"n": 2, "faithfulness": 4.0, "relevance": 4.0, "context": 3.0, "citation": 3.5}
    assert report["by_kind"]["N"]["n"] == 1
    assert report["overall_mean"] == round((19 + 10 + 20) / 12, 2)
    assert report["low"] == ["T2"]                      # any dimension at 2 or below


# ------------------------------------------------------------------ the command line, end to end

def test_export_then_fill_then_import_round_trips_through_real_csv_files(tmp_path, capsys):
    transcript = tmp_path / "answers.txt"
    transcript.write_text(TRANSCRIPT, encoding="utf-8")
    sheet_path, key_path = tmp_path / "sheet.csv", tmp_path / "key.csv"
    assert sheets.main(["x", "export", str(transcript), str(sheet_path), str(key_path)]) == 0

    with sheet_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for dimension in sheets.DIMENSIONS:
            row[dimension] = "4"
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sheets.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    capsys.readouterr()
    assert sheets.main(["x", "import", str(sheet_path), str(key_path)]) == 0
    assert "3 answers rated by a human" in capsys.readouterr().out


def test_importing_an_unfinished_sheet_exits_nonzero_with_a_reason(tmp_path, capsys):
    transcript = tmp_path / "answers.txt"
    transcript.write_text(TRANSCRIPT, encoding="utf-8")
    sheet_path, key_path = tmp_path / "sheet.csv", tmp_path / "key.csv"
    sheets.main(["x", "export", str(transcript), str(sheet_path), str(key_path)])
    capsys.readouterr()
    assert sheets.main(["x", "import", str(sheet_path), str(key_path)]) == 1
    assert "blank" in capsys.readouterr().out
