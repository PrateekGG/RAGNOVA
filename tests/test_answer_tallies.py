"""
The tallies printed by scripts/evaluate_answers.py.

"Answered, not refused" counted a bare "[2]" as a success. On 2026-10-07 the held-out
questions showed the model returning exactly that: 3 of 8 answers with trimmed context, 1
of 8 and 2 of 25 at the default. The tallies now separate a SUBSTANTIVE answer (at least
three real words once citation markers are removed) from a bare one, so a setting that
makes answers worse while keeping them "answered" can no longer look fine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("evaluate_answers", PROJECT_ROOT / "scripts" / "evaluate_answers.py")
evaluate_answers = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_answers"] = evaluate_answers
_spec.loader.exec_module(evaluate_answers)

REFUSAL = "I don't have enough information in the indexed documents to answer that."


def _result(answer: str):
    return SimpleNamespace(answer=answer, citations=[])


@pytest.mark.parametrize("answer,bare", [
    ("[2]", True),
    ("[2] [3] [4]", True),                                          # three markers are not three words
    ("Yes [1].", True),
    ("No prior knowledge is required to take the course. [2]", False),
    ("The prototype carries forty percent [1].", False),
    (REFUSAL, False),                                               # a refusal is its own category, not a bare answer
])
def test_a_bare_answer_is_one_with_fewer_than_three_real_words_besides_its_citations(answer, bare):
    assert evaluate_answers._bare(_result(answer)) is bare


def test_a_refusal_is_recognised_whatever_follows_the_opening():
    assert evaluate_answers._refused(_result(REFUSAL))
    assert not evaluate_answers._refused(_result("The prototype carries forty percent [1]."))


def test_the_tallies_report_bare_answers_separately_from_refusals(capsys):
    rows = [
        {"id": "T1", "tier": None, "result": _result("The prototype carries forty percent [1].")},
        {"id": "T2", "tier": None, "result": _result("[2]")},
        {"id": "T3", "tier": None, "result": _result(REFUSAL)},
    ]
    evaluate_answers.print_tallies(rows)
    out = capsys.readouterr().out
    assert "text rows answered (not refused):        2/3" in out          # the bare one still counts as not refused...
    assert "substantive (not bare [n]): 1/3" in out                       # ...but not as an answer
    assert "bare: ['T2']" in out
