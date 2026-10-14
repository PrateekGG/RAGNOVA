"""
src/core/compat.py: start when Windows blocks scikit-learn's OPTIONAL native accelerator (2026-10-12).

On 2026-10-12 the page showed a traceback instead of starting: Smart App Control refused
sklearn/metrics/_pairwise_distances_reduction/_argkmin.cp314-win_amd64.pyd (178 blocks that day, every retry failed,
machine online; the day before it had loaded), and sentence-transformers imports scikit-learn for functions this project
never calls. scikit-learn guards every use of that module with `is_usable_for(...)` and falls back to SciPy, so a
stand-in that answers "not usable" is the faithful behaviour. What these tests pin is the other half: it must be narrow.
A block on any other file, or any other import error, must still surface.

No real block is needed: the import is injected.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import compat

PACKAGE = "sklearn.metrics._pairwise_distances_reduction"
BLOCKED_MESSAGE = "DLL load failed while importing _argkmin: An Application Control policy has blocked this file."
BLOCKED_PATH = r"X:\RAGNova\.venv\Lib\site-packages\sklearn\metrics\_pairwise_distances_reduction\_argkmin.cp314-win_amd64.pyd"


@pytest.fixture(autouse=True)
def clean_modules(monkeypatch):
    """Whatever a test installs under the package's name is removed afterwards."""
    monkeypatch.delitem(sys.modules, PACKAGE, raising=False)
    yield
    sys.modules.pop(PACKAGE, None)


def _blocked(message=BLOCKED_MESSAGE, path=BLOCKED_PATH):
    def importer(name):
        raise ImportError(message, name="_argkmin", path=path)
    return importer


def test_when_the_import_works_nothing_is_installed():
    assert compat.tolerate_blocked_sklearn(importer=lambda name: object()) is False
    assert PACKAGE not in sys.modules


def test_without_scikit_learn_there_is_nothing_to_do():
    def missing(name):
        raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
    assert compat.tolerate_blocked_sklearn(importer=missing) is False
    assert PACKAGE not in sys.modules


def test_a_windows_block_of_the_accelerator_installs_a_stand_in_that_says_not_usable():
    assert compat.tolerate_blocked_sklearn(importer=_blocked()) is True
    stand_in = sys.modules[PACKAGE]
    for name in ("ArgKmin", "ArgKminClassMode", "RadiusNeighbors", "RadiusNeighborsClassMode", "BaseDistancesReductionDispatcher"):
        cls = getattr(stand_in, name)
        assert cls.is_usable_for([[0.0]], [[1.0]], "euclidean") is False      # scikit-learn then takes its SciPy path
        assert cls.valid_metrics() == []                                        # read at import time by sklearn.metrics.pairwise


def test_calling_a_stand_in_directly_fails_loudly_instead_of_returning_nothing():
    compat.tolerate_blocked_sklearn(importer=_blocked())
    with pytest.raises(RuntimeError, match="ArgKmin is unavailable"):
        sys.modules[PACKAGE].ArgKmin.compute(None, None)


def test_it_is_installed_once_and_a_second_call_changes_nothing():
    assert compat.tolerate_blocked_sklearn(importer=_blocked()) is True
    first = sys.modules[PACKAGE]
    assert compat.tolerate_blocked_sklearn(importer=_blocked()) is False
    assert sys.modules[PACKAGE] is first


@pytest.mark.parametrize("message,path", [
    # a Windows block of a DIFFERENT package's file: scipy, av, torch... still needs a retry or a decision about the policy
    (BLOCKED_MESSAGE, r"X:\RAGNova\.venv\Lib\site-packages\scipy\sparse\_csparsetools.cp314-win_amd64.pyd"),
    (BLOCKED_MESSAGE, r"X:\RAGNova\.venv\Lib\site-packages\sklearn\metrics\_dist_metrics.cp314-win_amd64.pyd"),
    # the accelerator's folder, but not a Windows block: a real bug or a broken install
    ("DLL load failed while importing _argkmin: The specified module could not be found.", BLOCKED_PATH),
    ("cannot import name 'ArgKmin'", BLOCKED_PATH),
    # no path at all
    (BLOCKED_MESSAGE, None),
])
def test_any_other_import_error_is_still_raised(message, path):
    with pytest.raises(ImportError):
        compat.tolerate_blocked_sklearn(importer=_blocked(message, path))
    assert PACKAGE not in sys.modules


def test_a_forward_slash_path_is_recognised_too():
    assert compat.tolerate_blocked_sklearn(importer=_blocked(path=BLOCKED_PATH.replace("\\", "/"))) is True


def test_the_stand_in_covers_every_class_the_installed_scikit_learn_imports_from_that_package():
    """If a scikit-learn release adds a class to the package, the stand-in has to follow, or sklearn.metrics would
    fail to import exactly when the file is blocked."""
    spec = importlib.util.find_spec("sklearn")
    if spec is None:
        pytest.skip("scikit-learn is not installed")
    init = Path(spec.submodule_search_locations[0]) / "metrics" / "_pairwise_distances_reduction" / "__init__.py"
    exported = set(re.search(r"__all__ = \[(.*?)\]", init.read_text(encoding="utf-8"), re.S).group(1).replace('"', "").replace("\n", " ").replace(" ", "").split(",")) - {""}
    needed = exported - {"sqeuclidean_row_norms"}              # a function, only called after is_usable_for said yes
    assert needed <= set(compat._NAMES), f"the stand-in is missing {sorted(needed - set(compat._NAMES))}"
