"""
Keep the app starting when Windows blocks an OPTIONAL native module (2026-10-12).

Smart App Control (Windows Security, App & browser control) refuses to load a native file it has no reputation for.
It does that per file, and its verdict changes: on 2026-10-11 `_argkmin.cp314-win_amd64.pyd` imported fine, on
2026-10-12 it was blocked 178 times in a day and every retry failed, with the machine online. The file belongs to
scikit-learn, which sentence-transformers imports for functions this project never calls, and `import
sentence_transformers` died with "DLL load failed while importing _argkmin: An Application Control policy has
blocked this file", so the page showed a traceback instead of starting.

scikit-learn treats that file as an accelerator, not a requirement: every use is guarded by
`ArgKmin.is_usable_for(...)` and falls back to SciPy/NumPy when the answer is no. So when, and only when, Windows
blocks that one native package, this stands in a module whose classes answer "not usable". Nothing is loaded that
the policy refused, and no policy is touched: the blocked file simply stays unloaded.

What it deliberately does NOT do: hide any other import error, or any block on any other file. A block elsewhere
(scipy, av, torch) still raises, and still needs a retry or a decision about the policy.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

_PACKAGE = "sklearn.metrics._pairwise_distances_reduction"
# What scikit-learn imports from that package (sklearn/metrics/_pairwise_distances_reduction/__init__.py).
_NAMES = ("ArgKmin", "ArgKminClassMode", "BaseDistancesReductionDispatcher", "RadiusNeighbors", "RadiusNeighborsClassMode")


class _NotUsable:
    """A stand-in for one of those classes: scikit-learn asks `is_usable_for(...)` first and takes its pure-Python
    path when the answer is False. Calling one directly would be a bug in the caller, so it fails loudly."""

    @classmethod
    def valid_metrics(cls) -> list:
        """Read at import time (sklearn/metrics/pairwise.py builds its parameter constraints from it): none."""
        return []

    @classmethod
    def is_usable_for(cls, *args, **kwargs) -> bool:
        return False

    @classmethod
    def compute(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__name__} is unavailable: Windows blocked scikit-learn's native module for it")


def _is_windows_block_of_the_accelerator(error: ImportError) -> bool:
    """True only for "Application Control blocked this file" about a file inside scikit-learn's pairwise-distance
    reduction package. `error.path` is the file Windows refused."""
    path = str(getattr(error, "path", "") or "").replace("\\", "/")
    return "Application Control" in str(error) and "/sklearn/metrics/_pairwise_distances_reduction/" in path


def tolerate_blocked_sklearn(importer=importlib.import_module) -> bool:
    """Make `import sklearn.metrics` work although Windows blocks scikit-learn's accelerator. Returns True when the
    stand-in was installed, False when nothing was needed (the usual case, or scikit-learn is not installed).
    `importer` is injected by tests."""
    if _PACKAGE in sys.modules:
        return False
    try:
        importer("sklearn.metrics")
        return False
    except ModuleNotFoundError:
        return False                                            # scikit-learn is not installed: nothing to do
    except ImportError as error:
        if not _is_windows_block_of_the_accelerator(error):
            raise
    stand_in = types.ModuleType(_PACKAGE)
    for name in _NAMES:
        setattr(stand_in, name, type(name, (_NotUsable,), {"__module__": _PACKAGE}))
    stand_in.__all__ = list(_NAMES)
    sys.modules[_PACKAGE] = stand_in
    logger.warning(
        "Windows (Application Control) blocked scikit-learn's native distance module; continuing without that "
        "optional accelerator. Nothing else is affected."
    )
    return True
