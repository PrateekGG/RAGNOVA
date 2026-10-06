"""
The RAGNOVA_OFFLINE switch (src/__init__.py) has to take effect BEFORE
huggingface_hub is imported, which is the whole reason it lives in the
package's __init__. Each test runs a fresh interpreter, because the setting
is read once at import time and cannot be observed from inside a process that
has already imported the libraries.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROBE = (
    "import sys; sys.path.insert(0, r'%s');"
    "import src.core.embeddings;"                       # imports sentence_transformers BEFORE src.core.config runs
    "import huggingface_hub.constants as c;"
    "import os;"
    "print('HF_HUB_OFFLINE=%%s TRANSFORMERS_OFFLINE=%%s' %% (c.HF_HUB_OFFLINE, os.environ.get('TRANSFORMERS_OFFLINE')))"
) % PROJECT_ROOT


def _run(**extra_env: str) -> str:
    env = {k: v for k, v in os.environ.items()
           if k not in ("RAGNOVA_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    env.update(extra_env)
    result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env,
                            cwd=PROJECT_ROOT, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    return result.stdout.strip().splitlines()[-1]


def test_the_switch_turns_huggingface_offline_even_though_embeddings_imports_it_first():
    assert _run(RAGNOVA_OFFLINE="1") == "HF_HUB_OFFLINE=True TRANSFORMERS_OFFLINE=1"


def test_without_the_switch_the_hub_stays_reachable_so_a_fresh_clone_can_download_models():
    assert _run() == "HF_HUB_OFFLINE=False TRANSFORMERS_OFFLINE=None"


def test_an_explicit_huggingface_setting_is_not_overridden_by_the_switch():
    # setdefault, not assignment: someone who deliberately sets HF_HUB_OFFLINE=0 keeps it.
    assert _run(RAGNOVA_OFFLINE="1", HF_HUB_OFFLINE="0").startswith("HF_HUB_OFFLINE=False")
