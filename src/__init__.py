"""
RAGNova.

The one thing done here, before any submodule is imported: the offline switch.

Set `RAGNOVA_OFFLINE=1` (in the environment or in `.env`) once the models have
been downloaded, and the Hugging Face libraries are told not to contact the
Hub. This has to happen HERE, not in `src/core/config.py`: `huggingface_hub`
reads `HF_HUB_OFFLINE` once, when it is first imported, and several modules
(`src/core/embeddings.py` imports `sentence_transformers` before it imports
the config) pull it in before the config could set anything. Python runs a
package's `__init__` before any of its submodules, so this is the earliest
point that is guaranteed.

Why it matters (measured 2026-10-05, scripts/verify_offline.py): with the
network unreachable and no switch, loading the text embedding model makes 31
attempts to reach the Hub, each retried with back-off, so the first question
stalls for ~49 s before falling back to the local cache; with the switch it
takes 0.6 s and makes no attempt at all. A fresh clone needs the network once
to download the models, so the switch is opt-in, not a default.
"""

import os

from dotenv import load_dotenv

load_dotenv()

if os.environ.get("RAGNOVA_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
