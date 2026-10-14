"""
Day-5 setup verification script.

Run with:   python scripts/verify_setup.py

Checks, in order, exactly the things Chapter 5 asks you to set up: the Python
environment, that Ollama is installed AND running AND has a model pulled,
and (optionally) that the embedding model downloads and runs. Each check is
independent and reports clearly rather than crashing on the first failure —
same "show every problem, not just the first" philosophy as
src/core/schemas.py's validate_chunk().

This script deliberately uses only the Python standard library for the
Ollama check (urllib, not `requests` or the `ollama` package) so it runs
correctly before requirements.txt has necessarily been installed — it's the
first thing you run, possibly before anything else.
"""

import json
import sys
import urllib.request
import urllib.error

# Windows terminals often default to a legacy codepage (cp1252) rather than
# UTF-8, which turns any non-ASCII character (an em-dash, a section sign)
# into "?" mid-sentence. Reconfiguring stdout explicitly avoids depending on
# the user's terminal settings. This same fix generalises to any script in
# this project that prints text containing such characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = "http://localhost:11434"
REQUIRED_MODEL_PREFIX = "llama3.2"  # matches any tag, e.g. llama3.2:3b


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    print(f"[1/4] Python version: {major}.{minor}.{sys.version_info.micro}")
    if (major, minor) != (3, 11):
        print(
            "      NOTE: this project targets Python 3.11 (Chapter 0 §A.3). "
            f"You're running {major}.{minor} — likely fine for this script, "
            "but if a later `pip install` fails with a build error, this is "
            "the first thing to check."
        )
        return False
    print("      OK — matches the recommended version.")
    return True


def check_ollama_reachable() -> bool:
    print(f"[2/4] Checking Ollama is running at {OLLAMA_URL} ...")
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"      FAILED — could not reach Ollama: {e.reason}")
        print("      Fix: open a terminal and run `ollama serve`, or on")
        print("      Windows/Mac just launch the Ollama app — it runs as a")
        print("      background service once started. Then re-run this script.")
        return False

    models = [m["name"] for m in data.get("models", [])]
    print(f"      OK — Ollama is running. Models available: {models or '(none)'}")

    if not any(m.startswith(REQUIRED_MODEL_PREFIX) for m in models):
        print(f"      WARNING: no model starting with '{REQUIRED_MODEL_PREFIX}' found.")
        print(f"      Fix: run `ollama pull llama3.2:3b` (~2 GB download).")
        return False
    return True


def check_ollama_generates() -> bool:
    print("[3/4] Sending a real test prompt to confirm generation works ...")
    payload = json.dumps({
        "model": "llama3.2:3b",
        "prompt": "Reply with exactly the word: OK",
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"      FAILED — generation request errored: {e}")
        return False

    response_text = data.get("response", "").strip()
    print(f"      Model replied: {response_text!r}")
    print(f"      Timing: total {data.get('total_duration', 0)/1e9:.1f}s, "
          f"eval {data.get('eval_count', '?')} tokens")
    print("      OK — the local LLM is reachable and generates text.")
    return True


def check_embedding_model() -> bool:
    print("[4/4] Checking sentence-transformers can load the embedding model ...")
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # this script is run from scripts/
        from src.core.compat import tolerate_blocked_sklearn

        tolerate_blocked_sklearn()          # Windows may block scikit-learn's optional native module (src/core/compat.py)
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        if "Application Control" in str(error):
            # Not a missing package: Windows (Smart App Control) refused a native file.
            print(f"      BLOCKED — Windows refused to load a library: {error}")
            print("      Wait a minute and run this again; if it keeps happening see docs/human-verification-checklist.md section 1.")
            return False
        print("      SKIPPED — sentence-transformers not installed yet.")
        print("      Fix: pip install -r requirements.txt")
        return False

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode("test sentence")
    print(f"      OK — embedded a test string into a {len(vector)}-dim vector.")
    return True


if __name__ == "__main__":
    print("RAGNova Day 5 setup check\n" + "=" * 40)
    results = {
        "python_version": check_python_version(),
        "ollama_reachable": check_ollama_reachable(),
    }
    # Only try generation if Ollama is actually reachable — no point sending
    # a request we already know will fail, and it keeps the failure message
    # focused on the one real problem instead of three restatements of it.
    if results["ollama_reachable"]:
        results["ollama_generates"] = check_ollama_generates()
    else:
        results["ollama_generates"] = False
        print("[3/4] SKIPPED — Ollama isn't reachable yet (see check 2 above).")

    results["embedding_model"] = check_embedding_model()

    print("\n" + "=" * 40)
    passed = sum(results.values())
    print(f"{passed}/{len(results)} checks passed.")
    if passed < len(results):
        print("Not all checks passed yet — that's normal on a fresh clone. "
              "Fix the failures above in order; later checks often depend "
              "on earlier ones.")
        sys.exit(1)
    print("Environment ready. Continue to Chapter 6.")
