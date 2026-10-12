"""
Does RAGNova work with no network, as far as that can be checked without the UI?

Run with:  python scripts/verify_offline.py                  (no offline settings)
           python scripts/verify_offline.py --hf-offline     (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 by hand)
           python scripts/verify_offline.py --offline-switch (RAGNOVA_OFFLINE=1, the project's own switch)
(needs `ollama serve` on this machine; the models must already be downloaded)

Methodology section 8.5 asks for the real test: switch the network adapter
off, then ingest a new file, answer a text query, answer an image query,
transcribe and answer a spoken query, and render citations. Switching an
adapter off is a system change this script will not make. Instead it installs
a guard inside the Python process (NetworkGuard below): every attempt to
reach anything but this machine (DNS lookups and connections to other
addresses) fails the way a dead network would, and is recorded. Talking to
the local Ollama server on 127.0.0.1 is allowed, since that is not the
network. Then it runs the checklist's steps through the same functions the
UI calls (src/ui/backend.py, src/pipelines/rag, src/ui/citations.py).

What this proves and what it does not: it proves no step of the checklist
*needs* the network, and shows every attempt that was made. It does not
replace the real run (adapter off, through the Streamlit page); the page is
wired now, so that one only needs a person: docs/human-verification-checklist.md, item 3. The guard only sees Python's own sockets, so a
native library opening its own connection would escape it.
"""

from __future__ import annotations

import argparse
import errno
import ipaddress
import os
import socket
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_local(host) -> bool:
    if host is None:
        return True
    host = str(host)
    if host.lower() in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class NetworkGuard:
    """While active, anything that would leave this machine fails like a dead
    network (a failed DNS lookup, or "network unreachable") and is recorded in
    `.blocked`. Loopback (localhost, 127.0.0.0/8, ::1) and Unix sockets pass."""

    def __init__(self) -> None:
        self.blocked: list[tuple[str, str]] = []

    def __enter__(self) -> "NetworkGuard":
        self._getaddrinfo = socket.getaddrinfo
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex
        guard = self

        def getaddrinfo(host, *args, **kwargs):
            if not _is_local(host):
                guard.blocked.append(("dns lookup", str(host)))
                raise socket.gaierror(socket.EAI_NONAME, "network blocked by NetworkGuard")
            return guard._getaddrinfo(host, *args, **kwargs)

        def _target(address):
            return (address[0], address[1]) if isinstance(address, tuple) and len(address) >= 2 else (None, None)

        def connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                host, port = _target(address)
                if not _is_local(host):
                    guard.blocked.append(("connect", f"{host}:{port}"))
                    raise OSError(errno.ENETUNREACH, "network blocked by NetworkGuard")
            return guard._connect(sock, address)

        def connect_ex(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                host, port = _target(address)
                if not _is_local(host):
                    guard.blocked.append(("connect", f"{host}:{port}"))
                    return errno.ENETUNREACH
            return guard._connect_ex(sock, address)

        socket.getaddrinfo = getaddrinfo
        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        return self

    def __exit__(self, *exc) -> None:
        socket.getaddrinfo = self._getaddrinfo
        socket.socket.connect = self._connect
        socket.socket.connect_ex = self._connect_ex


# ---------------------------------------------------------------------------
# the checklist
# ---------------------------------------------------------------------------

def run_steps(tmp: Path) -> list[tuple[str, bool, str, float]]:
    """Each step: (name, passed, detail, seconds). A failing step is recorded,
    not raised, so one failure does not hide the rest."""
    import fitz
    from PIL import Image

    from src.core.vector_store import get_client
    from src.pipelines.rag import answer_query
    from src.pipelines.rag.retrieve import retrieve
    from src.ui import backend
    from src.ui.citations import citation_views

    results: list[tuple[str, bool, str, float]] = []
    state: dict = {}

    def step(name):
        def decorator(fn):
            t0 = time.perf_counter()
            try:
                detail = fn()
                results.append((name, True, detail or "ok", time.perf_counter() - t0))
            except Exception as exc:  # noqa: BLE001 - a step failing is the result, not a crash
                results.append((name, False, f"{type(exc).__name__}: {str(exc)[:200]}", time.perf_counter() - t0))
            return fn
        return decorator

    client = get_client(persist_dir=tmp / "chroma")

    @step("1. ingest a NEW document (an upload) and index it")
    def _():
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "Offline test notice. The campus cafeteria opens at 7:45 on Mondays, "
                                             "and the quiet-study room closes at 21:30 every evening.")
        data = doc.tobytes()
        path = backend.save_upload("new_policy.pdf", data, data_root=tmp / "data")
        n = backend.index_file(path, client=client)
        assert n >= 1, "the new file produced no chunks"
        return f"{n} chunk(s) indexed from {path.name}"

    @step("2. answer a text question about it (embedding + search + local LLM)")
    def _():
        result = answer_query("What time does the cafeteria open on Mondays?", client=client)
        state["result"] = result
        assert "7:45" in result.answer, f"answer did not contain 7:45: {result.answer[:120]!r}"
        assert any(c.source.endswith("new_policy.pdf") for c in result.citations), "no citation to the new file"
        return f"answered: {result.answer.strip()[:90]!r}"

    @step("3. answer an IMAGE query (CLIP search by an uploaded picture + OCR of it)")
    def _():
        for name in ("screenshot_wifi_setup.png", "photo_lab_door_sign.png"):
            backend.index_file(PROJECT_ROOT / "data" / "images" / name, client=client)
        picture = Image.open(PROJECT_ROOT / "data" / "images" / "screenshot_wifi_setup.png").convert("RGB")
        hits = retrieve("", client=client, query_image=picture)
        assert hits and hits[0].source.endswith("screenshot_wifi_setup.png"), f"top image hit was {hits[0].source if hits else None}"
        text = backend.ocr_image(picture)
        assert text.strip(), "OCR returned no text (is Tesseract installed?)"
        return f"nearest image {Path(hits[0].source).name}; OCR read {len(text)} characters"

    @step("4. transcribe a SPOKEN query (Whisper)")
    def _():
        data = (PROJECT_ROOT / "data" / "audio" / "library_orientation_excerpt.wav").read_bytes()
        text = backend.transcribe_audio_bytes(data)
        assert "library" in text.lower(), f"transcript did not mention the library: {text[:100]!r}"
        return f"transcript: {text.strip()[:80]!r}"

    @step("5. build the citations the UI renders")
    def _():
        views = citation_views(state["result"].citations)
        assert views and views[0].file_exists, "no citation view, or its file is missing"
        return f"{len(views)} citation view(s); first: {views[0].title}"

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-offline", action="store_true",
                        help="set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before anything is imported")
    parser.add_argument("--offline-switch", action="store_true",
                        help="set only RAGNOVA_OFFLINE=1; src/__init__.py turns that into the Hugging Face settings")
    args = parser.parse_args()
    if args.offline_switch:
        os.environ["RAGNOVA_OFFLINE"] = "1"
    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    sys.path.insert(0, str(PROJECT_ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(PROJECT_ROOT)    # sources are stored relative to the project root

    mode = ("WITH RAGNOVA_OFFLINE=1 (the project's own switch)" if args.offline_switch
            else "WITH HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 set by hand" if args.hf_offline
            else "WITH NO OFFLINE SETTINGS")
    print(f"Offline verification {mode}\nNetwork guard: all non-loopback traffic is blocked and recorded.\n")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, NetworkGuard() as guard:
        results = run_steps(Path(tmp))

    for name, ok, detail, seconds in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   ({seconds:.1f}s)\n          {detail}")
    print(f"\nBlocked network attempts: {len(guard.blocked)}")
    for (kind, target), n in Counter(guard.blocked).most_common(12):
        print(f"   {n:4d} x {kind:10s} {target}")
    passed = all(ok for _, ok, _, _ in results)
    print(f"\nRESULT: {'all steps completed' if passed else 'AT LEAST ONE STEP FAILED'}; "
          f"{len(guard.blocked)} connection attempt(s) were made and blocked.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
