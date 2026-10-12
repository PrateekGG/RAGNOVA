"""
Attached image: does searching the index for LOOK-ALIKE images make the answer about the attachment worse?

When an image is attached, the page uses it twice: its text is context [1], and the picture itself is a search
key, so CLIP retrieves similar indexed images (five more notices, for a notice). The 2026-10-10 runs of
scripts/check_scoped_sources.py (page-style) showed "What does this image say?" answered with only the headline
("The image says LIBRARY NOTICE") in 2 of 2 such rows, the dates in [1] left out; with the look-alikes absent the
same question once returned the full text. One run each of a stochastic model proves nothing, so this repeats it.

DECLARED BEFORE RUNNING
  Rows: the four "describe it" rows of check_scoped_sources.py (A2, A5, A6, A7), each asked 3 times per variant.
  Variant A: as the page is (the picture is also a search key; look-alikes retrieved).
  Variant B: the attachment's text only (no image search key).
  An answer is COMPLETE if it contains the detail that is in the picture beyond its headline: for the library notice
  "october 13" or "9 am"; for the shuttle notice "8:15" or "5:45". A refusal is never complete.
  Rule: switch the page to B only if B has at least 3 more complete answers than A out of 12 and no more refusals.
  Otherwise the page stays as it is (a tie keeps the existing behaviour). Output goes to data/eval/.

    $env:CHROMA_PERSIST_DIR = "<scratch copy of chroma_db>"
    python scripts/probe_attachment_lookalikes.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.core.config import settings
from src.pipelines.rag.answer import answer_query
from src.pipelines.rag.attachments import image_attachment
from src.ui.backend import ocr_image

_spec = importlib.util.spec_from_file_location("chk", PROJECT_ROOT / "scripts" / "check_scoped_sources.py")
chk = importlib.util.module_from_spec(_spec)
sys.modules["chk"] = chk
_spec.loader.exec_module(chk)

ROWS = ["A2", "A5", "A6", "A7"]
REPEATS = 3
DETAIL = {"notice": ["october 13", "9 am"], "shuttle": ["8:15", "5:45"]}
REQUIRED_ADVANTAGE = 3


def main() -> int:
    real = (PROJECT_ROOT / "chroma_db").resolve()
    if Path(settings.CHROMA_PERSIST_DIR).resolve() == real:
        print("Point CHROMA_PERSIST_DIR at a copy of chroma_db/ (this script reads the index only, but keeps the habit).")
        return 2

    rows = [r for r in chk.ATTACHED if r[0] in ROWS]
    totals = {"A": {"complete": 0, "refused": 0}, "B": {"complete": 0, "refused": 0}}
    with tempfile.TemporaryDirectory() as tmp:
        pictures = {}
        for name, lines in chk.IMAGES.items():
            pictures[name] = Path(tmp) / f"{name}.png"
            chk.make_blank(pictures[name]) if lines is None else chk.make_text_image(pictures[name], lines)

        print(f"model {settings.OLLAMA_MODEL}, TOP_K {settings.TOP_K}; {len(rows)} rows x {REPEATS} repeats x 2 variants\n")
        for row_id, which, question, _ in rows:
            picture = Image.open(pictures[which])
            attachment = [image_attachment(pictures[which].name, ocr_image(picture))]
            for variant, key in (("A", picture), ("B", None)):
                for repeat in range(REPEATS):
                    result = answer_query(question, include_images=True, query_image=key, attachments=attachment)
                    answer = result.answer.strip()
                    refused = chk.refused(answer)
                    complete = not refused and any(d in answer.lower() for d in DETAIL[which])
                    totals[variant]["complete"] += complete
                    totals[variant]["refused"] += refused
                    print(f"[{variant}] {row_id} #{repeat + 1} {'COMPLETE' if complete else ('refused' if refused else 'partial ')} "
                          f"blocks={len(result.citations)}  {answer[:150]!r}")

    a, b = totals["A"], totals["B"]
    print(f"\nA (page as is, look-alikes searched): complete {a['complete']}/12, refused {a['refused']}")
    print(f"B (attachment text only):             complete {b['complete']}/12, refused {b['refused']}")
    switch = b["complete"] - a["complete"] >= REQUIRED_ADVANTAGE and b["refused"] <= a["refused"]
    print(f"rule (B needs +{REQUIRED_ADVANTAGE} complete and no more refusals): {'SWITCH to B' if switch else 'KEEP the page as it is'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
