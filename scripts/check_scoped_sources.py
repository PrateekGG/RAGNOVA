"""
Does answering from a picked file, and from an attached image, work with the real model?

Written 2026-10-10 after a CV uploaded through "Add to Corpus" could not be asked
about (5 of 6 questions refused: the corpus-wide relevance floor hides it) and two
screenshots attached to a question were refused although Tesseract had read them. The
page now lets a question be limited to chosen files (no floor) and shows an attached
image's text to the model. Taking the floor off in a scope is the risky half: the floor
is what refuses questions the corpus cannot answer, so this script also asks things the
scoped file does NOT contain, and each must still be refused.

It is a functional check, not an evaluation. A handful of rows, so it can show that a
behaviour exists or does not; it cannot show a rate. Every row's expectation is written
below BEFORE the run, and the output prints each answer so they can be read, not just
counted (a bare "[1]" would pass a keyword check on a citation).

The CV is synthetic (a made-up person), not anyone's real one. Nothing is added to the
real index: the script refuses to run unless CHROMA_PERSIST_DIR points somewhere else,
for example a copy of chroma_db/.

    $env:CHROMA_PERSIST_DIR = "<scratch copy of chroma_db>"
    python scripts/check_scoped_sources.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz
from PIL import Image, ImageDraw, ImageFont

from src.core.config import settings
from src.pipelines.documents.index import index_document_file
from src.pipelines.rag.answer import NOT_ENOUGH_INFO, answer_query
from src.pipelines.rag.attachments import image_attachment
from src.ui.backend import list_sources, ocr_image, sources_for_upload

CV_LINES = [
    "Maya Fernandes",
    "Pune, India | maya.fernandes@example.org",
    "",
    "ABOUT ME",
    "Final-year data science student who builds dashboards and small machine-learning tools.",
    "",
    "SKILLS",
    "Python, SQL, Tableau, scikit-learn, Git",
    "",
    "PROJECTS",
    "Campus Bike Share Analytics (2025): analysed 40,000 trips and cut idle bikes by 18 percent.",
    "Crop Disease Classifier (2024): a small CNN that reached 91 percent accuracy on leaf photos.",
    "",
    "EDUCATION",
    "B.Tech Computer Science, Westbridge Institute of Technology, 2022-2026, CGPA 8.4",
]
NOTICE_LINES = ["LIBRARY NOTICE", "The library is closed on October 12", "for maintenance.", "It reopens on October 13 at 9 AM."]
SHUTTLE_LINES = ["SHUTTLE TIMETABLE", "Buses leave the main gate", "at 8:15 AM and 5:45 PM.", "No service on Sundays."]

# (id, question, words of which at least one must appear in the answer; None = must be refused)
SCOPED = [
    ("S1", "Who is this person?", ["maya fernandes"]),
    ("S2", "What is in this document?", ["python", "bike", "crop", "westbridge"]),
    ("S3", "What skills are listed?", ["python"]),
    ("S4", "Which projects are mentioned?", ["bike", "crop"]),
    ("S5", "What is the email address?", ["maya.fernandes@example.org"]),
    ("S6", "Where did they study?", ["westbridge"]),
    ("SN1", "What is the capital of France?", None),
    ("SN2", "How many marks does the working prototype carry?", None),    # answered by the corpus, not by this CV
    ("SN3", "What is the campus Wi-Fi network name?", None),
    ("SN4", "Who won the 2018 football World Cup?", None),
]
UNSCOPED = [
    ("U1", "Who is Maya Fernandes?", ["maya fernandes"]),                 # names the person: the corpus-wide search can find it
    ("U2", "What skills are listed in Maya Fernandes's CV?", ["python"]),
    # U3, U4 were added after the first run, because U1 and U2 passed (the synthetic CV is short and the question names
    # its owner, so neither reproduces the reported failure). These refer to the file only as "this" or "listed": a
    # corpus-wide search cannot know which file is meant, so they are EXPECTED TO FAIL unscoped. That is why the scope exists.
    ("U3", "Who is this person?", ["maya fernandes"]),
    ("U4", "What skills are listed?", ["python"]),
]
ATTACHED = [
    # (id, which image, question, words of which one must appear; None = must be refused)
    ("A1", "notice", "When does the library reopen?", ["9"]),
    ("A2", "notice", "What does this image say?", ["library"]),
    ("A3", "blank", "What does this image show?", None),                  # no text in it: it must not invent a description
    ("A4", "notice", "What is the capital of France?", None),             # the attachment does not answer this
    # A5-A9 were added after the first run, in which A2 was answered from a DIFFERENT image already in the index ("this
    # image" was ambiguous among several image chunks). A7-A9 use a second image that fix was not written against.
    ("A5", "notice", "Summarise this image.", ["library", "october 12", "october 13", "maintenance"]),
    ("A6", "notice", "What is written in this picture?", ["library"]),
    ("A7", "shuttle", "What does this image say?", ["shuttle", "bus", "8:15", "5:45"]),
    ("A8", "shuttle", "When do the buses leave?", ["8:15", "5:45"]),
    ("A9", "shuttle", "How much does a bus ticket cost?", None),          # not on the notice
]
IMAGES = {"notice": NOTICE_LINES, "shuttle": SHUTTLE_LINES, "blank": None}


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def make_cv(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 72), "\n".join(CV_LINES), fontsize=11, lineheight=1.5)
    doc.save(path)


def make_text_image(path: Path, lines: list[str]) -> None:
    image = Image.new("RGB", (1100, 420), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((40, 30 + i * 90), line, fill="black", font=_font(44))
    image.save(path)


def make_blank(path: Path) -> None:
    image = Image.new("RGB", (600, 400), (200, 220, 245))
    ImageDraw.Draw(image).ellipse((200, 100, 400, 300), fill=(230, 60, 60))
    image.save(path)


def refused(text: str) -> bool:
    low = text.lower()
    return text.strip() == NOT_ENOUGH_INFO or "don't have enough information" in low or "cannot tell" in low or "can't tell" in low


def verdict(answer: str, words: list[str] | None) -> bool:
    if words is None:
        return refused(answer)
    return not refused(answer) and any(w in answer.lower() for w in words)


def main() -> int:
    real = (PROJECT_ROOT / "chroma_db").resolve()
    if Path(settings.CHROMA_PERSIST_DIR).resolve() == real:
        print("Refusing to run against the real chroma_db/: this adds a CV to the index. Point CHROMA_PERSIST_DIR at a copy.")
        return 2

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cv = tmp / "maya_cv.pdf"
        make_cv(cv)
        pictures = {}
        for name, lines in IMAGES.items():
            pictures[name] = tmp / f"{name}.png"
            if lines is None:
                make_blank(pictures[name])
            else:
                make_text_image(pictures[name], lines)
        chunks = index_document_file(cv)
        # The scope is the index's own spelling of the file's path, found the way the page finds it after an upload.
        scope = sources_for_upload(cv, list_sources())
        print(f"model {settings.OLLAMA_MODEL}, TOP_K {settings.TOP_K}, CV indexed as {chunks} chunk(s), scope {[Path(s).name for s in scope]}\n")
        assert chunks and scope, "the CV did not index: nothing to ask about"

        def run(row_id, kind, question, words, **kwargs):
            result = answer_query(question, include_images=True, **kwargs)
            ok = verdict(result.answer, words)
            rows.append((row_id, kind, ok))
            expect = "refuse" if words is None else "/".join(words)
            print(f"[{'PASS' if ok else 'FAIL'}] {row_id} ({kind}; expect {expect})\n  Q: {question}\n  A: {result.answer.strip()}\n"
                  f"  cites: {[c.source.split('/')[-1] for c in result.citations]}\n")

        for row_id, question, words in SCOPED:
            run(row_id, "scoped to the CV", question, words, sources=scope)
        for row_id, question, words in UNSCOPED:
            kind = "NOT scoped, names the person" if row_id in ("U1", "U2") else "NOT scoped, says only 'this' (expected to fail)"
            run(row_id, kind, question, words)
        for row_id, which, question, words in ATTACHED:
            path = pictures[which]
            picture = Image.open(path)
            text = ocr_image(picture)
            print(f"  (attached {path.name}: Tesseract read {len(text.split())} words: {text[:80]!r})")
            # As the page does: the picture is both an attachment (its text is context [1]) and a search key (look-alike
            # indexed images are retrieved). The first three runs passed only the attachment, so they under-reported the
            # noise the look-alikes add; from the fourth run on this mirrors the page.
            run(row_id, f"attached {which} image", question, words, attachments=[image_attachment(path.name, text)], query_image=picture)

    print("SUMMARY")
    for group in sorted({kind for _, kind, _ in rows}):
        ok = [r for r in rows if r[1] == group and r[2]]
        total = [r for r in rows if r[1] == group]
        print(f"  {group}: {len(ok)}/{len(total)} as expected   failed: {[r[0] for r in total if not r[2]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
