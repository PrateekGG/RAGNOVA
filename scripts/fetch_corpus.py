"""
Download the open-licence "study materials" corpus listed in data/SOURCES.md.

Run with:  python scripts/fetch_corpus.py              (download + write data/SOURCES.md)
           python scripts/fetch_corpus.py --dry-run    (only verify every licence, write nothing)

Why a script instead of hand-downloaded files: the report has to be able to
say where every corpus file came from and under which licence, and an
examiner has to be able to rebuild it. So each file below carries its source
URL, and BEFORE anything is saved this script re-checks the licence at the
source itself rather than trusting the manifest:

  - ACL Anthology papers: the paper page must carry the site's CC BY 4.0
    statement, and the paper must be from 2016 or later (older ACL material
    is CC BY-NC-SA, which we do not use).
  - arXiv paper: the abstract page must link the CC BY 4.0 licence (arXiv's
    default licence does NOT permit redistribution, which is why most
    arXiv papers are not in this corpus).
  - Algorithms (Erickson): the book page must state CC BY 4.0.
  - Zenodo records: the API's `license.id` must be `cc-by-4.0`, and the
    downloaded bytes must match Zenodo's md5.
  - Wikimedia Commons: the API's `LicenseShortName` must be on the allowed
    list below (no NC/ND), and the downloaded bytes must match Commons' sha1.

Audio is the one place we modify what we download: the Spoken Wikipedia
recordings are 15-37 minutes long, so each clip here is cut to about a
minute, at a quiet moment so it does not start or end mid-word, and
converted to mono 16 kHz WAV (what Whisper and tests/test_corpus_inventory.py
expect). That makes the clips adaptations of CC BY-SA material, so they stay
CC BY-SA 3.0, as data/SOURCES.md says.

The new files live in Git LFS (see .gitattributes); the Chapter 6 synthetic
starter files stay ordinary blobs. Everything is idempotent: re-running
overwrites the same files with the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Wikimedia asks for an identifying User-Agent. Deliberately names the
# project, not a person: no personal details go to third-party servers.
USER_AGENT = "RAGNova-college-project/0.1 (+https://github.com/Elitex07/RAGNova)"

# Only licences that allow redistribution with attribution. No NC, no ND.
COMMONS_ALLOWED = {
    "CC0", "Public domain", "CC BY 2.0", "CC BY 4.0",
    "CC BY-SA 2.0", "CC BY-SA 3.0", "CC BY-SA 4.0",
}


@dataclass
class Item:
    dest: str        # path under data/
    title: str
    licence: str     # what we EXPECT; verified at the source before saving
    check: str       # acl | arxiv | erickson | zenodo | commons
    ref: str         # ACL id / arXiv id / PDF URL / Zenodo record id / Commons file title
    author: str = ""
    page: str = ""   # human-readable page for SOURCES.md
    key: str = ""    # Zenodo file name inside the record
    max_side: int = 0  # >0: downscale images so the longest side is this many px
    note: str = ""


ALG = "http://jeffe.cs.illinois.edu/teaching/algorithms/book/"

ITEMS: list[Item] = [
    # ---- PDFs --------------------------------------------------------------
    Item("documents/acl_dense_passage_retrieval.pdf", "Dense Passage Retrieval for Open-Domain Question Answering",
         "CC BY 4.0", "acl", "2020.emnlp-main.550",
         "Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih"),
    Item("documents/acl_sentence_bert.pdf", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
         "CC BY 4.0", "acl", "D19-1410", "Nils Reimers, Iryna Gurevych"),
    Item("documents/acl_bert.pdf", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
         "CC BY 4.0", "acl", "N19-1423", "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova"),
    Item("documents/acl_lost_in_the_middle.pdf", "Lost in the Middle: How Language Models Use Long Contexts",
         "CC BY 4.0", "acl", "2024.tacl-1.9",
         "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang"),
    Item("documents/acl_fusion_in_decoder.pdf",
         "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
         "CC BY 4.0", "acl", "2021.eacl-main.74", "Gautier Izacard, Edouard Grave"),
    Item("documents/acl_in_context_ralm.pdf", "In-Context Retrieval-Augmented Language Models",
         "CC BY 4.0", "acl", "2023.tacl-1.75",
         "Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham"),
    Item("documents/acl_squad.pdf", "SQuAD: 100,000+ Questions for Machine Comprehension of Text",
         "CC BY 4.0", "acl", "D16-1264", "Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang"),
    Item("documents/arxiv_openclip_scaling_laws.pdf", "Reproducible scaling laws for contrastive language-image learning",
         "CC BY 4.0", "arxiv", "2212.07143",
         "Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, "
         "Christoph Schuhmann, Ludwig Schmidt, Jenia Jitsev"),
    Item("documents/algorithms_ch03_dynamic_programming.pdf", "Algorithms, Chapter 3: Dynamic Programming",
         "CC BY 4.0", "erickson", ALG + "03-dynprog.pdf", "Jeff Erickson", "http://jeffe.cs.illinois.edu/teaching/algorithms/"),
    Item("documents/algorithms_ch04_greedy.pdf", "Algorithms, Chapter 4: Greedy Algorithms",
         "CC BY 4.0", "erickson", ALG + "04-greedy.pdf", "Jeff Erickson", "http://jeffe.cs.illinois.edu/teaching/algorithms/"),
    Item("documents/algorithms_ch08_shortest_paths.pdf", "Algorithms, Chapter 8: Shortest Paths",
         "CC BY 4.0", "erickson", ALG + "08-sssp.pdf", "Jeff Erickson", "http://jeffe.cs.illinois.edu/teaching/algorithms/"),
    # ---- DOCX (Zenodo, CC BY 4.0) -------------------------------------------
    Item("documents/zenodo_first_steps_to_r.docx", "First Steps to R for Humanities & Social Sciences",
         "CC BY 4.0", "zenodo", "11209185", key="First steps to R-v2.0.docx"),
    Item("documents/zenodo_cpp_solved_problems.docx",
         "Collection of C++ Solved Problems Using Programming Methods - High-School Level",
         "CC BY 4.0", "zenodo", "18374967", key="COLLECTION OF SOLVED PROBLEMS.docx"),
    Item("documents/zenodo_learning_data_ethics.docx", "Learning Data Ethics for Data Sharing",
         "CC BY 4.0", "zenodo", "7291868", key="LearningDataEthicsforDataSharing_text_OER_v1.docx"),
    Item("documents/zenodo_ai_governance_syllabus.docx",
         "Good Governance for AI in Scientific Publications: Developing Policy for Reliability, Ethics, and Integrity (syllabus)",
         "CC BY 4.0", "zenodo", "13769486",
         key="Syllabus V08_ Good Governance for AI in Scientific Publications_ Developing Policy for "
             "Reliability, Ethics, and Integrity.docx"),
    Item("documents/zenodo_ten_simple_rules_identifiers.docx",
         "10 Simple rules for design, provision, and reuse of identifiers for web-based life science data",
         "CC BY 4.0", "zenodo", "31765", key="10RulesIdentifiers_MS_2015-09-24_Final_Clean.docx"),
    # ---- Images (Wikimedia Commons) -----------------------------------------
    Item("images/commons_diagram_transformer_architecture.png", "The Transformer model architecture",
         "CC BY-SA 3.0", "commons", "File:The-Transformer-model-architecture.png"),
    Item("images/commons_diagram_transformer_stacked_layers.png", "Transformer, stacked layers and sublayers",
         "CC BY 4.0", "commons", "File:Transformer, stacked layers and sublayers.png"),
    Item("images/commons_diagram_binary_search_flowchart.png", "Binary search flowchart",
         "Public domain", "commons", "File:BinarySearch.Flowchart.png"),
    Item("images/commons_screenshot_python_matplotlib_code.jpg", "Python code in a text editor with a matplotlib window",
         "CC BY-SA 4.0", "commons", "File:Screen-python-code-matplotlib-physics-simulation.jpg"),
    Item("images/commons_screenshot_matplotlib_figures_code.png", "matplotlib figures and source code",
         "CC BY-SA 4.0", "commons", "File:Mpl screenshot figures and code.png"),
    Item("images/commons_screenshot_rstudio_ide.png", "RStudio IDE", "CC BY-SA 4.0", "commons",
         "File:RStudio IDE screenshot.png"),
    Item("images/commons_photo_cap_computer_programmers_notes.jpg",
         "CAP computer: programmers' notes attached to the machine, Cambridge University Computer Laboratory",
         "CC BY-SA 3.0", "commons", "File:CAP Computer (notes) - Cambridge University.JPG"),
    Item("images/commons_photo_wifi_hotspot_sign.jpg", "Wi-Fi hotspot sign, Classon Playground, Brooklyn",
         "CC0", "commons", "File:CDSC wifi Classon jeh.jpg", max_side=1600,
         note="Downscaled so the longest side is 1600 px and re-saved as JPEG (quality 90)."),
    Item("images/commons_photo_digital_computer_laboratory.jpg",
         "Digital Computer Laboratory, University of Illinois Urbana-Champaign",
         "CC BY 2.0", "commons", "File:Digital Computer Laboratory.jpg", author="Dori",
         note="The API's artist field is empty; the author, Dori, is named on the Commons file page "
              "('Image taken by Dori'). The file page also offers GFDL; we rely on CC BY 2.0."),
    Item("images/commons_photo_former_computer_laboratory_tower.jpg",
         "Former Computer Laboratory tower, Cambridge (geograph)", "CC BY-SA 2.0", "commons",
         "File:Former Computer Laboratory tower - geograph.org.uk - 1704030.jpg"),
]


@dataclass
class Clip:
    dest: str
    source: str       # Commons file title of the full recording
    article: str      # the Wikipedia article being read
    start_s: float    # approximate start (snapped to a quiet moment)
    length_s: float   # approximate length (end snapped to a quiet moment)


CLIPS = [
    Clip("audio/spoken_wikipedia_neural_network_a.wav", "File:En-Neural network.ogg", "Neural network", 150, 65),
    Clip("audio/spoken_wikipedia_neural_network_b.wav", "File:En-Neural network.ogg", "Neural network", 1100, 65),
    Clip("audio/spoken_wikipedia_bioinformatics.wav", "File:En-Bioinformatics.ogg", "Bioinformatics", 400, 65),
    Clip("audio/spoken_wikipedia_wiki.wav", "File:En-Wiki2.ogg", "Wiki", 200, 65),
]
CLIP_SOURCES = {
    "File:En-Neural network.ogg": "CC BY-SA 3.0",
    "File:En-Bioinformatics.ogg": "CC BY-SA 3.0",
    "File:En-Wiki2.ogg": "CC BY-SA 3.0",
}


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def http_get(url: str, tries: int = 4) -> bytes:
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - any network failure is retried, then reported
            last = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def get_json(url: str) -> dict:
    import json
    return json.loads(http_get(url).decode("utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


# ---------------------------------------------------------------------------
# Licence verification (at the source, before anything is saved)
# ---------------------------------------------------------------------------

def _acl_year(ref: str) -> int:
    if re.match(r"^\d{4}\.", ref):          # new style: 2020.emnlp-main.550
        return int(ref[:4])
    return 2000 + int(re.match(r"^[A-Z](\d{2})-", ref).group(1))   # old style: D19-1410


def verify_acl(item: Item) -> tuple[str, str]:
    page = f"https://aclanthology.org/{item.ref}/"
    html = strip_tags(http_get(page).decode("utf-8", "replace"))
    if _acl_year(item.ref) < 2016:
        raise RuntimeError(f"{item.ref}: pre-2016 ACL material is CC BY-NC-SA, not usable")
    if "Creative Commons Attribution 4.0 International License" not in html:
        raise RuntimeError(f"{item.ref}: CC BY 4.0 statement not found on {page}")
    return item.licence, page


def verify_arxiv(item: Item) -> tuple[str, str]:
    page = f"https://arxiv.org/abs/{item.ref}"
    html = http_get(page).decode("utf-8", "replace")
    if "creativecommons.org/licenses/by/4.0" not in html:
        raise RuntimeError(f"arXiv {item.ref}: no CC BY 4.0 licence link on {page} (default arXiv licence is not enough)")
    return item.licence, page


def verify_erickson(item: Item) -> tuple[str, str]:
    html = strip_tags(http_get(item.page).decode("utf-8", "replace"))
    if "Creative Commons Attribution 4.0 International license" not in html:
        raise RuntimeError(f"Algorithms book page {item.page} no longer states CC BY 4.0")
    return item.licence, item.page


def zenodo_record(item: Item) -> dict:
    rec = get_json(f"https://zenodo.org/api/records/{item.ref}")
    licence = rec["metadata"].get("license", {}).get("id")
    if licence != "cc-by-4.0":
        raise RuntimeError(f"Zenodo {item.ref}: licence is {licence!r}, expected cc-by-4.0")
    return rec


def commons_info(titles: list[str]) -> dict[str, dict]:
    """One batched Commons API call: title -> {url, sha1, licence, artist, ...}."""
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        api = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
               "&iiprop=url|size|mime|sha1|extmetadata&titles=" + urllib.parse.quote("|".join(batch)))
        data = get_json(api)
        for page in data["query"]["pages"].values():
            if "imageinfo" not in page:
                raise RuntimeError(f"Commons file missing: {page.get('title')}")
            ii = page["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            out[page["title"]] = {
                "url": ii["url"].split("?")[0],
                "sha1": ii["sha1"],
                "licence": meta.get("LicenseShortName", {}).get("value", ""),
                "artist": strip_tags(meta.get("Artist", {}).get("value", "")),
                "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            }
    return out


# ---------------------------------------------------------------------------
# Audio: decode, cut at a quiet moment, write mono 16 kHz WAV
# ---------------------------------------------------------------------------

RATE = 16000


def decode_mono_16k(path: Path) -> np.ndarray:
    import av

    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    parts: list[np.ndarray] = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            parts.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        parts.append(out.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(parts)


def quietest_moment(samples: np.ndarray, around_s: float, window_s: float = 8.0) -> float:
    """Time (s) of the quietest ~0.2 s inside +/-window_s of `around_s`.
    Speakers pause between sentences, so this is almost always a sentence
    break; far better than cutting at an arbitrary offset mid-word."""
    lo = max(0, int((around_s - window_s) * RATE))
    hi = min(len(samples), int((around_s + window_s) * RATE))
    frame = int(0.02 * RATE)
    segment = samples[lo:hi].astype(np.float32)
    n = len(segment) // frame
    rms = np.sqrt((segment[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    width = 10
    smooth = np.convolve(rms, np.ones(width) / width, mode="valid")
    index = int(np.argmin(smooth))
    return (lo + (index + width / 2) * frame) / RATE


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="verify every licence at its source, write nothing")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "ragnova_corpus_cache",
                        help="scratch folder for the full-length audio recordings (never committed)")
    args = parser.parse_args()
    retrieved = date.today().isoformat()

    # ---- 1. Verify every licence at its source ----------------------------
    print("Verifying licences at each source ...")
    commons_titles = [i.ref for i in ITEMS if i.check == "commons"] + list(CLIP_SOURCES)
    commons = commons_info(commons_titles)
    zenodo: dict[str, dict] = {}
    resolved: dict[str, tuple[str, str]] = {}   # dest -> (licence, source page)
    for item in ITEMS:
        if item.check == "acl":
            resolved[item.dest] = verify_acl(item)
        elif item.check == "arxiv":
            resolved[item.dest] = verify_arxiv(item)
        elif item.check == "erickson":
            resolved[item.dest] = verify_erickson(item)
        elif item.check == "zenodo":
            rec = zenodo.setdefault(item.ref, zenodo_record(item))
            resolved[item.dest] = (item.licence, f"https://zenodo.org/records/{item.ref}")
            if not item.author:
                item.author = "; ".join(c["name"] for c in rec["metadata"].get("creators", []))
        elif item.check == "commons":
            info = commons[item.ref]
            if info["licence"] not in COMMONS_ALLOWED:
                raise RuntimeError(f"{item.ref}: licence {info['licence']!r} is not on the allowed list")
            if info["licence"] != item.licence:
                raise RuntimeError(f"{item.ref}: expected {item.licence!r}, Commons says {info['licence']!r}")
            if not item.author:
                item.author = info["artist"]
            if not item.author:
                raise RuntimeError(f"{item.ref}: no author on record; set one from the file page or drop the file")
            resolved[item.dest] = (info["licence"], info["page"])
        print(f"  ok  {item.dest}  [{resolved[item.dest][0]}]")
    for source, expected in CLIP_SOURCES.items():
        if commons[source]["licence"] != expected:
            raise RuntimeError(f"{source}: expected {expected}, Commons says {commons[source]['licence']!r}")
        print(f"  ok  {source}  [{expected}]")

    if args.dry_run:
        print("\nDry run: every licence verified, nothing written.")
        return 0

    # ---- 2. Download, verify bytes, write ---------------------------------
    rows: list[dict] = []
    for item in ITEMS:
        if item.check in ("acl", "arxiv"):
            pdf_url = (f"https://aclanthology.org/{item.ref}.pdf" if item.check == "acl"
                       else f"https://arxiv.org/pdf/{item.ref}")
            data = http_get(pdf_url)
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"{item.dest}: download is not a PDF")
            source_url = pdf_url
        elif item.check == "erickson":
            data = http_get(item.ref)
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"{item.dest}: download is not a PDF")
            source_url = item.ref
        elif item.check == "zenodo":
            files = {f["key"]: f for f in zenodo[item.ref]["files"]}
            record_file = files[item.key]
            data = http_get(record_file["links"]["self"])
            expected_md5 = record_file["checksum"].split(":", 1)[1]
            if hashlib.md5(data).hexdigest() != expected_md5:
                raise RuntimeError(f"{item.dest}: md5 mismatch against Zenodo's own checksum")
            if not data.startswith(b"PK"):
                raise RuntimeError(f"{item.dest}: download is not a DOCX (zip) file")
            source_url = record_file["links"]["self"]
        else:  # commons image
            info = commons[item.ref]
            data = http_get(info["url"])
            if hashlib.sha1(data).hexdigest() != info["sha1"]:
                raise RuntimeError(f"{item.dest}: sha1 mismatch against Commons' own checksum")
            source_url = info["url"]
            if item.max_side:
                with Image.open(io.BytesIO(data)) as img:
                    img = img.convert("RGB")
                    img.thumbnail((item.max_side, item.max_side))
                    out = io.BytesIO()
                    img.save(out, format="JPEG", quality=90)
                    data = out.getvalue()

        dest = DATA_DIR / item.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        rows.append({"file": f"data/{item.dest}", "title": item.title, "author": item.author,
                     "licence": resolved[item.dest][0], "source": source_url, "page": resolved[item.dest][1],
                     "sha256": sha256(data), "size": len(data), "note": item.note})
        print(f"  saved {item.dest}  ({len(data) / 1e6:.2f} MB)")

    # ---- 3. Audio: download once into the scratch cache, cut clips ---------
    args.cache.mkdir(parents=True, exist_ok=True)
    decoded: dict[str, np.ndarray] = {}
    for clip in CLIPS:
        info = commons[clip.source]
        if clip.source not in decoded:
            cached = args.cache / Path(info["url"]).name
            if not cached.exists() or hashlib.sha1(cached.read_bytes()).hexdigest() != info["sha1"]:
                data = http_get(info["url"])
                if hashlib.sha1(data).hexdigest() != info["sha1"]:
                    raise RuntimeError(f"{clip.source}: sha1 mismatch against Commons' own checksum")
                cached.write_bytes(data)
            decoded[clip.source] = decode_mono_16k(cached)
        samples = decoded[clip.source]
        start = quietest_moment(samples, clip.start_s)
        end = quietest_moment(samples, start + clip.length_s)
        data = wav_bytes(samples[int(start * RATE): int(end * RATE)])
        dest = DATA_DIR / clip.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        speaker = info["artist"].split("Authors of the article")[0].strip() or "volunteer speaker"
        rows.append({"file": f"data/{clip.dest}",
                     "title": f"Spoken Wikipedia: \"{clip.article}\" (clip, {start:.0f}s-{end:.0f}s of the recording)",
                     "author": f"{speaker}; text by the Wikipedia article's authors",
                     "licence": CLIP_SOURCES[clip.source], "source": info["url"], "page": info["page"],
                     "sha256": sha256(data), "size": len(data),
                     "note": f"Adaptation: cut from {start:.1f}s to {end:.1f}s of the original, converted to mono "
                             f"16 kHz WAV. Remains {CLIP_SOURCES[clip.source]} (ShareAlike)."})
        print(f"  saved {clip.dest}  ({len(data) / 1e6:.2f} MB, {end - start:.0f} s)")

    write_sources_md(rows, retrieved)
    print(f"\nWrote data/SOURCES.md ({len(rows)} files). Total {sum(r['size'] for r in rows) / 1e6:.1f} MB.")
    return 0


def write_sources_md(rows: list[dict], retrieved: str) -> None:
    lines = [
        "# Corpus provenance: downloaded files",
        "",
        "Every file below was downloaded from the source in its **Source** column, under the licence in its "
        f"**Licence** column, on **{retrieved}**. `scripts/fetch_corpus.py` regenerates them and, before saving "
        "anything, re-verifies each licence at its source. The Chapter 6 starter files "
        "(`notice.pdf`, `library_hours.pdf`, `it_onboarding.docx`, the 15 synthetic images, the 4 "
        "text-to-speech clips) are generated by `scripts/generate_*.py` and are not listed here.",
        "",
        "- **Attribution:** every entry names its author(s). CC BY / CC BY-SA require that.",
        "- **ShareAlike (CC BY-SA):** the Commons images and the audio clips stay under their own CC BY-SA "
        "licence. The audio clips are adaptations (cut and resampled), so they are CC BY-SA 3.0 as well. "
        "Nothing here relicenses the rest of the repository.",
        "- **Git LFS:** these files are stored in Git LFS (`.gitattributes`). Run `git lfs install` once before "
        "cloning or pulling, or the files arrive as small text pointers that no parser can open.",
        "- **sha256** is of the file as committed, so you can confirm you have the same bytes.",
        "",
        "| File | Title | Author(s) | Licence | Source | Notes | sha256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        title = r["title"].replace("|", "\\|")
        author = r["author"].replace("|", "\\|")
        note = r["note"].replace("|", "\\|")
        lines.append(f"| `{r['file']}` | {title} | {author} | {r['licence']} | [page]({r['page']}) · "
                     f"[file]({r['source']}) | {note} | `{r['sha256']}` |")
    (DATA_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
