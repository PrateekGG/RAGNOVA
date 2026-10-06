# Chapter 8 — Image Pipeline: CLIP + OCR (Day 9)

> **Deliverables today:** a complete image ingestion pipeline (`src/pipelines/images/`): Tesseract OCR for text extraction, OpenCLIP for visual embeddings, portable source paths, and the write path into ChromaDB's `image_index` collection; `ImageIngestionConfig` bound to `src.core.config.settings` for centralized CLIP model selection; real cross-modal search (text query → visual matches); and `tests/test_image_pipeline.py`.
>
> **Prerequisites:** [Chapter 7](ch07-embeddings-and-vector-database.md) (ChromaDB collections, the `add_chunks()` primitive this chapter reuses) and [ADR-003](../decisions/adr-003-two-vector-collections.md) (why a separate `image_index` collection, not merged with `text_index`).
>
> **The framing for today.** This is RAGNova's first multimodal track. Chapter 7 built `text_index` for documents; this chapter builds `image_index` for visual content — same ChromaDB backend, different embedding model (OpenCLIP instead of MiniLM), different modality-specific metadata (OCR text, image dimensions, sha256 hash). The two collections stay separate (ADR-003's modality gap), but Chapter 12 will merge their results by rank (ADR-007) so a single question can retrieve both a PDF page and a matching photo.

---

## How to read this chapter

| Part | What it does | Time |
|---|---|---|
| **Part 1 — LEARN: OpenCLIP in Practice** | Loading a vision-language model, `encode_image()` vs. `encode_text()`, the shared embedding space, normalization. | ~30 min |
| **Part 2 — LEARN: Tesseract OCR** | Installing Tesseract, extracting text from images, confidence scores, preprocessing. | ~30 min |
| **Part 3 — LEARN: Portable Image Paths** | The same absolute-path bug Chapter 7 fixed for documents, now for images. | ~15 min |
| **Part 4 — DECIDE** | Module layout, why CLIP config binds to `settings`, batch size tradeoffs, OCR normalization. | ~20 min |
| **Part 5 — BUILD** | Writing each file, indexing real images, running cross-modal search. | ~2 hours |
| **Part 6 — CHECK** | Rubric, question bank, troubleshooting, completion checklist. | ~30 min |

**Learning outcomes.** You will be able to: call OpenCLIP to embed images and text into a shared 512-d space; explain why the same model produces both visual and text embeddings; run Tesseract OCR on a PIL image; explain the config-drift bug this chapter fixes (hardcoded `"ViT-B-32"` vs. `settings.CLIP_MODEL`); and trace through the portable-path fix that prevents absolute paths from leaking into `image_index`.

---

# Part 1 — LEARN: OpenCLIP in Practice

## 1.1 Vision-language models vs. text-only embeddings

Chapter 7's MiniLM model (`all-MiniLM-L6-v2`) embeds *text only* — sentences → 384-d vectors. OpenCLIP is a **vision-language model**: it embeds both images and text into the *same* 512-d space, trained so that an image of a cat and the word "cat" have similar vectors. This shared space is what enables cross-modal search: embed a text query with `encode_text()`, search against image vectors produced by `encode_image()`, and cosine similarity ranks images by how well they match the text.

## 1.2 Loading and calling OpenCLIP

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
)
model.eval()

img = Image.open("photo.jpg")
img_tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
    img_features = model.encode_image(img_tensor)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)  # normalize

text_tokens = open_clip.tokenize(["a cat"])
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

similarity = (img_features @ text_features.T).item()  # cosine similarity
```

`preprocess` is a torchvision transform chain (resize, center-crop, normalize) specific to the chosen model — always apply it before encoding.

## 1.3 Normalization — the same lesson as Chapter 7 §1.3, for images

L2-normalizing the output vectors (dividing by their length) makes cosine similarity equivalent to dot product, and lets ChromaDB's `hnsw:space: "cosine"` report directly-interpretable distances. `src/pipelines/images/embedding.py` always normalizes when `config.normalize_embeddings=True` (the default), matching Chapter 7's text embedding behavior.

## 1.4 Model selection — why `"ViT-B-32/laion2b_s34b_b79k"`

ViT-B-32 is OpenCLIP's base Vision Transformer (about 150M parameters across its image and text encoders), pretrained on LAION-2B — a web-scale image-text dataset. It balances quality and inference speed for CPU-only deployment (Chapter 5's 8GB RAM budget). Larger models (`ViT-L-14`) would give better retrieval but exceed memory constraints. `laion2b_s34b_b79k` is the specific checkpoint ID; different pretraining datasets are available for the same architecture.

---

# Part 2 — LEARN: Tesseract OCR

## 2.1 Installing Tesseract

Tesseract is a separate binary, not a Python package:

- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** download the installer from [UB-Mannheim's repository](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env` to the full path (e.g. `C:/Program Files/Tesseract-OCR/tesseract.exe`)

The Python wrapper is `pytesseract` (already in `requirements.txt`), which calls the binary.

## 2.2 Extracting text from a PIL image

```python
import pytesseract
from PIL import Image

img = Image.open("scan.png")
text = pytesseract.image_to_string(img, lang="eng", config="--psm 3")
```

`--psm 3` is Tesseract's page segmentation mode 3 (fully automatic page segmentation, no orientation detection) — the default, suitable for typical document scans. Other modes (`--psm 6` for a single uniform block, `--psm 11` for sparse text) exist but aren't needed for this project's use case.

## 2.3 Confidence scores and word-level data

```python
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
```

Returns a dict with per-word bounding boxes, confidence scores (0–100), and text. `src/pipelines/images/ocr.py` uses this to compute an aggregate confidence for the entire image, stored in `ImageChunk.metadata`.

## 2.4 Preprocessing — grayscale, thresholding

Tesseract's accuracy improves on high-contrast images. `TesseractOCREngine` optionally converts to grayscale and applies binary thresholding before extraction (`ImageIngestionConfig.preprocess_for_ocr=True`). For clean, high-resolution scans this makes little difference; for low-quality phone photos it's often necessary.

---

# Part 3 — LEARN: Portable Image Paths

## 3.1 The bug — absolute paths leaking into `source`

`ImageIngestionPipeline._load_image()` resolves paths with `Path(image_input).expanduser().resolve()` — turning `"data/images/photo.png"` into `"/home/alice/RAGNova/data/images/photo.png"`. That absolute path is stored verbatim in `ImageChunk.source`, then written into ChromaDB. Citations later render this absolute path, baking Alice's checkout folder into the displayed source — the exact bug Chapter 7 §5.3 already fixed for documents.

## 3.2 The fix — `_relative_to_cwd()` in `src/pipelines/images/index.py`

```python
def _relative_to_cwd(source: str) -> str:
    try:
        return Path(source).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return source  # already relative, or outside the project
```

Called in `_to_index_chunk()` before the `Chunk` is written to ChromaDB, converting `/home/alice/RAGNova/data/images/photo.png` → `data/images/photo.png`. The same portable-path discipline Chapter 7 established, applied to images.

---

# Part 4 — DECIDE

## 4.1 Module layout

```
src/pipelines/images/
├── __init__.py
├── models.py          ← ImageIngestionConfig, ImageChunk, OCRResult, ImageMetadata
├── ocr.py             ← TesseractOCREngine (wraps pytesseract)
├── embedding.py       ← OpenCLIPEmbedder (wraps open-clip-torch)
├── ingest.py          ← ImageIngestionPipeline (orchestrates OCR + CLIP)
├── index.py           ← index_images_directory(), _to_index_chunk(), _relative_to_cwd()
├── search.py          ← search_images() (text-to-image and image-to-image search)
├── cli.py             ← command-line entry point
└── smoke_test.py      ← quick sanity check
```

Kept under `pipelines/images/` (not `core/`) because this is Track B's domain-specific code. Only the ChromaDB write primitive (`src/core/vector_store.add_chunks()`) and text normalization (`src/core/text_normalize.normalize_text()`) are shared with other tracks.

## 4.2 Why `ImageIngestionConfig` binds to `settings.CLIP_MODEL` / `settings.CLIP_PRETRAINED`

**The config-drift bug:** `models.py` originally hardcoded `model_name: str = "ViT-B-32"` and `pretrained: str = "laion2b_s34b_b79k"` as dataclass defaults, independent of `src/core/config.settings`. Changing the CLIP model required editing two places (`.env` and `models.py`), and forgetting one would silently use mismatched models for indexing vs. search — exactly the kind of drift Chapter 7's centralized `settings.TEXT_EMBEDDING_MODEL` was designed to prevent.

**The fix:** `ImageIngestionConfig` now defaults to `settings.CLIP_MODEL` and `settings.CLIP_PRETRAINED`:

```python
def _get_clip_model() -> str:
    from src.core.config import settings
    return settings.CLIP_MODEL

def _get_clip_pretrained() -> str:
    from src.core.config import settings
    return settings.CLIP_PRETRAINED

@dataclass
class ImageIngestionConfig:
    model_name: str = field(default_factory=_get_clip_model)
    pretrained: str = field(default_factory=_get_clip_pretrained)
```

The now-unused `DEFAULT_MODEL_NAME` / `DEFAULT_PRETRAINED` constants in `embedding.py` were deleted for the same reason: a second hardcoded copy of the model name is exactly the drift being fixed.

One source of truth, same discipline as audio (`settings.TEXT_EMBEDDING_MODEL`) and generation (`settings.OLLAMA_MODEL`).

## 4.3 Batch size — memory vs. throughput tradeoff

`ImageIngestionConfig.batch_size=16` is how many images are decoded and embedded together in one sub-batch. Larger batches are faster (one GPU/CPU forward pass for 16 images costs less than 16 separate passes), but hold 16 decoded images + 16 preprocessed tensors in memory simultaneously. Set too high on constrained hardware and ingestion OOMs before finishing. 16 is a starting point — lower it for high-resolution corpora on CPU-only machines.

## 4.4 OCR text normalization — reusing Chapter 6's `normalize_text()`

Tesseract output is full of stray newlines, double spaces, and inconsistent whitespace. `src/pipelines/images/index.py`'s `_to_index_chunk()` calls `normalize_text()` (Chapter 6's shared cleanup) before writing to ChromaDB — the exact same whitespace cleanup documents received, applied to OCR text.

---

# Part 5 — BUILD: the actual Day 9

## 5.1 Write `src/pipelines/images/models.py`

`ImageIngestionConfig`, `OCRResult`, `ImageMetadata`, `ImageChunk` — see Part 4.2 for the `settings` binding fix.

## 5.2 Write `src/pipelines/images/ocr.py`

`TesseractOCREngine.extract_text()` — see Part 2. Confirm Tesseract is installed and reachable:

```bash
tesseract --version
```

**Real captured output:**
```
tesseract 5.3.0
```

## 5.3 Write `src/pipelines/images/embedding.py`

`OpenCLIPEmbedder.embed_image()`, `.embed_text()`, `.embed_batch()` — see Part 1. Lazy-loads the model on first call, same pattern as Chapter 7's `embeddings.py`.

## 5.4 Write `src/pipelines/images/ingest.py`

`ImageIngestionPipeline.ingest_image()`, `.ingest_batch()`, `.ingest_directory()` — orchestrates OCR + CLIP, produces `ImageChunk` objects. Includes the per-file error handling (one corrupt image skipped, rest proceed) and sub-batch memory mitigation (Part 4.3).

## 5.5 Write `src/pipelines/images/index.py` and `search.py`

`index_images_directory()` + `_relative_to_cwd()` (the portable-path fix, Part 3.2) and `search_images()` (text-to-image and image-to-image search).

## 5.6 Test the pipeline with real images

Use a throwaway folder and a throwaway Chroma directory, not `data/images/` or the real `chroma_db/`, which hold the project's real corpus and index:

```bash
python -c "
import tempfile
from pathlib import Path
from PIL import Image
from src.core.vector_store import get_client
from src.pipelines.images.index import index_images_directory
from src.pipelines.images.search import search_images

# ignore_cleanup_errors: on Windows, ChromaDB can still hold its files open when the folder is deleted
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    images = Path(tmp) / 'images'
    images.mkdir()
    Image.new('RGB', (100, 100), (255, 0, 0)).save(images / 'red.png')
    Image.new('RGB', (100, 100), (0, 0, 255)).save(images / 'blue.png')

    client = get_client(persist_dir=Path(tmp) / 'chroma')
    print('Indexed', index_images_directory(images, client=client), 'images')

    for hit in search_images(query_text='a red square', client=client, top_k=2):
        print(f'{Path(hit.source).name}  score={hit.score:.3f}')
"
```

**Real captured output** (Windows, Python 3.14, real OpenCLIP `ViT-B-32`; your exact scores may differ slightly):
```
Indexed 2 images
red.png  score=0.275
blue.png  score=0.235
```

Cross-modal search works: the text query "a red square" ranks the red image first.

**Don't expect scores near 1.0.** CLIP's text-to-image cosine similarities sit around 0.2-0.35 even for a correct match. This is the *modality gap* (ADR-003, ADR-007), and it is why Chapter 12 gives images their own, lower relevance floor (ADR-010) and merges text and image results by rank, never by raw score.

## 5.7 Run `tests/test_image_pipeline.py`

Moved out of `test_integration.py` (its old Part 4), plus one new test, `test_image_config_uses_shared_clip_settings`, that locks in the §4.2 fix:

```bash
pytest tests/test_image_pipeline.py -v
```

**Real captured output:**
```
tests/test_image_pipeline.py::test_image_config_uses_shared_clip_settings PASSED [ 25%]
tests/test_image_pipeline.py::test_images_are_indexed_with_portable_sources_and_found_by_text PASSED [ 50%]
tests/test_image_pipeline.py::test_image_search_on_an_empty_index_returns_nothing_without_loading_clip PASSED [ 75%]
tests/test_image_pipeline.py::test_image_search_by_image PASSED          [100%]

======================== 4 passed, 1 warning in 9.39s =========================
```

## 5.8 Git hygiene for today

- [ ] `src/pipelines/images/{models,ocr,embedding,ingest,index,search}.py` committed
- [ ] `tests/test_image_pipeline.py` committed; image tests removed from `test_integration.py`
- [ ] `docs/chapters/ch08-image-pipeline-clip-and-ocr.md` committed
- [ ] Every team member has confirmed Tesseract is installed and `pytest tests/test_image_pipeline.py` passes

---

# Part 6 — CHECK

## 6.1 Rubric

| # | Criterion | Score |
|---|---|---|
| 1 | `tesseract --version` runs successfully | /3 |
| 2 | `ImageIngestionConfig` defaults to `settings.CLIP_MODEL` / `settings.CLIP_PRETRAINED` | /3 |
| 3 | `index_images_directory()` indexes real images into `image_index` | /3 |
| 4 | `search_images(query_text="...")` returns results ranked by similarity | /3 |
| 5 | `pytest tests/test_image_pipeline.py -v` fully passes | /3 |
| 6 | Indexed images have portable `source` paths (relative, not absolute) | /3 |
| 7 | Team can explain why OpenCLIP embeds both images and text | /3 |
| 8 | Team can explain the config-drift bug and how it was fixed | /3 |
| 9 | Team can explain why `batch_size=16` is a tradeoff, not a fixed constant | /3 |
| 10 | `data/images/` directory exists with at least 2 indexed images | /3 |
| | **Total** | **/30** |

## 6.2 Question bank

1. What's the difference between MiniLM (Chapter 7) and OpenCLIP (this chapter)? → §1.1; MiniLM embeds text only, OpenCLIP embeds images and text into a shared space.
2. Why does OpenCLIP embedding text and images into the same space enable cross-modal search? → §1.1; text query vector and image vectors are directly comparable via cosine similarity.
3. What does `preprocess` (from `create_model_and_transforms()`) actually do? → §1.2; torchvision transform chain (resize, center-crop, normalize) specific to the chosen model.
4. Why is `batch_size=16` not `batch_size=1` or `batch_size=1000`? → §4.3; 1 is too slow (no batching benefit), 1000 holds every decoded image in memory and OOMs on constrained hardware.
5. What did the config-drift bug cause before it was fixed? → §4.2; changing CLIP model in `.env` wouldn't change the model `ImageIngestionConfig` actually used, silently indexing with one model and searching with another.
6. Why does `_to_index_chunk()` call `normalize_text()` on OCR output? → §4.4; Tesseract output has stray newlines and double spaces, same cleanup documents received in Chapter 6.
7. What does `_relative_to_cwd()` prevent? → §3.1–3.2; absolute paths (e.g. `/home/alice/RAGNova/data/images/x.png`) leaking into citations.
8. A user asks "show me photos of a library." Which method handles this query? → §5.5; `search_images(query_text="show me photos of a library")` — text-to-image search.
9. A user uploads a photo and asks "find similar images." Which method handles this? → §5.5; `search_images(query_image=uploaded_img)` — image-to-image search.
10. Why does this chapter use `add_chunks()` (Chapter 7's function) instead of writing its own ChromaDB upsert logic? → §4.1; shared primitive, same upsert-over-add discipline, no duplication.

## 6.3 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every image comes back with empty OCR text, and the log warns that Tesseract isn't available | Tesseract binary not installed or not on PATH. `TesseractOCREngine` degrades (logs a warning, returns empty text) rather than raising | Install Tesseract (§2.1), or set `TESSERACT_CMD` in `.env` to the full path |
| OpenCLIP download is slow or times out | Hugging Face rate-limiting anonymous downloads | Wait it out; only worth a free Hugging Face account if repeated across the team |
| `IndexError: list index out of range` in `embed_batch()` | Passing an empty image list | Guard with `if not images: return []` — already in `embedding.py` |
| Images indexed with absolute paths despite the fix | `index.py` not imported, or an old version cached | Confirm `_relative_to_cwd()` is in `index.py` and the file was saved; `python -Bc "..."` to bypass bytecode cache |
| `ModuleNotFoundError: No module named 'open_clip'` | `open-clip-torch` not installed | `pip install -r requirements.txt` |
| High memory usage during `ingest_directory()` on 1000+ images | `batch_size` too large for available RAM | Lower `ImageIngestionConfig.batch_size` to 8 or 4 |

## 6.4 Day 9 completion checklist

- [ ] Tesseract installed and `tesseract --version` runs
- [ ] `ImageIngestionConfig` binds to `settings.CLIP_MODEL` / `settings.CLIP_PRETRAINED`
- [ ] At least 2 images indexed into `image_index`, portable sources confirmed
- [ ] `search_images(query_text="...")` returns ranked results
- [ ] `pytest tests/test_image_pipeline.py -v` fully passes
- [ ] Team can explain the config-drift bug and its fix
- [ ] Rubric §6.1 scored ≥ 24/30

---

**Next:** Chapter 9 — Audio Pipeline (Whisper transcription), where Track B builds the audio ingestion path — transcribing spoken content into text chunks with timestamp ranges, embedded by the same MiniLM model as documents, and written into `text_index` alongside PDFs and DOCX files.
