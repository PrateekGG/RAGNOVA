from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.schemas import validate_chunk
from src.core.vector_store import get_client, get_image_collection


def _unit(index: int, dimension: int = 512) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


class FakeClip:
    model_id = "fake-clip/test"

    def embed_batch(self, images):
        return [self.embed_image(image) for image in images]

    def embed_image(self, image):
        red, _green, blue = image.convert("RGB").getpixel((0, 0))
        return _unit(0) if red > blue else _unit(1)

    def embed_text(self, text):
        return _unit(0) if "red" in text.lower() else _unit(1)


@pytest.fixture
def image_pipeline():
    from src.pipelines.images.ingest import ImageIngestionPipeline
    from src.pipelines.images.models import ImageIngestionConfig

    return ImageIngestionPipeline(
        config=ImageIngestionConfig(ocr_enabled=False),
        embedder=FakeClip(),
    )


def test_image_config_uses_shared_clip_settings():
    from src.pipelines.images.models import ImageIngestionConfig

    config = ImageIngestionConfig()
    assert config.model_name == settings.CLIP_MODEL
    assert config.pretrained == settings.CLIP_PRETRAINED


def test_images_are_indexed_with_portable_sources_and_found_by_text(
    tmp_path,
    monkeypatch,
    image_pipeline,
):
    from src.pipelines.images.index import index_images_directory
    from src.pipelines.images.search import search_images

    monkeypatch.chdir(tmp_path)
    images = Path("data/images")
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(images / "red.png")
    Image.new("RGB", (8, 8), (0, 0, 255)).save(images / "blue.png")
    (images / "notes.txt").write_text("not an image")

    client = get_client(persist_dir=tmp_path / "chroma")
    assert index_images_directory(images, client=client, pipeline=image_pipeline) == 2

    stored = get_image_collection(client).get(include=["metadatas"])
    assert sorted(metadata["source"] for metadata in stored["metadatas"]) == [
        "data/images/blue.png",
        "data/images/red.png",
    ]

    hits = search_images(
        query_text="a red square",
        client=client,
        embedder=FakeClip(),
    )
    assert hits[0].source == "data/images/red.png"
    assert hits[0].score == pytest.approx(1.0)
    assert validate_chunk(hits[0]) == []


def test_index_images_directory_is_flat_by_default_and_recurses_on_request(
    tmp_path,
    monkeypatch,
    image_pipeline,
):
    from src.pipelines.images.index import index_images_directory

    monkeypatch.chdir(tmp_path)
    images = Path("data/images")
    (images / "nested").mkdir(parents=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(images / "red.png")
    Image.new("RGB", (8, 8), (0, 0, 255)).save(images / "nested" / "blue.png")

    flat = get_client(persist_dir=tmp_path / "chroma_flat")
    assert index_images_directory(images, client=flat, pipeline=image_pipeline) == 1

    deep = get_client(persist_dir=tmp_path / "chroma_deep")
    assert index_images_directory(images, client=deep, pipeline=image_pipeline, recursive=True) == 2


def test_image_search_on_an_empty_index_returns_nothing_without_loading_clip(tmp_path):
    from src.pipelines.images.search import search_images

    client = get_client(persist_dir=tmp_path / "chroma")
    assert search_images(query_text="anything", client=client, embedder=object()) == []


def test_image_search_by_image(tmp_path, monkeypatch, image_pipeline):
    from src.pipelines.images.index import index_image_files
    from src.pipelines.images.search import search_images

    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (8, 8), (0, 0, 255)).save("blue.png")
    Image.new("RGB", (8, 8), (255, 0, 0)).save("red.png")

    client = get_client(persist_dir=tmp_path / "chroma")
    index_image_files(["blue.png", "red.png"], client=client, pipeline=image_pipeline)

    hits = search_images(
        query_image=Image.new("RGB", (8, 8), (10, 0, 200)),
        client=client,
        embedder=FakeClip(),
    )
    assert hits[0].source == "blue.png"
