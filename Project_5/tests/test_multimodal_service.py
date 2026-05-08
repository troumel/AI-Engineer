"""Unit tests for the multi-modal service and supporting components."""

from pathlib import Path

import pytest

from app.services.multimodal_service import MultiModalService
from app.services.vision_provider import (
    HashingVisionProvider,
    cosine_similarity,
)


@pytest.fixture
def service(tmp_path: Path) -> MultiModalService:
    return MultiModalService(
        images_directory=str(tmp_path / "images"),
        metadata_file=str(tmp_path / "images" / "metadata.json"),
        vision_provider="hashing",
        embedding_dimensions=64,
        max_upload_bytes=1024 * 1024,
    )


def test_hashing_provider_aligns_text_and_image_embeddings():
    provider = HashingVisionProvider(dimensions=64)
    image_vector = provider.embed_image(b"\x89PNG\r\n\x1a\n-bytes", hints=["beach", "sunset"])
    matching_text = provider.embed_text("beach sunset")
    unrelated_text = provider.embed_text("server logs database")

    assert cosine_similarity(image_vector, matching_text) > cosine_similarity(
        image_vector, unrelated_text
    )


def test_hashing_provider_caption_uses_hints():
    provider = HashingVisionProvider(dimensions=64)
    caption = provider.caption(b"bytes", hints=["mountain", "snow"], prompt=None)
    assert "mountain" in caption.lower()
    assert "snow" in caption.lower()


def test_hashing_provider_answer_returns_yes_on_overlap():
    provider = HashingVisionProvider(dimensions=64)
    answer, confidence = provider.answer(b"bytes", "Is there a cat?", hints=["cat", "fluffy"])
    assert answer.lower().startswith("yes")
    assert confidence > 0.5


def test_upload_then_text_search_finds_image(service: MultiModalService):
    upload = service.upload_image(
        filename="apple.png",
        content_type="image/png",
        data=b"\x89PNG\r\n\x1a\napple",
        tags=["apple", "fruit", "red"],
    )

    results = service.search_by_text(query="red apple", top_k=3)

    assert results.count >= 1
    assert results.results[0].image_id == upload.image_id


def test_caption_persists_on_record(service: MultiModalService):
    upload = service.upload_image(
        filename="ocean.png",
        content_type="image/png",
        data=b"\x89PNG\r\n\x1a\nocean",
        tags=["ocean", "blue"],
    )

    response = service.caption_image(image_id=upload.image_id, prompt=None)
    info = service.get_image(upload.image_id)

    assert info.caption == response.caption
    assert "ocean" in response.caption.lower()


def test_upload_validates_size_limit(tmp_path: Path):
    tiny = MultiModalService(
        images_directory=str(tmp_path / "images"),
        metadata_file=str(tmp_path / "images" / "metadata.json"),
        vision_provider="hashing",
        embedding_dimensions=32,
        max_upload_bytes=4,
    )
    with pytest.raises(ValueError):
        tiny.upload_image(
            filename="big.png",
            content_type="image/png",
            data=b"\x89PNG-too-big",
            tags=["x"],
        )


def test_metadata_survives_service_restart(tmp_path: Path):
    images_dir = tmp_path / "images"
    metadata = images_dir / "metadata.json"

    first = MultiModalService(
        images_directory=str(images_dir),
        metadata_file=str(metadata),
        vision_provider="hashing",
        embedding_dimensions=32,
        max_upload_bytes=1024,
    )
    upload = first.upload_image(
        filename="x.png",
        content_type="image/png",
        data=b"\x89PNG\r\n\x1a\nx",
        tags=["alpha"],
    )

    reloaded = MultiModalService(
        images_directory=str(images_dir),
        metadata_file=str(metadata),
        vision_provider="hashing",
        embedding_dimensions=32,
        max_upload_bytes=1024,
    )
    listing = reloaded.list_images()

    assert listing.count == 1
    assert listing.images[0].image_id == upload.image_id
