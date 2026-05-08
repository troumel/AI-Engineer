"""Integration tests for the Project 5 multi-modal API endpoints."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.multimodal_service import MultiModalService


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _png_bytes(payload: bytes) -> bytes:
    """Return a deterministic byte string that mimics a tiny PNG file."""
    return PNG_HEADER + payload


@pytest.fixture(autouse=True)
def initialize_test_services(tmp_path: Path):
    """Inject an isolated multi-modal service before each test."""
    images_dir = tmp_path / "images"
    metadata_file = images_dir / "metadata.json"
    dependencies._multimodal_service = MultiModalService(
        images_directory=str(images_dir),
        metadata_file=str(metadata_file),
        vision_provider="hashing",
        embedding_dimensions=64,
        max_upload_bytes=1 * 1024 * 1024,
    )
    yield
    dependencies._multimodal_service = None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, *, name: str, payload: bytes, tags: str | None = None):
    files = {"file": (name, io.BytesIO(_png_bytes(payload)), "image/png")}
    data = {"tags": tags} if tags is not None else None
    return client.post("/images", files=files, data=data)


def test_root_returns_api_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_reports_hashing_provider(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["vision_provider"] == "hashing"
    assert payload["image_count"] == 0


def test_upload_image_persists_metadata(client):
    response = _upload(client, name="cat.png", payload=b"cat-bytes", tags="cat,animal")

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "cat.png"
    assert payload["content_type"] == "image/png"
    assert payload["tags"] == ["cat", "animal"]
    assert payload["size_bytes"] > 0


def test_upload_rejects_non_image_content_type(client):
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post("/images", files=files)
    assert response.status_code == 400


def test_list_and_get_image(client):
    upload = _upload(client, name="dog.png", payload=b"dog-bytes", tags="dog")
    image_id = upload.json()["image_id"]

    list_response = client.get("/images")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1

    detail = client.get(f"/images/{image_id}")
    assert detail.status_code == 200
    assert detail.json()["image_id"] == image_id

    missing = client.get("/images/does-not-exist")
    assert missing.status_code == 404


def test_caption_endpoint_includes_hint_tokens(client):
    upload = _upload(client, name="beach.png", payload=b"beach-bytes", tags="beach,sunset")
    image_id = upload.json()["image_id"]

    response = client.post(f"/images/{image_id}/caption", json={"prompt": "Describe it"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_id"] == image_id
    assert "beach" in payload["caption"].lower()
    assert payload["model"] == "hashing"


def test_vqa_uses_tags_when_question_overlaps(client):
    upload = _upload(client, name="park.png", payload=b"park-bytes", tags="dog,park")
    image_id = upload.json()["image_id"]

    response = client.post(
        f"/images/{image_id}/vqa",
        json={"question": "Is there a dog in the picture?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].lower().startswith("yes")
    assert "dog" in payload["answer"].lower()
    assert 0.0 < payload["confidence"] <= 1.0


def test_text_search_ranks_matching_image_first(client):
    cat = _upload(client, name="kitten.png", payload=b"kitten", tags="cat,kitten,furry").json()
    car = _upload(client, name="sedan.png", payload=b"sedan", tags="car,sedan,vehicle").json()

    response = client.post("/search/text", json={"query": "fluffy kitten cat", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    top_ids = [result["image_id"] for result in payload["results"]]
    assert top_ids[0] == cat["image_id"]
    assert car["image_id"] in top_ids


def test_image_to_image_search_excludes_query_image(client):
    first = _upload(client, name="forest1.png", payload=b"forest-a", tags="forest,trees").json()
    _upload(client, name="forest2.png", payload=b"forest-b", tags="forest,trees")
    _upload(client, name="city.png", payload=b"city", tags="city,buildings")

    response = client.post(
        f"/search/images/{first['image_id']}",
        json={"top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    returned_ids = {result["image_id"] for result in payload["results"]}
    assert first["image_id"] not in returned_ids
    assert payload["count"] == 2


def test_caption_404_for_missing_image(client):
    response = client.post("/images/missing/caption", json={})
    assert response.status_code == 404
