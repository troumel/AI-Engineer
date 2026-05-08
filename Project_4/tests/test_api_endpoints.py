"""Integration tests for the Project 4 API endpoints."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.ner_service import NERService


@pytest.fixture(autouse=True)
def initialize_test_services(tmp_path: Path):
    """Inject an isolated NER service before each test."""
    dependencies._ner_service = NERService(
        models_directory=str(tmp_path / "models"),
        default_model_version="baseline-v1",
        base_model_name="bert-base-cased",
        max_training_workers=2,
        default_candidate_percentage=0.5,
    )
    yield
    dependencies._ner_service = None


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_api_info(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check_returns_baseline_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["active_version"] == "baseline-v1"


def test_ner_extract_returns_seeded_entities(client):
    response = client.post(
        "/ner/extract",
        json={"text": "Sarah Johnson from Microsoft visited London."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "baseline-v1"
    assert [entity["label"] for entity in payload["entities"]] == ["PERSON", "ORG", "LOCATION"]


def test_training_job_can_run_synchronously(client):
    response = client.post(
        "/training/jobs",
        json={
            "version_name": "custom-v1",
            "run_async": False,
            "examples": [
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "completed"


def test_training_job_status_can_be_polled(client):
    response = client.post(
        "/training/jobs",
        json={
            "version_name": "async-v1",
            "run_async": True,
            "examples": [
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        },
    )
    job_id = response.json()["job_id"]

    status_payload = None
    for _ in range(20):
        status_response = client.get(f"/training/jobs/{job_id}")
        status_payload = status_response.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.02)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["created_version"] == "async-v1"


def test_models_endpoint_lists_new_version(client):
    client.post(
        "/training/jobs",
        json={
            "version_name": "custom-v1",
            "run_async": False,
            "examples": [
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        },
    )

    response = client.get("/models")

    assert response.status_code == 200
    assert {item["version_name"] for item in response.json()["models"]} == {"baseline-v1", "custom-v1"}


def test_activate_model_switches_active_version(client):
    client.post(
        "/training/jobs",
        json={
            "version_name": "custom-v1",
            "run_async": False,
            "examples": [
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        },
    )

    response = client.post("/models/activate", json={"version_name": "custom-v1"})

    assert response.status_code == 200
    assert response.json()["active_version"] == "custom-v1"


def test_rollout_can_route_to_candidate_model(client):
    client.post(
        "/training/jobs",
        json={
            "version_name": "custom-v1",
            "run_async": False,
            "examples": [
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        },
    )
    client.post(
        "/experiments/rollout",
        json={
            "primary_version": "baseline-v1",
            "candidate_version": "custom-v1",
            "candidate_percentage": 1.0,
        },
    )

    response = client.post(
        "/ner/extract",
        json={"text": "Acme Robotics hired Maya Patel in Berlin.", "audience_id": "tenant-a"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["routing_strategy"] == "ab_test_candidate"
    assert payload["model_version"] == "custom-v1"
    assert [entity["label"] for entity in payload["entities"]] == ["ORG", "PERSON", "LOCATION"]