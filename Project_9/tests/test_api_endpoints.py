"""Integration tests for the Project 9 API endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.inference_service import InferenceService
from scripts.prepare_dataset import prepare_demo_dataset
from scripts.train_model import train_support_ticket_model


@pytest.fixture(autouse=True)
def initialize_test_services(tmp_path: Path):
    project_root = tmp_path / "project"
    prepare_demo_dataset(project_root)
    train_support_ticket_model(
        data_dir=project_root / "data" / "processed",
        models_dir=project_root / "models",
        version_name="ticket_classifier_v1",
        base_model_name="distilbert-base-uncased",
    )

    dependencies._inference_service = InferenceService(
        models_directory=str(project_root / "models"),
        default_model_version="ticket_classifier_v1",
        base_model_name="distilbert-base-uncased",
    )
    dependencies._inference_service.load_model(allow_degraded_startup=False)
    yield
    dependencies._inference_service = None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_api_info(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check_returns_loaded_artifact_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_version"] == "ticket_classifier_v1"


def test_predict_returns_label_and_confidence(client):
    response = client.post(
        "/predict",
        json={"text": "My tracking page says delivered but the package is missing."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "shipping"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_rejects_blank_text(client):
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
