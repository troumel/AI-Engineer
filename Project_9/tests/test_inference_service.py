"""Tests for the Project 9 inference service."""

from app.services.inference_service import InferenceService
from scripts.prepare_dataset import prepare_demo_dataset
from scripts.train_model import train_support_ticket_model


def test_inference_service_predicts_from_saved_artifact(tmp_path):
    project_root = tmp_path / "project"
    prepare_demo_dataset(project_root)
    train_support_ticket_model(
        data_dir=project_root / "data" / "processed",
        models_dir=project_root / "models",
        version_name="ticket_classifier_v1",
        base_model_name="distilbert-base-uncased",
    )

    service = InferenceService(
        models_directory=str(project_root / "models"),
        default_model_version="ticket_classifier_v1",
        base_model_name="distilbert-base-uncased",
    )
    service.load_model(allow_degraded_startup=False)

    prediction = service.predict("I was charged twice and need a refund.")

    assert prediction.label == "billing"
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.model_version == "ticket_classifier_v1"


def test_inference_service_reports_degraded_without_artifact(tmp_path):
    service = InferenceService(
        models_directory=str(tmp_path / "models"),
        default_model_version="missing-model",
        base_model_name="distilbert-base-uncased",
    )
    service.load_model(allow_degraded_startup=True)

    health = service.get_health()

    assert health.status == "degraded"
    assert health.model_loaded is False
