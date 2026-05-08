"""Unit tests for the NER service."""

from pathlib import Path

from app.models.schemas import TrainingJobRequest
from app.services.ner_service import NERService


def build_service(tmp_path: Path) -> NERService:
    return NERService(
        models_directory=str(tmp_path / "models"),
        default_model_version="baseline-v1",
        base_model_name="bert-base-cased",
        max_training_workers=2,
        default_candidate_percentage=0.5,
    )


def test_baseline_model_exists(tmp_path: Path):
    service = build_service(tmp_path)

    models = service.list_models()

    assert models.active_version == "baseline-v1"
    assert len(models.models) == 1


def test_extract_entities_uses_explicit_version(tmp_path: Path):
    service = build_service(tmp_path)

    response = service.extract_entities(
        text="Sarah Johnson joined Microsoft in London.",
        model_version="baseline-v1",
        use_ab_test=False,
        audience_id=None,
    )

    assert response.routing_strategy == "explicit"
    assert [entity.label for entity in response.entities] == ["PERSON", "ORG", "LOCATION"]


def test_training_creates_new_version(tmp_path: Path):
    service = build_service(tmp_path)

    response = service.start_training_job(
        TrainingJobRequest(
            version_name="custom-v1",
            run_async=False,
            examples=[
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        )
    )

    assert response.status == "completed"
    assert {item.version_name for item in service.list_models().models} == {"baseline-v1", "custom-v1"}


def test_rollout_uses_candidate_when_percentage_is_one(tmp_path: Path):
    service = build_service(tmp_path)
    service.start_training_job(
        TrainingJobRequest(
            version_name="custom-v1",
            run_async=False,
            examples=[
                {
                    "text": "Acme Robotics hired Maya Patel in Berlin.",
                    "entities": [
                        {"start": 0, "end": 13, "label": "ORG"},
                        {"start": 20, "end": 30, "label": "PERSON"},
                        {"start": 34, "end": 40, "label": "LOCATION"},
                    ],
                }
            ],
        )
    )
    service.configure_rollout("baseline-v1", "custom-v1", 1.0)

    response = service.extract_entities(
        text="Acme Robotics hired Maya Patel in Berlin.",
        model_version=None,
        use_ab_test=True,
        audience_id="tenant-a",
    )

    assert response.routing_strategy == "ab_test_candidate"
    assert response.model_version == "custom-v1"