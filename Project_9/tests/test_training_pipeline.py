"""Tests for the Project 9 training pipeline scaffold."""

import json

from scripts.prepare_dataset import prepare_demo_dataset
from scripts.train_model import train_support_ticket_model


def test_training_pipeline_writes_expected_artifacts(tmp_path):
    project_root = tmp_path / "project"
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    prepare_demo_dataset(project_root)
    artifact_dir = train_support_ticket_model(
        data_dir=data_dir / "processed",
        models_dir=models_dir,
        version_name="ticket_classifier_v1",
        base_model_name="distilbert-base-uncased",
    )

    assert (artifact_dir / "model.pt").exists()
    assert (artifact_dir / "labels.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "training_config.json").exists()
    assert (artifact_dir / "tokenizer" / "tokenizer_config.json").exists()

    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics.keys()) == {"train", "validation"}
    assert 0.0 <= metrics["train"]["accuracy"] <= 1.0
