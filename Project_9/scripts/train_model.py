"""Train a scaffold ticket classifier artifact and write versioned outputs."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from app.config import settings
from app.services.inference_service import (
    TrainingExample,
    build_keyword_profiles,
    evaluate_examples,
)


def _read_examples(path: Path) -> list[TrainingExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TrainingExample(text=item["text"], label=item["label"]) for item in payload]


def _detect_training_device() -> str:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return "cpu"

    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def train_support_ticket_model(
    data_dir: Path,
    models_dir: Path,
    version_name: str,
    base_model_name: str,
) -> Path:
    """Build a versioned artifact using an offline-friendly keyword trainer."""
    train_examples = _read_examples(data_dir / "train.json")
    validation_examples = _read_examples(data_dir / "validation.json")

    labels = sorted({example.label for example in train_examples})
    keyword_profiles, label_priors = build_keyword_profiles(train_examples)

    train_metrics = evaluate_examples(
        train_examples, labels, keyword_profiles, label_priors
    )
    validation_metrics = evaluate_examples(
        validation_examples, labels, keyword_profiles, label_priors
    )

    artifact_dir = models_dir / version_name
    tokenizer_dir = artifact_dir / "tokenizer"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "labels.json").write_text(
        json.dumps(labels, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "keyword_profiles.json").write_text(
        json.dumps(
            {
                "keyword_profiles": keyword_profiles,
                "label_priors": label_priors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train": train_metrics,
                "validation": validation_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_config.json").write_text(
        json.dumps(
            {
                "model_version": version_name,
                "base_model_name": base_model_name,
                "training_backend": "keyword-fallback",
                "device": _detect_training_device(),
                "train_example_count": len(train_examples),
                "validation_example_count": len(validation_examples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model.pt").write_text(
        json.dumps({"placeholder": True, "version": version_name}, indent=2),
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "base_model_name": base_model_name,
                "tokenizer_backend": "placeholder",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    artifact_path = train_support_ticket_model(
        data_dir=root / "data" / "processed",
        models_dir=root / settings.models_directory,
        version_name=settings.default_model_version,
        base_model_name=settings.base_model_name,
    )
    print(f"Saved training artifact to {artifact_path}")
