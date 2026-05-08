"""Evaluate a saved Project 9 artifact on the test split."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.services.inference_service import InferenceService, TrainingExample, compute_accuracy, compute_macro_f1


def _read_examples(path: Path) -> list[TrainingExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TrainingExample(text=item["text"], label=item["label"]) for item in payload]


def evaluate_saved_model(data_dir: Path, models_dir: Path, version_name: str) -> dict[str, float | str]:
    """Load a saved artifact and evaluate it on the test split."""
    service = InferenceService(
        models_directory=str(models_dir),
        default_model_version=version_name,
        base_model_name=settings.base_model_name,
    )
    service.load_model(allow_degraded_startup=False)

    examples = _read_examples(data_dir / "test.json")
    predictions = [service.predict(example.text).label for example in examples]
    gold = [example.label for example in examples]
    return {
        "model_version": version_name,
        "accuracy": round(compute_accuracy(gold, predictions), 4),
        "macro_f1": round(compute_macro_f1(gold, predictions), 4),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    results = evaluate_saved_model(
        data_dir=root / "data" / "processed",
        models_dir=root / settings.models_directory,
        version_name=settings.default_model_version,
    )
    print(json.dumps(results, indent=2))
