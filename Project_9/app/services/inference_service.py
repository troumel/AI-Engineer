"""Artifact loading and inference service for Project 9."""

from __future__ import annotations

import importlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.models.schemas import HealthCheckResponse, PredictionResponse


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "the",
    "this",
    "to",
    "was",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9_']+")


@dataclass(frozen=True)
class TrainingExample:
    """A single labeled training example."""

    text: str
    label: str


@dataclass(frozen=True)
class LoadedArtifact:
    """Minimal training artifact used by the offline scaffold."""

    version_name: str
    labels: list[str]
    keyword_profiles: dict[str, dict[str, int]]
    label_priors: dict[str, int]
    base_model_name: str
    backend: str
    device: str


def tokenize_text(text: str) -> list[str]:
    """Convert free-form text into normalized tokens."""
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    ]


def compute_accuracy(gold_labels: list[str], predicted_labels: list[str]) -> float:
    """Compute exact-match accuracy."""
    if not gold_labels:
        return 0.0

    correct = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == pred)
    return correct / len(gold_labels)


def compute_macro_f1(gold_labels: list[str], predicted_labels: list[str]) -> float:
    """Compute macro F1 without requiring scikit-learn."""
    label_set = sorted(set(gold_labels) | set(predicted_labels))
    if not label_set:
        return 0.0

    f1_scores: list[float] = []
    for label in label_set:
        true_positive = sum(
            1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred == label
        )
        false_positive = sum(
            1 for gold, pred in zip(gold_labels, predicted_labels) if gold != label and pred == label
        )
        false_negative = sum(
            1 for gold, pred in zip(gold_labels, predicted_labels) if gold == label and pred != label
        )

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
            continue

        f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


def build_keyword_profiles(examples: Iterable[TrainingExample]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Build simple per-label keyword counts from labeled examples."""
    profiles: dict[str, Counter[str]] = {}
    priors: Counter[str] = Counter()

    for example in examples:
        priors[example.label] += 1
        label_profile = profiles.setdefault(example.label, Counter())
        label_profile.update(tokenize_text(example.text))

    normalized_profiles = {
        label: dict(counter.most_common(40))
        for label, counter in profiles.items()
    }
    return normalized_profiles, dict(priors)


def evaluate_examples(
    examples: Iterable[TrainingExample],
    labels: list[str],
    keyword_profiles: dict[str, dict[str, int]],
    label_priors: dict[str, int],
) -> dict[str, float]:
    """Evaluate the keyword classifier on a labeled dataset."""
    example_list = list(examples)
    if not example_list:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    gold_labels = [example.label for example in example_list]
    predicted_labels = [
        predict_from_profiles(example.text, labels, keyword_profiles, label_priors)[0]
        for example in example_list
    ]
    accuracy = compute_accuracy(gold_labels, predicted_labels)
    macro_f1 = compute_macro_f1(gold_labels, predicted_labels)
    loss = max(0.0, 1.0 - accuracy)
    return {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }


def predict_from_profiles(
    text: str,
    labels: list[str],
    keyword_profiles: dict[str, dict[str, int]],
    label_priors: dict[str, int],
) -> tuple[str, float]:
    """Classify text using simple keyword profiles with additive smoothing."""
    tokens = tokenize_text(text)
    token_counts = Counter(tokens)
    raw_scores: dict[str, float] = {}

    for label in labels:
        profile = keyword_profiles.get(label, {})
        prior = max(label_priors.get(label, 1), 1)
        score = 1.0 + math.log(prior + 1)
        for token, count in token_counts.items():
            score += profile.get(token, 0) * count
        raw_scores[label] = score

    best_label = max(labels, key=lambda item: (raw_scores[item], item))
    total_score = sum(raw_scores.values()) or 1.0
    confidence = max(0.0, min(1.0, raw_scores[best_label] / total_score))
    return best_label, round(confidence, 4)


class InferenceService:
    """Load a saved artifact and expose offline-friendly inference methods."""

    def __init__(
        self,
        models_directory: str,
        default_model_version: str,
        base_model_name: str,
    ) -> None:
        self.models_directory = Path(models_directory)
        self.default_model_version = default_model_version
        self.base_model_name = base_model_name
        self._artifact: LoadedArtifact | None = None
        self._backend = "keyword-fallback"
        self._device = self._detect_device()

    def _detect_device(self) -> str:
        """Return the available torch device when torch is installed."""
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return "cpu"

        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load_model(self, allow_degraded_startup: bool = True) -> None:
        """Load the default model artifact from disk."""
        artifact_dir = self.models_directory / self.default_model_version
        try:
            self._artifact = self._load_artifact(artifact_dir)
            self._backend = self._artifact.backend
            self._device = self._artifact.device
        except FileNotFoundError:
            self._artifact = None
            if not allow_degraded_startup:
                raise

    def _load_artifact(self, artifact_dir: Path) -> LoadedArtifact:
        """Load metadata and keyword profiles from disk."""
        labels_path = artifact_dir / "labels.json"
        keyword_profiles_path = artifact_dir / "keyword_profiles.json"
        training_config_path = artifact_dir / "training_config.json"

        if not labels_path.exists() or not keyword_profiles_path.exists() or not training_config_path.exists():
            raise FileNotFoundError(f"Missing artifact files in {artifact_dir}")

        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        payload = json.loads(keyword_profiles_path.read_text(encoding="utf-8"))
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))

        return LoadedArtifact(
            version_name=training_config.get("model_version", self.default_model_version),
            labels=labels,
            keyword_profiles=payload.get("keyword_profiles", {}),
            label_priors=payload.get("label_priors", {}),
            base_model_name=training_config.get("base_model_name", self.base_model_name),
            backend=training_config.get("training_backend", "keyword-fallback"),
            device=training_config.get("device", self._detect_device()),
        )

    def is_model_loaded(self) -> bool:
        """Return whether the default artifact is loaded."""
        return self._artifact is not None

    def predict(self, text: str) -> PredictionResponse:
        """Predict a label and confidence for the provided text."""
        if self._artifact is None:
            raise RuntimeError("Model artifact is not loaded.")

        label, confidence = predict_from_profiles(
            text=text,
            labels=self._artifact.labels,
            keyword_profiles=self._artifact.keyword_profiles,
            label_priors=self._artifact.label_priors,
        )
        return PredictionResponse(
            label=label,
            confidence=confidence,
            model_version=self._artifact.version_name,
            backend=self._artifact.backend,
        )

    def get_health(self) -> HealthCheckResponse:
        """Return health details for the inference service."""
        return HealthCheckResponse(
            status="healthy" if self._artifact is not None else "degraded",
            model_loaded=self._artifact is not None,
            model_version=(
                self._artifact.version_name if self._artifact is not None else self.default_model_version
            ),
            base_model_name=(
                self._artifact.base_model_name if self._artifact is not None else self.base_model_name
            ),
            device=self._device,
            backend=self._backend,
        )
