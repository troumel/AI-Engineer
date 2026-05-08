"""Local pattern-based NER model used for offline training and inference."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExtractedEntity:
    """Internal extracted entity representation."""

    text: str
    label: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class PatternDefinition:
    """Persisted entity pattern."""

    phrase: str
    label: str
    score: float


class PatternNERModel:
    """Simple phrase-based NER model artifact with persistence."""

    backend_name = "pattern-fallback"

    def __init__(
        self,
        version_name: str,
        base_model_name: str,
        patterns: list[PatternDefinition],
        created_at: str | None = None,
        training_example_count: int = 0,
    ):
        self.version_name = version_name
        self.base_model_name = base_model_name
        self.patterns = sorted(patterns, key=lambda item: len(item.phrase), reverse=True)
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.training_example_count = training_example_count

    @property
    def entity_labels(self) -> list[str]:
        return sorted({pattern.label for pattern in self.patterns})

    @classmethod
    def default_baseline(cls, version_name: str, base_model_name: str) -> "PatternNERModel":
        """Create a baseline model with a few seeded entities."""
        patterns = [
            PatternDefinition(phrase="Sarah Johnson", label="PERSON", score=0.92),
            PatternDefinition(phrase="Microsoft", label="ORG", score=0.96),
            PatternDefinition(phrase="OpenAI", label="ORG", score=0.96),
            PatternDefinition(phrase="London", label="LOCATION", score=0.90),
            PatternDefinition(phrase="Seattle", label="LOCATION", score=0.90),
        ]
        return cls(
            version_name=version_name,
            base_model_name=base_model_name,
            patterns=patterns,
            training_example_count=0,
        )

    @classmethod
    def from_examples(
        cls,
        version_name: str,
        base_model_name: str,
        examples: list[dict[str, object]],
        base_patterns: list[PatternDefinition],
    ) -> "PatternNERModel":
        """Train a new local NER model from character-span annotations."""
        label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for example in examples:
            text = str(example["text"])
            raw_entities = example["entities"]
            if not isinstance(raw_entities, list):
                raise ValueError("Training example entities must be a list.")
            entities = raw_entities
            for entity in entities:
                start = int(entity["start"])
                end = int(entity["end"])
                label = str(entity["label"]).upper()
                if start < 0 or end > len(text) or start >= end:
                    raise ValueError(f"Invalid entity span for training example: {entity}")
                phrase = text[start:end].strip()
                if not phrase:
                    continue
                label_counts[phrase][label] += 1

        merged_patterns: dict[str, PatternDefinition] = {
            pattern.phrase.lower(): pattern for pattern in base_patterns
        }
        for phrase, counts in label_counts.items():
            label, count = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0]
            score = min(0.99, 0.70 + (count * 0.05))
            merged_patterns[phrase.lower()] = PatternDefinition(phrase=phrase, label=label, score=score)

        return cls(
            version_name=version_name,
            base_model_name=base_model_name,
            patterns=list(merged_patterns.values()),
            training_example_count=len(examples),
        )

    @classmethod
    def load(cls, version_directory: Path) -> "PatternNERModel":
        """Load a model artifact from disk."""
        metadata = json.loads((version_directory / "metadata.json").read_text(encoding="utf-8"))
        patterns_payload = json.loads((version_directory / "patterns.json").read_text(encoding="utf-8"))
        patterns = [PatternDefinition(**item) for item in patterns_payload]
        return cls(
            version_name=metadata["version_name"],
            base_model_name=metadata["base_model_name"],
            patterns=patterns,
            created_at=metadata["created_at"],
            training_example_count=metadata["training_example_count"],
        )

    def save(self, models_directory: Path) -> None:
        """Persist the model artifact to disk."""
        version_directory = models_directory / self.version_name
        version_directory.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version_name": self.version_name,
            "backend": self.backend_name,
            "base_model_name": self.base_model_name,
            "created_at": self.created_at,
            "training_example_count": self.training_example_count,
            "entity_labels": self.entity_labels,
        }
        patterns_payload = [pattern.__dict__ for pattern in self.patterns]

        (version_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (version_directory / "patterns.json").write_text(
            json.dumps(patterns_payload, indent=2), encoding="utf-8"
        )

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using phrase matching with overlap protection."""
        lowered_text = text.lower()
        occupied = [False] * len(text)
        matches: list[ExtractedEntity] = []

        for pattern in self.patterns:
            phrase_lower = pattern.phrase.lower()
            search_start = 0
            while True:
                index = lowered_text.find(phrase_lower, search_start)
                if index == -1:
                    break
                end = index + len(pattern.phrase)
                if not self._is_word_boundary(text, index, end):
                    search_start = index + 1
                    continue
                if any(occupied[index:end]):
                    search_start = end
                    continue

                for cursor in range(index, end):
                    occupied[cursor] = True
                matches.append(
                    ExtractedEntity(
                        text=text[index:end],
                        label=pattern.label,
                        start=index,
                        end=end,
                        score=pattern.score,
                    )
                )
                search_start = end

        matches.sort(key=lambda item: item.start)
        return matches

    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end == len(text) or not text[end].isalnum()
        return before_ok and after_ok