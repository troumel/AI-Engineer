"""Prepare a small demo dataset for support ticket classification."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

DEMO_EXAMPLES = [
    {"text": "I was charged twice for my monthly subscription.", "label": "billing"},
    {"text": "Please refund the incorrect invoice for last month.", "label": "billing"},
    {"text": "The payment receipt shows the wrong amount.", "label": "billing"},
    {"text": "The app crashes every time I upload a file.", "label": "technical_issue"},
    {
        "text": "I keep seeing a server error when I save changes.",
        "label": "technical_issue",
    },
    {"text": "The dashboard is stuck loading after login.", "label": "technical_issue"},
    {
        "text": "I am locked out and the reset email never arrived.",
        "label": "account_access",
    },
    {
        "text": "My account was disabled and I cannot sign in.",
        "label": "account_access",
    },
    {"text": "Two-factor login is failing on my phone.", "label": "account_access"},
    {"text": "Could you add dark mode to the dashboard?", "label": "feature_request"},
    {
        "text": "We need export to CSV in the reporting page.",
        "label": "feature_request",
    },
    {"text": "It would help to have keyboard shortcuts.", "label": "feature_request"},
    {"text": "My order still has not arrived after a week.", "label": "shipping"},
    {
        "text": "The tracking page says delivered but nothing arrived.",
        "label": "shipping",
    },
    {"text": "The shipment is delayed and I need an update.", "label": "shipping"},
]


def write_dataset_split(data_dir: Path, examples: list[dict[str, str]]) -> None:
    """Write train, validation, and test splits into the processed data directory."""
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    grouped_examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        grouped_examples[example["label"]].append(example)

    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    test: list[dict[str, str]] = []

    for label_examples in grouped_examples.values():
        train.extend(label_examples[:2])
        validation.extend(label_examples[2:3])
        test.extend(label_examples[3:])

    (processed_dir / "train.json").write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    (processed_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    (processed_dir / "test.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )


def prepare_demo_dataset(project_root: Path) -> None:
    """Create a small demo dataset in the project data directory."""
    write_dataset_split(project_root / "data", DEMO_EXAMPLES)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    prepare_demo_dataset(root)
    print(f"Prepared demo dataset under {root / 'data' / 'processed'}")
