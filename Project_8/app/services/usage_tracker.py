"""Per-API-key + per-model usage tracking with JSON persistence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from app.models.schemas import UsageRecord, UsageReport


class UsageTracker:
    """Aggregate usage counters with simple JSON persistence."""

    def __init__(
        self,
        usage_file: str,
        cost_per_1k_prompt_tokens: float,
        cost_per_1k_completion_tokens: float,
    ) -> None:
        self.usage_file = Path(usage_file)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self.cost_per_1k_prompt_tokens = cost_per_1k_prompt_tokens
        self.cost_per_1k_completion_tokens = cost_per_1k_completion_tokens

        self._lock = Lock()
        self._total_requests = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._by_api_key: dict[str, dict[str, Any]] = defaultdict(self._empty_bucket)
        self._by_model: dict[str, dict[str, Any]] = defaultdict(self._empty_bucket)
        self._load()

    def record(
        self,
        api_key: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageRecord:
        cost = self._cost(prompt_tokens, completion_tokens)
        record = UsageRecord(
            api_key=api_key,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
        )
        with self._lock:
            self._total_requests += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_cost += cost

            self._increment_bucket(self._by_api_key[api_key], prompt_tokens, completion_tokens, cost)
            self._increment_bucket(self._by_model[model], prompt_tokens, completion_tokens, cost)
            self._persist_unlocked()
        return record

    def report(self) -> UsageReport:
        with self._lock:
            return UsageReport(
                total_requests=self._total_requests,
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_cost_usd=round(self._total_cost, 6),
                by_api_key={key: dict(value) for key, value in self._by_api_key.items()},
                by_model={key: dict(value) for key, value in self._by_model.items()},
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            (prompt_tokens / 1000.0) * self.cost_per_1k_prompt_tokens
            + (completion_tokens / 1000.0) * self.cost_per_1k_completion_tokens
        )

    @staticmethod
    def _empty_bucket() -> dict[str, Any]:
        return {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

    @staticmethod
    def _increment_bucket(
        bucket: dict[str, Any],
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        bucket["requests"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += prompt_tokens + completion_tokens
        bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 6)

    def _load(self) -> None:
        if not self.usage_file.exists():
            return
        try:
            payload = json.loads(self.usage_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self._total_requests = int(payload.get("total_requests", 0))
        self._total_prompt_tokens = int(payload.get("total_prompt_tokens", 0))
        self._total_completion_tokens = int(payload.get("total_completion_tokens", 0))
        self._total_cost = float(payload.get("total_cost_usd", 0.0))
        for key, bucket in payload.get("by_api_key", {}).items():
            self._by_api_key[key] = {**self._empty_bucket(), **bucket}
        for key, bucket in payload.get("by_model", {}).items():
            self._by_model[key] = {**self._empty_bucket(), **bucket}

    def _persist_unlocked(self) -> None:
        payload = {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "by_api_key": {key: dict(value) for key, value in self._by_api_key.items()},
            "by_model": {key: dict(value) for key, value in self._by_model.items()},
        }
        self.usage_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
