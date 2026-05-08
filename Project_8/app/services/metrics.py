"""Lightweight in-process metrics with Prometheus text exposition."""

from __future__ import annotations

import time
from threading import Lock


class Metrics:
    """Counters / histograms exposed in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total = 0
        self._errors_total = 0
        self._tokens_in_total = 0
        self._tokens_out_total = 0
        self._latency_sum_seconds = 0.0
        self._latency_count = 0
        self._batches_total = 0
        self._batched_requests_total = 0
        self._queue_depth = 0
        self._max_queue_depth = 0
        self._started_at = time.time()

    def inc_request(self, prompt_tokens: int, completion_tokens: int, latency: float) -> None:
        with self._lock:
            self._requests_total += 1
            self._tokens_in_total += prompt_tokens
            self._tokens_out_total += completion_tokens
            self._latency_sum_seconds += latency
            self._latency_count += 1

    def inc_error(self) -> None:
        with self._lock:
            self._errors_total += 1

    def record_batch(self, batch_size: int) -> None:
        with self._lock:
            self._batches_total += 1
            self._batched_requests_total += batch_size

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth
            self._max_queue_depth = max(self._max_queue_depth, depth)

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._queue_depth

    def render_prometheus(self) -> str:
        with self._lock:
            uptime = time.time() - self._started_at
            avg_latency = (
                self._latency_sum_seconds / self._latency_count
                if self._latency_count
                else 0.0
            )
            avg_batch = (
                self._batched_requests_total / self._batches_total
                if self._batches_total
                else 0.0
            )
            lines = [
                "# HELP llm_requests_total Total inference requests served.",
                "# TYPE llm_requests_total counter",
                f"llm_requests_total {self._requests_total}",
                "# HELP llm_errors_total Total inference errors.",
                "# TYPE llm_errors_total counter",
                f"llm_errors_total {self._errors_total}",
                "# HELP llm_tokens_in_total Total prompt tokens processed.",
                "# TYPE llm_tokens_in_total counter",
                f"llm_tokens_in_total {self._tokens_in_total}",
                "# HELP llm_tokens_out_total Total completion tokens generated.",
                "# TYPE llm_tokens_out_total counter",
                f"llm_tokens_out_total {self._tokens_out_total}",
                "# HELP llm_request_latency_seconds_avg Average request latency.",
                "# TYPE llm_request_latency_seconds_avg gauge",
                f"llm_request_latency_seconds_avg {avg_latency:.6f}",
                "# HELP llm_batch_size_avg Average batch size.",
                "# TYPE llm_batch_size_avg gauge",
                f"llm_batch_size_avg {avg_batch:.4f}",
                "# HELP llm_queue_depth Current scheduler queue depth.",
                "# TYPE llm_queue_depth gauge",
                f"llm_queue_depth {self._queue_depth}",
                "# HELP llm_queue_depth_max Maximum observed scheduler queue depth.",
                "# TYPE llm_queue_depth_max gauge",
                f"llm_queue_depth_max {self._max_queue_depth}",
                "# HELP llm_uptime_seconds Process uptime in seconds.",
                "# TYPE llm_uptime_seconds gauge",
                f"llm_uptime_seconds {uptime:.2f}",
            ]
        return "\n".join(lines) + "\n"
