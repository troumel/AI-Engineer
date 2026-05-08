"""Continuous-batching scheduler.

Real production servers (vLLM, TGI) implement *continuous* batching with
PagedAttention so new requests can join an in-flight batch on every decoding
step. This module captures the public contract of that scheduler — submit
requests one at a time, the scheduler bundles them up to `max_batch_size`
or `batch_timeout_ms`, and returns each caller's result — without requiring
a GPU. The same surface area can later be wired to vLLM's `AsyncLLMEngine`.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional

from app.services.llm_engine import GenerationRequest, GenerationResult, LLMEngine
from app.services.metrics import Metrics


@dataclass
class _PendingItem:
    request: GenerationRequest
    future: "asyncio.Future[GenerationResult]"


class ContinuousBatcher:
    """Async scheduler bundling requests into batches for the LLM engine."""

    def __init__(
        self,
        engine: LLMEngine,
        max_batch_size: int,
        max_queue_depth: int,
        batch_timeout_ms: int,
        metrics: Metrics,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_queue_depth = max_queue_depth
        self.batch_timeout = max(batch_timeout_ms, 0) / 1000.0
        self.metrics = metrics

        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue(maxsize=max_queue_depth)
        self._worker: Optional[asyncio.Task[None]] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = asyncio.create_task(self._run(), name="continuous-batcher")

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if not self._running:
            # Fall back to direct synchronous execution (handy in tests).
            return self.engine.generate(request)
        if self._queue.full():
            raise RuntimeError("scheduler queue is full")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()
        await self._queue.put(_PendingItem(request=request, future=future))
        self.metrics.set_queue_depth(self._queue.qsize())
        return await future

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def _run(self) -> None:
        try:
            while self._running:
                first = await self._queue.get()
                batch: list[_PendingItem] = [first]

                # Drain extra requests up to max_batch_size or batch_timeout.
                deadline = asyncio.get_running_loop().time() + self.batch_timeout
                while len(batch) < self.max_batch_size:
                    timeout = max(deadline - asyncio.get_running_loop().time(), 0.0)
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    batch.append(item)

                self.metrics.set_queue_depth(self._queue.qsize())
                self.metrics.record_batch(len(batch))
                results = self.engine.generate_batch([item.request for item in batch])
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)
        except asyncio.CancelledError:
            # Drain any waiting futures so nothing hangs forever.
            while not self._queue.empty():
                pending = self._queue.get_nowait()
                if not pending.future.done():
                    pending.future.cancel()
            raise
