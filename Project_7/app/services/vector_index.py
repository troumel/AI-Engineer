"""In-memory dense vector index with cosine search."""

from __future__ import annotations

from threading import Lock

from app.services.embeddings import cosine_similarity


class VectorIndex:
    """Tiny thread-safe cosine-similarity index keyed by chunk id."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._lock = Lock()

    def upsert(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._vectors[key] = list(vector)

    def remove(self, key: str) -> None:
        with self._lock:
            self._vectors.pop(key, None)

    def search(
        self,
        query: list[float],
        candidate_keys: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            iterator = (
                (key, cosine_similarity(query, vector))
                for key, vector in self._vectors.items()
                if candidate_keys is None or key in candidate_keys
            )
            scored = [item for item in iterator if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def __len__(self) -> int:
        return len(self._vectors)
