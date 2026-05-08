"""In-memory vector index keyed by image identifiers."""

from __future__ import annotations

from threading import Lock

from app.services.vision_provider import cosine_similarity


class VectorIndex:
    """Tiny in-memory cosine-similarity index.

    A real deployment would back this with ChromaDB or another vector
    database, but the API surface mirrors what those libraries expose so
    swapping backends later only touches this module.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._lock = Lock()

    def upsert(self, key: str, vector: list[float]) -> None:
        with self._lock:
            self._vectors[key] = list(vector)

    def remove(self, key: str) -> None:
        with self._lock:
            self._vectors.pop(key, None)

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            stored = self._vectors.get(key)
            return list(stored) if stored is not None else None

    def search(
        self,
        query: list[float],
        top_k: int,
        exclude_keys: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        excluded = exclude_keys or set()
        with self._lock:
            scored = [
                (key, cosine_similarity(query, vector))
                for key, vector in self._vectors.items()
                if key not in excluded
            ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._vectors)
