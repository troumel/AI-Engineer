"""Pure-Python BM25 keyword-search index.

Implements the Okapi BM25 ranking function over a tokenised corpus. The
production stack would use `rank-bm25`, Elasticsearch, or OpenSearch — this
implementation keeps the project dependency-light and deterministic for
unit tests.
"""

from __future__ import annotations

import math
from collections import Counter
from threading import Lock

from app.services.text_utils import tokenize


class BM25Index:
    """Stateful BM25 index keyed by chunk id."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = Lock()
        self._docs: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()

    def upsert(self, key: str, text: str) -> None:
        tokens = tokenize(text, drop_stopwords=True)
        with self._lock:
            if key in self._docs:
                self._remove_unlocked(key)
            counts = Counter(tokens)
            self._docs[key] = counts
            self._doc_lengths[key] = len(tokens)
            for term in counts:
                self._df[term] += 1

    def remove(self, key: str) -> None:
        with self._lock:
            self._remove_unlocked(key)

    def search(
        self,
        query: str,
        candidate_keys: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_tokens = tokenize(query, drop_stopwords=True)
        if not query_tokens or not self._docs:
            return []

        with self._lock:
            doc_count = len(self._docs)
            avg_length = (
                sum(self._doc_lengths.values()) / doc_count if doc_count else 0.0
            )
            df_snapshot = dict(self._df)
            docs_snapshot = self._docs
            lengths_snapshot = self._doc_lengths

            scores: list[tuple[str, float]] = []
            keys = candidate_keys or set(docs_snapshot.keys())
            for key in keys:
                counts = docs_snapshot.get(key)
                if counts is None:
                    continue
                length = lengths_snapshot[key]
                score = 0.0
                for term in query_tokens:
                    term_freq = counts.get(term)
                    if not term_freq:
                        continue
                    df = df_snapshot.get(term, 0)
                    if df == 0:
                        continue
                    idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                    denom = term_freq + self.k1 * (
                        1 - self.b + self.b * length / (avg_length or 1)
                    )
                    score += idf * (term_freq * (self.k1 + 1)) / denom
                if score > 0:
                    scores.append((key, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def __len__(self) -> int:
        return len(self._docs)

    def _remove_unlocked(self, key: str) -> None:
        counts = self._docs.pop(key, None)
        self._doc_lengths.pop(key, None)
        if counts is None:
            return
        for term in counts:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
