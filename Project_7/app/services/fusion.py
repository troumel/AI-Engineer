"""Reciprocal Rank Fusion + cross-encoder-style reranker stubs.

`reciprocal_rank_fusion` follows the standard RRF formula:

    score(d) = Σ 1 / (k + rank_in_list(d))

across multiple ranked lists. This is the canonical fusion method used in
hybrid retrieval and gives strong results without tuning.

`HeuristicReranker` mimics a cross-encoder's interface — it takes a query
and a chunk text and returns a relevance score. The production version
would use `cross-encoder/ms-marco-MiniLM-L-6-v2`; the heuristic here uses
weighted token overlap and proximity so unit tests stay deterministic.
"""

from __future__ import annotations

from collections import defaultdict

from app.services.text_utils import tokenize


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into a single ranking using RRF."""
    if k <= 0:
        raise ValueError("k must be positive.")
    fused: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (key, _score) in enumerate(ranked, start=1):
            fused[key] += 1.0 / (k + rank)
    items = list(fused.items())
    items.sort(key=lambda item: item[1], reverse=True)
    return items


class HeuristicReranker:
    """Cross-encoder-style reranker stub.

    Returns a relevance score in [0, 1] based on token overlap and bigram
    co-occurrence between the query and the chunk content. This is enough
    for the unit tests to verify that re-ranking lifts strongly relevant
    chunks above weakly relevant ones.
    """

    name = "heuristic"

    def score(self, query: str, text: str) -> float:
        query_tokens = tokenize(query, drop_stopwords=True)
        text_tokens = tokenize(text, drop_stopwords=True)
        if not query_tokens or not text_tokens:
            return 0.0

        text_token_set = set(text_tokens)
        unigram_overlap = sum(1 for token in query_tokens if token in text_token_set)
        unigram = unigram_overlap / len(query_tokens)

        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        text_bigrams = set(zip(text_tokens, text_tokens[1:]))
        bigram = (
            len(query_bigrams & text_bigrams) / len(query_bigrams)
            if query_bigrams
            else 0.0
        )

        density = unigram_overlap / max(len(text_tokens), 1)
        score = 0.6 * unigram + 0.3 * bigram + 0.1 * density
        return min(1.0, score)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        """Score `[(key, text)]` pairs and return them sorted descending."""
        scored = [(key, self.score(query, text)) for key, text in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored
