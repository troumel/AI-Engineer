"""Offline RAG evaluation metrics.

Approximates the three core RAG quality signals so the evaluation surface
can run without `ragas` or any external LLM:

* answer_relevance: how much the generated answer overlaps with the expected
  answer (token-level F1, stop-words removed).
* faithfulness: the share of answer tokens that are actually grounded in
  the retrieved chunks.
* context_precision: how many of the top-k retrieved chunks are relevant
  to the question — using either an explicit `relevant_document_id` or
  expected-answer overlap as the relevance signal.
"""

from __future__ import annotations

from app.services.text_utils import tokenize


def answer_relevance(answer: str, expected_answer: str) -> float:
    answer_tokens = set(tokenize(answer, drop_stopwords=True))
    expected_tokens = set(tokenize(expected_answer, drop_stopwords=True))
    if not answer_tokens or not expected_tokens:
        return 0.0

    overlap = answer_tokens & expected_tokens
    precision = len(overlap) / len(answer_tokens)
    recall = len(overlap) / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, retrieved_texts: list[str]) -> float:
    answer_tokens = [
        token for token in tokenize(answer, drop_stopwords=True) if len(token) > 2
    ]
    if not answer_tokens:
        return 0.0

    context_tokens = set()
    for text in retrieved_texts:
        context_tokens.update(tokenize(text, drop_stopwords=True))
    if not context_tokens:
        return 0.0
    grounded = sum(1 for token in answer_tokens if token in context_tokens)
    return grounded / len(answer_tokens)


def context_precision(
    expected_answer: str,
    retrieved_chunks: list[dict],
    relevant_document_id: str | None,
) -> float:
    if not retrieved_chunks:
        return 0.0

    if relevant_document_id is not None:
        hits = sum(
            1
            for chunk in retrieved_chunks
            if chunk.get("document_id") == relevant_document_id
        )
        return hits / len(retrieved_chunks)

    expected_tokens = set(tokenize(expected_answer, drop_stopwords=True))
    if not expected_tokens:
        return 0.0
    hits = 0
    for chunk in retrieved_chunks:
        chunk_tokens = set(tokenize(chunk.get("content", ""), drop_stopwords=True))
        if expected_tokens & chunk_tokens:
            hits += 1
    return hits / len(retrieved_chunks)
