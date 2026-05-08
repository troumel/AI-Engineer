"""Tokenization, text-chunking, and shared text helpers."""

from __future__ import annotations

import re


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
        "to", "was", "were", "will", "with", "this", "these", "those",
        "but", "not", "no", "do", "does", "did", "you", "your", "i", "we",
        "they", "their", "there", "what", "when", "where", "why", "how",
        "which", "who", "whom", "whose", "can", "could", "should", "would",
        "may", "might", "must", "shall", "if", "then", "else", "than",
        "so", "such", "about", "into", "over", "under", "between", "while",
    }
)


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercased alphanumeric tokens."""
    tokens = [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text or "")]
    if drop_stopwords:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks measured in tokens.

    Production code typically chunks by tokens of a real tokenizer; we use
    whitespace-delimited words here so the project stays dependency-light.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size).")

    words = (text or "").split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
