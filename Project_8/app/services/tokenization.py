"""Token counting and lightweight tokenization utilities.

Real LLM serving uses the model's own tokenizer (tiktoken, sentencepiece,
HuggingFace `AutoTokenizer`). The offline default here approximates token
counts by treating each whitespace-separated word + every 4 punctuation
characters as one token. The approximation is deterministic, fast, and good
enough for usage tracking in tests.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+")
_PUNCT = re.compile(r"[^\w\s]")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    # ~1 token per word + 1 token per 4 punctuation chars, min 1
    tokens = words + (punct + 3) // 4
    return max(tokens, 1)


def tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text or "")
