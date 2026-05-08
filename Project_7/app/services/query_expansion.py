"""Query expansion / hypothetical document embedding stub.

Returns 1 + N additional reformulations of the original query. The
deterministic offline expander adds keyword-only and synonym-style variants;
a production version would call an LLM to generate hypothetical answers
(HyDE) and embed those.
"""

from __future__ import annotations

import re

from app.services.text_utils import tokenize


_SYNONYMS: dict[str, list[str]] = {
    "build": ["create", "implement"],
    "use": ["leverage", "apply"],
    "fast": ["quick", "rapid"],
    "small": ["compact", "tiny"],
    "search": ["retrieve", "lookup"],
    "vector": ["embedding"],
    "document": ["doc", "passage"],
    "rerank": ["re-rank", "rerank"],
}


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    """Return the original query plus deterministic variants.

    The first variant strips stop-words to make a "keyword-only" form, which
    helps BM25; the second swaps known terms for synonyms; further variants
    are produced by trimming/duplicating informative tokens. Duplicates are
    removed and the original query is always at index 0.
    """
    if max_variants < 1:
        return [query]

    base = query.strip()
    variants: list[str] = [base]

    keyword_only = " ".join(tokenize(base, drop_stopwords=True))
    if keyword_only and keyword_only != base.lower():
        variants.append(keyword_only)

    synonym_form = _swap_synonyms(base)
    if synonym_form and synonym_form != base:
        variants.append(synonym_form)

    if len(variants) < max_variants:
        keyword_tokens = tokenize(base, drop_stopwords=True)
        if keyword_tokens:
            emphasised = " ".join(keyword_tokens + keyword_tokens[:1])
            if emphasised not in variants:
                variants.append(emphasised)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            deduplicated.append(variant)
        if len(deduplicated) >= max_variants:
            break
    return deduplicated


def _swap_synonyms(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        token = match.group(0)
        replacements = _SYNONYMS.get(token.lower())
        if not replacements:
            return token
        return replacements[0]

    return re.sub(r"[a-zA-Z]+", _replace, text)
