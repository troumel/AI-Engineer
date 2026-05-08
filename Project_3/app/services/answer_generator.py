"""Answer generator providers for the RAG system."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Iterable

from app.services.vector_store import RetrievedChunk


class AnswerGenerator(ABC):
    """Abstract answer generator used by the RAG service."""

    provider_name: str = "unknown"

    @abstractmethod
    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a grounded answer from retrieved chunks."""


class ExtractiveAnswerGenerator(AnswerGenerator):
    """Fallback answer generator that stitches together relevant excerpts."""

    provider_name = "extractive-fallback"

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find relevant context in the uploaded documents."

        question_terms = {term.lower() for term in question.split() if len(term) > 2}
        ranked_sentences: list[tuple[int, str, RetrievedChunk]] = []

        for chunk in chunks:
            sentences = [sentence.strip() for sentence in chunk.content.split(".") if sentence.strip()]
            for sentence in sentences:
                score = sum(1 for term in question_terms if term in sentence.lower())
                ranked_sentences.append((score, sentence, chunk))

        ranked_sentences.sort(key=lambda item: (item[0], item[2].score), reverse=True)
        selected = ranked_sentences[:2] if ranked_sentences else []

        if not selected:
            top_chunk = chunks[0]
            return f"Based on {top_chunk.source_name}, {top_chunk.content[:240].strip()}"

        fragments = [
            f"{sentence} [{chunk.source_name}#{chunk.chunk_index}]"
            for _, sentence, chunk in selected
        ]
        return " ".join(fragments)


class OpenAIAnswerGenerator(AnswerGenerator):
    """OpenAI-backed answer generator for grounded responses."""

    provider_name = "openai"

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        openai_module = importlib.import_module("openai")
        client_type = getattr(openai_module, "OpenAI")
        self.client = client_type(api_key=api_key)

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find relevant context in the uploaded documents."

        context = "\n\n".join(
            f"[{chunk.source_name}#{chunk.chunk_index}] {chunk.content}"
            for chunk in chunks
        )
        prompt = (
            "Answer based on the following context. If the answer is not supported by the "
            "context, say that clearly. Cite sources inline using the provided labels.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No answer was returned."


def create_answer_generator(api_key: str | None, model_name: str) -> AnswerGenerator:
    """Create the configured answer generator, falling back when needed."""
    if api_key:
        try:
            return OpenAIAnswerGenerator(api_key=api_key, model_name=model_name)
        except Exception:
            return ExtractiveAnswerGenerator()

    return ExtractiveAnswerGenerator()