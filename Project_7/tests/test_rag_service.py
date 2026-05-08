"""Unit tests for the RAG service and its building blocks."""

from __future__ import annotations

from app.models.schemas import (
    EvaluationExample,
    EvaluationRequest,
    IngestRequest,
    DocumentMetadata,
    MetadataFilter,
    SearchMode,
    SearchRequest,
)
from app.services.bm25 import BM25Index
from app.services.embeddings import HashingEmbeddingProvider, cosine_similarity
from app.services.fusion import HeuristicReranker, reciprocal_rank_fusion
from app.services.query_expansion import expand_query
from app.services.rag_service import AdvancedRagService
from app.services.text_utils import chunk_text, tokenize


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_tokenize_drops_punctuation_and_stopwords():
    tokens = tokenize("The quick, brown fox!", drop_stopwords=True)
    assert "the" not in tokens
    assert tokens == ["quick", "brown", "fox"]


def test_chunk_text_overlaps_correctly():
    text = " ".join([f"word{i}" for i in range(30)])
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) >= 3
    assert chunks[0].startswith("word0")
    # second chunk starts at word(10-2) = word8 because step = 8
    assert chunks[1].startswith("word8")


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbeddingProvider(dimensions=128)
    a = embedder.embed("vector databases for RAG")
    b = embedder.embed("vector databases for RAG")
    assert a == b
    norm_sq = sum(x * x for x in a)
    assert abs(norm_sq - 1.0) < 1e-6
    assert cosine_similarity(a, embedder.embed("vector databases")) > 0.4


def test_bm25_ranks_more_relevant_higher():
    index = BM25Index(k1=1.5, b=0.75)
    index.upsert("a", "Vector databases store embeddings for similarity search.")
    index.upsert("b", "Cooking recipes for pasta and pizza.")
    index.upsert("c", "Hybrid search combines BM25 and vector similarity.")
    results = index.search("vector similarity")
    assert results, "BM25 returned no results"
    assert results[0][0] in {"a", "c"}
    assert "b" not in {key for key, _ in results}


def test_reciprocal_rank_fusion_combines_rankings():
    list_a = [("x", 1.0), ("y", 0.5), ("z", 0.2)]
    list_b = [("y", 1.0), ("x", 0.4), ("w", 0.1)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    fused_keys = [key for key, _ in fused]
    # x and y should beat the singletons z and w
    assert fused_keys[0] in {"x", "y"}
    assert fused_keys[1] in {"x", "y"}
    assert "z" in fused_keys and "w" in fused_keys


def test_query_expansion_returns_unique_variants():
    variants = expand_query("how do I build a fast vector search?", max_variants=3)
    assert variants[0] == "how do I build a fast vector search?"
    assert len(variants) == len(set(variants))
    assert any("build" not in variant.lower() or "create" in variant.lower() for variant in variants)


def test_heuristic_reranker_prefers_overlapping_chunks():
    reranker = HeuristicReranker()
    high = reranker.score("vector similarity search", "Vector similarity search uses cosine.")
    low = reranker.score("vector similarity search", "Pizza recipe with mozzarella.")
    assert high > low


# ---------------------------------------------------------------------------
# RAG service end-to-end
# ---------------------------------------------------------------------------


def _ingest_corpus(service: AdvancedRagService) -> None:
    service.ingest(
        IngestRequest(
            document_id="vectors",
            title="Vector Databases",
            content=(
                "Vector databases store dense embeddings and serve similarity "
                "search using cosine distance. They power retrieval augmented "
                "generation systems by indexing chunked documents."
            ),
            metadata=DocumentMetadata(
                author="alice", category="ml", tags=["rag", "vectors"], date="2024-01-01"
            ),
        )
    )
    service.ingest(
        IngestRequest(
            document_id="cooking",
            title="Pasta Recipe",
            content=(
                "Boil water with salt. Add fresh pasta and cook until al dente. "
                "Toss with olive oil, garlic, and basil."
            ),
            metadata=DocumentMetadata(
                author="bob", category="cooking", tags=["food"], date="2023-06-01"
            ),
        )
    )
    service.ingest(
        IngestRequest(
            document_id="hybrid",
            title="Hybrid Search",
            content=(
                "Hybrid search combines BM25 keyword scores with dense vector "
                "similarity, fusing the rankings using Reciprocal Rank Fusion. "
                "A reranker then promotes the most relevant passages."
            ),
            metadata=DocumentMetadata(
                author="alice", category="ml", tags=["rag", "search"], date="2024-05-01"
            ),
        )
    )


def test_search_returns_relevant_results(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    response = rag_service.search(SearchRequest(query="vector similarity search", top_k=3))
    assert response.count > 0
    top_doc_ids = [chunk.document_id for chunk in response.results]
    # Cooking recipe must not be the top hit
    assert top_doc_ids[0] in {"vectors", "hybrid"}


def test_metadata_filter_excludes_other_documents(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    response = rag_service.search(
        SearchRequest(
            query="search",
            top_k=5,
            filters=MetadataFilter(category="cooking"),
        )
    )
    for chunk in response.results:
        assert chunk.document_id == "cooking"


def test_query_expansion_flag_returns_variants(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    response = rag_service.search(
        SearchRequest(
            query="build a vector search",
            top_k=3,
            mode=SearchMode(use_query_expansion=True),
        )
    )
    assert len(response.expanded_queries) >= 2


def test_ask_grounds_answer_in_corpus(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    response = rag_service.ask(SearchRequest(query="What is hybrid search?", top_k=3))
    assert response.sources, "No sources returned"
    assert response.answer
    assert "hybrid" in response.answer.lower() or "vector" in response.answer.lower()


def test_stream_ask_emits_sources_and_tokens(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    events = list(rag_service.stream_ask(SearchRequest(query="vector similarity", top_k=2)))
    assert any(event.startswith("event: sources") for event in events)
    assert any(event.startswith("event: token") for event in events)
    assert events[-1].startswith("event: done")


def test_evaluate_returns_metric_aggregates(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    response = rag_service.evaluate(
        EvaluationRequest(
            top_k=3,
            examples=[
                EvaluationExample(
                    question="What is hybrid search?",
                    expected_answer="Hybrid search combines BM25 keyword scoring with vector similarity using RRF.",
                    relevant_document_id="hybrid",
                ),
                EvaluationExample(
                    question="How do vector databases work?",
                    expected_answer="They store dense embeddings and serve similarity search using cosine distance.",
                    relevant_document_id="vectors",
                ),
            ],
        )
    )
    assert response.count == 2
    assert 0.0 <= response.average_answer_relevance <= 1.0
    assert 0.0 <= response.average_faithfulness <= 1.0
    assert 0.0 <= response.average_context_precision <= 1.0
    assert response.average_context_precision > 0.0


def test_persistence_across_service_restart(tmp_path):
    storage_file = str(tmp_path / "corpus.json")
    args = dict(
        embedding_dimensions=128,
        chunk_size=80,
        chunk_overlap=10,
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_k=60,
        reranker_enabled=True,
        answer_backend="rule_based",
    )
    first = AdvancedRagService(storage_file=storage_file, **args)
    first.ingest(
        IngestRequest(
            document_id="persist",
            title="Persisted",
            content="Reciprocal rank fusion fuses keyword and dense rankings.",
            metadata=DocumentMetadata(category="ml"),
        )
    )

    second = AdvancedRagService(storage_file=storage_file, **args)
    docs = second.list_documents()
    assert docs.count == 1
    assert docs.documents[0].document_id == "persist"
    response = second.search(SearchRequest(query="rank fusion", top_k=3))
    assert response.count > 0
    assert response.results[0].document_id == "persist"


def test_remove_document_clears_indexes(rag_service: AdvancedRagService):
    _ingest_corpus(rag_service)
    rag_service.remove_document("cooking")
    response = rag_service.search(
        SearchRequest(query="pasta", top_k=5, mode=SearchMode(use_reranker=False))
    )
    for chunk in response.results:
        assert chunk.document_id != "cooking"
