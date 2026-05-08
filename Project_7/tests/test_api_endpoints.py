"""End-to-end API tests for Project 7."""

from __future__ import annotations


def _ingest(client, **overrides):
    payload = {
        "document_id": overrides.get("document_id", "doc-1"),
        "title": overrides.get("title", "Vector Databases"),
        "content": overrides.get(
            "content",
            "Vector databases index dense embeddings for similarity search. "
            "They power retrieval augmented generation by serving the most "
            "similar chunks for a given query.",
        ),
        "metadata": overrides.get(
            "metadata",
            {
                "author": "alice",
                "category": "ml",
                "tags": ["rag"],
                "date": "2024-01-01",
                "extra": {},
            },
        ),
    }
    response = client.post("/documents", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_endpoint_reports_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["embedding_dimensions"] == 128


def test_ingest_list_and_delete_document(client):
    body = _ingest(client)
    assert body["document_id"] == "doc-1"
    assert body["chunk_count"] >= 1

    listed = client.get("/documents").json()
    assert listed["count"] == 1
    assert listed["documents"][0]["title"] == "Vector Databases"

    deleted = client.delete("/documents/doc-1").json()
    assert deleted["document_id"] == "doc-1"
    assert deleted["removed_chunks"] >= 1
    assert client.get("/documents").json()["count"] == 0


def test_delete_missing_document_returns_404(client):
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404


def test_search_endpoint_returns_relevant_chunks(client):
    _ingest(client, document_id="vectors")
    _ingest(
        client,
        document_id="cooking",
        title="Pasta Recipe",
        content="Boil pasta and toss with olive oil, garlic, and basil.",
        metadata={
            "author": "bob",
            "category": "cooking",
            "tags": ["food"],
            "date": "2023-01-01",
            "extra": {},
        },
    )

    response = client.post(
        "/search",
        json={"query": "vector similarity", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert body["results"][0]["document_id"] == "vectors"


def test_search_with_filters_and_expansion(client):
    _ingest(client, document_id="vectors")
    _ingest(
        client,
        document_id="cooking",
        title="Pasta",
        content="Boil pasta and toss with olive oil.",
        metadata={
            "author": "bob",
            "category": "cooking",
            "tags": ["food"],
            "date": "2023-01-01",
            "extra": {},
        },
    )

    response = client.post(
        "/search",
        json={
            "query": "build a vector search",
            "top_k": 3,
            "mode": {
                "use_vector": True,
                "use_bm25": True,
                "use_query_expansion": True,
                "use_reranker": True,
            },
            "filters": {"category": "ml"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["expanded_queries"]) >= 2
    for chunk in body["results"]:
        assert chunk["document_id"] == "vectors"


def test_ask_endpoint_returns_grounded_answer(client):
    _ingest(client, document_id="vectors")
    response = client.post(
        "/ask",
        json={"query": "What do vector databases do?", "top_k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]


def test_ask_stream_emits_sse_events(client):
    _ingest(client, document_id="vectors")
    with client.stream(
        "POST",
        "/ask/stream",
        json={"query": "vector similarity search", "top_k": 2},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "event: sources" in body
    assert "event: token" in body
    assert "event: done" in body


def test_evaluate_endpoint_returns_aggregates(client):
    _ingest(client, document_id="vectors")
    response = client.post(
        "/evaluate",
        json={
            "top_k": 3,
            "examples": [
                {
                    "question": "What do vector databases store?",
                    "expected_answer": "Dense embeddings for similarity search.",
                    "relevant_document_id": "vectors",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert 0.0 <= body["average_context_precision"] <= 1.0
    assert body["items"][0]["retrieved_document_ids"]


def test_validation_rejects_empty_query(client):
    response = client.post("/search", json={"query": "", "top_k": 1})
    assert response.status_code == 422
