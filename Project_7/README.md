# Project 7 — Vector Database Advanced RAG System

A FastAPI service that demonstrates a production-style **advanced RAG pipeline**: hybrid retrieval (dense + BM25), reciprocal rank fusion, cross-encoder-style re-ranking, query expansion, metadata filtering, streaming answers, and offline RAG-quality evaluation.

The service ships with **fully offline defaults** — every component (embeddings, BM25, reranker, evaluator, generator) has a deterministic in-process implementation so `pytest` works with no external services. Production upgrade slots are marked in the source.

## Features

- Document ingest with chunking, metadata, and JSON persistence (`data/corpus.json`).
- **Hybrid retrieval**: BM25 + dense vector cosine search fused with **Reciprocal Rank Fusion**.
- Optional **query expansion** (keyword-only and synonym variants — drop-in slot for HyDE/LLM expansion).
- Optional **re-ranking** stage (heuristic offline cross-encoder; replaceable with `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Metadata filters** (author, category, tag, date range) applied before retrieval.
- **Ask** endpoint with grounded answer + source chunks; **streaming** variant emits Server-Sent Events.
- **Offline RAG evaluation** computing answer relevance, faithfulness, and context precision.

## Project layout

```
Project_7/
  app/
    main.py                FastAPI entry point (port 8006)
    config.py              pydantic-settings config
    dependencies.py        Singleton DI for the RAG service
    models/schemas.py      Pydantic request/response DTOs
    routers/               health, documents, search, ask, evaluate
    services/
      text_utils.py        Tokenizer + chunker
      embeddings.py        Hashing-based offline embedder
      bm25.py              Pure-Python BM25 index
      vector_index.py      In-memory cosine index
      fusion.py            RRF + heuristic reranker
      query_expansion.py   Offline query-expansion variants
      corpus_store.py      JSON-persisted document/chunk store
      evaluation.py        Offline RAG metrics
      rag_service.py       High-level orchestrator
  tests/                   Unit + API tests
  docker/                  Dockerfile and docker-compose.yml
```

## Running locally

```powershell
# from repo root, reusing the workspace virtualenv
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m pip install -r Project_7/requirements.txt

cd Project_7
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m uvicorn app.main:app --reload --port 8006
```

Then visit http://127.0.0.1:8006/docs.

## Tests

```powershell
cd Project_7
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m pytest tests -q
```

## API

| Method | Path             | Purpose                                       |
|--------|------------------|-----------------------------------------------|
| GET    | `/health`        | Service status, document/chunk counts         |
| POST   | `/documents`     | Ingest a document (auto-chunked)              |
| GET    | `/documents`     | List ingested documents                       |
| DELETE | `/documents/{id}`| Remove a document and its chunks              |
| POST   | `/search`        | Hybrid retrieval (vector + BM25 + RRF + rerank) |
| POST   | `/ask`           | Grounded answer + sources                     |
| POST   | `/ask/stream`    | SSE stream of sources, then token events      |
| POST   | `/evaluate`      | Offline answer-relevance / faithfulness / context-precision |

## Production upgrade slots

| Concern | Offline default | Production swap |
|---------|------------------|-----------------|
| Embeddings | `HashingEmbeddingProvider` | `sentence-transformers` or OpenAI `text-embedding-3-small` |
| Vector store | In-memory `VectorIndex` | Qdrant / pgvector / Pinecone |
| BM25 | Pure-Python `BM25Index` | `rank-bm25`, Elasticsearch, OpenSearch |
| Reranker | `HeuristicReranker` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generator | `rule_based` rewriter | OpenAI Chat Completions / vLLM |
| Evaluation | Token-based metrics | `ragas`, TruLens |

Each component is constructed via `AdvancedRagService` so swapping it is a one-line change.
