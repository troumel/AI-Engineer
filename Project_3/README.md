# Project 3: Basic RAG (Retrieval Augmented Generation) System

**Difficulty:** ⭐⭐ Medium  
**Duration:** 1-2 weeks  
**Tech Stack:** FastAPI, ChromaDB, OpenAI-compatible API, PyPDF2

## Overview

This project implements a document question-answering API. You upload PDF or TXT files, the app splits them into chunks, generates embeddings, stores them in a vector store, retrieves the most relevant chunks for a question, and returns a grounded answer with source citations.

The structure stays aligned with Projects 1 and 2 so the main new concepts are the retrieval pipeline and provider abstractions rather than a new app shape.

## Features

- Upload `.txt` and `.pdf` documents
- Chunk documents using a recursive splitter with a manual fallback
- Embed chunks with OpenAI when configured, or a deterministic local fallback for development
- Store vectors in ChromaDB when available, or an in-memory fallback during local testing
- Ask grounded questions over indexed documents
- Return source citations for the retrieved chunks
- Health endpoint with provider and indexing status
- Manifest file for indexed document metadata
- Docker setup and test suite

## Architecture

1. `POST /documents/upload` saves the raw document and extracts text.
2. The text is chunked with `chunk_size=1000` and `chunk_overlap=200` by default.
3. Each chunk is embedded and stored in the vector store.
4. `POST /qa/ask` embeds the question, retrieves the top `k` chunks, and generates an answer.
5. The API returns the answer plus citations that identify the source document and chunk.

## Project Structure

```text
Project_3/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── documents.py
│   │   ├── health.py
│   │   └── qa.py
│   └── services/
│       ├── answer_generator.py
│       ├── document_processor.py
│       ├── embedding_provider.py
│       ├── rag_service.py
│       └── vector_store.py
├── data/
│   ├── chroma/
│   └── uploads/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_api_endpoints.py
│   └── test_rag_service.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```powershell
cd "c:\Users\Theo\source\repos\AI Engineer\Project_3"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure `OPENAI_API_KEY` if you want to use live OpenAI embeddings and answer generation. If no key is present, the app still runs using deterministic local fallback providers for development and tests.

## Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

Swagger UI will be available at `http://localhost:8002/docs`.

## Example Requests

### Upload a document

```powershell
curl -X POST "http://localhost:8002/documents/upload" \
  -F "file=@sample.txt"
```

### Ask a question

```json
{
  "question": "What does the document say about vector databases?",
  "top_k": 3
}
```

## Notes on Providers

- Embeddings: OpenAI `text-embedding-3-small` when `OPENAI_API_KEY` is set, otherwise a deterministic local hash-based embedding fallback.
- Answer generation: OpenAI chat completions when configured, otherwise an extractive answer fallback built from the retrieved chunks.
- Vector store: ChromaDB when installed and configured, otherwise an in-memory vector store.

That fallback path exists so the project remains runnable and testable before API keys are configured.

## Testing

The tests inject fake embedding and answer providers, so they do not call external APIs.

```powershell
pytest tests -q
```

## .NET Parallels

- `app/models/schemas.py` is equivalent to DTOs with validation attributes.
- `app/services/rag_service.py` is the orchestration service layer.
- `app/routers/*.py` maps closely to ASP.NET controllers.
- `app/dependencies.py` plays the same role as singleton registration in `Program.cs`.