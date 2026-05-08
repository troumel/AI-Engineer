# Project 2: Text Classification API (Sentiment Analysis)

**Difficulty:** ⭐⭐ Easy-Medium  
**Duration:** 5-7 days  
**Tech Stack:** FastAPI, transformers, Redis, Docker

## Overview

This project exposes a FastAPI service that classifies text sentiment using a HuggingFace transformer model. It adds practical API concerns on top of model inference: request validation, batch processing, response caching, rate limiting, and request logging.

The structure mirrors Project 1 so the move from classical ML inference to transformer-backed NLP stays incremental.

## Features

- Single-text sentiment endpoint
- Batch sentiment endpoint
- HuggingFace transformer integration with `distilbert-base-uncased-finetuned-sst-2-english`
- Redis-backed response caching with in-memory fallback for local development
- Per-client rate limiting
- Request logging middleware
- Health endpoint showing model and cache status
- Docker and docker-compose setup
- Unit and integration tests

## Project Structure

```text
Project_2/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── health.py
│   │   └── predictions.py
│   └── services/
│       ├── cache_service.py
│       ├── rate_limiter.py
│       └── sentiment_service.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_api_endpoints.py
│   └── test_sentiment_service.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Create or activate a virtual environment

```powershell
cd "c:\Users\Theo\source\repos\AI Engineer\Project_2"
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and adjust values if needed.

Key settings:

- `SENTIMENT_MODEL_NAME`: HuggingFace model name
- `REDIS_URL`: Redis connection string
- `NEUTRAL_THRESHOLD`: Scores below this threshold are normalized to `neutral`
- `RATE_LIMIT_REQUESTS`: Requests allowed per client within the configured window

## Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Open the Swagger UI at `http://localhost:8001/docs`.

## API Endpoints

### `POST /sentiment`

```json
{
  "text": "I love how simple this API is."
}
```

### `POST /sentiment/batch`

```json
{
  "texts": [
    "This response was excellent.",
    "The experience was terrible."
  ]
}
```

### `GET /health`

Returns service health, model availability, cache backend, and configured model name.

## Notes on Sentiment Labels

The underlying SST-2 model is binary (`positive` or `negative`). This project adds a `neutral` API label by mapping low-confidence predictions below `NEUTRAL_THRESHOLD` to `neutral`.

## Testing

The tests inject a fake classifier so they do not need to download the transformer model.

```powershell
pytest tests -v
```

## Docker

Run the API and Redis together:

```powershell
cd docker
docker compose up --build
```

## .NET Parallels

- `app/models/schemas.py` is similar to DTOs with validation attributes
- `app/services/sentiment_service.py` is the business logic layer
- `app/routers/predictions.py` plays the same role as an ASP.NET controller
- `app/dependencies.py` is the DI registration seam