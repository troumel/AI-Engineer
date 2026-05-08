# Project 4: Named Entity Recognition (NER) API with Fine-tuning

**Difficulty:** ⭐⭐⭐ Medium  
**Duration:** 1-2 weeks  
**Tech Stack:** FastAPI, transformers, datasets, torch

## Overview

This project exposes a FastAPI service for named entity extraction and model lifecycle management. It includes:

- Named entity extraction endpoint
- Training job endpoint for custom annotated datasets
- Model version registry
- A/B rollout between versions
- Training job status tracking

The local default implementation uses a persisted pattern-based trainer so the project remains runnable and testable without GPU setup. The API shape mirrors what you would keep if you later swap the trainer to HuggingFace `Trainer` with a token-classification model.

## Features

- `POST /ner/extract` for entity extraction
- `POST /training/jobs` to create training jobs
- `GET /training/jobs/{job_id}` for job status
- `GET /models` to list model versions
- `POST /models/activate` to switch active versions
- `POST /experiments/rollout` to configure A/B rollout
- Persistent model registry under `models/`
- Health endpoint with active model and running job count

## Architecture

1. A baseline model version is created on first startup.
2. Extraction requests can use an explicit version, the active version, or a deterministic A/B rollout.
3. Training jobs accept annotated examples using character spans.
4. Completed jobs persist a new model version and can optionally auto-activate it.
5. Rollout settings split traffic between a primary and candidate version.

## Project Structure

```text
Project_4/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── experiments.py
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── ner.py
│   │   └── training.py
│   └── services/
│       ├── ner_service.py
│       └── pattern_ner.py
├── models/
├── docker/
├── tests/
├── requirements.txt
└── README.md
```

## Training Data Format

Training examples use character offsets instead of token indices:

```json
{
  "version_name": "custom-v1",
  "auto_activate": true,
  "run_async": true,
  "examples": [
    {
      "text": "Acme Robotics hired Maya Patel in Berlin.",
      "entities": [
        {"start": 0, "end": 13, "label": "ORG"},
        {"start": 20, "end": 30, "label": "PERSON"},
        {"start": 34, "end": 40, "label": "LOCATION"}
      ]
    }
  ]
}
```

## Setup

```powershell
cd "c:\Users\Theo\source\repos\AI Engineer\Project_4"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to customize ports or model storage paths.

## Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003
```

Swagger UI will be available at `http://localhost:8003/docs`.

## Notes on Fine-tuning

This implementation keeps the external API and model lifecycle that a HuggingFace fine-tuning workflow would need, but the default local trainer is intentionally lightweight and deterministic. That makes the project easy to run before you add:

- HuggingFace `Trainer`
- token-classification datasets
- GPU-backed training
- experiment tracking with MLflow or Weights & Biases

In other words, the training job flow, registry, rollout, and status tracking are implemented now, and the underlying trainer can be upgraded later without changing the API contract.

## Testing

```powershell
pytest tests -q
```

## .NET Parallels

- `app/models/schemas.py` corresponds to DTOs and request models.
- `app/services/ner_service.py` is the service layer and orchestration boundary.
- `app/routers/*.py` maps to ASP.NET controllers.
- `app/dependencies.py` is the singleton DI registration seam.