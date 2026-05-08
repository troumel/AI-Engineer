# Project 9: PyTorch Fine-Tuning and Training Pipeline

**Difficulty:** ⭐⭐⭐ Medium  
**Duration:** 7-10 days  
**Tech Stack:** FastAPI, PyTorch, transformers, pytest

## Overview

This project introduces the missing step between using pretrained models and operating production AI systems: building a repeatable training pipeline that produces a deployable model artifact.

The scaffold in this repository is intentionally offline-friendly. The inference path loads a saved artifact and serves predictions behind FastAPI. The training scripts create a versioned artifact folder and reserve the upgrade path for real PyTorch fine-tuning.

## Features

- Versioned training artifact layout under `models/`
- Offline dataset preparation script
- Training script that writes labels, metrics, config, and tokenizer placeholders
- Evaluation script for saved artifacts
- FastAPI inference API with `/predict` and `/health`
- Degraded startup behavior when artifacts are missing
- Offline tests for training pipeline and API wiring

## Project Structure

```text
Project_9/
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
│       └── inference_service.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   └── evaluate_model.py
├── tests/
│   ├── test_api_endpoints.py
│   ├── test_inference_service.py
│   └── test_training_pipeline.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Create or activate a virtual environment

```powershell
cd "c:\Users\Theo\source\repos\AI Engineer\Project_9"
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Prepare demo data and train an artifact

```powershell
python scripts/prepare_dataset.py
python scripts/train_model.py
```

### 4. Run the API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8008
```

Open the Swagger UI at `http://localhost:8008/docs`.

## Notes

The current scaffold uses a deterministic keyword backend so the project stays runnable offline and testable in CI. The files and interfaces are shaped so you can replace the training internals with a real PyTorch fine-tuning loop later without changing the API contract.

## Testing

```powershell
pytest tests -v
```

## Suggested next upgrade

Replace the keyword trainer in `scripts/train_model.py` with a real HuggingFace + PyTorch fine-tuning loop that writes the same artifact shape.
