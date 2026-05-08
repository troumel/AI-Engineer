# Project 5 — Multi-Modal AI Application (Image + Text)

A FastAPI service that processes both images and text. It supports image
upload, captioning, visual question answering (VQA), and cross-modal
search (text → images and image → images).

## Highlights

- **Image upload** with persistent metadata stored alongside files
- **Caption generation** with optional natural-language prompts
- **Visual Question Answering** endpoint
- **Cross-modal search**: find images by text or by another image
- **Offline-first**: a deterministic hashing vision provider keeps the
  service runnable and testable without downloading BLIP / CLIP weights
- **Pluggable**: a HuggingFace provider slot is reserved (BLIP for
  captioning, CLIP for image/text embeddings) for production upgrades

## Project structure

```
Project_5/
├── app/
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── models/schemas.py
│   ├── routers/{health,images,search}.py
│   └── services/
│       ├── image_store.py        # disk-backed metadata + bytes
│       ├── multimodal_service.py # orchestration
│       ├── vector_index.py       # in-memory cosine search
│       └── vision_provider.py    # hashing fallback + abstract API
├── data/images/                  # runtime image storage
├── docker/{Dockerfile,docker-compose.yml}
├── requirements.txt
└── tests/
    ├── test_api_endpoints.py
    └── test_multimodal_service.py
```

## Running locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8004
```

Open http://localhost:8004/docs for the interactive Swagger UI.

## Running tests

```powershell
pytest tests -q
```

## Endpoints

| Method | Path                       | Description                              |
| ------ | -------------------------- | ---------------------------------------- |
| GET    | `/`                        | API metadata                             |
| GET    | `/health`                  | Service + provider status                |
| POST   | `/images`                  | Upload image (multipart) + optional tags |
| GET    | `/images`                  | List uploaded images                     |
| GET    | `/images/{image_id}`       | Get image metadata                       |
| POST   | `/images/{image_id}/caption` | Generate (or refresh) caption          |
| POST   | `/images/{image_id}/vqa`   | Answer a question about the image        |
| POST   | `/search/text`             | Find images matching a text query        |
| POST   | `/search/images/{image_id}` | Find images visually similar to one    |

## Upgrading to BLIP + CLIP

1. Uncomment `transformers`, `torch`, `Pillow` in `requirements.txt` and
   reinstall.
2. Set `VISION_PROVIDER=huggingface` in your environment.
3. Implement `app/services/huggingface_vision.py` with a class
   `HuggingFaceVisionProvider` exposing `embed_text`, `embed_image`,
   `caption`, and `answer`. The factory in `vision_provider.build_vision_provider`
   already routes to it and silently falls back to the hashing provider
   if the import fails.

## Notes for production

- Replace local disk storage with S3 / Azure Blob / MinIO and return
  pre-signed URLs in the upload response.
- Replace the in-memory vector index with ChromaDB / Qdrant / Weaviate.
- Add authentication / rate limiting at the API gateway layer.
