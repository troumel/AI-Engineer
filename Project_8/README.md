# Project 8 — Custom LLM Deployment with Optimization

A FastAPI service that exposes a self-hosted, OpenAI-compatible LLM inference API. It demonstrates the production architecture you'd build around vLLM / TGI / llama.cpp:

- **OpenAI-compatible** endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).
- **Continuous-batching scheduler** that bundles concurrent requests up to `max_batch_size` or `batch_timeout_ms`.
- **Token streaming** via Server-Sent Events with the OpenAI chunk format and `data: [DONE]` terminator.
- **Per-API-key + per-model usage tracking** with cost computation persisted to JSON.
- **Token-bucket rate limiting** on both requests-per-minute and tokens-per-minute.
- **Prometheus-style `/metrics`** exposing latency, batch size, queue depth, error counts, and uptime.
- **Multi-model registry** with quantization and context-window metadata (Phi-3, Mistral 7B, Llama 3 8B placeholders).
- **Bearer-token auth** (optional — controlled by the `API_KEYS` env var).

The service runs **fully offline** by default: the `EchoLLMEngine` produces deterministic completions so the OpenAI shape, streaming, batching, usage tracking, and rate limiter can be validated in CI without a GPU. Real inference engines slot in via `build_engine("vllm" | "transformers" | "llamacpp")`.

## Project layout

```
Project_8/
  app/
    main.py                FastAPI entry point (port 8007)
    config.py              pydantic-settings config
    dependencies.py        Singleton DI + bearer auth
    models/schemas.py      OpenAI-compatible DTOs
    routers/
      health.py            /, /health
      models.py            /v1/models, /v1/models/{id}
      chat.py              /v1/chat/completions (+ streaming)
      completions.py       /v1/completions
      observability.py     /usage, /metrics (Prometheus text)
    services/
      llm_engine.py        Engine protocol + EchoLLMEngine + factory
      model_registry.py    Multi-model registry with quantization metadata
      batcher.py           Continuous-batching async scheduler
      rate_limiter.py      Per-key token bucket (RPM + TPM)
      usage_tracker.py     Per-key + per-model token & cost ledger (JSON)
      metrics.py           Prometheus text exposition
      tokenization.py      Lightweight token counter
      llm_service.py       High-level orchestrator
  tests/                   Unit + API tests (pytest-asyncio)
  docker/                  Dockerfile + docker-compose.yml
```

## Running locally

```powershell
cd Project_8
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m pip install -r requirements.txt
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m uvicorn app.main:app --reload --port 8007
```

Then visit http://127.0.0.1:8007/docs.

## Tests

```powershell
cd Project_8
& "C:/Users/Theo/source/repos/AI Engineer/.venv/Scripts/python.exe" -m pytest -q
```

## Example requests

The wire format matches OpenAI exactly, so the official `openai` Python SDK works against this server with `base_url="http://localhost:8007/v1"`.

```bash
curl -X POST http://localhost:8007/v1/chat/completions \
  -H "Authorization: Bearer my-key" -H "Content-Type: application/json" \
  -d '{"model":"phi-3-mini-offline","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

Streaming:

```bash
curl -N -X POST http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"phi-3-mini-offline","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":32}'
```

## Observability

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Backend, queue depth, registered model count |
| `GET /usage` | Aggregated tokens + cost per API key and per model |
| `GET /metrics` | Prometheus text — `llm_requests_total`, `llm_tokens_in_total`, `llm_tokens_out_total`, `llm_batch_size_avg`, `llm_queue_depth`, `llm_request_latency_seconds_avg`, `llm_uptime_seconds`, ... |

## Production upgrade slots

| Concern | Offline default | Production swap |
|---------|------------------|-----------------|
| Inference engine | `EchoLLMEngine` | vLLM (`AsyncLLMEngine`), HuggingFace TGI, `llama.cpp` (GGUF), `transformers.generate` |
| Quantization | Metadata only (`int4-awq`, `int4-gptq`, `int8`) | `bitsandbytes`, AutoGPTQ, AWQ, GGUF Q4_K_M |
| Tokenization | Word-based approximation | `tiktoken`, model-specific `AutoTokenizer` |
| Batching | In-process `ContinuousBatcher` | vLLM PagedAttention + continuous batching |
| Rate limiting | In-memory token buckets | Redis-backed bucket + NGINX `limit_req` |
| Metrics | Manual Prometheus text | `prometheus-client` + Grafana dashboards |
| Auth | Static bearer-token list | OIDC / API gateway (Kong, Envoy) |
| Deployment | Single container | Kubernetes + HPA + NGINX/Envoy load balancer |

Each component is constructed in `dependencies.initialize_services()` so swapping a real backend is a one-line change.
