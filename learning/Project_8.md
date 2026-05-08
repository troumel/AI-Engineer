# Project 8 — Tutor Walkthrough: Custom LLM Deployment & Optimization

> A guided tour of [Project_8](../Project_8/) for a junior AI engineer. Until now you've **consumed** LLMs through OpenAI/Anthropic SDKs. In this final project you become the *provider*: you build the kind of inference server that vLLM, TGI, and llama.cpp ship — OpenAI-compatible endpoints, continuous batching, token streaming, per-key usage and cost tracking, rate limiting, multi-model registry, and Prometheus metrics. The actual model is mocked offline so the architecture stays portable, but every component has a marked production swap.

---

## 1. What this project actually does

You expose, on port 8007, a self-hosted **OpenAI-compatible** LLM API:

- `GET /v1/models` and `/v1/models/{id}` — registry of served models with quantization + context-window metadata.
- `POST /v1/chat/completions` — chat with optional `stream=true` for SSE token streaming.
- `POST /v1/completions` — legacy text completion (same plumbing).
- `GET /usage` — per-API-key + per-model token + cost ledger.
- `GET /metrics` — Prometheus text-format metrics.
- `GET /health` — backend, queue depth, registered model count.

The wire format is **byte-for-byte OpenAI**, so the official `openai` Python SDK works against your server with `base_url="http://localhost:8007/v1"`. That's the headline feature — your service is a drop-in replacement.

The default backend is `EchoLLMEngine`, a deterministic offline mock. Real production inference plugs in via `build_engine("vllm" | "transformers" | "llamacpp")`.

---

## 2. Why this project matters — the production LLM serving stack

For most of 2023–2024, the AI engineering job was "wire up OpenAI". That's commodity skill now. The real money in 2026 sits in **self-hosting**: open-weight models (Llama 3, Mistral, Phi-3, Qwen, DeepSeek) running on your own hardware for cost, privacy, latency, or compliance reasons. Companies that move from `gpt-4o` to a fine-tuned 8B model see 10× cost reductions, single-digit-millisecond first-token latency, and *no data leaving the VPC*.

The skills you need to do this:

- **OpenAI-compatible APIs** so existing client code keeps working.
- **Continuous batching** to extract GPU throughput. A naïve "one request → one forward pass" server wastes 95% of GPU time.
- **Streaming** because users won't wait 8 seconds for a complete response.
- **Rate limiting + auth** because *you* are now the API gateway, not OpenAI.
- **Usage + cost tracking** because Finance will ask "who is using how much" within a week of going live.
- **Metrics** because "is the GPU saturated?" is a question you must answer.
- **Quantization metadata** because int4-vs-int8-vs-fp16 is a real deployment decision.

This project gives you the *shape* of all of that. Swap the engine for vLLM at the end and you have a real serving stack.

---

## 3. Architecture in one picture

```
                       Client (openai SDK / curl / langchain)
                                       │  Bearer ${API_KEY}
                                       ▼
                          ┌─────────────────────────┐
                          │   FastAPI app (8007)    │
                          │  CORS + access logs     │
                          └─────────────────────────┘
                                       │
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
               authenticate     ModelRegistry     LLMService
               (Bearer key)     (3 models)        (orchestrator)
                                                       │
                  ┌────────────────────────────────────┼────────────────────────────────────┐
                  ▼                                    ▼                                    ▼
            RateLimiter                        ContinuousBatcher                       UsageTracker
          (TPM + RPM bucket)                   ┌────────────────┐                  (per-key, per-model
              per key                          │  asyncio.Queue │                   tokens + cost)
                                               │  worker task   │                       │
                                               └────────────────┘                       ▼
                                                       │                            data/usage.json
                                                       ▼
                                                  LLMEngine
                                              (echo|vllm|TGI|...)
                                                       │
                                                  Metrics  ◄── /metrics (Prometheus)
```

Eight collaborators, one orchestrator. Same architectural pattern as every previous project — service composes pluggable parts via DI — but at production scale.

---

## 4. Concept-by-concept walkthrough

### 4.1 OpenAI compatibility — the most valuable feature

Look at [`schemas.py`](../Project_8/app/models/schemas.py): `ChatCompletionRequest`, `ChatCompletionResponse`, `Usage`, `ChatChunkDelta`, etc. Every field name (`prompt_tokens`, `completion_tokens`, `finish_reason: "stop"|"length"`, `object: "chat.completion"`) matches OpenAI exactly. Why?

Because **the world's LLM client code is written against the OpenAI shape**. LangChain, LlamaIndex, the `openai` SDK, every chatbot UI, every agent framework — they all expect OpenAI-shaped JSON. If your self-hosted server speaks OpenAI, you inherit the entire ecosystem for free.

This is why vLLM, TGI, llama-cpp-python, Ollama, LiteLLM, and every other serving project converged on this shape. **Compatibility is the moat.**

The two endpoints worth memorising:

- `POST /v1/chat/completions` — the modern endpoint. Takes `messages: [{role, content}]`. Returns `{choices: [{message, finish_reason}], usage}`.
- `POST /v1/completions` — legacy text-in/text-out. Same plumbing, different shape.

The streaming variant uses **Server-Sent Events** with chunks shaped like `{choices: [{delta: {content: "..."}, finish_reason: null}]}` and a final `data: [DONE]\n\n` sentinel. See [`stream_chat_completion`](../Project_8/app/services/llm_service.py#L121-L208).

### 4.2 The LLM engine Protocol — the swap point

[`LLMEngine`](../Project_8/app/services/llm_engine.py#L40-L55) is a `Protocol` with three methods:

```python
def generate(self, request: GenerationRequest) -> GenerationResult: ...
def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...
def stream(self, request: GenerationRequest) -> Iterator[str]: ...
```

That's the *entire* contract. Any engine — `EchoLLMEngine` (offline mock), a future `VLLMEngine` wrapping `AsyncLLMEngine`, a `TransformersEngine` calling `model.generate()`, a `LlamaCppEngine` — implements those three methods.

Below the Protocol is `EchoLLMEngine`: a deterministic offline mock that produces a templated response from prompt words. It exists for one reason — **so the rest of the system (batching, streaming, usage, metrics) can be tested without a GPU**. Tests run in 2 seconds. CI is happy. The architecture is unchanged whether the engine is mocked or real.

`build_engine(backend)` is a dispatcher. The vllm/transformers/llamacpp branches currently fall back to echo (the upgrade point); you'd put the real imports there.

### 4.3 Continuous batching — the optimisation that makes self-hosting viable

This is **the most important idea in the project**. Read it twice.

**Naïve serving:** one request arrives, run a forward pass, return result. The GPU has thousands of cores; one request uses a tiny fraction of them. GPU utilisation: ~5%.

**Static batching:** wait for N requests, batch them, run one forward pass over all of them. Better utilisation, but the slowest request in the batch holds up everyone else. And what if request 1 finishes at token 20 while request 2 needs 200? Request 1 waits.

**Continuous batching** (vLLM, TGI, this project): the scheduler maintains a queue. On every decoding step, it can:
- Add new requests to the in-flight batch (they join mid-flight).
- Remove finished requests (so the batch shrinks dynamically).
- Pad the rest with the next-token forward pass.

Result: GPU utilisation goes from ~5% to ~80%, throughput jumps 5–20×. **This is why vLLM dominates LLM serving.**

[`ContinuousBatcher`](../Project_8/app/services/batcher.py) captures the *public contract* of this scheduler:

```python
async def submit(self, request) -> GenerationResult:
    future = loop.create_future()
    await self._queue.put(_PendingItem(request, future))
    return await future
```

A worker task ([`_run`](../Project_8/app/services/batcher.py#L88-L110)) loops:

```python
first = await self._queue.get()
batch = [first]
deadline = now + self.batch_timeout
while len(batch) < self.max_batch_size:
    timeout = max(deadline - now, 0)
    if timeout <= 0: break
    try: batch.append(await asyncio.wait_for(self._queue.get(), timeout=timeout))
    except TimeoutError: break
results = self.engine.generate_batch([item.request for item in batch])
for item, result in zip(batch, results):
    item.future.set_result(result)
```

The two parameters that govern behaviour:

- **`max_batch_size`** — how many requests bundle into one forward pass. Larger = better GPU utilisation but more memory pressure (each request adds a sequence to KV cache).
- **`batch_timeout_ms`** — how long the first request in a new batch waits for company. Larger = better batching at low traffic but worse p99 latency. Default 20ms is a reasonable starting point.

The futures-based design means *callers don't know they're being batched*. They `await batcher.submit(req)` and get a result. The batching is invisible.

> **Note:** this implementation does **request-level** batching (whole requests go in/out together). Real continuous batching does **token-level** batching (requests can join mid-generation). The interface is the same; the engine implementation is what differs. vLLM's `AsyncLLMEngine` plugs into this exact `submit()` shape.

### 4.4 Streaming — line by line

[`stream_chat_completion`](../Project_8/app/services/llm_service.py#L121-L208) does the dance:

1. **Validate the model** explicitly *before* opening the SSE stream. If you raised inside the generator, FastAPI would have already started a 200 response with the SSE content type — too late to send a 404.
2. **Count tokens** and **enforce rate limit** *before* generation starts. Same reason: you can't 429 mid-stream.
3. **First chunk** announces the role (`delta: {role: "assistant"}`). OpenAI's wire format requires this so clients know the role of the upcoming content.
4. **Token chunks** — for each piece from `engine.stream(request)`, emit `delta: {content: piece}`. The `await asyncio.sleep(0)` after each yield is critical — it lets the event loop preempt the worker so the chunks actually flush to the client (without it, a fast generator can starve the I/O loop and the user sees buffered chunks all at once at the end).
5. **Final chunk** — `delta: {}` (empty) with `finish_reason: "stop" | "length"`. OpenAI's contract.
6. **Sentinel** — `data: [DONE]\n\n`. Tells the client the stream is over.
7. **After the stream** — record metrics + usage. Streaming requests bill exactly like non-streaming ones; the user just sees the answer faster.

The non-stream path is simpler ([`create_chat_completion`](../Project_8/app/services/llm_service.py#L93-L120)): submit one `GenerationRequest` to the batcher, await the result, package into `ChatCompletionResponse`. The batcher is shared between streaming and non-streaming requests in real engines (the streaming path bypasses `submit()` here only because `EchoLLMEngine.stream` is synchronous; vLLM's stream is itself batched).

### 4.5 Token-bucket rate limiting — TPM + RPM

[`RateLimiter`](../Project_8/app/services/rate_limiter.py) implements **two independent token buckets per API key**:

- **Requests per minute (RPM).** Capacity = configured RPM. Refill rate = capacity/60 per second. Each request consumes 1.
- **Tokens per minute (TPM).** Capacity = configured TPM. Each request consumes its `prompt_tokens + max_tokens` estimate.

Why two buckets? Because the failure modes differ. A user spamming 1000 one-token requests per second is a *request* abuser. A user submitting 5 requests with 64K context each is a *token* abuser. Either rate-limits independently. **OpenAI's own rate limiting works exactly this way.**

The classic token-bucket update on each request:

```python
elapsed = now - bucket.updated_at
bucket.tokens = min(capacity, bucket.tokens + elapsed * refill_rate)
bucket.requests = min(capacity, bucket.requests + elapsed * refill_rate)
bucket.updated_at = now

if bucket.requests < 1: deny
if bucket.tokens < estimated: deny
bucket.requests -= 1
bucket.tokens -= estimated
```

The `estimated_tokens` is `prompt_tokens + max_tokens` — the **upper bound** before generation. You debit pessimistically: if the model finishes early, the user got a small free lunch. The alternative (debit after generation) lets users blow through the bucket on long outputs before any check fires.

**Production swap:** the in-memory bucket dies when the process restarts and doesn't share state across multiple workers. Real systems put the bucket state in **Redis** with atomic `INCRBY` + `EXPIRE` (or use NGINX `limit_req` at the gateway). The interface stays the same.

### 4.6 Usage tracking — the cost ledger

[`UsageTracker`](../Project_8/app/services/usage_tracker.py) records every successful request keyed by `(api_key, model)`:

```python
record(api_key, model, prompt_tokens, completion_tokens) → UsageRecord
```

It maintains:
- Global counters (total requests, prompt tokens, completion tokens, USD cost).
- Per-API-key aggregates.
- Per-model aggregates.

Cost is derived from configurable `cost_per_1k_prompt_tokens` and `cost_per_1k_completion_tokens`. The defaults ($0.0005 prompt / $0.0015 completion per 1K) are roughly GPT-4o-mini-shaped, but the point is the **mechanism**: you set the price, the ledger computes the cost. Self-hosted models often have an *implicit* cost (GPU-hours / token); same calculation, different number.

Persisted to `data/usage.json` after every record. Same JSON-as-DB pattern from prior projects. **Production swap is a real database** (Postgres with one row per request, materialised view for aggregates), and you'd bill on it. Stripe metering APIs accept a stream of usage events; this is the per-request log they want.

### 4.7 Prometheus metrics — observability is non-optional

[`Metrics`](../Project_8/app/services/metrics.py) maintains in-memory counters and gauges, and emits them in [Prometheus text format](https://prometheus.io/docs/instrumenting/exposition_formats/) at `GET /metrics`:

```
# HELP llm_requests_total Total inference requests served.
# TYPE llm_requests_total counter
llm_requests_total 142
# HELP llm_tokens_in_total Total prompt tokens processed.
# TYPE llm_tokens_in_total counter
llm_tokens_in_total 9842
...
```

The metrics worth memorising:

| Metric | Question it answers |
|---|---|
| `llm_requests_total` | Throughput |
| `llm_errors_total` | Are we failing? |
| `llm_tokens_in_total` / `llm_tokens_out_total` | Token economics; capacity planning |
| `llm_request_latency_seconds_avg` | User-perceived latency |
| `llm_batch_size_avg` | Is continuous batching working? Should be near `max_batch_size` under load |
| `llm_queue_depth` / `llm_queue_depth_max` | Are we saturated? Climbing queue depth = GPU bottleneck |
| `llm_uptime_seconds` | Has the service restarted? |

Prometheus scrapes `/metrics` every N seconds, stores time series, and Grafana plots them. The two graphs that matter:

1. **`llm_queue_depth_max` over time** — if this hits the cap, you need more GPUs.
2. **`llm_request_latency_seconds_avg` vs `llm_batch_size_avg`** — if latency rises while batch size stays low, your batch timeout is too short or traffic is too sparse.

**Production swap:** use the official `prometheus-client` library instead of hand-rolled text. Adds histograms, labels, gauge children. Same endpoint, same scrape contract.

### 4.8 Multi-model registry + quantization metadata

[`ModelRegistry`](../Project_8/app/services/model_registry.py) lists three models by default:

```python
ModelInfo(id="phi-3-mini-offline", quantization="int4-awq", context_window=4096, ...)
ModelInfo(id="mistral-7b-offline", quantization="int4-gptq", context_window=8192, ...)
ModelInfo(id="llama-3-8b-offline", quantization="int8", context_window=8192, ...)
```

The metadata is **what production tooling actually wants**:

- **`quantization`** — `int4-awq`, `int4-gptq`, `int8`, `fp16`, `Q4_K_M` (GGUF). Affects VRAM footprint and quality. A 7B int4-quantised model fits in 5GB; the same model in fp16 needs 14GB. Quantization is *the* lever for "can this run on a single A10 / 3090 / Mac Studio?".
- **`context_window`** — max input tokens the model accepts. Server should reject prompts > this, and clients should know the limit.
- **`max_output_tokens`** — service-level cap on generation length. Independent of context window — used to bound rate-limit estimates and prevent runaway generation.
- **`backend`** — which engine serves it. In a multi-engine deployment, one model might be on vLLM, another on llama.cpp.

When a request arrives with `model="phi-3-mini-offline"`, the service validates the ID against the registry, returns 404 if missing, otherwise dispatches to the engine. Multi-tenancy / multi-model is just routing on the `model` field.

### 4.9 Auth: bearer tokens, optional

[`authenticate`](../Project_8/app/dependencies.py#L74-L99) is a FastAPI dependency that reads `Authorization: Bearer <key>`:

- If `API_KEYS` env is empty: auth disabled, but if a Bearer header is present it's still extracted and used as the *usage key* (so you get per-user usage even without enforced auth).
- If `API_KEYS` is set: token must match one of the configured keys; otherwise 401.

For a real deployment you'd front this with an API gateway (Kong, Envoy, or a managed offering) that handles OIDC / JWT / OAuth, and the inference server only sees pre-authenticated traffic. But knowing the bearer-token mechanic is the floor — you should never expose a self-hosted LLM unauthenticated to the internet (cost runaways, abuse, prompt-injection).

### 4.10 Async lifespan: starting the batcher worker

[`lifespan`](../Project_8/app/main.py#L19-L29) calls `await initialize_services()` on startup, which does `await batcher.start()`. The batcher worker is an `asyncio.create_task(self._run())` that runs for the life of the process, dequeuing items and dispatching batches.

On shutdown, `await batcher.stop()` cancels the worker and drains pending futures. Without this, pending requests would hang forever; with it, in-flight callers get `CancelledError` and FastAPI returns proper errors.

**This is the first project where the service has a long-running async task** beyond the request handlers. Note the discipline:
- Task created inside `lifespan` so it lives exactly as long as the app.
- Cancelled cleanly on shutdown.
- Pending futures cancelled rather than leaked.

When you wire vLLM, the same pattern applies — its `AsyncLLMEngine` runs background tasks; you wrap them in lifespan management.

---

## 5. Worked example — tracing one chat request

Setup: default config, no API keys configured, healthy state.

**Request:**
```
POST /v1/chat/completions
Authorization: Bearer alice
Content-Type: application/json
{"model":"phi-3-mini-offline","messages":[{"role":"user","content":"Explain BM25"}],"max_tokens":32}
```

1. `authenticate` reads Bearer header → returns `"alice"` as the usage key.
2. `LLMService.create_chat_completion("alice", request)` is invoked.
3. `_render_chat_prompt` builds `"<|user|>\nExplain BM25\n<|assistant|>\n"`. (When you swap to a real model, replace this with the model's actual chat template — Llama uses `<|begin_of_text|>...`, Mistral uses `[INST]...[/INST]`, etc.)
4. `_run_generation`:
    - `_validate_model("phi-3-mini-offline")` passes.
    - `count_tokens(prompt)` → e.g. 6.
    - `_effective_max_tokens(32)` → `min(max(32,1), 256) = 32`.
    - `_enforce_rate_limit("alice", 6+32=38)` → bucket has plenty; allowed.
    - `batcher.submit(GenerationRequest(prompt, max_tokens=32, temp=0, top_p=1, stop=None))`.
5. Inside the batcher: `alice`'s request hits the queue. The worker dequeues it, waits up to 20ms for company, finds none (low traffic), runs a batch of size 1: `engine.generate_batch([req])`.
6. `EchoLLMEngine.generate` extracts salient words from the prompt, builds `"Synthesised response based on your prompt: explain bm25..."`, truncates to 32 words, returns `GenerationResult(text=..., finish_reason="stop")`.
7. Future resolves; `_run_generation` continues:
    - `count_tokens(result.text)` → e.g. 28.
    - `metrics.inc_request(prompt_tokens=6, completion_tokens=28, latency=0.012)`.
    - `usage_tracker.record("alice", "phi-3-mini-offline", 6, 28)` → ledger updated, JSON rewritten.
8. Response packaged:
    ```json
    {
      "id": "chatcmpl-<uuid>",
      "object": "chat.completion",
      "created": 1714530000,
      "model": "phi-3-mini-offline",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthesised..."}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 6, "completion_tokens": 28, "total_tokens": 34}
    }
    ```

That JSON is **indistinguishable from OpenAI's** — and that's the point.

---

## 6. Self-quiz

1. Why is OpenAI wire-format compatibility the *single most valuable* feature of a self-hosted LLM server?
2. Explain continuous batching in one paragraph. Why does it lift GPU utilisation from ~5% to ~80%?
3. What do `max_batch_size` and `batch_timeout_ms` trade off against each other?
4. Why does `stream_chat_completion` validate the model and rate-limit *before* opening the SSE stream?
5. What is the role of `await asyncio.sleep(0)` between SSE chunks? What goes wrong without it?
6. Why two independent token buckets (TPM + RPM) instead of one?
7. Why is `estimated_tokens = prompt + max_tokens` (an upper bound) used at rate-limit time rather than the actual completion length?
8. What does the `quantization` field on a `ModelInfo` actually affect at deployment time? Compare int4-awq vs fp16 for a 7B model.
9. The batcher's worker is started inside `lifespan` rather than at import time. Why does that matter?
10. How would you swap `EchoLLMEngine` for vLLM? Which method signatures stay; which gain async-ness? What changes elsewhere in the system?
11. Where would `data/usage.json` start to fail in production, and what's the minimum viable replacement?
12. Why expose `llm_queue_depth_max` separately from `llm_queue_depth`?

---

## 7. Hands-on next steps

- **Wire real vLLM.** Replace `EchoLLMEngine` with a class that wraps `vllm.AsyncLLMEngine`. Adapt the `Iterator[str]` stream contract to vLLM's `AsyncIterator`. Most of `LLMService` is unchanged.
- **Replace tokenization.** Use `tiktoken` for OpenAI-compatible models or `transformers.AutoTokenizer` for the model you actually serve. Update `count_tokens` and the chat prompt template.
- **Move rate limiter to Redis.** Use `INCRBY` + `EXPIRE` for atomic token-bucket updates that survive restarts and shard across workers.
- **Add per-key quotas.** Beyond rate limits, add monthly / daily token budgets in `UsageTracker` and reject when exceeded.
- **Switch metrics to `prometheus-client`.** Use `Histogram` for latency (gives you p50/p95/p99 percentiles, not just averages).
- **Add structured logging** with request IDs propagated through every log line. Essential for debugging "why was this request slow?".
- **Add a `/v1/embeddings` endpoint.** Same OpenAI shape, different engine. Most production stacks serve both.
- **Run two engines side-by-side** — `phi-3-mini-offline` on vLLM, a tiny model on llama.cpp — and let the registry route per-model. Demonstrates the multi-backend architecture.
- **Build a Grafana dashboard** with: requests/sec, error rate, p95 latency, queue depth, cost/min by API key. This is the day-1 production dashboard.
- **Front it with NGINX** for TLS termination + a basic `limit_req` zone as belt-and-braces rate limiting at the edge.

---

## 8. .NET parallels

| Concept here | .NET equivalent |
|---|---|
| `LLMEngine` Protocol | `IInferenceEngine` interface |
| `EchoLLMEngine` mock | A test double registered via DI in test profile |
| `ContinuousBatcher` | A `BackgroundService` (`IHostedService`) draining a `Channel<T>` |
| `asyncio.Future` | `TaskCompletionSource<T>` |
| `RateLimiter` token bucket | `RateLimiter`/`TokenBucketRateLimiter` from `System.Threading.RateLimiting` |
| `UsageTracker` | EF Core `DbSet<UsageRecord>` + a `IUsageMetering` service |
| `Metrics` Prometheus text | `prometheus-net` library on `/metrics` |
| `authenticate` Bearer | `AddAuthentication().AddJwtBearer(...)` |
| `lifespan` startup/shutdown | `IHostApplicationLifetime.ApplicationStarted/Stopping` events |
| OpenAI-compatible DTOs | Manually authored DTO records mirroring the OpenAI shape |

---

## 9. File cheat-sheet

| File | Purpose | Key idea |
|---|---|---|
| [app/config.py](../Project_8/app/config.py) | Settings | Backend, model, batch params, rate limits, costs, API keys |
| [app/main.py](../Project_8/app/main.py) | App wiring | Lifespan starts/stops batcher; 5 routers |
| [app/dependencies.py](../Project_8/app/dependencies.py) | DI + auth | Singleton `LLMService`, Bearer-token `authenticate` |
| [app/models/schemas.py](../Project_8/app/models/schemas.py) | OpenAI DTOs | `ChatCompletionRequest/Response`, `Usage`, `ChatCompletionChunk`, `ModelInfo` |
| [app/services/llm_engine.py](../Project_8/app/services/llm_engine.py) | Engine Protocol | `generate`, `generate_batch`, `stream`; `EchoLLMEngine`; `build_engine` |
| [app/services/model_registry.py](../Project_8/app/services/model_registry.py) | Model catalogue | id, quantization, context window, backend |
| [app/services/batcher.py](../Project_8/app/services/batcher.py) | Continuous batching | Async queue + worker task draining up to `max_batch_size` / `batch_timeout_ms` |
| [app/services/rate_limiter.py](../Project_8/app/services/rate_limiter.py) | Rate limit | Per-key token-bucket for TPM + RPM |
| [app/services/usage_tracker.py](../Project_8/app/services/usage_tracker.py) | Cost ledger | Per-key + per-model token + cost aggregates, JSON persisted |
| [app/services/metrics.py](../Project_8/app/services/metrics.py) | Metrics | Prometheus text exposition |
| [app/services/tokenization.py](../Project_8/app/services/tokenization.py) | Token count | Word + punctuation approximation |
| [app/services/llm_service.py](../Project_8/app/services/llm_service.py) | Orchestrator | Chat, completion, streaming, validation, rate-limit, usage |
| [app/routers/chat.py](../Project_8/app/routers/chat.py) | Chat | `POST /v1/chat/completions` (with stream branch) |
| [app/routers/completions.py](../Project_8/app/routers/completions.py) | Legacy | `POST /v1/completions` |
| [app/routers/models.py](../Project_8/app/routers/models.py) | Registry | `GET /v1/models`, `GET /v1/models/{id}` |
| [app/routers/observability.py](../Project_8/app/routers/observability.py) | Ops | `GET /usage`, `GET /metrics` |
| [app/routers/health.py](../Project_8/app/routers/health.py) | Liveness | `GET /` and `/health` |

---

## 10. The single most important takeaway

> **Self-hosting an LLM is 10% the model and 90% the serving stack around it.**
>
> The model is a frozen artefact you download. Everything that turns it into a *production API* — OpenAI-compatible routing, continuous batching, streaming, rate limiting, usage tracking, metrics, multi-model registry, auth — is what you have to build. Or, more accurately: what tools like vLLM and TGI build for you, and what you must understand to operate them sanely.
>
> Every component in this project is intentionally simple so the *shape* is visible. The shape is what's universal. The specific engine (echo today, vLLM tomorrow, whatever-comes-next next year) is replaceable. The OpenAI wire format will outlive any single model. Continuous batching will be relevant as long as transformers do autoregressive decoding. Metrics + cost tracking + rate limiting will be relevant as long as users + finance teams exist.
>
> You are now equipped to read the vLLM source code and recognise every concept. That, more than any single project, is what the AI Engineering Roadmap was building toward.
