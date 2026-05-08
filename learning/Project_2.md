# Project 2 — Tutor Walkthrough: Sentiment Analysis API

> A guided tour of [Project_2](../Project_2/) for a junior AI engineer coming from a backend background. Read this side-by-side with the source files. Every concept below is grounded in a specific file you can open.

---

## 1. What this project actually does

You expose a small HTTP API. A client sends text, the API replies with a sentiment label (`positive`, `negative`, `neutral`) and a confidence score. Internally we:

1. Validate the request (so junk input never reaches the model).
2. Look up a cached answer first.
3. Otherwise run inference on a **pre-trained HuggingFace transformer**.
4. Normalise the model's raw output into our API's contract.
5. Cache the answer and return it.
6. Apply a per-client **rate limit** so one noisy client can't saturate the model.

That's the whole game. Everything below is just *how* we do those six steps cleanly.

---

## 2. Why this project is the next step after Project 1

| Concern | Project 1 | Project 2 |
|--------|-----------|-----------|
| Model | Trained yourself (scikit-learn) | **Pre-trained transformer downloaded from HuggingFace Hub** |
| Inputs | Numerical features | **Free-form natural language text** |
| Output | Numerical prediction | Label + confidence |
| Latency profile | Microseconds | Tens to hundreds of milliseconds → caching matters |
| Memory profile | Small | Hundreds of MB resident → load **once at startup** |
| External deps | None | **Redis** (optional, with fallback) |
| New cross-cutting concerns | — | Caching, rate limiting, request logging, async offloading |

The structural pattern (config → dependencies → services → routers) is identical to Project 1. That's deliberate — you reuse the muscle memory and only learn the new AI/NLP-specific pieces.

---

## 3. Mental model: where each new concept lives

```
Client ──HTTP──▶ FastAPI app
                  │
                  ├── middleware (logging, CORS)
                  ├── /health
                  └── /sentiment, /sentiment/batch
                       │
                       ├── enforce_rate_limit   ◀── FixedWindowRateLimiter
                       ├── Pydantic validation  ◀── SentimentRequest / BatchSentimentRequest
                       └── SentimentService.predict(...)
                              │
                              ├── 1. Hash text → cache key
                              ├── 2. CacheBackend.get_json(key)        (Redis or in-memory)
                              ├── 3. classifier(text)  ─── HuggingFace pipeline
                              ├── 4. _normalize_result(...)            (raw → API contract)
                              └── 5. CacheBackend.set_json(key, ...)
```

Every box maps to a real symbol in the code. As you read, keep this picture next to you.

---

## 4. Concept-by-concept walkthrough

### 4.1 What a HuggingFace transformer pipeline really is

A *transformer* is the neural network architecture behind models like BERT, GPT, and DistilBERT. For sentiment we use [`distilbert-base-uncased-finetuned-sst-2-english`](../Project_2/app/config.py#L19). Three concepts you must internalise:

- **Pre-training**: Someone else trained the base DistilBERT on huge text corpora to learn general language. This is expensive and rare.
- **Fine-tuning**: Someone (Hugging Face) then continued training it on the **SST-2** dataset (Stanford Sentiment Treebank), which is a labelled set of movie reviews with `positive`/`negative` labels. That fine-tuning is what makes it a *sentiment* model.
- **Inference**: What you do — feed it text, get a label + confidence. No training happens.

The `transformers.pipeline("sentiment-analysis", model=...)` call hides three things behind one function: a **tokenizer** (converts your text to integer IDs), the **model** (does the math), and a **post-processor** (turns logits into labels and softmax probabilities).

The library does the heavy lifting — your job is to *integrate it into a production-shaped service*.

### 4.2 Loading the model exactly once — and why

Open [Project_2/app/services/sentiment_service.py](../Project_2/app/services/sentiment_service.py#L40-L60). Notice that `load_model` is called from `initialize_services()` in [app/dependencies.py](../Project_2/app/dependencies.py#L20-L46), which is itself called from FastAPI's `lifespan` in [app/main.py](../Project_2/app/main.py#L22-L31).

Why? A transformer takes hundreds of MB and **seconds** to load from disk into RAM. If you loaded it inside the request handler:

- The first request would be unusably slow.
- Concurrent requests would race to load it, each consuming hundreds of MB.
- Memory would balloon, your container would OOM, and you'd blame "AI being slow" when really you blamed yourself.

This is the **#1 production pitfall for AI APIs**. The fix is the singleton pattern: one instance, created at startup, injected via `Depends(get_sentiment_service)`.

> **.NET parallel:** This is `services.AddSingleton<SentimentService>()` in `Program.cs`. The mechanism differs (FastAPI uses module-level globals plus a `Depends` accessor), but the *intent* is identical.

### 4.3 The "degraded startup" pattern

Look closely at [`load_model`](../Project_2/app/services/sentiment_service.py#L42-L59):

```python
try:
    transformers = importlib.import_module("transformers")
    ...
except Exception as exc:
    logger.warning(...)
    if not allow_degraded_startup:
        raise
```

This is a deliberate design choice: the API can boot **without** the model (e.g. when running tests, or in CI without internet access to HuggingFace Hub). In that state the `/health` endpoint returns `degraded` and `/sentiment` returns 503.

Why care? Because:

- It lets you run the test suite without a 250 MB model download.
- It lets the orchestrator (Kubernetes) keep the container alive, see the failing health check, and trigger alerts — instead of crashing in a restart loop you can't observe.

This is a **production-readiness pattern**. The API surface stays stable; only the *capability* degrades.

### 4.4 Pydantic models = your validation contract

[`app/models/schemas.py`](../Project_2/app/models/schemas.py) defines every request and response shape. Junior-engineer tip: **don't skip this file** — it is your API contract, your OpenAPI doc, and your runtime validator, in one place.

Three things worth highlighting:

**a. `Field(min_length=1, max_length=5000)`** on `SentimentRequest.text` — junk input is rejected at the boundary with a 422 response. The endpoint code never has to defend against blank or 10 MB text.

**b. The custom validators**:

```python
@field_validator("text")
@classmethod
def validate_text(cls, value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Text must not be blank.")
    return stripped
```

A whitespace-only string would pass `min_length=1` but isn't useful input. The validator strips whitespace and rejects truly empty strings. Pydantic also lets you mutate the value during validation — the stripped form is what the rest of the code sees.

**c. `Literal["positive", "negative", "neutral"]`** as the `SentimentLabel` type. This isn't just docs — it's enforced. The service literally cannot return `"sad"` and have it pass response validation. Type discipline catches bugs before users do.

> **.NET parallel:** Pydantic models are DTOs with `[Required]`, `[StringLength]`, `[RegularExpression]` baked in plus FluentValidation-style custom rules — but evaluated automatically by the framework on every request.

### 4.5 Binary model → ternary API: the neutral-threshold trick

**This is one of the most important AI engineering lessons in this project.**

The SST-2 model is **binary** — it only knows `positive` or `negative`. But your product probably wants `neutral` too ("The package arrived." is neither happy nor angry).

Look at [`_map_label`](../Project_2/app/services/sentiment_service.py#L142-L153):

```python
if score < self.neutral_threshold:
    return "neutral"
```

If the model is *uncertain* (confidence below `NEUTRAL_THRESHOLD`, default `0.70`), we override the binary label and report `neutral`. This is a thin **post-processing layer** between the model's native contract and your product's contract.

Why it matters:

- You did not retrain the model — that would cost time, GPUs, and labelled data.
- You shaped the model's output to your product semantics with a single threshold.
- The threshold is a config value (`NEUTRAL_THRESHOLD`), not a magic number — you can tune it.

This pattern — **wrapping a generic model in a product-specific normalisation layer** — recurs everywhere in AI engineering. Internalise it.

### 4.6 Caching: why hash, why TTL, why two backends

Open [`app/services/cache_service.py`](../Project_2/app/services/cache_service.py).

**Why cache at all?** Inference on a transformer is 10–100× slower than a Redis lookup. If two users send "Great product!" we shouldn't burn GPU cycles twice.

**Why hash the text into the key?** Three reasons in [`_build_cache_key`](../Project_2/app/services/sentiment_service.py#L114-L116):

```python
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
return f"sentiment:{digest}"
```

1. Bounded key length — Redis keys above ~512 bytes hurt performance.
2. No special characters — colons, spaces, newlines in raw text would break key parsing.
3. **Privacy-friendly** — your Redis DB stores hashes, not raw user text. (For real PII you'd want stronger guarantees, but this is a meaningful default.)

**Why TTL?** Stored predictions can become stale if you upgrade the model. A 5-minute TTL is a cheap insurance policy: cache speeds up bursts, but you naturally re-validate over time.

**Why two backends?** [`create_cache_backend`](../Project_2/app/services/cache_service.py#L83-L92) is a tiny factory: it tries Redis, and if it can't connect (no Redis running locally, no network), it silently falls back to an in-memory TTL map. Same `CacheBackend` interface, different implementation. This means:

- The dev experience stays painless (no need to start Redis just to run the API locally).
- Tests work without a Redis container.
- Production gets the real distributed cache.

This is the **strategy pattern with a graceful default** — the same idea behind ASP.NET's `IDistributedCache` / `MemoryDistributedCache`.

### 4.7 Rate limiting: protecting the expensive thing

[`FixedWindowRateLimiter`](../Project_2/app/services/rate_limiter.py#L31-L66) is the simplest practical algorithm. For each client (here keyed by IP), we track:

- when the current window started (`window_started_at`)
- how many requests they've made in this window (`request_count`)

If the window has elapsed → reset both. If they're under quota → allow. Otherwise → raise `RateLimitExceeded` and the dependency turns it into a `429 Too Many Requests` with a `Retry-After` header.

The dependency [`enforce_rate_limit`](../Project_2/app/dependencies.py#L70-L91) also writes the `X-RateLimit-*` headers on every response — a small touch that lets clients see how close they are to the limit, instead of getting smacked by 429s with no warning.

**Limitations to know (you'll be asked in interviews):**

- Fixed-window has **boundary spikes**: a client can do 30 requests at second 59 and 30 more at second 61 — 60 requests in 2 seconds. *Sliding-window* or *token-bucket* algorithms (we use the latter in Project 8) fix this.
- It's **per-process in-memory**. Two API replicas behind a load balancer each have their own counter, so the effective limit is `2 × N`. Real production uses Redis-backed limiters.

The current implementation is **good enough** for one container and very honest about what it does. Trade-offs > complexity.

### 4.8 Async, threads, and "don't block the event loop"

Look at the route handler in [`app/routers/predictions.py`](../Project_2/app/routers/predictions.py#L29-L33):

```python
async def classify_sentiment(...):
    return await asyncio.to_thread(service.predict, request.text)
```

Why not just `service.predict(request.text)`?

FastAPI runs your handler on a single-thread **asyncio event loop**. If you call a CPU-heavy synchronous function (and transformer inference is exactly that) directly inside an `async def`, you **freeze the event loop** — every other request stalls until your inference returns. With concurrent traffic, throughput collapses.

`asyncio.to_thread` punts the synchronous call to a worker thread, releases the event loop, and `await`s the thread's result. Other requests keep flowing.

Rules of thumb:

- I/O-bound (DB, HTTP, Redis): use real `async` libraries → `await` directly.
- CPU-bound (model inference, image processing): wrap in `asyncio.to_thread` (or use a process pool, or — better — a dedicated batching server like vLLM, which we explore in Project 8).

> **.NET parallel:** Same story as `Task.Run(() => CpuBoundWork())` in ASP.NET Core. Don't block the request thread on CPU-heavy work.

### 4.9 The lifespan context manager

[`lifespan`](../Project_2/app/main.py#L22-L31) is FastAPI's **startup/shutdown hook** — equivalent to `IHostedService.StartAsync` / `StopAsync` in .NET. Anything that must run *exactly once* (load model, open DB pool, start background workers) goes in the `before yield` half. Cleanup goes after `yield`.

Two reasons this matters:

- Tests using `TestClient` actually trigger the lifespan when used as a context manager (`with TestClient(app) as client:`). That means `initialize_services()` runs in tests too — so dependencies behave the same way.
- It composes with `asynccontextmanager` so you can `await` async setup work (open async DB pools, warm up async clients, etc.).

### 4.10 The request-logging middleware

[`log_requests`](../Project_2/app/main.py#L51-L64) wraps every request, measures latency with `time.perf_counter()`, and logs `METHOD PATH -> STATUS in DURATIONms`. In production you'd ship this to a structured logger (ELK, Datadog, Loki). The pattern itself — middleware that observes timing — is the same.

`time.perf_counter()` (not `time.time()`) is the right tool for measuring elapsed time: it's monotonic and ignores system clock changes.

### 4.11 Test seam: the injected fake classifier

The single most important *testing* idea in this project lives in [`SentimentService.__init__`](../Project_2/app/services/sentiment_service.py#L25-L37):

```python
def __init__(self, ..., classifier: Optional[ClassifierType] = None):
    ...
    self.classifier = classifier
```

And in [`load_model`](../Project_2/app/services/sentiment_service.py#L42-L45):

```python
if self.classifier is not None:
    return
```

If a test passes its own `classifier` callable (any function `str -> [{"label":..., "score":...}]`), `load_model` is a no-op and the test never touches HuggingFace. This is **constructor-injected dependency seam-ing** — the same idea as taking an `IFooClient` parameter so a test can pass a fake.

Concretely, the test in `tests/test_sentiment_service.py` injects a `lambda text: [{"label": "POSITIVE", "score": 0.99}]` and validates the *normalisation*, *cache lookup*, and *threshold* logic without ever loading a 250 MB model.

**Lesson:** AI services should be designed so the model layer is an injected dependency, not a hard import. Tests, local dev, and CI all benefit.

---

## 5. The end-to-end request lifecycle (in 9 steps)

When a client `POST`s to `/sentiment`:

1. **CORS middleware** matches the origin.
2. **Logging middleware** starts a timer.
3. FastAPI matches the `/sentiment` route in [`predictions.py`](../Project_2/app/routers/predictions.py#L19-L33).
4. **Pydantic** parses the JSON into `SentimentRequest`, runs `validate_text`. If invalid → 422 (early exit).
5. **`enforce_rate_limit` dependency** runs. The IP is checked against the limiter; under quota → counter increments and `X-RateLimit-*` headers are set; over quota → 429.
6. **`get_sentiment_service` dependency** returns the singleton.
7. The handler calls `await asyncio.to_thread(service.predict, request.text)` — offloading CPU work.
8. Inside `predict`: hash the text → cache lookup. **Hit** returns immediately with `cached=True`. **Miss** → run the classifier → `_normalize_result` → cache store.
9. The `SentimentResponse` flows back. Pydantic validates the **outbound** shape too (this catches bugs where a refactor breaks the contract). The middleware logs the latency.

Trace through one specific request in your head before moving on. If a step is fuzzy, re-read its section above.

---

## 6. What a junior engineer should be able to explain after reading this

Quiz yourself. If you can answer these without looking, you've absorbed the project.

1. Why is the model loaded in `lifespan` instead of inside the route handler?
2. Why does `_map_label` exist? What problem does the `NEUTRAL_THRESHOLD` solve?
3. Why are texts SHA-256-hashed before being used as cache keys?
4. What happens if Redis is down at startup? What about at runtime?
5. Why is `asyncio.to_thread` used in the route handler?
6. How are tests able to validate `SentimentService` without downloading a 250 MB model?
7. What are two limitations of `FixedWindowRateLimiter`, and what would replace it in production?
8. Why does the API have a "degraded" health state and what is its production purpose?
9. Where exactly does Pydantic validation run in the request lifecycle, and what does it protect against?
10. If the SST-2 model returned `LABEL_1` (raw) instead of `POSITIVE`, would this code still work? Why?

---

## 7. Suggested next steps for hands-on learning

- **Replace the threshold with calibration.** Read about *temperature scaling* and *Platt scaling*. Implement one in `_map_label` and compare neutral coverage on a small labelled set you assemble yourself.
- **Swap caches.** Add a third backend that uses SQLite. The `CacheBackend` ABC was designed for exactly this.
- **Trade fixed-window for token-bucket.** Re-read [Project_8's `RateLimiter`](../Project_8/app/services/rate_limiter.py) and port the algorithm. Notice how the `RateLimiter` *interface* used by callers barely changed.
- **Add a `/sentiment/explain` endpoint.** Use HuggingFace's `inputs_embeds` + gradient norms (or a simple library like [`captum`](https://captum.ai/)) to highlight which tokens drove the score. This is your first taste of *model interpretability*.
- **Profile cold start.** Use `time` and a Python profiler to measure how long `pipeline(...)` takes. Now you know exactly why singleton initialisation matters.

---

## 8. Cheat-sheet of the files

| File | One-line purpose | Key idea you must remember |
|------|------------------|----------------------------|
| [app/config.py](../Project_2/app/config.py) | All tunables in one place | 12-factor config via `pydantic-settings` |
| [app/main.py](../Project_2/app/main.py) | App wiring | `lifespan` runs once; middleware logs latency |
| [app/dependencies.py](../Project_2/app/dependencies.py) | DI container | Singletons + `Depends(...)` accessor pattern |
| [app/models/schemas.py](../Project_2/app/models/schemas.py) | Request / response contracts | Pydantic = validation + docs + types |
| [app/services/sentiment_service.py](../Project_2/app/services/sentiment_service.py) | Inference orchestration | Cache → classifier → normalise; injected classifier seam |
| [app/services/cache_service.py](../Project_2/app/services/cache_service.py) | Cache strategy | ABC + Redis impl + in-memory fallback factory |
| [app/services/rate_limiter.py](../Project_2/app/services/rate_limiter.py) | Per-client throttle | Fixed-window, thread-safe, returns headers |
| [app/routers/predictions.py](../Project_2/app/routers/predictions.py) | API surface | `asyncio.to_thread` for CPU-bound inference |
| [app/routers/health.py](../Project_2/app/routers/health.py) | Liveness / readiness | Reports degraded state when model not loaded |

---

## 9. The single most important takeaway

> **The transformer model is the smallest part of this project.** The bulk of the code is about *running it safely in production* — load it once, validate inputs, cache outputs, throttle abuse, normalise its raw shape into your product contract, and stay testable without it.
>
> AI engineering is mostly **engineering**. The "AI" is one well-isolated dependency you call through a clean interface.

Once you internalise that, every subsequent project in this roadmap is just a variation on the same theme.
