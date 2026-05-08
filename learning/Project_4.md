# Project 4 — Tutor Walkthrough: NER API with Model Lifecycle

> A guided tour of [Project_4](../Project_4/) for a junior AI engineer. The first project that introduces **the model lifecycle**: training, versioning, registry, activation, and A/B rollout — the operational scaffolding that real ML teams obsess over.

---

## 1. What this project actually does

It exposes an HTTP API that:

1. Extracts **named entities** from text (`PERSON`, `ORG`, `LOCATION`, etc.) — `POST /ner/extract`.
2. Accepts **annotated training data** and produces a new model **version** as a background job — `POST /training/jobs`.
3. Maintains a **model registry** on disk so versions survive restarts — `GET /models`.
4. Lets ops **activate** a specific version — `POST /models/activate`.
5. Routes a configurable percentage of traffic to a candidate version for **A/B rollout** — `POST /experiments/rollout`.
6. Reports background **job progress** — `GET /training/jobs/{job_id}`.

The "model" itself is intentionally a tiny phrase-based pattern matcher. **That is a feature, not a limitation.** It lets you focus on the *lifecycle plumbing*, which is what's hard, while the inference component remains trivially understandable.

---

## 2. Why this project matters — the lifecycle is the lesson

Most beginner AI tutorials stop at "I trained a model on my laptop". Production teams spend the next two years answering questions like:

- Which version is currently serving traffic?
- How do I roll out a new version without breaking everything?
- If the new version is worse, how fast can I revert?
- Where is each version stored, and how do I reproduce it?
- How do I run training jobs without blocking the API?
- How do I report progress to the team while a job runs for hours?

This project is a miniature, working answer to all of those — using a 200-line "model" so the operational concepts stand out cleanly. When you swap in a real HuggingFace `Trainer` later, **the API contract and lifecycle don't change** — only `PatternNERModel` does.

That separation is the whole point of the design.

---

## 3. What "Named Entity Recognition" actually is

NER is the task of finding spans of text that name real-world things and labelling them. Given:

> *"Sarah Johnson joined Microsoft in Seattle last March."*

A NER model returns:

| start | end | text | label |
|------:|----:|------|-------|
| 0 | 13 | Sarah Johnson | PERSON |
| 21 | 30 | Microsoft | ORG |
| 34 | 41 | Seattle | LOCATION |

Two important details junior engineers miss:

- **Spans are character offsets, not token indices.** This decision propagates everywhere — the training data format, [`EntitySpan`](../Project_4/app/models/schemas.py#L7-L11), the model's `extract` output. Character offsets are universal; token offsets depend on the tokenizer. You almost always want to expose character offsets at the API boundary.
- **NER is sequence labelling, not classification.** Project 2 said "this whole sentence is positive". NER says "characters 0-13 of this sentence are a PERSON". Subtle but huge: now your training data needs span annotations, your model needs span outputs, and your evaluation needs span-level F1 — none of which apply to plain classification.

In the real world, a HuggingFace token-classification model (e.g. `bert-base-cased`) is fine-tuned on a CoNLL-style BIO-tagged dataset. Tokens get tags like `B-PERSON` (begin), `I-PERSON` (inside), `O` (outside). The post-processor merges adjacent `B`/`I` tokens back into character spans. We skip all of that; `PatternNERModel` does straight phrase matching. But the **data shape we expose** — character spans in, character spans out — is the same.

---

## 4. The pattern-based "model" — why it's fine

[`PatternNERModel`](../Project_4/app/services/pattern_ner.py) holds a list of `PatternDefinition(phrase, label, score)`. `extract(text)` lowercases the input, walks the patterns, finds case-insensitive matches respecting **word boundaries** (so "Apple" doesn't match inside "pineapple"), and rejects overlaps via an `occupied: list[bool]` sentinel.

`from_examples` "trains" by:

1. Counting (phrase, label) pairs across the training set.
2. For each unique phrase, picking the most common label.
3. Assigning a score `min(0.99, 0.70 + count * 0.05)` — more sightings → higher confidence.
4. **Merging** the new patterns onto the base version's patterns (later wins on conflict).

Patterns are sorted **longest-first** in `__init__` so "Sarah Johnson" matches before "Sarah". This is a lesson worth keeping when you replace it with a transformer: even neural NER pipelines often have a *gazetteer* layer for known entities, and longest-match-wins is the standard tie-breaker.

> **The point:** this is *not* meant to be a great NER model. It is meant to be a model artifact you can **train, serialise, version, activate, and roll out**. Treat it as a stand-in for a 400 MB transformer checkpoint.

### 4.1 The artifact format

Every version persists as a folder under `models/<version_name>/` containing:

- `metadata.json` — version name, base model, created_at, training_example_count, entity_labels
- `patterns.json` — list of pattern dicts

This is the same shape a HuggingFace checkpoint has (a folder of files). Crucially:

- The **registry** is a separate file, [`models/registry.json`](../Project_4/app/services/ner_service.py#L298-L309), that records the *active version* and *rollout config*.
- On startup, [`_load_or_seed_models`](../Project_4/app/services/ner_service.py#L317-L334) scans the directory, loads every artifact, and seeds a `baseline-v1` if none exist.
- If `registry.active_version` points to a missing version, it self-heals to the first available one.

This is **deployment-resilient**. Whoever next ships a version-3 onto an existing server doesn't break the boot sequence.

---

## 5. The model registry pattern

[`NERService`](../Project_4/app/services/ner_service.py) is the registry. Internally:

```python
self._models: dict[str, PatternNERModel] = {}   # in-memory by version_name
self._registry: dict = {                        # persisted as registry.json
    "active_version": "baseline-v1",
    "rollout": {
        "primary_version": "baseline-v1",
        "candidate_version": None,
        "candidate_percentage": 0.5,
    },
}
```

Three operations on it:

- `activate_model(version)` writes `active_version` to disk.
- `configure_rollout(primary, candidate, pct)` validates both versions exist and writes the rollout block.
- `list_models()` returns the registry plus per-version metadata.

The registry file is the **source of truth**. The in-memory `dict` is a cache. Restart the process → reload from disk → state is preserved.

This is a faithful miniature of MLflow Model Registry, AWS SageMaker Model Registry, and similar systems. They add UI, ACL, and lineage tracking, but the core data model is `(version_name → artifact_path) + (active stage → version_name)`.

---

## 6. Background training: how it actually works

[`start_training_job`](../Project_4/app/services/ner_service.py#L99-L131) is the most operationally interesting code in the project.

```python
self._executor = ThreadPoolExecutor(max_workers=max_training_workers)
...
if request.run_async:
    self._executor.submit(self._run_training_job, job_id, request, base_version)
    return TrainingJobResponse(job_id=..., status="queued", ...)
```

What's happening:

1. The HTTP handler returns `202 Accepted` immediately with a `job_id`. The client doesn't wait.
2. The training function runs on a **background thread** owned by the service. The thread updates `state.status`, `state.progress`, `state.current_step` as it works.
3. The client polls `GET /training/jobs/{job_id}` to see progress.
4. On completion, the new version is saved to disk, registered, and (optionally) auto-activated.
5. On failure, the job state's `status="failed"` and `error` is populated. **Crucially the API doesn't crash** — the exception is caught inside `_run_training_job`.

A few subtle things to internalise:

- `_executor.submit` returns a `Future`, but we ignore it. The job state in `self._jobs` is the only handle we need; the client reads it via the `/training/jobs/{job_id}` endpoint.
- `max_training_workers=2` (from settings) caps concurrency. If you submit 5 jobs, only 2 run at once — the rest queue inside the executor.
- The training job lookup raises `ValueError` and the [router maps that to 404](../Project_4/app/routers/training.py#L42-L46). Different from `extract_entities` which maps `ValueError` to 400. **The same exception type can mean different HTTP semantics** in different routes — the router is where you make that mapping decision.

> **.NET parallel:** This is the equivalent of enqueuing a background work item on a hosted service / `BackgroundService`, with a thread-safe job dictionary as the progress board. For real production scale you'd swap the in-process executor for Celery, Sidekiq, RQ, or AWS Batch — same API contract, different worker.

### 6.1 The progress reporting trick

Notice [`_update_job`](../Project_4/app/services/ner_service.py#L240-L259):

```python
def _update_job(self, state, status=None, progress=None, current_step=None, ...):
    if status is not None: state.status = status
    if progress is not None: state.progress = progress
    if current_step is not None: state.current_step = current_step
    ...
```

It's a **partial-update helper**. The training routine calls it at meaningful checkpoints:

```python
self._update_job(state, status="running", progress=10, current_step="Validating dataset")
...
self._update_job(state, progress=45, current_step="Building entity patterns")
...
self._update_job(state, progress=80, current_step="Persisting model version")
...
self._update_job(state, status="completed", progress=100, ...)
```

Why this matters: training jobs in production take hours. **Users hate seeing a spinner with no information.** Even fake progress bars (Vercel deploys, npm installs) are better than blackbox waits. Always give your training jobs at least three meaningful checkpoints with descriptive `current_step` strings.

### 6.2 The synchronous escape hatch

`run_async=False` (in [`TrainingJobRequest`](../Project_4/app/models/schemas.py#L73-L78)) skips the executor and runs in-line. Why expose this?

- **Tests.** The test suite can submit a job and assert on the result without polling or sleeping.
- **Debugging.** A developer reproducing a training failure wants the stack trace immediately, not "status=failed, error=KeyError" via polling.

Both modes share the *exact same code path* (`_run_training_job`). That's the test discipline talking — never have a "tested mode" and a "production mode" that differ in the function actually called.

---

## 7. A/B rollout: deterministic traffic splitting

[`_select_version`](../Project_4/app/services/ner_service.py#L262-L278):

```python
if explicit_version:
    return explicit_version, "explicit"

rollout = self.get_rollout_config()
if use_ab_test and rollout.primary_version and rollout.candidate_version:
    routing_key = audience_id or text
    ratio = self._routing_ratio(routing_key)
    if ratio < rollout.candidate_percentage:
        return rollout.candidate_version, "ab_test_candidate"
    return rollout.primary_version, "ab_test_primary"

return self.active_version, "active"
```

And [`_routing_ratio`](../Project_4/app/services/ner_service.py#L280-L283):

```python
digest = sha256(value.encode("utf-8")).hexdigest()[:8]
return int(digest, 16) / 0xFFFFFFFF
```

The crucial property: **same `audience_id` → same ratio → same version**, every time, deterministically. This is called **stable bucketing** and it's what makes A/B testing valid.

Why? If a user's bucket flipped randomly per request, you couldn't measure their experience — you'd be averaging the candidate and primary inside a single user. Worse, the user would see the system behave inconsistently. Stable bucketing means user `42` always gets the candidate (or always doesn't).

The fallback to hashing the *text* (when no `audience_id`) preserves determinism within a single document but doesn't bucket *users* — fine for offline replay, dangerous for live experimentation. Always pass an `audience_id` in production.

The **routing strategy is reported in the response** (`routing_strategy: "ab_test_candidate"`). This is non-negotiable for A/B tests: when comparing metrics across versions you must know which version actually served each request.

> **Limit to know:** SHA-256-based bucketing is uniform but not fancy. For multi-variant experiments, layered tests, or holdout groups, you'd want a real experimentation framework (Statsig, GrowthBook). The principle — *deterministic hash of a stable key* — is the same.

---

## 8. The version selection priority

```
explicit_version  >  A/B rollout  >  active_version
```

Three escape hatches in priority order:

1. **`model_version="custom-v3"`** — power users / debug clients can pin a version. `routing_strategy: "explicit"`.
2. **A/B rollout** — for any client that doesn't pin, the rollout config decides primary vs candidate. `routing_strategy: "ab_test_primary"` or `"ab_test_candidate"`.
3. **`use_ab_test=False`** or no rollout configured → fall through to the active version. `routing_strategy: "active"`.

This priority order is exactly how you want production to behave. Engineers debugging a regression can pin a version. Product can run experiments. End users get the "right answer" by default.

---

## 9. Threading & state safety

The service holds **mutable state** (the registry, the job dict, the model dict) accessed from:

- The FastAPI request worker thread (extracting, listing, activating).
- The training executor threads (writing new versions, mutating job state).

The code uses a `Lock` in two places:

- `_store_document`-style write of a new model into `_models` ([line 209](../Project_4/app/services/ner_service.py#L209-L210)).
- The lock is held only across the dict mutation, not during the slow training itself — *minimise the critical section*.

The `_jobs` dict mutations (status updates) use no lock. Why is that safe? CPython's GIL serialises individual dict assignments, and we never mutate a `TrainingJobState` field from two threads simultaneously — the executor thread writes, the request thread only reads. This is **good enough for in-process** but would not survive a multi-process worker setup. If you ran with `uvicorn --workers 4`, each worker would have its own `_jobs` dict and you'd silently lose status visibility — a pitfall worth flagging if you scale this out.

> **Production fix:** persist job state to Redis/DB. Same as why you'd swap the in-memory vector store in Project 3.

---

## 10. End-to-end request flow examples

### Inference

```
POST /ner/extract  {"text": "...", "audience_id": "user-42"}
  └─ asyncio.to_thread → NERService.extract_entities
        └─ _select_version → ("baseline-v1", "active")  [or candidate, etc.]
        └─ _get_model("baseline-v1").extract(text) → list[ExtractedEntity]
        └─ ExtractResponse(text, model_version, routing_strategy, entities)
```

### Training

```
POST /training/jobs  {"version_name": "v2", "examples": [...], "auto_activate": true}
  └─ NERService.start_training_job
        └─ TrainingJobState(status="queued") → self._jobs[job_id]
        └─ _executor.submit(_run_training_job, ...)  [background thread]
  └─ 202 Accepted with job_id
                                                             [meanwhile...]
[bg thread] _run_training_job
  ├─ progress=10 "Validating dataset"
  ├─ progress=45 "Building entity patterns"
  │     PatternNERModel.from_examples(...)
  ├─ progress=80 "Persisting model version"
  │     model.save(models_directory)  +  self._models[v2] = model
  ├─ if auto_activate: activate_model("v2")  → registry.json updated
  └─ progress=100 status="completed"

GET /training/jobs/{job_id}  →  current snapshot of TrainingJobState
```

---

## 11. Self-quiz

1. Why does the NER API expose **character** offsets instead of **token** offsets at the boundary?
2. What is the role of `models/registry.json` versus the per-version `metadata.json` files?
3. What three pieces of state does `_run_training_job` mutate, and how does the client observe each?
4. Why does the training endpoint return `202 Accepted` instead of `200 OK`?
5. Explain stable bucketing in three sentences. What property must the routing key have?
6. The same `ValueError` raised by `start_training_job` becomes 400, but raised by `get_job_status` becomes 404. Why is that not a contradiction?
7. What happens at startup if the configured `active_version` no longer exists on disk?
8. If you ran `uvicorn --workers 4`, what would silently break? Why?
9. Why does `PatternNERModel.__init__` sort patterns longest-first?
10. Where is the `routing_strategy` reported, and why is reporting it necessary for valid A/B testing?

---

## 12. Hands-on next steps

- **Replace `PatternNERModel` with a HuggingFace token classifier.** Keep the same interface (`extract`, `save`, `load`, `from_examples`). The lifecycle, registry, and rollout code shouldn't need to change at all — that's the design test.
- **Add Server-Sent Events for training progress.** A `GET /training/jobs/{job_id}/stream` endpoint that emits `progress` events instead of forcing the client to poll. (Project 8 has SSE patterns you can crib.)
- **Add holdout traffic.** A third bucket that sees neither primary nor candidate (e.g. a stale baseline) — useful for "is the new model even better than doing nothing different" experiments.
- **Persist jobs.** Move `self._jobs` to SQLite. Now a process restart doesn't orphan running jobs, and `--workers N` works.
- **Add evaluation on training data.** After training, run the new model over a held-out portion of the input examples and compute span-level precision/recall. Store the numbers in `metadata.json` so `/models` returns them. Now activating a worse model is a deliberate choice, not an accident.
- **Add `DELETE /models/{version_name}`** with a guard preventing deletion of the active or rollout-bound version.

---

## 13. .NET parallels table

| Concept here | .NET equivalent |
|---|---|
| `NERService` singleton | `services.AddSingleton<NerService>()` |
| `ThreadPoolExecutor` for jobs | `BackgroundService` / `IHostedService` queue + workers |
| `_jobs` dict + polling endpoint | `IMemoryCache` + status endpoint, or Hangfire jobs |
| `models/registry.json` | EF Core entity in SQLite |
| `PatternNERModel.save/load` | `JsonSerializer.Serialize` to a known directory |
| Pydantic schemas | DTOs + `[Required]` + FluentValidation |
| `Depends(get_ner_service)` | Constructor-injected singleton |
| `lifespan` startup | `Program.cs` startup configuration |

---

## 14. File cheat-sheet

| File | Purpose | Key idea |
|------|---------|----------|
| [app/config.py](../Project_4/app/config.py) | Settings | `models_directory`, `default_model_version`, `max_training_workers` |
| [app/main.py](../Project_4/app/main.py) | App wiring | Lifespan + 5 routers (`health`, `ner`, `training`, `models`, `experiments`) |
| [app/dependencies.py](../Project_4/app/dependencies.py) | DI | One singleton `NERService` |
| [app/models/schemas.py](../Project_4/app/models/schemas.py) | Pydantic | `ExtractRequest/Response`, `TrainingJob*`, `Rollout*`, `ModelVersionInfo` |
| [app/services/pattern_ner.py](../Project_4/app/services/pattern_ner.py) | Model artifact | `PatternNERModel.save/load/from_examples/extract` |
| [app/services/ner_service.py](../Project_4/app/services/ner_service.py) | Lifecycle orchestration | Registry, jobs, A/B routing |
| [app/routers/ner.py](../Project_4/app/routers/ner.py) | Inference | `POST /ner/extract` |
| [app/routers/training.py](../Project_4/app/routers/training.py) | Training | `POST /training/jobs`, `GET /training/jobs/{id}` |
| [app/routers/models.py](../Project_4/app/routers/models.py) | Registry | `GET /models`, `POST /models/activate` |
| [app/routers/experiments.py](../Project_4/app/routers/experiments.py) | Rollout | `POST /experiments/rollout` |
| [app/routers/health.py](../Project_4/app/routers/health.py) | Liveness | Active version + running job count |

---

## 15. The single most important takeaway

> **The model is the smallest, most replaceable component of an ML system. The lifecycle is the system.**
>
> In this project the "model" is a 200-line phrase matcher and the lifecycle is most of the code. That ratio is honest: in real ML platforms, the training framework, registry, rollout, monitoring, and serving infrastructure dominate the codebase. The neural network is a versioned binary blob in a folder.
>
> Design every ML API so the model is one named, swappable artifact behind a stable interface. Then training, A/B testing, rollback, and audit all become *operations on a registry* — boring, observable, and safe.
