# Project 5 — Tutor Walkthrough: Multi-Modal AI (Image + Text)

> A guided tour of [Project_5](../Project_5/) for a junior AI engineer. This is your first project where the model has to **see**, not just read. Concepts: image upload, captioning, visual question answering, and the unifying idea behind it all — **a shared embedding space for text and images**.

---

## 1. What this project actually does

Five capabilities exposed as HTTP endpoints:

1. **Upload** an image (`POST /images`) with optional tags. The bytes are stored on disk, metadata in JSON, and an embedding goes straight into a vector index.
2. **Caption** an image (`POST /images/{id}/caption`). Returns a natural-language description, optionally biased by a prompt.
3. **Visual Question Answering (VQA)** (`POST /images/{id}/vqa`). "What colour is the dog?" → "Brown."
4. **Text-to-image search** (`POST /search/text`). "sunset over a beach" → top-k matching images.
5. **Image-to-image search** (`POST /search/images/{id}`). Find visually similar images.

The service runs **without any neural network** by default — a deterministic hashing provider stands in for BLIP/CLIP. Same as previous projects, that exists so you focus on the *architecture* of multi-modal systems, not on downloading 2 GB of weights.

---

## 2. The single most important idea: shared embedding space

This is the concept that makes "search images by typing words" *possible*.

A model like **CLIP** is trained on hundreds of millions of (image, caption) pairs from the web with a contrastive objective: pull the embedding of *each image* close to the embedding of *its caption*, and push it away from every other caption in the batch. After training:

- "a photo of a golden retriever" and a real photo of a golden retriever land at **similar coordinates**.
- "the stock market crashed" and that same dog photo land **far apart**.

Once text embeddings and image embeddings live in the **same vector space**, cross-modal search is just cosine similarity — the exact same operation you used in Project 3 for RAG. **Cross-modal retrieval is RAG retrieval with a different encoder.**

The hashing provider in [`HashingVisionProvider`](../Project_5/app/services/vision_provider.py#L48-L101) deliberately preserves this property so the rest of the code is honest. Look at [`embed_image`](../Project_5/app/services/vision_provider.py#L73-L82):

```python
text_component = self._token_vector(hint_tokens)         # tokens from tags/filename/caption
image_component = self._image_signature(image_bytes)     # SHA-256 of bytes
combined = [text_component[i] + 0.25 * image_component[i] for i in range(self.dimensions)]
return _normalize(combined)
```

The image and the text contribute to the **same** vector. So `embed_text("dog")` and `embed_image(<dog photo>, hints=["dog"])` will overlap. It's not real understanding, but it's a faithful structural placeholder for what CLIP does.

> **Key insight:** every multi-modal AI architecture you'll meet — vision-language search, text-to-image generation, multi-modal RAG, video Q&A — is built on the same trick: encode each modality into a shared vector space. Internalise it once.

---

## 3. The architecture in one picture

```
                            ┌──────────────────┐
POST /images ───────────────▶│ MultiModalService│
                            │                  │
                            │ ┌──────────────┐ │
                            │ │  ImageStore  │ │   data/images/<id>.png
                            │ │ disk + meta  │ │   data/images/metadata.json
                            │ └──────────────┘ │
                            │                  │
                            │ ┌──────────────┐ │
                            │ │VisionProvider│ │   hashing  ←→  HF/CLIP/BLIP
                            │ └──────────────┘ │
                            │                  │
                            │ ┌──────────────┐ │
                            │ │ VectorIndex  │ │   in-memory cosine
                            │ └──────────────┘ │
                            └──────────────────┘
                                    ▲
POST /search/text ──────────────────┘  (embed query → search index)
POST /search/images/{id} ───────────┘  (lookup vector → search index)
POST /images/{id}/caption ──────────┘  (load bytes → vision.caption → store)
POST /images/{id}/vqa ──────────────┘  (load bytes → vision.answer)
```

Three collaborators, one orchestrator. Familiar pattern from Projects 3 and 4.

---

## 4. Concept-by-concept walkthrough

### 4.1 The vision provider abstraction

[`VisionProvider`](../Project_5/app/services/vision_provider.py#L34-L55) is a `Protocol` (Python's structural-typing answer to interfaces):

```python
class VisionProvider(Protocol):
    name: str
    def embed_text(self, text: str) -> list[float]: ...
    def embed_image(self, image_bytes: bytes, hints: Iterable[str]) -> list[float]: ...
    def caption(self, image_bytes: bytes, hints, prompt) -> str: ...
    def answer(self, image_bytes: bytes, question: str, hints) -> tuple[str, float]: ...
```

Four capabilities. **The multi-modal service knows nothing about which backend is active** — it calls these four methods. The factory [`build_vision_provider`](../Project_5/app/services/vision_provider.py#L154-L172) is the routing point:

- `"hashing"` → `HashingVisionProvider` (default, no deps)
- `"huggingface"` → tries to import `HuggingFaceVisionProvider`; falls back to hashing on `ImportError`

That fallback is the same pattern from earlier projects, with one twist: **swapping providers is a one-line config change**, not a code change. The HF implementation lives in `huggingface_vision.py` (not committed) and gets routed in by the factory only if the optional `transformers`/`torch`/`Pillow` stack is installed.

> **.NET parallel:** Pure interface-driven design. `IVisionProvider` registered in `Program.cs`, swapped via configuration. Same idea.

### 4.2 Why a `Protocol` and not an `ABC`?

Project 3 used `class CacheBackend(ABC)`. Project 5 uses `class VisionProvider(Protocol)`. The difference matters:

- **ABC** → an implementation must explicitly inherit. Strong nominal typing.
- **Protocol** → any class with matching methods/attributes satisfies it. Structural typing (duck typing with type-checker support).

`Protocol` is the right choice when you anticipate a third-party class (HuggingFace's `pipeline`, OpenAI's client) that you cannot make inherit from your ABC. You just describe the *shape* you need, and any compliant object works. Use `Protocol` for **adapter boundaries with the outside world**, `ABC` when you control all implementations.

### 4.3 The hashing fallback: structural fidelity, zero quality

Read `HashingVisionProvider.embed_text`:

```python
def embed_text(self, text: str) -> list[float]:
    return _normalize(self._token_vector(_tokenize(text)))
```

For each token, it picks a stable index via MD5, increments that bucket. Then L2-normalises. **This is not semantic** — "dog" and "puppy" hash to unrelated indices. But:

- "dog" always maps to the same index → deterministic.
- "dog cat" and "cat dog" produce identical vectors → bag-of-words.
- The vector space is shared with `embed_image` (via `_token_vector` on hints), so text and image embeddings are comparable.

Captions and VQA answers are similarly templated — `caption` stitches together the top-5 hint tokens; `answer` checks for question/hint token overlap. Useless for production. **Perfect for tests** because every input has a deterministic, reasonable-shape output.

The same lesson as Project 3's hash embeddings: when you see `hashing` in `/health`, *retrieval and answers are nonsensical content-wise but structurally valid*. Always confirm the production provider is selected before judging quality.

### 4.4 The image store: bytes + metadata, persisted

[`ImageStore`](../Project_5/app/services/image_store.py) does two jobs:

**Bytes on disk** — `save_image` writes raw bytes to `data/images/<uuid>.<ext>`. Extensions are derived from the filename suffix or a content-type lookup table ([`_EXTENSION_MAP`](../Project_5/app/services/image_store.py#L13-L21)). Falls back to `.bin` if neither is recognised.

**Metadata in JSON** — every record's metadata (id, filename, content type, size, timestamp, tags, caption) is appended to `data/images/metadata.json`. The whole file is rewritten on each change, under a `Lock`. That's:

- atomic-enough for a single-process API.
- terrible for high write throughput.
- great for transparency — you can `cat metadata.json` and see exactly what's indexed.

The `Lock` only wraps the dict mutation + file write. Reads (`get`, `list_records`) are unlocked. CPython's GIL makes single dict reads safe, and we never observe a half-written record because we don't yield to other threads inside the critical section.

`update_caption(image_id, caption)` does an in-place mutation of the existing record. This is why captions persist between API calls — running `/images/{id}/caption` twice with different prompts overwrites the stored text, then re-embeds the image (with the new caption now appearing in `_record_hints`), then re-upserts into the vector index. Captions feed back into search quality.

### 4.5 The vector index: small, fast, ready to swap

[`VectorIndex`](../Project_5/app/services/vector_index.py) is dictionary-of-vectors with cosine search. Three observations:

- **Same shape as Chroma's interface** — `upsert`, `remove`, `get`, `search(query, top_k, exclude_keys)`. When you outgrow this, the swap is local: replace `VectorIndex` with a Chroma client. The `MultiModalService` doesn't change.
- **`exclude_keys`** is the small but vital feature for image-to-image search: when finding "similar to image X", you must exclude X itself or it always wins with score 1.0.
- **Linear scan** — every search compares against every stored vector. Fine until you have ~10k images. Beyond that, you need an HNSW index (Chroma, FAISS, Qdrant).

### 4.6 The orchestrator: `MultiModalService`

[`MultiModalService`](../Project_5/app/services/multimodal_service.py) is the conductor. Each public method maps to one endpoint. The interesting ones:

#### Upload flow

```python
record = self.image_store.save_image(...)
embedding = self.vision.embed_image(data, hints=self._record_hints(record))
self.index.upsert(record.image_id, embedding)
```

Three lines, three collaborators. **The order matters**: store first (so we have a stable id), embed second (so the embedding includes hint context derived from the saved metadata), index third (so a search immediately after upload finds the image).

#### Hint construction

[`_record_hints`](../Project_5/app/services/multimodal_service.py#L155-L162):

```python
hints.extend(record.tags)
if record.caption:
    hints.append(record.caption)
if record.filename:
    stem = Path(record.filename).stem.replace("_", " ").replace("-", " ")
    hints.append(stem)
```

The hashing provider has no real understanding of pixels. So we feed it whatever textual hints we have: user-supplied tags, the caption (once generated), and the filename stem (e.g. `golden_retriever_at_park.jpg` → `"golden retriever at park"`). With CLIP, this matters far less — CLIP looks at pixels — but even then, **enriching embeddings with sidecar metadata is a known trick** for improving retrieval. Filenames carry signal.

#### Caption flow with re-embedding

```python
caption_text = self.vision.caption(data, hints=..., prompt=prompt)
updated = self.image_store.update_caption(image_id, caption_text)
embedding = self.vision.embed_image(data, hints=self._record_hints(updated))
self.index.upsert(updated.image_id, embedding)
```

After captioning, **the embedding is recomputed** and the index updated. Why? The new caption is now in the hints, so the embedding should reflect it. If you ever swap to CLIP, you might drop this re-embed (CLIP doesn't use hints) — but conceptually it's correct: any time the metadata that influenced the embedding changes, the embedding should be refreshed.

#### Image-to-image search

```python
query_vector = self.index.get(image_id)
matches = self.index.search(query_vector, top_k=top_k, exclude_keys={image_id})
```

Two lines. We *don't* re-embed the source image — we look up its already-stored vector. This is faster, cheaper, and crucially **identical** to what `embed_image` produced at upload time. Reusing stored vectors for similarity searches is a tiny optimisation that compounds at scale.

#### Re-indexing on startup

[`_reindex_existing_records`](../Project_5/app/services/multimodal_service.py#L194-L201):

```python
for record in self.image_store.list_records():
    data = Path(record.storage_path).read_bytes()
    embedding = self.vision.embed_image(data, hints=self._record_hints(record))
    self.index.upsert(record.image_id, embedding)
```

The vector index is **in-memory**. Image bytes and metadata are **on disk**. So on every startup we rebuild the index from the persistent store. With 100 images this is instant; with a million you'd persist embeddings too (hello, ChromaDB). But notice the upgrade path is just swapping the index — startup re-indexing logic stays clean.

### 4.7 Multipart uploads and the `Form`/`File` dance

[`upload_image`](../Project_5/app/routers/images.py#L51-L74):

```python
async def upload_image(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(default=None),
    service: MultiModalService = Depends(get_multimodal_service),
) -> ImageUploadResponse:
```

Multipart uploads in FastAPI work via `File(...)` for binary parts and `Form(...)` for textual parts. **You cannot mix `multipart/form-data` with a JSON body** — once you accept a `File`, every other field must come from the form. That's why `tags` is a comma-separated or JSON-encoded string parsed inside the handler ([`_parse_tags`](../Project_5/app/routers/images.py#L33-L48)) rather than a Pydantic model.

`_parse_tags` accepts two formats:

- `"red,blue,beach"` — a friendly comma-separated string.
- `'["red","blue","beach"]'` — JSON array literal for clients that prefer structured data.

This dual-format trick is a small UX win for API consumers. It's worth doing for any optional list field in a multipart endpoint.

### 4.8 Async, threads, and binary uploads

Same pattern from Project 3:

- `await file.read()` — async-native FastAPI I/O.
- `await asyncio.to_thread(service.upload_image, ...)` — blocking work (disk write, embedding) goes to a worker thread.

`embed_image` in the hashing provider is fast, but the moment you swap in CLIP (a real neural model), `embed_image` becomes a **multi-second GPU/CPU call**. Putting it on a thread now means you don't have to retrofit the whole pipeline later. **Design for the slow case from the start.**

### 4.9 Error mapping summary

| Where | Trigger | Status |
|---|---|---|
| `upload_image` | Empty file, oversized, non-image content type | 400 |
| `caption_image`, `answer_question`, `get_image`, `search_by_image` | `KeyError` (image_id not found) | 404 |
| `_parse_tags` | Invalid JSON in tags | 400 |
| Anywhere | Unexpected `Exception` | 500 (default FastAPI) |

Different exception types per business meaning, mapped to different HTTP statuses by the routers. Same discipline from Project 4.

### 4.10 Where the design will bend at production scale

Read this section as "what to fix when this becomes real":

- **Local disk → object storage.** `ImageStore` should write to S3/Azure Blob/MinIO and return pre-signed URLs in `ImageUploadResponse`. Right now `storage_path` is a server-side path leaking implementation details.
- **In-memory index → vector DB.** `VectorIndex` is fine for dev. Production wants Chroma, Qdrant, or pgvector.
- **JSON metadata → SQL.** `metadata.json` is a single-writer file. Postgres or SQLite is the natural step.
- **Hashing provider → CLIP/BLIP.** Real understanding of pixels. The factory is already wired.
- **Single-process state → distributed.** Once you scale `--workers > 1`, the in-memory `VectorIndex` and the per-worker metadata cache desync. The fix is the same: move state to external services.

The codebase is structured so each of these is a localised change. That's the design test.

---

## 5. Self-quiz

1. What is a "shared embedding space" and why does it make text→image search possible?
2. The hashing provider produces semantically meaningless embeddings. What property does it preserve so the rest of the code is still meaningful?
3. Why does `Protocol` make sense for `VisionProvider` instead of `ABC`?
4. After captioning an image, the service re-embeds and re-upserts it. Why?
5. Why does `search_by_image` use `exclude_keys={image_id}`?
6. Why is `_reindex_existing_records` called at startup, and when would you stop doing this?
7. Why can't you mix multipart `File` uploads with a JSON request body in FastAPI?
8. Why is filename stem ("golden_retriever_at_park") added to the embedding hints? Would CLIP still benefit from this?
9. Where would you change code to swap in a real CLIP/BLIP provider, and where would you *not* need to?
10. If you ran with `uvicorn --workers 4`, what would silently break? Why?

---

## 6. Hands-on next steps

- **Implement `huggingface_vision.py`.** Use `transformers.pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")` for `caption`, `transformers.AutoModel.from_pretrained("openai/clip-vit-base-patch32")` plus its processor for `embed_text`/`embed_image`, and a VQA pipeline for `answer`. The factory already routes to it.
- **Persist embeddings.** Add an `embedding: list[float]` column to `metadata.json` and skip the startup re-index. Measure cold-start improvement.
- **Replace `VectorIndex` with Chroma.** Same interface; port `upsert`, `search`, `get`, `remove`. Validate with the test suite — if it still passes, your abstraction was honest.
- **Add caption confidence and language detection.** Some captioners return token-level log-probs; expose an aggregate confidence in `CaptionResponse`.
- **Add a `/search/hybrid` endpoint** that combines a text query AND a reference image (weighted sum of two query vectors). This is how real visual product search works.
- **Add an OCR layer.** Run Tesseract on uploaded images and append the OCR text to hints/caption. Now searching "menu pizza" can find a photo of a restaurant menu.

---

## 7. .NET parallels

| Concept here | .NET equivalent |
|---|---|
| `VisionProvider` Protocol | `IVisionProvider` interface |
| `build_vision_provider` factory | DI registration with conditional resolution |
| `ImageStore` | `IFileStore` over local disk / Azure Blob |
| `VectorIndex` | `IVectorIndex` over in-memory or Chroma |
| `MultiModalService` | Application service / orchestrator |
| `UploadFile` + `Form` | `IFormFile` + `[FromForm]` parameters |
| `metadata.json` rewrite | EF Core entity in SQLite |
| `Depends(get_multimodal_service)` | Constructor-injected singleton |

---

## 8. File cheat-sheet

| File | Purpose | Key idea |
|---|---|---|
| [app/config.py](../Project_5/app/config.py) | Settings | `vision_provider`, `embedding_dimensions`, `max_upload_bytes` |
| [app/main.py](../Project_5/app/main.py) | App wiring | Lifespan + 3 routers (`health`, `images`, `search`) |
| [app/dependencies.py](../Project_5/app/dependencies.py) | DI | One singleton `MultiModalService` |
| [app/models/schemas.py](../Project_5/app/models/schemas.py) | Pydantic | `ImageUploadResponse`, `CaptionResponse`, `VQAResponse`, `SearchResponse` |
| [app/services/vision_provider.py](../Project_5/app/services/vision_provider.py) | Vision capability | Protocol + hashing fallback + factory |
| [app/services/image_store.py](../Project_5/app/services/image_store.py) | Persistence | Bytes on disk + JSON metadata index |
| [app/services/vector_index.py](../Project_5/app/services/vector_index.py) | Similarity search | In-memory cosine, `exclude_keys` for self-similarity |
| [app/services/multimodal_service.py](../Project_5/app/services/multimodal_service.py) | Orchestration | Upload, caption, VQA, text/image search, re-index |
| [app/routers/images.py](../Project_5/app/routers/images.py) | Image lifecycle | Multipart upload, list/get, caption, VQA |
| [app/routers/search.py](../Project_5/app/routers/search.py) | Cross-modal search | text→image, image→image |
| [app/routers/health.py](../Project_5/app/routers/health.py) | Health | Reports active provider + image count |

---

## 9. The single most important takeaway

> **Multi-modal AI is single-modal AI plus a shared embedding space.**
>
> Once text encoders and image encoders agree on the same coordinate system, every cross-modal capability — search, retrieval, alignment, multi-modal RAG — collapses into operations you already know: cosine similarity and top-k. The hard part is not the search. The hard part is the encoders.
>
> Design every multi-modal API around the abstraction "encode → store vector → search vector" with the encoders as swappable providers. Then upgrading from a hashing fallback to CLIP, or from CLIP to whatever 2030 brings, is a configuration change — not a rewrite.
