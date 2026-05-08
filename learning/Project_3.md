# Project 3 — Tutor Walkthrough: Basic RAG (Retrieval-Augmented Generation)

> A guided tour of [Project_3](../Project_3/) for a junior AI engineer. This is your first **RAG** system. Read with the source open — every concept here points to a real file and line.

---

## 1. What this project actually does

You upload a `.txt` or `.pdf`. The API splits it into chunks, embeds each chunk into a vector, and stores those vectors. Later you ask a question. The API embeds the question, finds the most similar chunks, hands them to an LLM as **context**, and returns a grounded answer **plus citations** to the chunks that supported it.

That is the entire RAG pattern: **Retrieve relevant context, then Augment the LLM's prompt with it, then Generate**.

---

## 2. Why RAG exists — the single most important idea in this project

LLMs have two weaknesses you must internalise:

1. **They don't know your private data.** ChatGPT can't answer "what does *our internal Q3 onboarding doc* say about device returns?"
2. **They hallucinate.** Asked something they don't know, they invent confident-sounding nonsense.

You could solve (1) by **fine-tuning** the model on your data — but that's expensive, slow, and goes stale every time the data changes. RAG instead **looks up your data at request time** and gives the LLM the exact passages it needs as part of the prompt. This:

- Is cheap (no training).
- Updates instantly when documents change (just re-index).
- Reduces hallucinations because the LLM is told to answer **only from the provided context**.
- Gives you **citations** for free — you know which chunk produced the answer.

If you understand only one thing from this project, understand that. Everything else is plumbing.

---

## 3. The end-to-end picture

```
─── INGESTION (one-time, per document) ─────────────────────────────────
Client ──upload──▶ /documents/upload
                     │
                     ├── DocumentProcessor.process_upload
                     │     ├── save raw bytes to disk
                     │     ├── parse text  (PDF → PyPDF2,  TXT → utf-8)
                     │     └── chunk text  (langchain splitter, fallback to manual)
                     ├── EmbeddingProvider.embed_texts(chunks)   →  list[vector]
                     ├── VectorStore.upsert_chunks(records)      (Chroma or in-memory)
                     └── manifest write  (data/uploads/documents.json)

─── QUERY (every question) ──────────────────────────────────────────────
Client ──ask──▶ /qa/ask
                     │
                     ├── EmbeddingProvider.embed_texts([question]) → query vector
                     ├── VectorStore.query(vector, top_k)          → top-k chunks
                     ├── AnswerGenerator.generate_answer(question, chunks)
                     │     (OpenAI chat with grounded prompt, or extractive fallback)
                     └── citations built from retrieved chunks
```

Each box maps to a file. The whole project is just clean implementations of those nine boxes.

---

## 4. Concept-by-concept walkthrough

### 4.1 Document ingestion: parsing and chunking

[`DocumentProcessor`](../Project_3/app/services/document_processor.py) is the entry point.

**Why parse?** PDFs are not text — they're a layout format. `PyPDF2` extracts the text layer page-by-page. TXT files are just decoded as UTF-8.

**Why chunk?** LLM context windows are finite (e.g. 128k tokens). You can't shove a 500-page PDF into a prompt. More importantly, **embeddings work best on small focused passages**: a single embedding for an entire book would be a meaningless average of every topic. You want one embedding per *idea*.

The defaults — `chunk_size=1000`, `chunk_overlap=200` — are industry common ground:

- ~1000 chars ≈ ~250 tokens ≈ a few paragraphs. Big enough to carry meaning, small enough to be specific.
- 200 chars of **overlap** between consecutive chunks ensures a sentence that straddles a chunk boundary appears whole in at least one chunk. Without overlap, you would shred ideas at arbitrary positions.

**Why a recursive splitter?** Look at [`_chunk_text`](../Project_3/app/services/document_processor.py#L84-L96):

```python
splitter_module = importlib.import_module("langchain_text_splitters")
splitter_type = getattr(splitter_module, "RecursiveCharacterTextSplitter")
```

The `RecursiveCharacterTextSplitter` tries to split on natural boundaries first — paragraphs (`\n\n`), then lines (`\n`), then sentences (`. `), and only falls back to character-level cuts as a last resort. Naive fixed-size cuts break in the middle of words.

If `langchain_text_splitters` isn't installed, [`_manual_chunk_text`](../Project_3/app/services/document_processor.py#L98-L107) does dumb fixed-window chunking with overlap. **The fallback exists so the project runs in a fresh venv with minimal deps.** This pattern — *try the good library, fall back to a working baseline* — recurs throughout the project.

### 4.2 Embeddings: turning text into geometry

[`EmbeddingProvider`](../Project_3/app/services/embedding_provider.py) is the core idea of RAG.

An **embedding** is a list of floats (here ~1500 floats for OpenAI, 96 for the fallback) such that **semantically similar texts have geometrically close vectors**. "I love dogs" and "puppies are great" point in roughly the same direction. "I love dogs" and "the stock market crashed" don't.

You then measure similarity with **cosine similarity** — the angle between two vectors, normalized to `[-1, 1]` (or in this codebase, rescaled to `[0, 1]`).

There are two providers:

**`OpenAIEmbeddingProvider`** — calls the real `text-embedding-3-small` model. Production-quality. Costs money. Requires `OPENAI_API_KEY`.

**`HashEmbeddingProvider`** — a deterministic fake. For each token in the text, it hashes it, picks an index in the vector, and adds a signed magnitude. The vector is then L2-normalised. This is **not a real embedding** — it has no semantic understanding ("dog" and "puppy" hash to unrelated indices). Its only purpose is:

- Same input → same vector (deterministic; tests can assert outputs).
- Different inputs → different vectors (so retrieval doesn't always return the same chunk).
- No external dependency, no API key, no network.

This is the **provider abstraction** — the same Strategy pattern you saw with cache backends in Project 2. The rest of the code doesn't care which provider is active. [`create_embedding_provider`](../Project_3/app/services/embedding_provider.py#L65-L74) is the factory:

```python
if api_key:
    try:
        return OpenAIEmbeddingProvider(...)
    except Exception:
        return HashEmbeddingProvider()
return HashEmbeddingProvider()
```

> **Junior engineer trap:** the fallback provider lets the API *run*, but its retrieval quality is **garbage**. Always check `/health` to see which `embedding_provider` is active. If it says `hash-fallback` in production, you have a config bug, not a working system.

### 4.3 Vector stores: the database for embeddings

[`VectorStore`](../Project_3/app/services/vector_store.py) is the second Strategy abstraction.

A regular SQL database can't help you with "give me the rows whose embedding is closest to *this* vector". You need a **vector index**: a data structure that supports nearest-neighbour search efficiently (typical algorithms: HNSW, IVF, FAISS).

Two implementations:

**`InMemoryVectorStore`** — stores records in a Python list, computes cosine similarity to **every** stored vector on each query (linear scan), sorts, returns top-k. This is `O(n)` per query. Fine for tests and small dev sets. Catastrophic at 10 million chunks.

```python
similarity = float(np.dot(query_vector, embedding) / denominator)
similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))   # [-1,1] → [0,1]
```

**`ChromaVectorStore`** — wraps [ChromaDB](https://www.trychroma.com/), a real vector database with HNSW indexing. It persists to disk (`data/chroma/`) so embeddings survive restarts. It also handles its own metadata storage — notice [`upsert_chunks`](../Project_3/app/services/vector_store.py#L121-L138) passing `metadatas={"document_id":..., "source_name":..., "chunk_index":...}`. Chroma round-trips that metadata back to you in the query response.

Note in `ChromaVectorStore.query`:

```python
similarity = 1.0 / (1.0 + float(distance))
```

Chroma returns L2 **distance** (lower = better). We convert to a similarity score in `[0,1]` (higher = better) so callers see one consistent shape regardless of backend. This is a normalisation pattern — *paper over backend differences in the abstraction layer, not at every call site*.

> **.NET parallel:** `IDistributedCache` exposes the same surface for in-memory, Redis, or SQL Server. Different backends, one interface.

### 4.4 The retrieval step itself

[`RagService.answer_question`](../Project_3/app/services/rag_service.py#L106-L129):

```python
query_embedding = self.embedding_provider.embed_texts([question])[0]
retrieved = self.vector_store.query(query_embedding=query_embedding, top_k=effective_top_k)
answer = self.answer_generator.generate_answer(question=question, chunks=retrieved)
```

**Why embed the question with the *same* provider as the documents?** Because cosine similarity is only meaningful if both vectors live in the same vector space. Embedding documents with OpenAI and the question with the hash provider would produce nonsense — like comparing latitude in degrees to longitude in radians.

**Why `top_k=3`?** Trade-off:

- Too few chunks → the relevant passage might be the 4th match and you miss it.
- Too many chunks → you waste context window on irrelevant text, increase cost, and *dilute* the LLM's attention. Empirically, 3–5 is a strong default for short questions.

`top_k` is also a per-request parameter (`QuestionRequest.top_k`, capped at 10) so power users can tune it.

### 4.5 Answer generation: grounded vs hallucinated

[`AnswerGenerator`](../Project_3/app/services/answer_generator.py) is the third Strategy abstraction. Two implementations again, but the interesting one is the prompt in `OpenAIAnswerGenerator.generate_answer`:

```python
context = "\n\n".join(
    f"[{chunk.source_name}#{chunk.chunk_index}] {chunk.content}"
    for chunk in chunks
)
prompt = (
    "Answer based on the following context. If the answer is not supported by the "
    "context, say that clearly. Cite sources inline using the provided labels.\n\n"
    f"Context:\n{context}\n\nQuestion: {question}"
)
```

Three deliberate choices:

1. **"Answer based on the following context"** — anchors the LLM to your data, not its pre-training knowledge.
2. **"If the answer is not supported by the context, say that clearly"** — the *anti-hallucination* instruction. Without it the LLM will happily fabricate.
3. **`[source_name#chunk_index]` labels** baked into the context — the LLM can cite them inline. This is a cheap, robust citation scheme; no need for the model to "know" how to cite.

`temperature=0.2` keeps the answer deterministic and grounded. RAG with `temperature=0.9` is asking the model to be creative *with your facts* — almost never what you want.

The `ExtractiveAnswerGenerator` fallback is much simpler: it scores sentences by overlap with question terms and stitches the top two together with citation tags. Not as fluent, but **always grounded** (it physically cannot hallucinate — it can only quote retrieved text). For a free-tier dev environment this is fine.

### 4.6 Citations: the trust layer

In production, **users will not trust an LLM answer they can't verify**. [`SourceCitation`](../Project_3/app/models/schemas.py#L51-L58) and [`_build_citation`](../Project_3/app/services/rag_service.py#L143-L150) return:

- `document_id` — which file
- `source_name` — its filename (human-readable)
- `chunk_index` — which chunk inside the file
- `similarity_score` — how confident the retrieval was
- `excerpt` — the first 240 chars so the user can sanity-check without re-fetching

A real frontend would render these as clickable links to a "show source" pane. **Always design RAG APIs with citations from day one.**

### 4.7 The document manifest — why a JSON file?

[`RagService`](../Project_3/app/services/rag_service.py#L34-L57) stores metadata about every uploaded document in `data/uploads/documents.json`. Why a JSON file and not a real DB?

- **Source of truth for "which documents are indexed".** The vector store can answer "which chunks exist?" but not "which logical documents has the user uploaded?".
- **Survives restarts** without a database server.
- **Survives backend swaps** — if you switch from in-memory to Chroma, the manifest still lists the documents (you'd need to re-index, but the *list* is preserved).

Threading: notice the `Lock` and `_store_document` writing the *whole* file atomically:

```python
with self._lock:
    self._documents[entry.document_id] = entry
    serialized = [asdict(item) for item in self._documents.values()]
    self.manifest_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
```

For production scale you'd swap this for a real DB — but the abstraction (`_load_manifest`/`_store_document`) keeps that swap localised.

### 4.8 The shape of `RagService` — orchestration over implementation

Look at [`RagService.__init__`](../Project_3/app/services/rag_service.py#L40-L57). It receives **four collaborators** as constructor parameters — processor, embedding provider, vector store, answer generator. It owns *none* of them. It only **orchestrates** them.

This is the core of the design:

- Each collaborator is an interface (ABC) with two implementations (real + fallback).
- `RagService` is the only place that knows the right *order* — chunk → embed → store, embed → retrieve → generate.
- Tests inject fakes for any/all collaborators. The service logic is exercised without HuggingFace, OpenAI, or Chroma.

> **.NET parallel:** This is textbook constructor injection. `RagService` is your application service; the four parameters are the registered services in `Program.cs`.

### 4.9 Async, file uploads, and `asyncio.to_thread`

In [`documents.py`](../Project_3/app/routers/documents.py#L23-L48):

```python
content = await file.read()
...
return await asyncio.to_thread(
    service.ingest_document,
    filename=file.filename,
    content_type=file.content_type,
    data=content,
)
```

Why two different async patterns?

- `file.read()` is **I/O-bound** and FastAPI's `UploadFile` is async-native — `await` it directly.
- `service.ingest_document` is a **synchronous** function that does CPU-bound work (parsing PDFs, computing embeddings, vector store writes). Calling it directly inside an `async def` would freeze the event loop. `asyncio.to_thread` punts it to a worker thread.

Same rule from Project 2. Same reasoning. Internalise it.

The router also enforces an upload size limit (`max_upload_size_mb=10`) **after** reading the body. For real production you'd want streaming validation to avoid buffering 1 GB into memory before rejecting it — but for a learning project this is fine.

### 4.10 Error mapping

| Where | Trigger | Status |
|-------|---------|--------|
| `DocumentProcessor` | Unsupported extension, empty text, unreadable PDF | `ValueError` → 400 |
| `documents.upload_document` | File too large | 413 |
| `documents.upload_document` | Anything unexpected | 500 |
| `RagService.answer_question` | No documents indexed | `ValueError` → 400 |
| `qa.ask_question` | Anything unexpected | 500 |

The pattern: **business-rule errors** become 4xx (caller's fault); **unexpected exceptions** become 500 (our fault). Never let a stack trace leak — the generic 500 detail says "An unexpected error occurred" with the exception string only.

### 4.11 The provider visibility in `/health`

[`HealthCheckResponse`](../Project_3/app/models/schemas.py#L80-L88) returns:

```json
{
  "status": "healthy",
  "vector_store_backend": "chroma",
  "embedding_provider": "openai",
  "answer_provider": "openai",
  "indexed_documents": 7,
  "indexed_chunks": 142
}
```

This is **observability built into the API**. Without it, "the API is up but answers are weirdly bad" is debugged by reading code. With it, one curl tells you "ah, embedding_provider is `hash-fallback`, the API key didn't load". Spend the five lines of code. Always.

---

## 5. The mental model you must walk away with

RAG = **three pluggable components plus an orchestrator**:

1. **Embedder** — text → vector (semantic geometry).
2. **Retriever** — vector + corpus → top-k chunks (similarity search).
3. **Generator** — question + chunks → grounded answer (LLM with strict prompt).
4. **Orchestrator** (`RagService`) — wires the three together with citations and manifest.

Every "advanced" RAG technique you'll meet later — re-ranking, hybrid (keyword + vector) search, query rewriting, multi-hop, contextual compression — is a *modification of one of those three components*. Project 7 (Advanced RAG) is exactly that: same skeleton, smarter components.

---

## 6. The .NET parallels table

| Concept here | .NET / ASP.NET equivalent |
|--------------|---------------------------|
| `EmbeddingProvider` ABC + factory | `IEmbeddingClient` interface + DI registration |
| `VectorStore` ABC | `IVectorStore` (think `IDistributedCache` for vectors) |
| `RagService` | Application/orchestration service |
| `Depends(get_rag_service)` | Constructor-injected singleton |
| `Pydantic` models | DTOs with `[Required]`, `[StringLength]`, FluentValidation |
| `lifespan` | `IHostedService` / `Program.cs` startup |
| `BaseSettings` | `IOptions<T>` + `appsettings.json` |
| `APIRouter` | Controller with `[Route]` |
| `UploadFile` | `IFormFile` |
| `documents.json` manifest | EF Core entity in a SQLite file |

---

## 7. Self-quiz (answer without looking)

1. Why do we chunk documents? Why with overlap?
2. Why must question and document embeddings come from the same provider?
3. What does cosine similarity actually measure, and why is it the right metric here?
4. Why does the system have a "hash" embedding fallback that produces nonsensical retrieval? What is its purpose?
5. What three things does the OpenAI prompt do to **prevent hallucination**?
6. Why is the `ExtractiveAnswerGenerator` literally incapable of hallucinating?
7. What is `top_k` and why is bigger not better?
8. Where does the API enforce the upload size limit, and what's a more robust alternative for very large files?
9. Why is there both a vector store **and** a `documents.json` manifest? What does each own?
10. If `/health` shows `embedding_provider: "hash-fallback"` in production, what is broken and where would you look?

---

## 8. Hands-on next steps

- **Add a re-ranker.** After `vector_store.query` returns top-k=10, run a cross-encoder (e.g. `sentence-transformers/ms-marco-MiniLM-L-6-v2`) to re-score the chunks against the question and keep the top-3. Retrieval quality often jumps significantly.
- **Add a "no answer" guardrail.** If the highest similarity score in `retrieved` is below e.g. `0.4`, short-circuit and return "I don't have information about this in the indexed documents" without calling the LLM. Stops cost burn and forced hallucination on out-of-scope questions.
- **Replace JSON manifest with SQLite.** Same interface (`_load_manifest`, `_store_document`). Add an `is_deleted` column and a `DELETE /documents/{id}` endpoint that soft-deletes manifest entries and removes chunks from the vector store.
- **Add chunk metadata filters.** Let `/qa/ask` accept `document_ids: list[str]` and pass it to Chroma's `where={"document_id": {"$in": [...]}}`. Now users can ask questions scoped to specific documents.
- **Add an evaluation harness.** Hand-write 20 (question, expected_excerpt) pairs. Score: did `expected_excerpt` appear in any retrieved chunk? This is your **retrieval recall** number — the single most useful metric in early RAG.

---

## 9. File cheat-sheet

| File | Purpose | Key idea |
|------|---------|----------|
| [app/config.py](../Project_3/app/config.py) | Settings | Provider toggles + chunking knobs in one place |
| [app/main.py](../Project_3/app/main.py) | App wiring | `lifespan` initialises the RAG service once |
| [app/dependencies.py](../Project_3/app/dependencies.py) | DI container | One singleton `RagService`, four collaborators wired in |
| [app/models/schemas.py](../Project_3/app/models/schemas.py) | Pydantic contracts | `QuestionRequest`, `AnswerResponse` with `SourceCitation[]` |
| [app/services/document_processor.py](../Project_3/app/services/document_processor.py) | Parse + chunk | Recursive splitter with manual fallback |
| [app/services/embedding_provider.py](../Project_3/app/services/embedding_provider.py) | Text → vector | OpenAI real, hash fallback |
| [app/services/vector_store.py](../Project_3/app/services/vector_store.py) | Vector DB | Chroma persistent, in-memory fallback; cosine similarity |
| [app/services/answer_generator.py](../Project_3/app/services/answer_generator.py) | LLM answer | OpenAI grounded prompt, extractive fallback |
| [app/services/rag_service.py](../Project_3/app/services/rag_service.py) | Orchestration | Ingest, retrieve, answer, manifest, citations |
| [app/routers/documents.py](../Project_3/app/routers/documents.py) | Upload + list | Size limit, async file read, threadpool ingest |
| [app/routers/qa.py](../Project_3/app/routers/qa.py) | Ask | Single endpoint, threadpool answer |
| [app/routers/health.py](../Project_3/app/routers/health.py) | Health | Exposes which providers are active |

---

## 10. The single most important takeaway

> **RAG is not magic — it's a search engine that pipes its top results into an LLM with strict instructions.**
>
> Most of your engineering effort is *retrieval quality* (chunking, embedding, vector store, re-ranking) and *prompting discipline* (grounding, anti-hallucination, citation tags). The LLM is the easiest, most replaceable component.
>
> When a RAG system gives bad answers, **look at retrieval first**, prompt second, model third. Get this debugging order right and you'll outperform engineers who blame the model.
