# Project 7 — Tutor Walkthrough: Vector Database & Advanced RAG

> A guided tour of [Project_7](../Project_7/) for a junior AI engineer. Project 3 introduced a basic RAG pipeline (embed → retrieve → answer). This project is the **production-grade upgrade**: hybrid retrieval, reciprocal rank fusion, query expansion, re-ranking, metadata filtering, streaming answers, and offline RAG-quality evaluation. Every component you'll meet here has a real-world counterpart used at scale (Qdrant, OpenAI embeddings, cross-encoders, Ragas).

---

## 1. What this project actually does

You expose a corpus-backed Q&A API on port 8006. The pipeline:

1. **Ingest** — chunk a document into overlapping windows, store with metadata, index in BM25 + a vector index.
2. **Search** — given a query, run BM25 *and* dense vector retrieval, optionally with query expansion variants, fuse the rankings via Reciprocal Rank Fusion (RRF), then optionally re-rank the top pool with a cross-encoder-style scorer.
3. **Ask** — same retrieval as search, then generate an answer grounded in the top chunks. Stream variant emits Server-Sent Events.
4. **Evaluate** — run a Q&A dataset through the full pipeline and compute three RAG-quality metrics: answer relevance, faithfulness, context precision.

Every component has an offline default (deterministic, no API key, no network) and a marked production-swap slot. This is how you make a RAG system **testable in CI** while staying honest about what the production stack will look like.

---

## 2. Why this project matters — RAG done right

Naïve RAG (Project 3) is "embed everything, do top-k cosine, stuff into prompt". It works on toy corpora and breaks on real ones. The reasons:

- **Lexical queries** ("error code E_TIMEOUT_42") are murdered by dense retrievers — embeddings smear the rare token across nearby concepts.
- **Semantic queries** ("how do I make this faster?") are murdered by BM25 — keyword overlap is zero with the actual answer.
- **Top-k cosine** has no notion of *cross-encoder relevance* — the embedding similarity says "these chunks are about the same topic", not "this chunk answers this question".
- **No filters** = users wait while the system retrieves over the entire corpus when they only care about docs from author X in 2024.
- **No evaluation** = you can't tell whether your last "improvement" actually improved anything. RAG quality regression is silent and brutal.

This project's structure addresses each of those failures explicitly. Every subsystem maps to a real-world failure mode you'll meet in production.

---

## 3. Architecture in one picture

```
                POST /ask  {query, top_k, mode, filters}
                              │
                              ▼
                ┌──────────────────────────┐
                │   AdvancedRagService     │
                └──────────────────────────┘
                              │
                  _retrieve(request)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   query_expansion     metadata filter         (loop over
   → 1..N variants     → candidate_keys         each variant)
                                                     │
                                       ┌─────────────┴─────────────┐
                                       ▼                           ▼
                                 VectorIndex                  BM25Index
                                 (cosine)                  (Okapi BM25)
                                       │                           │
                                       └────────── fuse ───────────┘
                                                  RRF
                                                  │
                                          top-pool (k*3)
                                                  │
                                        HeuristicReranker
                                          (cross-encoder slot)
                                                  │
                                              top_k chunks
                                                  │
                                       _generate_answer
                                       (sentence selection,
                                        LLM slot)
                                                  │
                                          AskResponse
                                          {answer, sources}
```

Eight components behind one service. That's a lot, but each does *one thing* and only `AdvancedRagService` knows about the others. The orchestrator is the only place where you'd ever change the wiring.

---

## 4. Concept-by-concept walkthrough

### 4.1 Chunking — the silent quality lever

[`chunk_text`](../Project_7/app/services/text_utils.py) splits text into overlapping windows of `chunk_size` words with `overlap` words shared between adjacent chunks. Defaults: 400 words, 80 overlap.

Why overlap? Because answers often straddle chunk boundaries. Without overlap, "the timeout is set in the config — see `MAX_TIMEOUT` in settings.py" might split between "the timeout is set in the" and "config — see `MAX_TIMEOUT`...". Neither chunk alone retrieves on the query "where is MAX_TIMEOUT defined?". Overlap mitigates this by ~20% redundancy.

Chunking strategy is **the single biggest quality lever in RAG**. Word-count chunking (here) is the simplest. Production swaps:
- **Token-based** with the actual model tokenizer (so chunks fit the context window precisely).
- **Sentence-aware** (chunks always end at a sentence boundary).
- **Semantic chunking** (split where embedding similarity drops sharply between paragraphs).
- **Document-structure-aware** (one chunk per markdown heading, code-aware splitter for source code).

If you take only one optimisation away from this project, it's "spend a day tuning your chunker before tuning anything else".

### 4.2 Hashing embeddings — fake but useful

[`HashingEmbeddingProvider`](../Project_7/app/services/embeddings.py) hashes each token to a deterministic index in a 128-dim vector, increments that component, then L2-normalises. Cosine similarity then approximately measures token-set overlap.

**It is not a real semantic embedding.** It cannot tell that "automobile" and "car" are similar. It exists so the project's tests are deterministic and offline.

The production swap is one line in [`AdvancedRagService.__init__`](../Project_7/app/services/rag_service.py#L51):

```python
# self.embedder = HashingEmbeddingProvider(dimensions=embedding_dimensions)
self.embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
# or:
self.embedder = OpenAIEmbedder("text-embedding-3-small")
```

The interface is two methods (`embed`, `embed_many`) returning `list[float]`. Every component downstream is unchanged.

### 4.3 BM25 — why you cannot drop keyword search

[`BM25Index`](../Project_7/app/services/bm25.py) is a pure-Python implementation of Okapi BM25. The formula:

$$\text{score}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

with $\text{idf}(t) = \log\!\left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$.

Three things are doing real work:
- **IDF** down-weights common terms ("the", "of"). Without it, every document looks equally relevant to every query.
- **Term frequency saturation** ($k_1$, default 1.5) — the 10th occurrence of a term in a doc adds far less than the 2nd. Prevents keyword stuffing from dominating.
- **Length normalisation** ($b$, default 0.75) — penalises very long documents that match many terms purely by length.

**Why bother with BM25 when we have embeddings?** Because dense retrieval has a known weakness: rare/exact-match tokens. Error codes, product SKUs, rare proper nouns, function names. BM25 handles these natively because it's anchored to surface forms. Hybrid retrieval (BM25 + vector) is the production answer in 2026 — basically every serious system runs both.

### 4.4 Reciprocal Rank Fusion — the boringly correct way to combine rankings

You have two rankings: BM25 says `[A, B, C, D]`, the vector index says `[B, A, E, C]`. How do you merge them into one ranking?

Naïve approach: combine raw scores. **This breaks** because BM25 scores are unbounded positive reals while cosine similarities are in [-1, 1]. Normalising them is fragile (max-min normalisation explodes when an outlier appears).

[`reciprocal_rank_fusion`](../Project_7/app/services/fusion.py#L24-L36) ignores scores entirely and uses **ranks**:

$$\text{score}(d) = \sum_{\text{list } L} \frac{1}{k + \text{rank}_L(d)}$$

with $k = 60$ (the original paper's value, almost never re-tuned). Properties:
- Bounded, scale-free — works for any number of input rankers.
- Documents in multiple rankings get boosted (they appear in multiple sums).
- Top-of-list dominance: position 1 contributes $1/61$, position 10 contributes $1/70$ — sub-linear decay rewards consensus near the top.

**RRF is the default fusion in production hybrid retrieval.** Memorise the formula. It's three lines of code that consistently outperforms anything more complex.

### 4.5 Query expansion — when the user's wording is wrong

Users write bad queries. "fast db" doesn't textually overlap with "high-throughput SQL", but the user wants the same thing. Query expansion generates *additional* phrasings of the same query and runs all of them through retrieval, fusing the results.

[`expand_query`](../Project_7/app/services/query_expansion.py) produces up to 3 variants:
1. **Original** — always preserved.
2. **Keyword-only** — stop-words stripped. Helps BM25 by removing noise.
3. **Synonym-swapped** — token-level dictionary swaps ("build" → "create").

This is deliberately weak — the production swap is **HyDE (Hypothetical Document Embeddings)**: ask an LLM to *write a hypothetical answer* to the query, then embed *that* and retrieve. Often massively beats query embedding because the hypothetical answer matches the *wording style* of the actual answer in the corpus.

The point: the `_retrieve` loop runs once per variant and lets RRF fuse the results. **Expansion is just "more rankings to fuse"** — no special case in the rest of the pipeline.

### 4.6 Re-ranking — where your top-3 actually gets fixed

After RRF you have a candidate pool (top `k*3`). Re-ranking re-orders that pool with a *more expensive but more accurate* relevance model.

The architectural distinction:

| Stage | Cost per chunk | Cost is paid... | Model type |
|---|---|---|---|
| Retrieval (vector) | One vector dot-product | At index time (embedding) + query time (1 dot-product per chunk) | **Bi-encoder** — query and doc embedded *separately* |
| Re-ranking | One full forward pass | At query time, per chunk | **Cross-encoder** — query and doc concatenated, scored together |

Bi-encoders are cheap and approximate. Cross-encoders are accurate but ~100× slower per pair. **You can't run a cross-encoder over a million docs.** The two-stage pattern (cheap retrieval narrows to a candidate pool, expensive reranker re-orders) is universal.

[`HeuristicReranker`](../Project_7/app/services/fusion.py#L39-L83) is a fake cross-encoder — token overlap + bigram overlap + token density, all heuristic. It exists so the pipeline runs offline. The production swap is a real cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` — small, fast, very good). Same `score(query, text) → float` interface; one-line replacement.

### 4.7 Metadata filtering — pre-filter is the secret to RAG latency

[`_candidate_keys`](../Project_7/app/services/rag_service.py#L249-L279) walks the chunk store and returns the chunk IDs matching the filter (author, category, tag, date range). The vector and BM25 searches then receive `candidate_keys` and **only score those chunks**.

Why pre-filter rather than post-filter? Because retrieval cost scales with the candidate count, and post-filtering throws away work after you paid for it. Production vector DBs (Qdrant, pgvector, Pinecone) all support this pattern: filtered search returns top-k *over the filtered subset*, not top-k followed by filtering.

The **deeper design principle**: a search request should let the user *narrow the haystack before searching*. Categories, dates, tenant IDs, language tags. Latency drops, relevance rises (no more cross-tenant leakage), and your bill drops too.

### 4.8 The retrieve loop — the heart of the service

[`AdvancedRagService._retrieve`](../Project_7/app/services/rag_service.py#L188-L247) is 60 lines you should read line-by-line. The structure:

```python
candidate_keys = self._candidate_keys(request.filters)  # pre-filter
queries = expand_query(...) if mode.use_query_expansion else [query]
ranked_lists = []
for q in queries:
    if mode.use_vector:
        ranked_lists.append(vector_index.search(embed(q), candidate_keys))
    if mode.use_bm25:
        ranked_lists.append(bm25_index.search(q, candidate_keys))
fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
pool = fused[: top_k * 3]
ordered = reranker.rerank(query, [(id, content) for id, _ in pool]) if reranker_enabled else pool
return ordered[:top_k]
```

Three things to notice:

1. **Every retrieval mode is a toggle.** `mode.use_vector`, `mode.use_bm25`, `mode.use_query_expansion`, `mode.use_reranker` — the request controls which subsystems run. This is what makes A/B testing tractable: you can ship one endpoint and let an experiment framework flip toggles.
2. **Score collection happens during retrieval.** `vector_scores` and `bm25_scores` dicts capture the per-chunk scores so the `RetrievedChunk` response can show *which signal flagged this chunk*. Observability for the RAG team.
3. **Empty results are handled at three levels.** No candidate keys after filtering → empty result. No ranked lists (e.g. all toggles off) → empty. Otherwise, the pool may be smaller than `top_k * 3`, and `min(...)` clips it. **Always think about empty cases first** — RAG silently produces nonsense answers when the retrieval is empty, and the answerer needs the "no context" branch (see 4.9).

### 4.9 Answer generation — grounded extraction, not free text

[`_generate_answer`](../Project_7/app/services/rag_service.py#L297-L320) is intentionally dumb:

```python
if not retrieved:
    return "I could not find relevant context in the corpus to answer that question."
# Extract sentences from top chunks that share tokens with the query.
# Stitch up to 3 of them.
return f"Based on '{title}': {joined}"
```

It is **extractive**, not generative. It cannot hallucinate because it returns sentences that literally appear in the retrieved chunks. The production swap calls an LLM with a grounded-answer prompt:

```
You are a question-answering assistant. Answer the user's question using ONLY the
provided context. If the context does not contain the answer, say so.
Context: {top chunks with citations}
Question: {query}
```

Two production-critical bits stay even when you swap in an LLM:

- **The "no context" branch.** Always handle empty retrieval with an explicit refusal. Without it, the LLM will make something up.
- **Sources returned alongside the answer.** `AskResponse.sources` is non-optional. Users (and downstream auditors) need to see which chunks grounded the answer. UI: render the answer with footnote-style citations linking to the chunk metadata.

### 4.10 Streaming — UX that hides latency

[`stream_ask`](../Project_7/app/services/rag_service.py#L141-L154) emits **Server-Sent Events** (SSE) — a long-lived HTTP response with `Content-Type: text/event-stream` and lines like `event: token\ndata: ...\n\n`. The order:

1. `event: sources` — the retrieved chunks (so the UI can show "I'm reading these documents…").
2. `event: token` × N — the answer split into tokens.
3. `event: done` — terminal marker.

Why SSE rather than WebSockets? Because RAG streaming is one-way: server → client. SSE works through corporate proxies, supports auto-reconnect via the browser's `EventSource` API, and is stateless on the server side. WebSockets are the wrong tool when you don't need bidirectional traffic.

Streaming **doesn't make your system faster** — total time is unchanged. It makes the *perceived* latency dramatically better because the user sees output starting in 200ms instead of 4 seconds. For LLM-generated answers (where total generation time can be 10+ seconds), this is non-optional UX.

### 4.11 RAG evaluation — three metrics, three failure modes

[`evaluation.py`](../Project_7/app/services/evaluation.py) implements the three classic RAG quality metrics. Each addresses a *different* failure mode.

**`answer_relevance`** — token-level F1 between generated answer and expected answer.
- Catches: "the answer is technically correct but doesn't address what the user asked".
- Failure example: user asks "how does X work?", model answers "X is awesome" — high faithfulness, low relevance.

**`faithfulness`** — fraction of answer tokens grounded in the retrieved context.
- Catches: **hallucination**. The answer says things not supported by retrieval.
- This is the metric that tells you whether your generator is making things up.

**`context_precision`** — fraction of retrieved chunks that are actually relevant.
- Catches: retrieval pulling junk into the prompt. If you retrieve 10 chunks but only 2 are useful, the LLM gets distracted by 8 irrelevant ones.
- Use either an explicit `relevant_document_id` (gold-standard label) or expected-answer overlap as a proxy.

These three together form the **RAG triad**. The production stack equivalents:
- `ragas` library — same three metrics with LLM-as-judge instead of token overlap.
- TruLens — adds latency and cost tracking.
- DeepEval — pytest-style assertions over RAG behaviour.

**Eval is not optional.** Build the eval harness *before* you ship the system. Every change to chunker, embedder, reranker, prompt is a regression risk. Without eval, you're flying blind.

### 4.12 Persistence + reindexing on startup

[`_reindex_existing`](../Project_7/app/services/rag_service.py#L335-L337) iterates `store.list_chunks()` on startup and re-builds the BM25 + vector indexes from scratch. The corpus is persisted in `data/corpus.json` (documents + chunks); the indexes are *not*, because they're trivially rebuildable.

This is a deliberate trade-off:
- **Pro:** simpler — no index serialisation/migration code, no risk of stale indexes.
- **Pro:** safe — index format changes (new BM25 params) don't require data migration.
- **Con:** slow startup with a large corpus (re-embedding takes time).

In production, you'd persist the indexes too (FAISS dump, Qdrant snapshot, Elasticsearch). But for a project where rebuild is a few hundred ms, this is correct.

### 4.13 The thread-safety pattern

Both [`BM25Index`](../Project_7/app/services/bm25.py) and [`VectorIndex`](../Project_7/app/services/vector_index.py) hold an internal `Lock`. Mutations (upsert, remove) take the lock; reads (search) take it briefly to grab a *snapshot* of the data structures, then compute the score outside the lock.

```python
with self._lock:
    df_snapshot = dict(self._df)
    docs_snapshot = self._docs       # reference, not deep copy — read-only access OK
    ...
    # scoring happens here, still inside lock for simplicity
```

Lock granularity matters in real systems. Holding the lock during scoring (as here) is fine for a few hundred chunks; for production scale you'd snapshot pointers and release the lock first. The lesson: **mutable shared state needs synchronisation**, even in CPython, because thread-switches between dict mutations corrupt iterators.

---

## 5. Worked example — tracing one query

Setup: 3 documents ingested, default `mode={vector, bm25, reranker}`, `top_k=2`.

**Request:** `POST /ask {"query": "how does BM25 score documents?", "top_k": 2}`

1. `_candidate_keys` — no filters, returns all 12 chunk IDs.
2. `expand_query` skipped (default off). `queries = ["how does BM25 score documents?"]`.
3. **Vector retrieval** — embed the query into a 128-dim hash vector, cosine-search against the index. Returns `[(chunk_4, 0.82), (chunk_2, 0.71), (chunk_7, 0.43), ...]`.
4. **BM25 retrieval** — tokenise to `["bm25", "score", "documents"]`, score every chunk. Returns `[(chunk_4, 9.4), (chunk_5, 4.1), (chunk_2, 3.7)]`.
5. **RRF** — fuse the two rankings with `k=60`:
    - chunk_4: $1/61 + 1/61 = 0.0328$
    - chunk_2: $1/62 + 1/63 = 0.0320$
    - chunk_5: $1/62 = 0.0161$
    - chunk_7: $1/63 = 0.0159$
6. **Pool** = top 6 fused.
7. **Rerank** — heuristic scorer evaluates `(query, chunk_content)` for each. chunk_4 stays #1, chunk_2 jumps over chunk_5 because of bigram overlap.
8. **Top 2** — chunk_4 and chunk_2.
9. `_generate_answer` — extract sentences from chunk_4 + chunk_2 that share tokens with the query. Return `"Based on 'BM25 Reference': BM25 scores documents using a saturating term-frequency function..."`.
10. Response: `{query, expanded_queries, answer, sources: [chunk_4, chunk_2]}`. Each source carries `score`, `vector_score`, `bm25_score`, `rerank_score` so the client can see which signals contributed.

---

## 6. Self-quiz

1. Why does hybrid retrieval (BM25 + dense) consistently beat either alone? Give a concrete example of a query that breaks each individually.
2. Explain RRF in one sentence. Why is it preferable to score-based fusion?
3. What is the difference between a **bi-encoder** and a **cross-encoder**, and why is the two-stage retrieve→rerank pipeline necessary?
4. What problem does **chunk overlap** solve? Give a query example.
5. Why is the candidate-key pre-filter applied *before* retrieval rather than after?
6. The streaming endpoint sends `event: sources` *before* any tokens. Why this order, not the other way around?
7. Define **faithfulness** in RAG evaluation. What failure mode does it catch that **answer relevance** misses?
8. The `HashingEmbeddingProvider` cannot tell that "automobile" and "car" are similar. Why is the project still useful for testing the rest of the pipeline?
9. What's the production upgrade path for query expansion? (Two letters.)
10. Why does the service rebuild BM25 + vector indexes on every startup instead of persisting them?

---

## 7. Hands-on next steps

- **Swap embeddings to `sentence-transformers/all-MiniLM-L6-v2`.** One line in `__init__`. Re-run the eval harness; observe answer-relevance lift.
- **Add a real cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Replace `HeuristicReranker` with a class that wraps the model.
- **Implement HyDE query expansion.** Add an `LLMHyDEExpander` that calls an LLM to write a hypothetical answer, returns `[query, hypothetical]`. Compare retrieval quality with and without.
- **Add a Qdrant backend.** Implement `QdrantVectorIndex` with the same `upsert/remove/search(candidate_keys)` interface. Toggle via `VECTOR_BACKEND` env var. The rest of the code is unchanged — that's the point of the abstraction.
- **Wire `ragas` into the `/evaluate` endpoint** as an alternative metric backend. Compare token-overlap metrics vs LLM-as-judge metrics on the same corpus.
- **Add BM25 + vector score logging** to OpenTelemetry. Tag every retrieval with the score breakdown so you can answer "why did this query return that chunk?" in production.
- **Build a small evaluation dataset** (20 question/answer pairs over your favourite open-source docs) and treat it as the regression suite for every change.

---

## 8. .NET parallels

| Concept here | .NET equivalent |
|---|---|
| `AdvancedRagService` | An application service / use-case orchestrator |
| `BM25Index`, `VectorIndex` | Custom in-memory indexes; production = Lucene.NET, Elasticsearch.NET, Qdrant client |
| `HashingEmbeddingProvider` interface | `IEmbeddingProvider`, registered via DI |
| `HeuristicReranker` interface | `IReranker` strategy pattern |
| `SearchMode` toggles | Feature flags via `IConfiguration` |
| `MetadataFilter` pre-filter | EF Core `.Where(...)` clause before vector search |
| RRF | A pure function — no .NET equivalent needed; copy-paste 10 lines |
| `_generate_answer` | A `IAnswerComposer` strategy; production = OpenAI client |
| Streaming SSE | `IAsyncEnumerable<string>` returned from a controller, written via `Response.WriteAsync` |
| `evaluation.py` metrics | A `RagEvaluator` test harness in xUnit |

---

## 9. File cheat-sheet

| File | Purpose | Key idea |
|---|---|---|
| [app/config.py](../Project_7/app/config.py) | Settings | Chunk size, BM25 params, RRF k, reranker toggle, answer backend |
| [app/main.py](../Project_7/app/main.py) | App wiring | Lifespan + 5 routers |
| [app/dependencies.py](../Project_7/app/dependencies.py) | DI | One singleton `AdvancedRagService` |
| [app/models/schemas.py](../Project_7/app/models/schemas.py) | Pydantic | `IngestRequest`, `SearchRequest` (mode + filters), `AskResponse`, `EvaluationResponse` |
| [app/services/text_utils.py](../Project_7/app/services/text_utils.py) | Tokenizer + chunker | Stop-words, overlapping windows |
| [app/services/embeddings.py](../Project_7/app/services/embeddings.py) | Offline embedder | Hashing → dense vector, cosine helper |
| [app/services/bm25.py](../Project_7/app/services/bm25.py) | Keyword index | Okapi BM25 with k1, b |
| [app/services/vector_index.py](../Project_7/app/services/vector_index.py) | Dense index | Cosine similarity over candidate keys |
| [app/services/fusion.py](../Project_7/app/services/fusion.py) | RRF + reranker | Three-line RRF, heuristic cross-encoder stub |
| [app/services/query_expansion.py](../Project_7/app/services/query_expansion.py) | Variants | Original + keyword-only + synonym swap |
| [app/services/corpus_store.py](../Project_7/app/services/corpus_store.py) | Persistence | JSON file under `Lock` for documents + chunks |
| [app/services/evaluation.py](../Project_7/app/services/evaluation.py) | RAG metrics | Answer relevance, faithfulness, context precision |
| [app/services/rag_service.py](../Project_7/app/services/rag_service.py) | Orchestration | Ingest, retrieve, ask, stream, evaluate |
| [app/routers/ask.py](../Project_7/app/routers/ask.py) | Q&A | `POST /ask` and SSE `/ask/stream` |
| [app/routers/search.py](../Project_7/app/routers/search.py) | Hybrid search | `POST /search` |
| [app/routers/documents.py](../Project_7/app/routers/documents.py) | Ingest | `POST /documents`, `GET`, `DELETE /{id}` |
| [app/routers/evaluate.py](../Project_7/app/routers/evaluate.py) | Eval | `POST /evaluate` |
| [app/routers/health.py](../Project_7/app/routers/health.py) | Health | Document/chunk counts, backends |

---

## 10. The single most important takeaway

> **RAG quality is a chain, and the weakest link sets the ceiling.**
>
> A perfect reranker cannot save bad retrieval. Perfect retrieval cannot save a bad chunker. A perfect generator cannot save retrieval that returned irrelevant chunks. And no amount of clever architecture can save you if you don't measure quality.
>
> The advanced RAG playbook is mechanical: hybrid retrieval (BM25 + dense), fuse with RRF, rerank the top pool with a cross-encoder, pre-filter aggressively on metadata, ground the answer in retrieved chunks, return sources, stream for UX, and **evaluate every change against a fixed eval set**.
>
> Every component in this project is a stand-in for a production version. The architecture *is* the product — the specific embedder, reranker, or LLM is a configuration choice that will be different next year. Build the slots correctly, and your system will outlive any single model.
