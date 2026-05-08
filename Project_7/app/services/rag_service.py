"""High-level orchestration: ingestion, hybrid retrieval, answering, eval."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from app.models.schemas import (
    AskResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentMetadata,
    EvaluationItemResult,
    EvaluationRequest,
    EvaluationResponse,
    HealthCheckResponse,
    IngestRequest,
    IngestResponse,
    MetadataFilter,
    RetrievedChunk,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from app.services.bm25 import BM25Index
from app.services.corpus_store import ChunkRecord, CorpusStore
from app.services.embeddings import HashingEmbeddingProvider
from app.services.evaluation import (
    answer_relevance,
    context_precision,
    faithfulness,
)
from app.services.fusion import HeuristicReranker, reciprocal_rank_fusion
from app.services.query_expansion import expand_query
from app.services.text_utils import chunk_text, tokenize
from app.services.vector_index import VectorIndex


class AdvancedRagService:
    """Coordinate corpus storage, hybrid retrieval, and answer generation."""

    def __init__(
        self,
        storage_file: str,
        embedding_dimensions: int,
        chunk_size: int,
        chunk_overlap: int,
        bm25_k1: float,
        bm25_b: float,
        rrf_k: int,
        reranker_enabled: bool,
        answer_backend: str,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rrf_k = rrf_k
        self.reranker_enabled = reranker_enabled
        self.answer_backend_name = answer_backend or "rule_based"

        self.store = CorpusStore(storage_path=storage_file)
        self.embedder = HashingEmbeddingProvider(dimensions=embedding_dimensions)
        self.vector_index = VectorIndex()
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.reranker = HeuristicReranker()

        self._reindex_existing()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, request: IngestRequest) -> IngestResponse:
        chunks = chunk_text(
            text=request.content,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document content produced no chunks.")

        metadata_dict = self._metadata_to_dict(request.metadata)
        document, chunk_records = self.store.add_document(
            title=request.title,
            chunks=chunks,
            metadata=metadata_dict,
            document_id=request.document_id,
        )

        for record in chunk_records:
            self._index_chunk(record)

        return IngestResponse(
            document_id=document.document_id,
            chunk_count=len(chunk_records),
        )

    def list_documents(self) -> DocumentListResponse:
        documents = self.store.list_documents()
        return DocumentListResponse(
            count=len(documents),
            documents=[
                DocumentInfo(
                    document_id=document.document_id,
                    title=document.title,
                    chunk_count=len(document.chunk_ids),
                    created_at=document.created_at,
                    metadata=DocumentMetadata(**self._metadata_from_dict(document.metadata)),
                )
                for document in documents
            ],
        )

    def remove_document(self, document_id: str) -> int:
        chunk_ids = self.store.remove_document(document_id)
        for chunk_id in chunk_ids:
            self.vector_index.remove(chunk_id)
            self.bm25_index.remove(chunk_id)
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Search / Ask
    # ------------------------------------------------------------------

    def search(self, request: SearchRequest) -> SearchResponse:
        retrieved, expanded_queries = self._retrieve(request)
        return SearchResponse(
            query=request.query,
            expanded_queries=expanded_queries,
            count=len(retrieved),
            results=retrieved,
        )

    def ask(self, request: SearchRequest) -> AskResponse:
        retrieved, expanded_queries = self._retrieve(request)
        answer = self._generate_answer(request.query, retrieved)
        return AskResponse(
            query=request.query,
            expanded_queries=expanded_queries,
            answer=answer,
            sources=retrieved,
        )

    def stream_ask(self, request: SearchRequest) -> Iterator[str]:
        """Yield Server-Sent Events: first sources, then the answer in chunks."""
        retrieved, expanded_queries = self._retrieve(request)
        sources_payload = AskResponse(
            query=request.query,
            expanded_queries=expanded_queries,
            answer="",
            sources=retrieved,
        ).model_dump_json()
        yield f"event: sources\ndata: {sources_payload}\n\n"

        answer = self._generate_answer(request.query, retrieved)
        for token in answer.split():
            yield f"event: token\ndata: {token}\n\n"
        yield "event: done\ndata: {}\n\n"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        items: list[EvaluationItemResult] = []
        for example in request.examples:
            search_request = SearchRequest(query=example.question, top_k=request.top_k)
            retrieved, _ = self._retrieve(search_request)
            answer = self._generate_answer(example.question, retrieved)

            retrieved_dicts = [chunk.model_dump() for chunk in retrieved]
            ar = answer_relevance(answer, example.expected_answer)
            fa = faithfulness(answer, [chunk.content for chunk in retrieved])
            cp = context_precision(
                example.expected_answer,
                retrieved_dicts,
                relevant_document_id=example.relevant_document_id,
            )

            items.append(
                EvaluationItemResult(
                    question=example.question,
                    answer=answer,
                    answer_relevance=round(ar, 4),
                    faithfulness=round(fa, 4),
                    context_precision=round(cp, 4),
                    retrieved_document_ids=[chunk.document_id for chunk in retrieved],
                )
            )

        return EvaluationResponse(
            count=len(items),
            average_answer_relevance=round(self._mean(item.answer_relevance for item in items), 4),
            average_faithfulness=round(self._mean(item.faithfulness for item in items), 4),
            average_context_precision=round(self._mean(item.context_precision for item in items), 4),
            items=items,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="healthy",
            answer_backend=self.answer_backend_name,
            document_count=self.store.document_count(),
            chunk_count=self.store.chunk_count(),
            embedding_dimensions=self.embedder.dimensions,
            reranker_enabled=self.reranker_enabled,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        request: SearchRequest,
    ) -> tuple[list[RetrievedChunk], list[str]]:
        candidate_keys = self._candidate_keys(request.filters)
        if not candidate_keys:
            return [], [request.query]

        queries = (
            expand_query(request.query, max_variants=3)
            if request.mode.use_query_expansion
            else [request.query]
        )

        ranked_lists: list[list[tuple[str, float]]] = []
        vector_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for query in queries:
            if request.mode.use_vector:
                vector_query = self.embedder.embed(query)
                vector_ranked = self.vector_index.search(vector_query, candidate_keys=candidate_keys)
                if vector_ranked:
                    ranked_lists.append(vector_ranked)
                    for key, score in vector_ranked:
                        vector_scores[key] = max(vector_scores.get(key, 0.0), score)

            if request.mode.use_bm25:
                bm25_ranked = self.bm25_index.search(query, candidate_keys=candidate_keys)
                if bm25_ranked:
                    ranked_lists.append(bm25_ranked)
                    for key, score in bm25_ranked:
                        bm25_scores[key] = max(bm25_scores.get(key, 0.0), score)

        if not ranked_lists:
            return [], queries

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        # Take a generous candidate pool, then optionally re-rank.
        pool_size = min(len(fused), max(request.top_k * 3, request.top_k))
        pool = fused[:pool_size]

        rerank_scores: dict[str, float] = {}
        if self.reranker_enabled and request.mode.use_reranker:
            reranker_input = [
                (key, self.store.get_chunk(key).content) for key, _ in pool
            ]
            reranked = self.reranker.rerank(request.query, reranker_input)
            rerank_scores = dict(reranked)
            ordered_keys = [key for key, _ in reranked]
        else:
            ordered_keys = [key for key, _ in pool]

        ordered_keys = ordered_keys[: request.top_k]
        return (
            [
                self._build_retrieved_chunk(
                    chunk_id=chunk_id,
                    fused_score=dict(fused).get(chunk_id, 0.0),
                    vector_score=vector_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    rerank_score=rerank_scores.get(chunk_id),
                )
                for chunk_id in ordered_keys
            ],
            queries,
        )

    def _candidate_keys(self, filters: MetadataFilter) -> set[str]:
        if not any(
            value is not None
            for value in (
                filters.author,
                filters.category,
                filters.tag,
                filters.date_from,
                filters.date_to,
            )
        ):
            return {chunk.chunk_id for chunk in self.store.list_chunks()}

        keys: set[str] = set()
        for chunk in self.store.list_chunks():
            metadata = chunk.metadata
            if filters.author and metadata.get("author") != filters.author:
                continue
            if filters.category and metadata.get("category") != filters.category:
                continue
            if filters.tag and filters.tag not in metadata.get("tags", []):
                continue
            date = metadata.get("date") or ""
            if filters.date_from and date < filters.date_from:
                continue
            if filters.date_to and date > filters.date_to:
                continue
            keys.add(chunk.chunk_id)
        return keys

    def _build_retrieved_chunk(
        self,
        chunk_id: str,
        fused_score: float,
        vector_score: float | None,
        bm25_score: float | None,
        rerank_score: float | None,
    ) -> RetrievedChunk:
        chunk = self.store.get_chunk(chunk_id)
        primary_score = rerank_score if rerank_score is not None else fused_score
        return RetrievedChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            content=chunk.content,
            score=round(float(primary_score), 6),
            vector_score=None if vector_score is None else round(float(vector_score), 6),
            bm25_score=None if bm25_score is None else round(float(bm25_score), 6),
            rerank_score=None if rerank_score is None else round(float(rerank_score), 6),
            metadata=DocumentMetadata(**self._metadata_from_dict(chunk.metadata)),
        )

    def _generate_answer(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "I could not find relevant context in the corpus to answer that question."

        query_tokens = set(tokenize(query, drop_stopwords=True))
        # Pull the sentences from the top chunks that share tokens with the query.
        selected_sentences: list[str] = []
        for chunk in retrieved:
            for sentence in self._split_sentences(chunk.content):
                sentence_tokens = set(tokenize(sentence, drop_stopwords=True))
                if query_tokens & sentence_tokens:
                    selected_sentences.append(sentence.strip())
                    if len(selected_sentences) >= 3:
                        break
            if len(selected_sentences) >= 3:
                break

        if not selected_sentences:
            selected_sentences = [retrieved[0].content[:300]]

        joined = " ".join(selected_sentences)
        title = retrieved[0].title
        return f"Based on '{title}': {joined}".strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Lightweight sentence splitter; production code would use spaCy/nltk.
        rough = []
        buffer = ""
        for char in text:
            buffer += char
            if char in ".!?":
                rough.append(buffer)
                buffer = ""
        if buffer.strip():
            rough.append(buffer)
        return [item for item in (sentence.strip() for sentence in rough) if item]

    def _index_chunk(self, record: ChunkRecord) -> None:
        embedding = self.embedder.embed(record.content)
        self.vector_index.upsert(record.chunk_id, embedding)
        self.bm25_index.upsert(record.chunk_id, record.content)

    def _reindex_existing(self) -> None:
        for chunk in self.store.list_chunks():
            self._index_chunk(chunk)

    @staticmethod
    def _metadata_to_dict(metadata: DocumentMetadata) -> dict[str, Any]:
        payload = metadata.model_dump()
        # Pydantic stores `extra` under that name; flatten only if needed.
        return payload

    @staticmethod
    def _metadata_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            "author": payload.get("author"),
            "category": payload.get("category"),
            "date": payload.get("date"),
            "tags": list(payload.get("tags", [])),
            "extra": dict(payload.get("extra", {})),
        }
        return cleaned

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        items = list(values)
        if not items:
            return 0.0
        return sum(items) / len(items)
