"""Pydantic schemas for the advanced RAG API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Optional document-level metadata used for filtering."""

    author: Optional[str] = None
    category: Optional[str] = None
    date: Optional[str] = Field(default=None, description="ISO date string YYYY-MM-DD.")
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Add a single document to the corpus."""

    document_id: Optional[str] = None
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class IngestResponse(BaseModel):
    """Result of ingesting a document."""

    document_id: str
    chunk_count: int


class DocumentInfo(BaseModel):
    """Document metadata plus chunk count."""

    document_id: str
    title: str
    chunk_count: int
    created_at: str
    metadata: DocumentMetadata


class DocumentListResponse(BaseModel):
    """List of stored documents."""

    count: int
    documents: list[DocumentInfo]


class MetadataFilter(BaseModel):
    """Optional filter applied to retrieval results."""

    author: Optional[str] = None
    category: Optional[str] = None
    tag: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SearchMode(BaseModel):
    """Toggle individual retrieval components."""

    use_vector: bool = True
    use_bm25: bool = True
    use_query_expansion: bool = False
    use_reranker: bool = True


class SearchRequest(BaseModel):
    """Hybrid search request."""

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchMode = Field(default_factory=SearchMode)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)


class RetrievedChunk(BaseModel):
    """A single retrieval result returned to the client."""

    document_id: str
    chunk_id: str
    title: str
    content: str
    score: float
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: DocumentMetadata


class SearchResponse(BaseModel):
    """Hybrid search response."""

    query: str
    expanded_queries: list[str] = Field(default_factory=list)
    count: int
    results: list[RetrievedChunk]


class AskRequest(SearchRequest):
    """Question answering request."""

    pass


class AskResponse(BaseModel):
    """Answer + the chunks used to ground it."""

    query: str
    expanded_queries: list[str] = Field(default_factory=list)
    answer: str
    sources: list[RetrievedChunk]


class EvaluationExample(BaseModel):
    """A single question/expected-answer pair for evaluation."""

    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    relevant_document_id: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Evaluate retrieval + generation quality on a small dataset."""

    examples: list[EvaluationExample] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class EvaluationItemResult(BaseModel):
    """Per-example evaluation metrics."""

    question: str
    answer: str
    answer_relevance: float
    faithfulness: float
    context_precision: float
    retrieved_document_ids: list[str]


class EvaluationResponse(BaseModel):
    """Aggregated evaluation report."""

    count: int
    average_answer_relevance: float
    average_faithfulness: float
    average_context_precision: float
    items: list[EvaluationItemResult]


class HealthCheckResponse(BaseModel):
    """Health-check payload."""

    status: str
    answer_backend: str
    document_count: int
    chunk_count: int
    embedding_dimensions: int
    reranker_enabled: bool
