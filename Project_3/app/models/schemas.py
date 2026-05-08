"""Pydantic request and response models for the RAG API."""

from pydantic import BaseModel, Field, field_validator


class DocumentUploadResponse(BaseModel):
    """Response returned after a document has been uploaded and indexed."""

    document_id: str = Field(..., description="Generated identifier for the uploaded document.")
    filename: str = Field(..., description="Original filename.")
    content_type: str = Field(..., description="Detected content type.")
    chunks_indexed: int = Field(..., ge=1, description="Number of chunks stored for retrieval.")
    stored_path: str = Field(..., description="Path where the raw upload was saved.")


class DocumentSummary(BaseModel):
    """Metadata summary for an indexed document."""

    document_id: str = Field(..., description="Unique document identifier.")
    filename: str = Field(..., description="Original filename.")
    content_type: str = Field(..., description="Detected content type.")
    chunks_indexed: int = Field(..., ge=1, description="Number of chunks created for the document.")
    uploaded_at: str = Field(..., description="UTC timestamp when the document was indexed.")


class DocumentsResponse(BaseModel):
    """Collection of indexed documents."""

    total_documents: int = Field(..., ge=0, description="Number of indexed documents.")
    documents: list[DocumentSummary] = Field(..., description="Indexed document summaries.")


class QuestionRequest(BaseModel):
    """Request model for asking a grounded question over indexed documents."""

    question: str = Field(..., min_length=3, max_length=2000, description="User question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional number of chunks to retrieve before answer generation.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question must not be blank.")
        return stripped


class SourceCitation(BaseModel):
    """Source metadata for one retrieved chunk."""

    document_id: str = Field(..., description="Source document identifier.")
    source_name: str = Field(..., description="Filename of the source document.")
    chunk_index: int = Field(..., ge=0, description="Chunk number inside the source document.")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized retrieval similarity.")
    excerpt: str = Field(..., description="Short excerpt from the retrieved chunk.")


class AnswerResponse(BaseModel):
    """Response for grounded question answering."""

    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Generated answer grounded in retrieved context.")
    retrieval_count: int = Field(..., ge=0, description="Number of retrieved chunks used.")
    answer_provider: str = Field(..., description="Provider used to generate the answer.")
    embedding_provider: str = Field(..., description="Provider used to generate embeddings.")
    citations: list[SourceCitation] = Field(..., description="Retrieved chunks used as context.")


class HealthCheckResponse(BaseModel):
    """Health response with indexing and provider state."""

    status: str = Field(..., description="Service health status.")
    vector_store_backend: str = Field(..., description="Active vector store backend.")
    embedding_provider: str = Field(..., description="Active embedding provider.")
    answer_provider: str = Field(..., description="Active answer generator provider.")
    indexed_documents: int = Field(..., ge=0, description="Number of indexed documents.")
    indexed_chunks: int = Field(..., ge=0, description="Total number of stored chunks.")