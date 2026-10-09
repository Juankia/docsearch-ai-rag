from __future__ import annotations

from pydantic import BaseModel, Field

from docsearch.domain import Filters


class DocumentIngested(BaseModel):
    doc_id: str
    source: str
    doc_type: str
    chunks: int
    replaced_chunks: int = Field(
        description="Chunks of a previous version that were deleted before re-indexing."
    )


class DocumentDeleted(BaseModel):
    doc_id: str
    deleted_chunks: int


class DocumentSummary(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: Filters | None = Field(
        default=None,
        description='Equality filters on chunk metadata, e.g. {"doc_type": "markdown"}.',
        examples=[{"doc_id": "employee-handbook"}],
    )


class Source(BaseModel):
    id: str = Field(description="Citation id: `doc_id:chunk_index`.")
    doc_id: str
    chunk_index: int
    source: str
    score: float
    cited: bool
    text: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[str]
    sources: list[Source]
    generator: str | None = None
    latency_ms: float


class Health(BaseModel):
    status: str
    version: str
    vector_store: str
    embedding_provider: str
    llm_provider: str
    documents: int
    chunks: int
