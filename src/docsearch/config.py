"""Application settings, loaded from environment variables / a `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docsearch.domain import DocType


class ChunkingConfig(BaseModel):
    """Chunk size/overlap (in characters) for one document type."""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)


# Defaults tuned for the kind of content each type usually carries:
# - markdown: structured, heading-delimited sections -> medium chunks, split on headings first
# - text: free-flowing prose -> slightly larger chunks with more overlap
# - pdf: extracted text is noisy (hyphenation, page breaks) -> larger chunks, generous overlap
DEFAULT_CHUNKING: dict[DocType, ChunkingConfig] = {
    DocType.MARKDOWN: ChunkingConfig(chunk_size=700, chunk_overlap=80),
    DocType.TEXT: ChunkingConfig(chunk_size=800, chunk_overlap=120),
    DocType.PDF: ChunkingConfig(chunk_size=1000, chunk_overlap=150),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings
    embedding_provider: Literal["hashing", "openai", "ollama"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512

    # Answer generation
    llm_provider: Literal["extractive", "openai", "anthropic", "ollama"] = "extractive"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Vector store
    vector_store: Literal["memory", "milvus"] = "memory"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "docsearch_chunks"

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=-1.0, le=1.0)

    # Chunking (per document type)
    chunking: dict[DocType, ChunkingConfig] = Field(default_factory=lambda: dict(DEFAULT_CHUNKING))

    # Provider credentials / endpoints
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    max_upload_bytes: int = 10 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
