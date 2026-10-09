from __future__ import annotations

from docsearch.config import Settings
from docsearch.embeddings.base import Embedder
from docsearch.embeddings.hashing import HashingEmbedder


def build_embedder(settings: Settings) -> Embedder:
    match settings.embedding_provider:
        case "hashing":
            return HashingEmbedder(dimension=settings.embedding_dim)
        case "openai":
            from docsearch.embeddings.remote import OpenAIEmbedder

            return OpenAIEmbedder(
                settings.embedding_model, settings.embedding_dim, settings.openai_api_key
            )
        case "ollama":
            from docsearch.embeddings.remote import OllamaEmbedder

            return OllamaEmbedder(
                settings.embedding_model, settings.embedding_dim, settings.ollama_base_url
            )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
