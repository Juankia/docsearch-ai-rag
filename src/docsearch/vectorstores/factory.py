from __future__ import annotations

from docsearch.config import Settings
from docsearch.vectorstores.base import VectorStore
from docsearch.vectorstores.memory import InMemoryVectorStore


def build_vector_store(settings: Settings, dimension: int) -> VectorStore:
    match settings.vector_store:
        case "memory":
            return InMemoryVectorStore(dimension=dimension)
        case "milvus":
            from docsearch.vectorstores.milvus import MilvusVectorStore

            return MilvusVectorStore(
                dimension=dimension,
                uri=settings.milvus_uri,
                collection=settings.milvus_collection,
                token=settings.milvus_token,
            )
    raise ValueError(f"Unknown vector store: {settings.vector_store}")
