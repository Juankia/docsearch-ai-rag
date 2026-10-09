"""Vector store contract. Implementations must be safe to call from FastAPI workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from docsearch.domain import Chunk, Filters, Metadata, ScoredChunk
from docsearch.embeddings.base import Matrix, Vector


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        """Insert or overwrite chunks keyed by `chunk.chunk_id`."""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove every chunk of a document. Returns the number of chunks removed."""

    @abstractmethod
    def search(self, query: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        """Top-k chunks by cosine similarity, restricted to chunks matching `filters`."""

    @abstractmethod
    def count(self, doc_id: str | None = None) -> int:
        """Number of stored chunks (optionally for a single document)."""

    @abstractmethod
    def list_documents(self) -> dict[str, int]:
        """Map of doc_id -> chunk count."""

    def replace_document(self, doc_id: str, chunks: Sequence[Chunk], vectors: Matrix) -> int:
        """Delete-then-insert so a re-ingested document never leaves stale vectors behind.

        A plain upsert is not enough: if v2 of a document produces fewer chunks than v1,
        chunks `n..m` of v1 would survive and keep matching queries with outdated text.
        Returns the number of chunks that were replaced.
        """
        removed = self.delete_by_doc_id(doc_id)
        if chunks:
            self.upsert(chunks, vectors)
        return removed

    @staticmethod
    def _validate(chunks: Sequence[Chunk], vectors: Matrix, dimension: int) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
        if vectors.ndim != 2 or (len(chunks) and vectors.shape[1] != dimension):
            raise ValueError(f"expected vectors of dimension {dimension}, got {vectors.shape}")


def matches_filters(metadata: Metadata, filters: Filters | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
