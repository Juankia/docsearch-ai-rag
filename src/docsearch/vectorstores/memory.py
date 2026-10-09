"""In-process vector store backed by a numpy matrix (exact cosine search).

Exact brute-force search is O(N * d) per query, which is perfectly fine up to a few
hundred thousand chunks on one machine and has perfect recall - a good default for
development, tests and small deployments.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from docsearch.domain import Chunk, Filters, ScoredChunk
from docsearch.embeddings.base import Matrix, Vector, normalize_rows
from docsearch.vectorstores.base import VectorStore, matches_filters


class InMemoryVectorStore(VectorStore):
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._matrix: Matrix = np.zeros((0, dimension), dtype=np.float32)
        self._lock = threading.RLock()

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        self._validate(chunks, vectors, self._dimension)
        vectors = normalize_rows(vectors) if len(chunks) else vectors
        with self._lock:
            position = {cid: i for i, cid in enumerate(self._ids)}
            new_rows: list[np.ndarray] = []
            for chunk, vec in zip(chunks, vectors, strict=True):
                cid = chunk.chunk_id
                if cid in position:
                    self._matrix[position[cid]] = vec
                else:
                    position[cid] = len(self._ids)
                    self._ids.append(cid)
                    new_rows.append(vec)
                self._chunks[cid] = chunk
            if new_rows:
                self._matrix = np.vstack([self._matrix, np.vstack(new_rows)])

    def delete_by_doc_id(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, cid in enumerate(self._ids) if self._chunks[cid].doc_id != doc_id]
            removed = len(self._ids) - len(keep)
            if removed:
                for cid in self._ids:
                    if self._chunks[cid].doc_id == doc_id:
                        del self._chunks[cid]
                self._ids = [self._ids[i] for i in keep]
                self._matrix = self._matrix[keep]
            return removed

    def search(self, query: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        with self._lock:
            if not self._ids:
                return []
            q = normalize_rows(query)[0]
            scores = self._matrix @ q
            candidates = [
                i
                for i, cid in enumerate(self._ids)
                if matches_filters(self._chunks[cid].metadata, filters)
            ]
            if not candidates:
                return []
            cand = np.asarray(candidates)
            order = cand[np.argsort(-scores[cand], kind="stable")][:k]
            return [
                ScoredChunk(chunk=self._chunks[self._ids[i]], score=float(scores[i])) for i in order
            ]

    def count(self, doc_id: str | None = None) -> int:
        with self._lock:
            if doc_id is None:
                return len(self._ids)
            return sum(1 for c in self._chunks.values() if c.doc_id == doc_id)

    def list_documents(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for chunk in self._chunks.values():
                out[chunk.doc_id] = out.get(chunk.doc_id, 0) + 1
            return out
