"""Milvus implementation using the `pymilvus.MilvusClient` API.

Schema (one row per chunk):
    id          VARCHAR  primary key, "doc_id:chunk_index"
    vector      FLOAT_VECTOR(dim), HNSW index, COSINE metric
    doc_id      VARCHAR  (scalar index -> fast delete-by-doc and filtering)
    doc_type    VARCHAR
    source      VARCHAR
    chunk_index INT64
    text        VARCHAR
    metadata    JSON     (arbitrary user metadata, filterable via metadata["key"])
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from docsearch.domain import Chunk, Filters, MetadataValue, ScoredChunk
from docsearch.embeddings.base import Matrix, Vector, normalize_rows
from docsearch.errors import ProviderError
from docsearch.vectorstores.base import VectorStore

_SCALAR_FIELDS = {"doc_id", "doc_type", "source", "chunk_index"}
_OUTPUT_FIELDS = ["doc_id", "doc_type", "source", "chunk_index", "text", "metadata"]
_MAX_TEXT = 65_535


def _literal(value: MetadataValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    return json.dumps(str(value))  # double-quoted with escapes


def build_filter_expr(filters: Filters | None) -> str:
    """Translate equality filters into a Milvus boolean expression.

    Known scalar columns are filtered directly (indexed); anything else goes through
    the JSON `metadata` column.
    """
    if not filters:
        return ""
    clauses: list[str] = []
    for key, expected in sorted(filters.items()):
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid filter key: {key!r}")
        field = key if key in _SCALAR_FIELDS else f'metadata["{key}"]'
        if isinstance(expected, list):
            values = ", ".join(_literal(v) for v in expected)
            clauses.append(f"{field} in [{values}]")
        else:
            clauses.append(f"{field} == {_literal(expected)}")
    return " and ".join(clauses)


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        dimension: int,
        uri: str = "http://localhost:19530",
        collection: str = "docsearch_chunks",
        token: str = "",
        client: Any | None = None,
    ) -> None:
        self._dimension = dimension
        self._collection = collection
        if client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover - optional extra
                raise ProviderError("Install the 'milvus' extra: pip install '.[milvus]'") from exc
            client = MilvusClient(uri=uri, token=token or None)
        self._client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self._collection):
            return
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dimension)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=_MAX_TEXT)
        schema.add_field("metadata", DataType.JSON)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(field_name="doc_id", index_type="INVERTED")
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            # Strong consistency: a query right after re-ingestion must not see
            # vectors that were just deleted. Trade a few ms of latency for correctness.
            consistency_level="Strong",
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Matrix) -> None:
        self._validate(chunks, vectors, self._dimension)
        if not chunks:
            return
        vectors = normalize_rows(vectors)
        rows = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            extra = {k: v for k, v in chunk.metadata.items() if k not in _SCALAR_FIELDS}
            rows.append(
                {
                    "id": chunk.chunk_id,
                    "vector": vec.tolist(),
                    "doc_id": chunk.doc_id,
                    "doc_type": str(chunk.metadata.get("doc_type", "")),
                    "source": str(chunk.metadata.get("source", "")),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text[:_MAX_TEXT],
                    "metadata": extra,
                }
            )
        self._client.upsert(collection_name=self._collection, data=rows)

    def delete_by_doc_id(self, doc_id: str) -> int:
        expr = build_filter_expr({"doc_id": doc_id})
        result = self._client.delete(collection_name=self._collection, filter=expr)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return int(getattr(result, "delete_count", 0) or 0)

    def search(self, query: Vector, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        q = normalize_rows(query)[0]
        results = self._client.search(
            collection_name=self._collection,
            data=[q.tolist()],
            limit=k,
            filter=build_filter_expr(filters),
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, k * 4)}},
        )
        hits = results[0] if results else []
        return [self._to_scored(hit) for hit in hits]

    @staticmethod
    def _to_scored(hit: dict[str, Any]) -> ScoredChunk:
        entity = hit.get("entity", hit)
        metadata = dict(entity.get("metadata") or {})
        metadata.update(
            doc_id=entity["doc_id"],
            doc_type=entity["doc_type"],
            source=entity["source"],
            chunk_index=int(entity["chunk_index"]),
        )
        chunk = Chunk(
            doc_id=entity["doc_id"],
            chunk_index=int(entity["chunk_index"]),
            text=entity["text"],
            metadata=metadata,
        )
        # With metric COSINE, Milvus returns similarity in `distance` (higher = closer).
        return ScoredChunk(chunk=chunk, score=float(hit["distance"]))

    def count(self, doc_id: str | None = None) -> int:
        expr = build_filter_expr({"doc_id": doc_id}) if doc_id else ""
        rows = self._client.query(
            collection_name=self._collection, filter=expr, output_fields=["count(*)"]
        )
        return int(rows[0]["count(*)"]) if rows else 0

    def list_documents(self) -> dict[str, int]:
        out: dict[str, int] = {}
        iterator = self._client.query_iterator(
            collection_name=self._collection, batch_size=1000, filter="", output_fields=["doc_id"]
        )
        try:
            while batch := iterator.next():
                for row in batch:
                    out[row["doc_id"]] = out.get(row["doc_id"], 0) + 1
        finally:
            iterator.close()
        return out
