import numpy as np
import pytest

from docsearch.domain import Chunk
from docsearch.vectorstores.base import matches_filters
from docsearch.vectorstores.memory import InMemoryVectorStore
from docsearch.vectorstores.milvus import MilvusVectorStore, build_filter_expr


def _chunk(doc_id: str, idx: int, **meta: str) -> Chunk:
    return Chunk(doc_id, idx, f"{doc_id} chunk {idx}", {"doc_id": doc_id, **meta})


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore(dimension=3)
    s.upsert(
        [_chunk("a", 0, team="hr"), _chunk("a", 1, team="hr"), _chunk("b", 0, team="eng")],
        np.vstack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0)]),
    )
    return s


def test_search_orders_by_cosine(store: InMemoryVectorStore) -> None:
    hits = store.search(_unit(1, 0.1, 0), k=3)
    assert [h.chunk.chunk_id for h in hits] == ["a:0", "b:0", "a:1"]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_with_metadata_filter(store: InMemoryVectorStore) -> None:
    hits = store.search(_unit(1, 0, 0), k=3, filters={"team": "eng"})
    assert [h.chunk.chunk_id for h in hits] == ["b:0"]
    hits = store.search(_unit(1, 0, 0), k=3, filters={"doc_id": ["a", "zzz"]})
    assert {h.chunk.doc_id for h in hits} == {"a"}
    assert store.search(_unit(1, 0, 0), k=3, filters={"team": "sales"}) == []


def test_upsert_overwrites_same_chunk_id(store: InMemoryVectorStore) -> None:
    store.upsert([Chunk("a", 0, "new text", {"doc_id": "a"})], _unit(0, 0, 1).reshape(1, -1))
    assert store.count() == 3
    top = store.search(_unit(0, 0, 1), k=1)[0]
    assert top.chunk.text == "new text"


def test_delete_by_doc_id(store: InMemoryVectorStore) -> None:
    assert store.delete_by_doc_id("a") == 2
    assert store.count() == 1
    assert store.list_documents() == {"b": 1}
    assert store.delete_by_doc_id("a") == 0
    assert all(h.chunk.doc_id == "b" for h in store.search(_unit(1, 0, 0), k=5))


def test_replace_document_removes_stale_chunks(store: InMemoryVectorStore) -> None:
    # v2 of doc "a" has a single chunk; the old chunk a:1 must not survive.
    replaced = store.replace_document("a", [_chunk("a", 0)], _unit(1, 0, 0).reshape(1, -1))
    assert replaced == 2
    assert store.count("a") == 1
    assert "a:1" not in {h.chunk.chunk_id for h in store.search(_unit(0, 1, 0), k=5)}


def test_dimension_mismatch_rejected() -> None:
    s = InMemoryVectorStore(dimension=3)
    with pytest.raises(ValueError):
        s.upsert([_chunk("a", 0)], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        s.upsert([_chunk("a", 0), _chunk("a", 1)], np.ones((1, 3), dtype=np.float32))


def test_empty_store_search() -> None:
    assert InMemoryVectorStore(dimension=3).search(_unit(1, 0, 0), k=3) == []


def test_matches_filters() -> None:
    meta = {"doc_type": "pdf", "year": 2026}
    assert matches_filters(meta, None)
    assert matches_filters(meta, {"doc_type": "pdf", "year": 2026})
    assert not matches_filters(meta, {"doc_type": "markdown"})
    assert matches_filters(meta, {"doc_type": ["pdf", "text"]})


# --------------------------------------------------------------------------- Milvus


def test_build_filter_expr() -> None:
    assert build_filter_expr(None) == ""
    assert build_filter_expr({"doc_id": 'hand"book'}) == 'doc_id == "hand\\"book"'
    expr = build_filter_expr({"team": ["hr", "eng"], "doc_type": "pdf", "year": 2026})
    assert expr == (
        'doc_type == "pdf" and metadata["team"] in ["hr", "eng"] and metadata["year"] == 2026'
    )
    with pytest.raises(ValueError):
        build_filter_expr({"x) or (1==1": "y"})


class FakeMilvusClient:
    """Captures calls so we can assert on the Milvus requests without a server."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []

    def has_collection(self, name: str) -> bool:
        return True

    def upsert(self, collection_name: str, data: list[dict]) -> dict:
        self.calls.append(("upsert", {"n": len(data)}))
        for row in data:
            self.rows[row["id"]] = row
        return {"upsert_count": len(data)}

    def delete(self, collection_name: str, filter: str) -> dict:
        self.calls.append(("delete", {"filter": filter}))
        doc_id = filter.split('"')[1]
        ids = [k for k, r in self.rows.items() if r["doc_id"] == doc_id]
        for k in ids:
            del self.rows[k]
        return {"delete_count": len(ids)}

    def search(self, collection_name: str, data, limit, filter, output_fields, search_params):  # type: ignore[no-untyped-def]
        self.calls.append(("search", {"filter": filter, "limit": limit}))
        q = np.asarray(data[0])
        hits = [
            {"id": r["id"], "distance": float(np.dot(q, r["vector"])), "entity": r}
            for r in self.rows.values()
        ]
        hits.sort(key=lambda h: -h["distance"])
        return [hits[:limit]]


def test_milvus_store_roundtrip_with_fake_client() -> None:
    fake = FakeMilvusClient()
    store = MilvusVectorStore(dimension=3, client=fake)
    chunk = Chunk(
        "a",
        0,
        "hello",
        {"doc_id": "a", "doc_type": "text", "source": "a.txt", "chunk_index": 0, "team": "hr"},
    )
    store.replace_document("a", [chunk], _unit(1, 0, 0).reshape(1, -1))

    assert fake.calls[0] == ("delete", {"filter": 'doc_id == "a"'})
    assert fake.rows["a:0"]["metadata"] == {"team": "hr"}

    hits = store.search(_unit(1, 0, 0), k=2, filters={"team": "hr"})
    assert fake.calls[-1] == ("search", {"filter": 'metadata["team"] == "hr"', "limit": 2})
    assert hits[0].chunk.chunk_id == "a:0"
    assert hits[0].chunk.metadata["team"] == "hr"
    assert hits[0].score == pytest.approx(1.0)
    assert store.delete_by_doc_id("a") == 1
