import json

from fastapi.testclient import TestClient

from docsearch.rag.prompting import REFUSAL
from tests.fakes import minimal_pdf

HANDBOOK = (
    b"# Handbook\n\n## Vacation\nEmployees receive 22 days of paid vacation per year.\n\n"
    b"## Expenses\nExpenses must be submitted within 30 days with a receipt.\n"
)


def _upload(client: TestClient, name: str = "handbook.md", body: bytes = HANDBOOK, **form: str):  # type: ignore[no-untyped-def]
    return client.post("/documents", files={"file": (name, body, "text/markdown")}, data=form)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake-llm"
    assert body["documents"] == 0


def test_upload_query_delete_flow(client: TestClient) -> None:
    resp = _upload(client, metadata=json.dumps({"department": "hr"}))
    assert resp.status_code == 201, resp.text
    assert resp.json()["doc_id"] == "handbook"
    assert resp.json()["replaced_chunks"] == 0

    resp = client.post("/query", json={"question": "How many paid vacation days do employees get?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["citations"][0].startswith("handbook:")
    top = body["sources"][0]
    assert top["doc_id"] == "handbook" and top["cited"] is True
    assert 0 < top["score"] <= 1
    assert "22 days" in top["text"]

    assert client.get("/documents").json() == [{"doc_id": "handbook", "chunks": 1}]

    resp = client.delete("/documents/handbook")
    assert resp.status_code == 200
    assert resp.json()["deleted_chunks"] >= 1
    assert client.delete("/documents/handbook").status_code == 404

    resp = client.post("/query", json={"question": "How many paid vacation days?"})
    assert resp.json()["answer"] == REFUSAL
    assert resp.json()["grounded"] is False


def test_reupload_reports_replaced_chunks(client: TestClient) -> None:
    first = _upload(client).json()
    second = _upload(client).json()
    assert second["replaced_chunks"] == first["chunks"]
    assert client.get("/health").json()["chunks"] == second["chunks"]


def test_query_with_filters(client: TestClient) -> None:
    _upload(client, doc_id="hb")
    _upload(client, "travel.txt", b"Vacation travel is booked by the employee.", doc_id="travel")
    resp = client.post(
        "/query", json={"question": "vacation", "filters": {"doc_id": "travel"}, "top_k": 5}
    )
    assert {s["doc_id"] for s in resp.json()["sources"]} == {"travel"}


def test_upload_pdf(client: TestClient) -> None:
    resp = client.post(
        "/documents",
        files={"file": ("refunds.pdf", minimal_pdf("Refunds take five days"), "application/pdf")},
    )
    assert resp.status_code == 201
    assert resp.json()["doc_type"] == "pdf"


def test_upload_validation_errors(client: TestClient) -> None:
    assert _upload(client, "photo.png", b"\x89PNG").status_code == 415
    assert _upload(client, "empty.txt", b"   ").status_code == 422
    assert _upload(client, metadata="not-json").status_code == 422
    assert _upload(client, metadata=json.dumps({"doc_id": "x"})).status_code == 422
    assert _upload(client, metadata=json.dumps({"nested": {"a": 1}})).status_code == 422
    assert _upload(client, doc_id="../etc/passwd").status_code == 422


def test_query_validation(client: TestClient) -> None:
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    too_many = {"question": "valid question", "top_k": 99}
    assert client.post("/query", json=too_many).status_code == 422
