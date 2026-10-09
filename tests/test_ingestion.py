import pytest

from docsearch.domain import DocType
from docsearch.errors import EmptyDocumentError, UnsupportedDocumentError
from docsearch.ingestion.loaders import detect_doc_type, load_document, make_doc_id
from tests.fakes import minimal_pdf


@pytest.mark.parametrize(
    ("filename", "content_type", "expected"),
    [
        ("guide.md", None, DocType.MARKDOWN),
        ("NOTES.TXT", None, DocType.TEXT),
        ("report.pdf", None, DocType.PDF),
        ("blob", "application/pdf", DocType.PDF),
        ("readme", "text/markdown; charset=utf-8", DocType.MARKDOWN),
    ],
)
def test_detect_doc_type(filename: str, content_type: str | None, expected: DocType) -> None:
    assert detect_doc_type(filename, content_type) is expected


def test_detect_doc_type_rejects_unknown() -> None:
    with pytest.raises(UnsupportedDocumentError):
        detect_doc_type("image.png", "image/png")


def test_make_doc_id_is_stable_and_url_safe() -> None:
    assert make_doc_id("Employee Handbook (v2).md") == "employee-handbook-v2"
    assert make_doc_id("Política_Seguridad.txt") == "politica-seguridad"
    assert make_doc_id("???.md") == "document"


def test_load_markdown_normalizes_whitespace() -> None:
    doc = load_document(b"# Title\r\n\r\n\r\n\r\nBody   \r\n", "a.md", metadata={"team": "hr"})
    assert doc.doc_id == "a"
    assert doc.doc_type is DocType.MARKDOWN
    assert doc.text == "# Title\n\nBody"
    assert doc.metadata == {"team": "hr"}


def test_load_text_with_explicit_doc_id_and_bom() -> None:
    doc = load_document("﻿hello world".encode(), "x.txt", doc_id="custom-id")
    assert doc.doc_id == "custom-id"
    assert doc.text == "hello world"


def test_load_pdf_extracts_text() -> None:
    doc = load_document(minimal_pdf("Refunds are processed in five days"), "policy.pdf")
    assert doc.doc_type is DocType.PDF
    assert "Refunds are processed in five days" in doc.text


def test_load_corrupt_pdf_raises() -> None:
    with pytest.raises(UnsupportedDocumentError):
        load_document(b"%PDF-1.4 garbage", "broken.pdf")


def test_empty_document_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        load_document(b"   \n\n  ", "empty.txt")
