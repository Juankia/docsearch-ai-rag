"""Turn uploaded bytes (PDF, Markdown, plain text) into normalized text."""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import PurePath

from pypdf import PdfReader

from docsearch.domain import DocType, LoadedDocument, Metadata
from docsearch.errors import EmptyDocumentError, UnsupportedDocumentError

_EXTENSIONS: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    ".txt": DocType.TEXT,
    ".text": DocType.TEXT,
}
_CONTENT_TYPES: dict[str, DocType] = {
    "application/pdf": DocType.PDF,
    "text/markdown": DocType.MARKDOWN,
    "text/x-markdown": DocType.MARKDOWN,
    "text/plain": DocType.TEXT,
}


def detect_doc_type(filename: str, content_type: str | None = None) -> DocType:
    """Resolve the document type from the extension, or the MIME type if there is none."""
    suffix = PurePath(filename).suffix.lower()
    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]
    # Only trust the MIME type when the filename has no extension: a `.png` sent as
    # `text/plain` is still a PNG.
    if not suffix and content_type:
        base = content_type.split(";")[0].strip().lower()
        if base in _CONTENT_TYPES:
            return _CONTENT_TYPES[base]
    raise UnsupportedDocumentError(
        f"Unsupported file '{filename}'. Supported extensions: {sorted(_EXTENSIONS)}"
    )


def make_doc_id(filename: str) -> str:
    """Derive a stable, URL-safe id from a filename.

    Stable ids are what make re-ingestion idempotent: uploading `handbook.md` twice
    resolves to the same `doc_id`, so the old vectors are replaced instead of duplicated.
    """
    stem = PurePath(filename).stem
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_stem).strip("-").lower()
    return slug or "document"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises a variety of errors on malformed input
        raise UnsupportedDocumentError(f"Could not parse PDF: {exc}") from exc
    # Re-join words hyphenated across line breaks ("employ-\nees" -> "employees").
    text = "\n\n".join(pages)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)  # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip()


def load_document(
    data: bytes,
    filename: str,
    *,
    content_type: str | None = None,
    doc_id: str | None = None,
    metadata: Metadata | None = None,
) -> LoadedDocument:
    doc_type = detect_doc_type(filename, content_type)
    raw = _extract_pdf(data) if doc_type is DocType.PDF else _decode_text(data)
    text = normalize_text(raw)
    if not text:
        raise EmptyDocumentError(f"No extractable text found in '{filename}'.")
    return LoadedDocument(
        doc_id=doc_id or make_doc_id(filename),
        source=PurePath(filename).name,
        doc_type=doc_type,
        text=text,
        metadata=dict(metadata or {}),
    )
