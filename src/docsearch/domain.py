"""Core domain types shared by every layer. No framework imports here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]
# Equality filters; a list value means "any of" (SQL `IN`).
Filters = dict[str, MetadataValue | list[MetadataValue]]


class DocType(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Raw text extracted from an uploaded file, before chunking."""

    doc_id: str
    source: str
    doc_type: DocType
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    metadata: Metadata = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Stable identifier, also used as the citation label: `doc_id:chunk_index`."""
        return f"{self.doc_id}:{self.chunk_index}"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is better


@dataclass(frozen=True, slots=True)
class IngestResult:
    doc_id: str
    source: str
    doc_type: DocType
    chunks: int
    replaced_chunks: int


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    grounded: bool
    sources: list[ScoredChunk]
    cited: list[str]
    extra: dict[str, Any] = field(default_factory=dict)
