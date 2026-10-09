"""Split documents into overlapping chunks with per-document-type settings."""

from __future__ import annotations

from collections.abc import Mapping

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docsearch.config import DEFAULT_CHUNKING, ChunkingConfig
from docsearch.domain import Chunk, DocType, LoadedDocument

# Separator priority per type. Markdown splits on headings first so a chunk rarely
# straddles two sections; prose falls back to paragraphs, sentences, then words.
_SEPARATORS: dict[DocType, list[str]] = {
    DocType.MARKDOWN: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    DocType.TEXT: ["\n\n", "\n", ". ", " ", ""],
    DocType.PDF: ["\n\n", "\n", ". ", " ", ""],
}


class Chunker:
    def __init__(self, configs: Mapping[DocType, ChunkingConfig] | None = None) -> None:
        self._configs = {**DEFAULT_CHUNKING, **(configs or {})}
        self._splitters: dict[DocType, RecursiveCharacterTextSplitter] = {}
        for doc_type, cfg in self._configs.items():
            if cfg.chunk_overlap >= cfg.chunk_size:
                raise ValueError(f"{doc_type}: chunk_overlap must be smaller than chunk_size")
            self._splitters[doc_type] = RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                separators=_SEPARATORS[doc_type],
                keep_separator=True,
                strip_whitespace=True,
            )

    def config_for(self, doc_type: DocType) -> ChunkingConfig:
        return self._configs[doc_type]

    def split(self, document: LoadedDocument) -> list[Chunk]:
        pieces = self._splitters[document.doc_type].split_text(document.text)
        chunks: list[Chunk] = []
        for index, text in enumerate(p for p in pieces if p.strip()):
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_index=index,
                    text=text.strip(),
                    metadata={
                        **document.metadata,
                        "doc_id": document.doc_id,
                        "source": document.source,
                        "doc_type": document.doc_type.value,
                        "chunk_index": index,
                    },
                )
            )
        return chunks
