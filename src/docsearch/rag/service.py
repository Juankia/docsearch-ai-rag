"""Application service orchestrating ingestion and question answering."""

from __future__ import annotations

import logging
import time

from docsearch.domain import Answer, Filters, IngestResult, Metadata, ScoredChunk
from docsearch.embeddings.base import Embedder
from docsearch.errors import DocumentNotFoundError
from docsearch.ingestion.chunking import Chunker
from docsearch.ingestion.loaders import load_document
from docsearch.rag.generation import AnswerGenerator
from docsearch.rag.prompting import REFUSAL, extract_citations, is_refusal
from docsearch.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        chunker: Chunker,
        generator: AnswerGenerator,
        top_k: int = 4,
        min_score: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    # ------------------------------------------------------------------ ingestion
    def ingest(
        self,
        data: bytes,
        filename: str,
        *,
        content_type: str | None = None,
        doc_id: str | None = None,
        metadata: Metadata | None = None,
    ) -> IngestResult:
        document = load_document(
            data, filename, content_type=content_type, doc_id=doc_id, metadata=metadata
        )
        chunks = self.chunker.split(document)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        replaced = self.store.replace_document(document.doc_id, chunks, vectors)
        logger.info(
            "ingested doc_id=%s chunks=%d replaced=%d", document.doc_id, len(chunks), replaced
        )
        return IngestResult(
            doc_id=document.doc_id,
            source=document.source,
            doc_type=document.doc_type,
            chunks=len(chunks),
            replaced_chunks=replaced,
        )

    def delete(self, doc_id: str) -> int:
        removed = self.store.delete_by_doc_id(doc_id)
        if removed == 0:
            raise DocumentNotFoundError(f"Document '{doc_id}' is not indexed.")
        return removed

    # ------------------------------------------------------------------ querying
    def retrieve(
        self, question: str, k: int | None = None, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        query_vec = self.embedder.embed_query(question)
        return self.store.search(query_vec, k or self.top_k, filters)

    def answer(
        self, question: str, *, k: int | None = None, filters: Filters | None = None
    ) -> Answer:
        started = time.perf_counter()
        hits = self.retrieve(question, k, filters)
        relevant = [h for h in hits if h.score >= self.min_score]

        if not relevant:
            # Refuse *before* calling the LLM: cheaper, faster and no hallucination risk.
            return Answer(
                answer=REFUSAL,
                grounded=False,
                sources=hits,
                cited=[],
                extra={"reason": "no_relevant_context", "latency_ms": _ms(started)},
            )

        text = self.generator.generate(question, relevant)
        cited = extract_citations(text, allowed=[h.chunk.chunk_id for h in relevant])
        grounded = not is_refusal(text) and bool(cited)
        return Answer(
            answer=text if not is_refusal(text) else REFUSAL,
            grounded=grounded,
            sources=relevant,
            cited=cited,
            extra={"generator": self.generator.name, "latency_ms": _ms(started)},
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
