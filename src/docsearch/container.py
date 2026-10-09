"""Composition root: wires concrete implementations from settings."""

from __future__ import annotations

from docsearch.config import Settings
from docsearch.embeddings.factory import build_embedder
from docsearch.ingestion.chunking import Chunker
from docsearch.llm.factory import build_llm
from docsearch.rag.generation import AnswerGenerator, ExtractiveGenerator, LLMGenerator
from docsearch.rag.service import RAGService
from docsearch.vectorstores.factory import build_vector_store


def build_service(settings: Settings) -> RAGService:
    embedder = build_embedder(settings)
    store = build_vector_store(settings, embedder.dimension)
    llm = build_llm(settings)
    generator: AnswerGenerator = LLMGenerator(llm) if llm else ExtractiveGenerator()
    return RAGService(
        embedder=embedder,
        store=store,
        chunker=Chunker(settings.chunking),
        generator=generator,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )
