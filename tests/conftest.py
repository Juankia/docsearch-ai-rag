from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docsearch.api.app import create_app
from docsearch.config import Settings
from docsearch.ingestion.chunking import Chunker
from docsearch.rag.generation import LLMGenerator
from docsearch.rag.service import RAGService
from docsearch.vectorstores.memory import InMemoryVectorStore
from tests.fakes import FakeEmbedder, FakeLLM

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=64)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(embedder: FakeEmbedder, fake_llm: FakeLLM) -> RAGService:
    return RAGService(
        embedder=embedder,
        store=InMemoryVectorStore(dimension=embedder.dimension),
        chunker=Chunker(),
        generator=LLMGenerator(fake_llm),
        top_k=3,
        min_score=0.2,
    )


@pytest.fixture
def client(service: RAGService) -> TestClient:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    return TestClient(create_app(service=service, settings=settings))


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES
