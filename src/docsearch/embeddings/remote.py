"""Network-backed embedders (OpenAI, Ollama). Imported lazily by the factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from docsearch.embeddings.base import Matrix, Vector, first_row, normalize_rows
from docsearch.errors import ProviderError


class OpenAIEmbedder:
    def __init__(self, model: str, dimension: int, api_key: str, batch_size: int = 128) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError("Install the 'openai' extra: pip install '.[openai]'") from exc
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self._client: Any = OpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            resp = self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dimension
            )
            rows.extend(item.embedding for item in resp.data)
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return normalize_rows(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        return first_row(self.embed_documents([text]))


class OllamaEmbedder:
    """Calls Ollama's `/api/embed` endpoint (e.g. model `nomic-embed-text`, dim 768)."""

    def __init__(self, model: str, dimension: int, base_url: str, timeout: float = 60.0) -> None:
        self._model = model
        self._dimension = dimension
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            resp = self._client.post(
                "/api/embed", json={"model": self._model, "input": list(texts)}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama embedding request failed: {exc}") from exc
        matrix = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        if matrix.shape[1] != self._dimension:
            raise ProviderError(
                f"Ollama returned dim {matrix.shape[1]}, expected EMBEDDING_DIM={self._dimension}"
            )
        return normalize_rows(matrix)

    def embed_query(self, text: str) -> Vector:
        return first_row(self.embed_documents([text]))
