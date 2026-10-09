"""Embedding interface. Every implementation returns L2-normalized float32 vectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Embed many texts at once; returns an array of shape (len(texts), dimension)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single query; returns an array of shape (dimension,)."""
        ...


def normalize_rows(matrix: NDArray[np.floating]) -> Matrix:
    """L2-normalize each row so that dot product == cosine similarity."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = (arr / norms).astype(np.float32)
    return out


def first_row(matrix: Matrix) -> Vector:
    row: Vector = matrix[0]
    return row
