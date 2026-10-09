"""Deterministic, offline embedder based on feature hashing.

This is *not* a semantic model. It is a lexical baseline (roughly a hashed TF-IDF
without the IDF) that lets the whole system - API, eval script, tests - run with zero
network access and fully reproducible results. Swap in `openai` or `ollama` for real
semantic embeddings.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from docsearch.embeddings.base import Matrix, Vector, first_row, normalize_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its
    me my of on or our so that the their them then there these this to us was we
    what when where which who why will with you your""".split()
)


def _stem(token: str) -> str:
    # Tiny suffix stripper: good enough to match "days"/"day", "requested"/"request".
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbedder:
    def __init__(self, dimension: int = 512, use_bigrams: bool = True) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._use_bigrams = use_bigrams

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0  # signed hashing reduces collision bias
        return (value >> 1) % self._dimension, sign

    def _embed_one(self, text: str) -> Vector:
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = tokenize(text)
        features = list(tokens)
        if self._use_bigrams:
            features += [f"{a}_{b}" for a, b in pairwise(tokens)]
        counts: dict[str, int] = {}
        for f in features:
            counts[f] = counts.get(f, 0) + 1
        for feature, count in counts.items():
            idx, sign = self._bucket(feature)
            vec[idx] += sign * (1.0 + np.log(count))  # sublinear term frequency
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return normalize_rows(np.vstack([self._embed_one(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        return first_row(normalize_rows(self._embed_one(text)))
