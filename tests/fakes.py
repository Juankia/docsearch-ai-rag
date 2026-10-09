"""Deterministic test doubles: no network, no API keys, no randomness."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from docsearch.embeddings.base import Matrix, Vector, normalize_rows


class FakeEmbedder:
    """Bag-of-words embedder: each lowercase word hashes to one dimension.

    Two texts sharing words get positive cosine similarity; disjoint texts get ~0.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dimension, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension
            v[idx] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.calls += 1
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return normalize_rows(np.vstack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        return normalize_rows(self._vec(text))[0]


class FakeLLM:
    """Records the prompt and replies with a scripted answer.

    By default it cites the first excerpt id found in the prompt, which mimics a
    well-behaved grounded model.
    """

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake-llm"

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if self.reply is not None:
            return self.reply
        match = re.search(r'<excerpt id="([^"]+)"', user)
        assert match, "prompt should contain at least one excerpt"
        return f"According to the documents, this is the answer [{match.group(1)}]."


def minimal_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF containing `text` (valid enough for pypdf)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
