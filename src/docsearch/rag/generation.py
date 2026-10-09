"""Answer generation strategies: LLM-backed (grounded prompt) or offline extractive."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from docsearch.domain import ScoredChunk
from docsearch.embeddings.hashing import tokenize
from docsearch.llm.base import LLMClient
from docsearch.rag.prompting import REFUSAL, build_messages


class AnswerGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> str: ...


class LLMGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return self._llm.name

    def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> str:
        system, user = build_messages(question, chunks)
        return self._llm.complete(system, user).strip() or REFUSAL


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveGenerator:
    """No-LLM fallback: returns the retrieved sentences that best overlap the question.

    Useful for local demos and CI where no API key is available. Every sentence it
    returns is copied verbatim from the context, so it cannot hallucinate.
    """

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1) -> None:
        self._max_sentences = max_sentences
        self._min_overlap = min_overlap

    @property
    def name(self) -> str:
        return "extractive"

    def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> str:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, str, str]] = []
        for rank, sc in enumerate(chunks):
            for sentence in _SENTENCE_RE.split(sc.chunk.text):
                clean = sentence.strip().lstrip("#-* ").strip()
                if len(clean) < 20:
                    continue
                overlap = len(q_terms & set(tokenize(clean)))
                if overlap >= self._min_overlap:
                    # Prefer overlap, then retrieval rank.
                    candidates.append((overlap + sc.score, -rank, clean, sc.chunk.chunk_id))
        if not candidates:
            return REFUSAL
        candidates.sort(reverse=True)
        picked = candidates[: self._max_sentences]
        return " ".join(f"{text} [{cid}]" for _, _, text, cid in picked)
