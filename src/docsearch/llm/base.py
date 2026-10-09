"""Minimal chat-completion interface shared by every LLM provider."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, user: str) -> str:
        """Return the assistant's text reply for a single-turn prompt."""
        ...
