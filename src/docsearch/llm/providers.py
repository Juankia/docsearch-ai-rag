"""LLM provider adapters. SDK imports are lazy so unused providers cost nothing."""

from __future__ import annotations

from typing import Any

import httpx

from docsearch.errors import ProviderError


class OpenAIChat:
    def __init__(self, model: str, api_key: str, temperature: float, max_tokens: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderError("Install the 'openai' extra: pip install '.[openai]'") from exc
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        self._client: Any = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return str(resp.choices[0].message.content or "")


class AnthropicChat:
    """Claude via the official `anthropic` SDK (Messages API)."""

    def __init__(self, model: str, api_key: str, max_tokens: int) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ProviderError(
                "Install the 'anthropic' extra: pip install '.[anthropic]'"
            ) from exc
        # With no explicit key the SDK resolves credentials from the environment.
        self._client: Any = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Grounded Q&A over short context does not need deep reasoning.
            output_config={"effort": "low"},
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in resp.content if block.type == "text")


class OllamaChat:
    """Local models through Ollama's `/api/chat` endpoint (no API key needed)."""

    def __init__(self, model: str, base_url: str, temperature: float, timeout: float = 120.0):
        self._model = model
        self._temperature = temperature
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama chat request failed: {exc}") from exc
        return str(resp.json()["message"]["content"])
