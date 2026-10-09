from __future__ import annotations

from docsearch.config import Settings
from docsearch.llm.base import LLMClient

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
    "ollama": "llama3.1",
}


def build_llm(settings: Settings) -> LLMClient | None:
    """Return the configured LLM, or None for the offline extractive answerer."""
    provider = settings.llm_provider
    if provider == "extractive":
        return None
    model = settings.llm_model or DEFAULT_MODELS[provider]
    if provider == "openai":
        from docsearch.llm.providers import OpenAIChat

        return OpenAIChat(
            model, settings.openai_api_key, settings.llm_temperature, settings.llm_max_tokens
        )
    if provider == "anthropic":
        from docsearch.llm.providers import AnthropicChat

        return AnthropicChat(model, settings.anthropic_api_key, settings.llm_max_tokens)
    if provider == "ollama":
        from docsearch.llm.providers import OllamaChat

        return OllamaChat(model, settings.ollama_base_url, settings.llm_temperature)
    raise ValueError(f"Unknown LLM provider: {provider}")
