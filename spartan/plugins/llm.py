"""Minimal LLM clients returning structured JSON (used by optional plugins only).

* :class:`OllamaClient` - local, offline (default). Uses JSON-schema constrained
  decoding and temperature 0 for deterministic extraction.
* :class:`AzureOpenAIClient` - optional cloud backend, configured purely through
  environment variables (``SPARTAN_AZURE_OPENAI_*``).

Every failure (server down, invalid JSON, timeout) is logged and reported as
``None`` so that callers can fall back to their deterministic logic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from ..config import LlmProvider, Settings, get_settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise information-extraction engine. Answer only with JSON that "
    "follows the requested schema. Never invent values: use an empty string or an "
    "empty list when the information is not present."
)


class LLMClient(Protocol):
    name: str

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None: ...


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{[\s\S]*\}", text or "")
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    log.warning("LLM response is not valid JSON")
    return None


class OllamaClient:
    def __init__(self, settings: Settings):
        import ollama

        self.name = f"ollama:{settings.ollama_model}"
        self._model = settings.ollama_model
        self._client = ollama.Client(host=settings.ollama_host, timeout=settings.llm_timeout)
        self._disabled = False

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        if self._disabled:
            return None
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": 0},
            )
        except Exception as exc:  # connection refused, model missing, timeout...
            log.warning("Ollama request failed (%s); continuing without LLM", exc)
            self._disabled = True
            return None
        return _parse_json(response.message.content)


class AzureOpenAIClient:
    def __init__(self, settings: Settings):
        from openai import AzureOpenAI

        if not (settings.azure_openai_endpoint and settings.azure_openai_key):
            raise RuntimeError("Azure OpenAI selected but SPARTAN_AZURE_OPENAI_ENDPOINT / SPARTAN_AZURE_OPENAI_KEY "
                               "are not set. Add your own endpoint and key to .env (see .env.example).")
        self.name = f"azure:{settings.azure_openai_deployment}"
        self._deployment = settings.azure_openai_deployment
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            timeout=settings.llm_timeout,
        )

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{prompt}\n\nJSON schema:\n{json.dumps(schema)}"},
                ],
            )
        except Exception as exc:
            log.warning("Azure OpenAI request failed (%s); continuing without LLM", exc)
            return None
        return _parse_json(response.choices[0].message.content or "")


def create_llm_client(settings: Settings | None = None, provider: LlmProvider | str | None = None) -> LLMClient | None:
    """Build the configured LLM client, or ``None`` when disabled/unavailable."""
    settings = settings or get_settings()
    provider = LlmProvider(provider) if provider else settings.llm_provider
    if provider is LlmProvider.NONE:
        return None
    try:
        if provider is LlmProvider.OLLAMA:
            return OllamaClient(settings)
        if provider is LlmProvider.AZURE:
            return AzureOpenAIClient(settings)
    except Exception as exc:
        log.warning("LLM provider '%s' unavailable: %s", provider.value, exc)
    return None
