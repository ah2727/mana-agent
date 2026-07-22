"""Provider-neutral inference connection resolution.

This is deliberately the only place that maps persisted provider credentials
to a runtime transport.  Callers retain the selected provider alongside the
model instead of guessing it from a model ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mana_agent.config.provider_registry import PROVIDERS


@dataclass(frozen=True, slots=True)
class InferenceConnection:
    provider: str
    api_key: str
    base_url: str
    headers: dict[str, str]


class ProviderConfigurationError(ValueError):
    """A selected provider has no usable local configuration."""


def resolve_inference_connection(settings: Any, *, provider: str | None = None, require_api_key: bool = True) -> InferenceConnection:
    provider_id = str(provider or getattr(settings, "mana_ai_provider", "openai") or "openai").strip().lower()
    definition = PROVIDERS.get(provider_id)
    if provider_id == "openrouter":
        api_key = str(getattr(settings, "openrouter_api_key", "") or "")
        base_url = str(getattr(settings, "openrouter_base_url", "") or definition.default_base_url)
        headers = {
            key: value
            for key, value in {
                "HTTP-Referer": str(getattr(settings, "openrouter_http_referer", "") or ""),
                "X-OpenRouter-Title": str(getattr(settings, "openrouter_title", "") or ""),
            }.items()
            if value
        }
    else:
        api_key = str(getattr(settings, "openai_api_key", "") or "")
        base_url = str(getattr(settings, "openai_base_url", "") or definition.default_base_url)
        headers = dict(definition.default_headers)
    if require_api_key and not api_key:
        raise ProviderConfigurationError(
            f"{definition.display_name} authentication is not configured. Set {definition.api_key_env}."
        )
    return InferenceConnection(provider=provider_id, api_key=api_key, base_url=base_url.rstrip("/"), headers=headers)
