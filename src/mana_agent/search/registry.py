from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchProviderDefinition:
    id: str
    display_name: str
    requires_api_key: bool = True
    requires_engine_id: bool = False
    requires_endpoint: bool = False


SEARCH_PROVIDERS: tuple[SearchProviderDefinition, ...] = (
    SearchProviderDefinition("tavily", "Tavily"),
    SearchProviderDefinition("brave", "Brave Search"),
    SearchProviderDefinition("exa", "Exa"),
    SearchProviderDefinition("serpapi", "SerpAPI"),
    SearchProviderDefinition("google_cse", "Google Programmable Search", requires_engine_id=True),
    SearchProviderDefinition("bing", "Bing-compatible search"),
    SearchProviderDefinition("custom", "Custom HTTP provider", requires_api_key=False, requires_endpoint=True),
)


def get_search_provider(provider_id: str) -> SearchProviderDefinition:
    normalized = str(provider_id or "").strip().lower()
    for provider in SEARCH_PROVIDERS:
        if provider.id == normalized:
            return provider
    raise KeyError(f"Unsupported web search provider: {provider_id}")
