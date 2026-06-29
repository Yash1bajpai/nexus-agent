from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, RateLimitError
from ..utils.config import ConfigError

# Lazy imports — only load a provider SDK if we actually try to use it
def _make_gemini():
    from .gemini_provider import GeminiProvider
    return GeminiProvider()

def _make_anthropic():
    from .anthropic_provider import AnthropicProvider
    return AnthropicProvider()

def _make_openai():
    from .openai_provider import OpenAIProvider
    return OpenAIProvider()

FALLBACK_CHAIN = [
    ("gemini", _make_gemini),
    ("anthropic", _make_anthropic),
    ("openai", _make_openai),
]

class FallbackProvider(BaseProvider):
    """
    Meta-provider that tries providers in fallback order.
    On RateLimitError or ConfigError (missing API key), automatically
    switches to the next provider and logs a warning.
    """

    def __init__(self, start_provider: str = "gemini"):
        self._start = start_provider.lower()
        self._current_name: Optional[str] = None
        self._current_provider: Optional[BaseProvider] = None
        self._warn_fn = None  # injected by app.py for rich output

        # Re-order chain so the preferred provider is first
        names = [n for n, _ in FALLBACK_CHAIN]
        if self._start in names:
            idx = names.index(self._start)
            self._chain = FALLBACK_CHAIN[idx:] + FALLBACK_CHAIN[:idx]
        else:
            self._chain = FALLBACK_CHAIN

        self._init_first()

    def _init_first(self):
        """Try to initialize the first available provider in the chain."""
        for name, factory in self._chain:
            try:
                provider = factory()
                self._current_name = name
                self._current_provider = provider
                return
            except ConfigError:
                continue
        raise ConfigError(
            "No API keys found for any provider (gemini, anthropic, openai). "
            "Please set at least one key in your .env file."
        )

    def _switch_next(self, failed_name: str):
        """Switch to the next available provider after a failure."""
        names = [n for n, _ in self._chain]
        try:
            current_idx = names.index(failed_name)
        except ValueError:
            current_idx = -1

        for name, factory in self._chain[current_idx + 1:]:
            try:
                provider = factory()
                self._current_name = name
                self._current_provider = provider
                if self._warn_fn:
                    self._warn_fn(failed_name, name)
                return
            except ConfigError:
                continue

        raise RateLimitError(
            failed_name,
            f"Rate limit hit and no fallback providers available. "
            f"All remaining providers in chain missing API keys."
        )

    @property
    def model(self) -> str:
        return getattr(self._current_provider, "model", "unknown")

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        while True:
            try:
                return self._current_provider.complete(messages, tools, system)  # type: ignore
            except RateLimitError:
                failed = self._current_name or "unknown"
                self._switch_next(failed)

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        # streams can't easily retry mid-flight, so fall back to complete()
        return self.complete(messages, tools, system)

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return self._current_provider.format_tool_result_message(tool_call_id, result)  # type: ignore
