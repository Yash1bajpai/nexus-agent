from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, RateLimitError
from ..utils.config import ConfigError

# Lazy imports — only load a provider SDK if we actually try to use it
def _make_gemini():
    from .gemini_provider import GeminiProvider
    return GeminiProvider()

def _make_openrouter():
    """OpenRouter free tier — Laguna M.1 free model (no per-minute rate limit)."""
    import os
    from .openai_provider import OpenAIProvider
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if not or_key:
        raise ConfigError("OPENROUTER_API_KEY not set in .env")
    return OpenAIProvider(
        model="poolside/laguna-m.1:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=or_key,
    )

def _make_anthropic():
    from .anthropic_provider import AnthropicProvider
    return AnthropicProvider()

def _make_openai():
    from .openai_provider import OpenAIProvider
    return OpenAIProvider()

FALLBACK_CHAIN = [
    ("gemini",      _make_gemini),
    ("openrouter",  _make_openrouter),   # ← free fallback, no per-minute limit
    ("anthropic",   _make_anthropic),
    ("openai",      _make_openai),
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
            except Exception:
                continue
        raise ConfigError(
            "No valid API keys found or initialization failed for all providers (gemini, anthropic, openai). "
            "Please check your .env file."
        )

    def _switch_next(self, failed_name: str, reason: str = ""):
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
                    try:
                        self._warn_fn(failed_name, name, reason)
                    except TypeError:
                        self._warn_fn(failed_name, name)
                return
            except Exception:
                continue

        raise RateLimitError(
            failed_name,
            f"Provider '{failed_name}' failed ({reason}) and no remaining fallback providers available."
        )

    @property
    def model(self) -> str:
        return getattr(self._current_provider, "model", "unknown")

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        while True:
            try:
                return self._current_provider.complete(messages, tools, system)  # type: ignore
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                failed = self._current_name or "unknown"
                self._switch_next(failed, reason=str(e))

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        try:
            if hasattr(self._current_provider, "stream") and self._current_provider is not None:
                for chunk in self._current_provider.stream(messages, tools, system):
                    yield chunk
                return
        except Exception as e:
            self._switch_next(self._current_name or "unknown", reason=str(e))
        res = self.complete(messages, tools, system)
        if res.text:
            yield res.text
        yield res

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return self._current_provider.format_tool_result_message(tool_call_id, result)  # type: ignore
