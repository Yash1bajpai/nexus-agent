"""Documented synchronous model limits and per-session configuration.

Known-model overrides above a documented maximum emit ``UserWarning`` through
``warnings.warn`` and are clamped. Unknown models have no verified maxima:
``model_capabilities`` returns ``None`` and cloud resolution warns when using the
unverified 32,768/16,384 fallback; this is not a verified endpoint capacity.
"""

from dataclasses import dataclass
import os
import re
import warnings


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_size: int
    max_output_tokens: int


# Exact documented IDs only. Never infer capacity from a family-name prefix.
_MODEL_LIMITS: dict[tuple[str, str], ModelLimits] = {
    # https://developers.openai.com/api/docs/models/gpt-4o-mini
    ("openai", "gpt-4o-mini"): ModelLimits(128_000, 16_384),
    ("openai", "gpt-4o-mini-2024-07-18"): ModelLimits(128_000, 16_384),
    # https://developers.openai.com/api/docs/models/gpt-4o
    ("openai", "gpt-4o"): ModelLimits(128_000, 16_384),
    ("openai", "gpt-4o-2024-08-06"): ModelLimits(128_000, 16_384),
    ("openai", "gpt-4o-2024-11-20"): ModelLimits(128_000, 16_384),
    # Google documents an input limit and a separate output limit.
    # https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite
    ("gemini", "gemini-2.5-flash-lite"): ModelLimits(1_048_576, 65_536),
    # https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash
    ("gemini", "gemini-2.5-flash"): ModelLimits(1_048_576, 65_536),
    # https://ai.google.dev/gemini-api/docs/models/gemini-2.0-flash
    ("gemini", "gemini-2.0-flash"): ModelLimits(1_048_576, 8_192),
    ("gemini", "gemini-2.0-flash-001"): ModelLimits(1_048_576, 8_192),
    # https://platform.claude.com/docs/en/models/sonnet-4-6/overview
    # 128K is synchronous; 300K requires the Batches-only output beta.
    ("anthropic", "claude-sonnet-4-6"): ModelLimits(1_000_000, 128_000),
    # https://platform.claude.com/docs/en/models/haiku-4-5/overview
    ("anthropic", "claude-haiku-4-5"): ModelLimits(200_000, 64_000),
    ("anthropic", "claude-haiku-4-5-20251001"): ModelLimits(200_000, 64_000),
    # Historical limits, not a promise that these retired IDs remain available.
    # https://www.anthropic.com/news/claude-3-5-sonnet (200K context)
    # https://platform.claude.com/docs/en/release-notes/overview#august-19th-2024
    # https://www.anthropic.com/news/3-5-models-and-computer-use
    ("anthropic", "claude-3-5-sonnet-20240620"): ModelLimits(200_000, 8_192),
    ("anthropic", "claude-3-5-sonnet-20241022"): ModelLimits(200_000, 8_192),
    # https://docs.anthropic.com/en/docs/about-claude/models (legacy table)
    # https://www.anthropic.com/news/claude-3-family (200K context)
    ("anthropic", "claude-3-sonnet-20240229"): ModelLimits(200_000, 4_096),
}

_CLOUD_FALLBACK = ModelLimits(32_768, 16_384)
_LOCAL_DEFAULTS = ModelLimits(4_096, 2_048)


def model_capabilities(provider_name: str, model: str) -> ModelLimits | None:
    """Return immutable documented maxima, or ``None`` for an unverified ID.

    This lookup is independent of environment and session overrides. Provider
    names are case-insensitive; model IDs must match a documented entry exactly.
    Local model limits depend on the loaded weights and are not inferred here.
    """
    return _MODEL_LIMITS.get((provider_name.strip().lower(), model))


def _configured_limit(
    value: int | str | None,
    field: str,
    env_name: str,
    default: int,
    maximum: int | None,
    minimum: int,
    provider_name: str,
    model: str,
) -> int:
    raw = value if value is not None else os.environ.get(env_name)
    if raw is None:
        return default
    error = (
        f"{field} ({env_name}) must be an integer of at least {minimum} tokens; "
        f"got {raw!r}. Set {field} or {env_name} to a valid token count."
    )
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise ValueError(error)
    if isinstance(raw, str) and re.fullmatch(r"[+-]?[0-9]+", raw.strip()) is None:
        raise ValueError(error)
    try:
        resolved = int(raw)
    except ValueError as exc:
        raise ValueError(error) from exc
    if resolved < minimum:
        raise ValueError(error)
    if maximum is not None and resolved > maximum:
        warnings.warn(
            f"{provider_name}/{model}: {field}={resolved} exceeds the documented "
            f"maximum {maximum}; clamping to {maximum} tokens.",
            UserWarning,
            stacklevel=3,
        )
        return maximum
    return resolved


def resolve_limits(
    provider_name: str,
    model: str,
    context_size: int | str | None = None,
    max_output_tokens: int | str | None = None,
    *,
    local: bool = False,
) -> ModelLimits:
    """Resolve args > NEXUS environment > model maxima > cloud fallback.

    Each field resolves independently. Known-model maxima cap explicit arguments
    and environment settings, with a visible UserWarning. Local defaults remain
    4096/2048, context must be at least 512, and neither local setting has an
    inferred upper bound: the user must choose within model and RAM limits.
    The unknown-cloud fallback is configurable and is not a verified capacity.
    """
    local = local or provider_name.strip().lower() == "local"
    maxima = None if local else model_capabilities(provider_name, model)
    defaults = _LOCAL_DEFAULTS if local else (maxima or _CLOUD_FALLBACK)
    if (not local and maxima is None and provider_name.strip().lower() != "fallback"
            and context_size is None and "NEXUS_CONTEXT_SIZE" not in os.environ):
        warnings.warn(
            f"{provider_name}/{model}: model capacity is unverified; using the conservative "
            f"{defaults.context_size}-token context fallback. Set NEXUS_CONTEXT_SIZE or "
            "--context-size to the endpoint's documented limit.",
            UserWarning,
            stacklevel=2,
        )
    return ModelLimits(
        context_size=_configured_limit(
            context_size, "context_size", "NEXUS_CONTEXT_SIZE",
            defaults.context_size, maxima.context_size if maxima else None,
            512 if local else 1, provider_name, model,
        ),
        max_output_tokens=_configured_limit(
            max_output_tokens, "max_output_tokens", "NEXUS_MAX_OUTPUT_TOKENS",
            defaults.max_output_tokens, maxima.max_output_tokens if maxima else None,
            1, provider_name, model,
        ),
    )
