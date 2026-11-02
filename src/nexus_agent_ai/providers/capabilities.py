"""Documented synchronous model limits and per-session configuration.

Known-model overrides above a documented maximum emit ``UserWarning`` through
``warnings.warn`` and are clamped. Unknown models have no verified maxima:
``model_capabilities`` returns ``None`` and cloud resolution warns when using the
unverified 32,768/16,384 fallback; this is not a verified endpoint capacity.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal
import os
import re
import warnings


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_size: int
    max_output_tokens: int


CapabilitySource = Literal["model-reported", "explicit-table", "assumed"]


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    """Per-field evidence; ``None`` means no value or evidence is available."""

    safe_context: CapabilitySource
    max_context: CapabilitySource | None
    kv_warn_bytes: CapabilitySource | None


@dataclass(frozen=True, slots=True)
class ContextCapabilities:
    """Advisory context configuration, not new provider execution limits.

    ``max_context`` and ``kv_warn_bytes`` stay ``None`` when unknown. A reported
    GGUF context length describes the weights' training extent, not measured
    quality, available RAM, or the running server's configured context size.
    """

    safe_context: int
    max_context: int | None
    kv_warn_bytes: int | None
    provenance: ContextProvenance


class ContextQualityWarning(UserWarning):
    """A requested context exceeds the registry's advisory safe setting."""


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
_LOCAL_DEFAULTS = ModelLimits(4_096, 512)  # measured: typical turns 20-120 tokens; long writes get one bounded boost retry


def model_capabilities(provider_name: str, model: str) -> ModelLimits | None:
    """Return immutable documented maxima, or ``None`` for an unverified ID.

    This lookup is independent of environment and session overrides. Provider
    names are case-insensitive; model IDs must match a documented entry exactly.
    Local model limits depend on the loaded weights and are not inferred here.
    """
    return _MODEL_LIMITS.get((provider_name.strip().lower(), model))


def _reported_context(metadata: Mapping[str, object] | None) -> dict[str, int]:
    """Read already available model information without probing any endpoint."""
    if not isinstance(metadata, Mapping):
        return {}
    reported = {}
    for field in ("safe_context", "max_context", "kv_warn_bytes"):
        value = metadata.get(field)
        if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            reported[field] = value
    if "max_context" not in reported:
        # Use only model capacity fields. n_ctx/num_ctx and tokenizer sentinel
        # lengths describe runtime settings or placeholders, not model maxima.
        architecture = metadata.get("general.architecture")
        gguf_key = f"{architecture}.context_length" if isinstance(architecture, str) else None
        keys = ("context_length", "n_ctx_train", "max_position_embeddings", "input_token_limit")
        if gguf_key:
            keys = (gguf_key, *keys)
        else:
            gguf_keys = [key for key in metadata if isinstance(key, str) and key.endswith(".context_length")]
            if len(gguf_keys) == 1:
                keys = (*keys, gguf_keys[0])
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
                value = int(value)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                reported["max_context"] = value
                break
    return reported


def resolve_context_capabilities(
    provider_name: str,
    model: str,
    *,
    metadata: Mapping[str, object] | None = None,
    local: bool = False,
) -> ContextCapabilities:
    """Resolve model report > documented exact-ID table > existing defaults.

    Callers may supply existing cloud model-info, local model config, or GGUF
    metadata; this function performs no I/O and ignores environment overrides.
    Normalized ``safe_context``, ``max_context`` and ``kv_warn_bytes`` are
    accepted, alongside GGUF context_length, Transformers max_position_embeddings
    and cloud input_token_limit. No KV threshold is inferred from model size.
    Registry values are advisory: resolve_limits retains all existing defaults
    and documented maximum/clamping policies even when a report is supplied.
    """
    local = local or provider_name.strip().lower() == "local"
    documented = None if local else model_capabilities(provider_name, model)
    reported = _reported_context(metadata)
    safe = reported.get("safe_context")
    safe_source: CapabilitySource = "model-reported"
    if safe is None:
        safe = documented.context_size if documented else (
            _LOCAL_DEFAULTS.context_size if local else _CLOUD_FALLBACK.context_size
        )
        safe_source = "explicit-table" if documented else "assumed"
    maximum = reported.get("max_context")
    maximum_source: CapabilitySource | None = "model-reported" if maximum is not None else None
    if maximum is None and documented:
        maximum = documented.context_size
        maximum_source = "explicit-table"
    kv = reported.get("kv_warn_bytes")
    return ContextCapabilities(
        safe_context=safe,
        max_context=maximum,
        kv_warn_bytes=kv,
        provenance=ContextProvenance(safe_source, maximum_source, "model-reported" if kv is not None else None),
    )


def warn_context_quality(
    provider_name: str,
    model: str,
    context_size: int,
    capabilities: ContextCapabilities,
) -> None:
    """Warn without changing the requested context or allocating KV memory."""
    if context_size > capabilities.safe_context:
        warnings.warn(
            f"QUALITY + RAM WARNING: {provider_name}/{model}: context_size={context_size} "
            f"exceeds safe_context={capabilities.safe_context} "
            f"({capabilities.provenance.safe_context}). Larger context can degrade "
            "response quality and increase KV-cache RAM use; available RAM and "
            "model quality are not guaranteed. This advisory does not clamp context.",
            ContextQualityWarning,
            stacklevel=3,
        )


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
    metadata: Mapping[str, object] | None = None,
) -> ModelLimits:
    """Resolve args > NEXUS environment > model maxima > cloud fallback.

    Each field resolves independently. Known-model maxima cap explicit arguments
    and environment settings, with a visible UserWarning. Local defaults remain
    4096/512, context must be at least 512, and neither local setting has an
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
    limits = ModelLimits(
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
    raw_context = context_size if context_size is not None else os.environ.get("NEXUS_CONTEXT_SIZE")
    if raw_context is not None and provider_name.strip().lower() != "fallback":
        # _configured_limit has already validated this value. Compare the actual
        # requested count, even when the existing documented maximum clamps it.
        warn_context_quality(
            provider_name, model, int(raw_context),
            resolve_context_capabilities(provider_name, model, metadata=metadata, local=local),
        )
    return limits
