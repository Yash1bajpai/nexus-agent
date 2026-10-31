from dataclasses import FrozenInstanceError
import os
import warnings

import pytest

from nexus_agent_ai.providers.capabilities import (
    ModelLimits,
    model_capabilities,
    resolve_limits,
)


@pytest.fixture(autouse=True)
def clear_limit_environment(monkeypatch):
    monkeypatch.delenv("NEXUS_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("NEXUS_MAX_OUTPUT_TOKENS", raising=False)


# Expected public specifications, including snapshots whose limits differ from
# the unverified-cloud fallback. Keep these explicit rather than copying the
# implementation's private table.
DOCUMENTED_MODELS = [
    ("openai", "gpt-4o-mini", 128_000, 16_384),
    ("openai", "gpt-4o-mini-2024-07-18", 128_000, 16_384),
    ("openai", "gpt-4o", 128_000, 16_384),
    ("openai", "gpt-4o-2024-08-06", 128_000, 16_384),
    ("openai", "gpt-4o-2024-11-20", 128_000, 16_384),
    ("gemini", "gemini-2.5-flash-lite", 1_048_576, 65_536),
    ("gemini", "gemini-2.5-flash", 1_048_576, 65_536),
    ("gemini", "gemini-2.0-flash", 1_048_576, 8_192),
    ("gemini", "gemini-2.0-flash-001", 1_048_576, 8_192),
    ("anthropic", "claude-sonnet-4-6", 1_000_000, 128_000),
    ("anthropic", "claude-haiku-4-5", 200_000, 64_000),
    ("anthropic", "claude-haiku-4-5-20251001", 200_000, 64_000),
    ("anthropic", "claude-3-5-sonnet-20240620", 200_000, 8_192),
    ("anthropic", "claude-3-5-sonnet-20241022", 200_000, 8_192),
    ("anthropic", "claude-3-sonnet-20240229", 200_000, 4_096),
]


@pytest.mark.parametrize("provider,model,context,output", DOCUMENTED_MODELS)
def test_documented_models_default_to_actual_maxima(provider, model, context, output):
    expected = ModelLimits(context, output)
    assert model_capabilities(provider, model) == expected
    assert resolve_limits(provider, model) == expected
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_limits(provider, model, context, output) == expected
    assert caught == []


@pytest.mark.parametrize("provider,model,context,output", DOCUMENTED_MODELS)
def test_known_maxima_clamp_each_field_with_visible_warnings(
    provider, model, context, output,
):
    with pytest.warns(UserWarning) as caught:
        limits = resolve_limits(provider, model, context + 1, output + 1)
    assert limits == ModelLimits(context, output)
    assert len(caught) == 2
    assert "context_size" in str(caught[0].message)
    assert str(context) in str(caught[0].message)
    assert "max_output_tokens" in str(caught[1].message)
    assert str(output) in str(caught[1].message)
    assert all(model in str(item.message) for item in caught)
    assert model_capabilities(provider, model) == limits


@pytest.mark.parametrize("field,env_name,maximum", [
    ("context_size", "NEXUS_CONTEXT_SIZE", 128_000),
    ("max_output_tokens", "NEXUS_MAX_OUTPUT_TOKENS", 16_384),
])
def test_environment_over_maximum_is_also_clamped(monkeypatch, field, env_name, maximum):
    monkeypatch.setenv(env_name, str(maximum + 1))
    with pytest.warns(UserWarning, match=field):
        limits = resolve_limits("openai", "gpt-4o-mini")
    assert getattr(limits, field) == maximum
    assert os.environ[env_name] == str(maximum + 1)


@pytest.mark.parametrize("provider,model", [
    ("openai", "unlisted-model"),
    ("openai", "gpt-4o-mini-future"),
    ("openai", "gpt-4o-mini-2099-01-01"),
    ("gemini", "gemini-2.5-flash-lite-unverified"),
    ("anthropic", "claude-sonnet-4-6-20990101"),
    ("anthropic", "claude-haiku-future"),
    ("openai", "gemini-2.5-flash-lite"),
    ("custom-endpoint", "gpt-4o-mini"),
])
def test_unknown_capacity_is_not_inferred_from_model_prefix(provider, model):
    assert model_capabilities(provider, model) is None
    assert resolve_limits(provider, model) == ModelLimits(200_000, 16_384)
    assert resolve_limits(provider, model, 400_000, 80_000) == ModelLimits(400_000, 80_000)


def test_documented_capabilities_are_immutable_and_environment_independent(monkeypatch):
    limits = model_capabilities(" OpenAI ", "gpt-4o-mini")
    assert limits == ModelLimits(128_000, 16_384)
    with pytest.raises(FrozenInstanceError):
        limits.context_size = 1
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "4096")
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "2048")
    assert model_capabilities("openai", "gpt-4o-mini") == limits
    assert resolve_limits("openai", "gpt-4o-mini") == ModelLimits(4096, 2048)
    assert model_capabilities("openai", "GPT-4O-MINI") is None


def test_argument_environment_and_defaults_resolve_independently(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "8192")
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "4096")
    assert resolve_limits("openai", "gpt-4o-mini") == ModelLimits(8192, 4096)
    assert resolve_limits("openai", "gpt-4o-mini", context_size=2048) == ModelLimits(2048, 4096)
    assert resolve_limits("openai", "gpt-4o-mini", max_output_tokens=1024) == ModelLimits(8192, 1024)
    assert resolve_limits("openai", "gpt-4o-mini", 1024, 512) == ModelLimits(1024, 512)


def test_arguments_override_even_malformed_environment(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "bad-context")
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "bad-output")
    assert resolve_limits("openai", "gpt-4o-mini", 8192, 4096) == ModelLimits(8192, 4096)
    with pytest.raises(ValueError, match="NEXUS_MAX_OUTPUT_TOKENS"):
        resolve_limits("openai", "gpt-4o-mini", context_size=8192)
    with pytest.raises(ValueError, match="NEXUS_CONTEXT_SIZE"):
        resolve_limits("openai", "gpt-4o-mini", max_output_tokens=4096)


@pytest.mark.parametrize("field,env_name", [
    ("context_size", "NEXUS_CONTEXT_SIZE"),
    ("max_output_tokens", "NEXUS_MAX_OUTPUT_TOKENS"),
])
@pytest.mark.parametrize("value", [0, -1, 1.5, 4096.0, True, False, "", " ", "bad", "1.5", "1e4", "1_000", [], {}])
def test_malformed_arguments_raise_actionable_errors(field, env_name, value):
    with pytest.raises(ValueError, match=env_name):
        resolve_limits("openai", "gpt-4o-mini", **{field: value})


@pytest.mark.parametrize("env_name", ["NEXUS_CONTEXT_SIZE", "NEXUS_MAX_OUTPUT_TOKENS"])
@pytest.mark.parametrize("value", ["0", "-1", "", " ", "bad", "2.5", "1e3", "1_000"])
def test_malformed_environment_raises_actionable_errors(monkeypatch, env_name, value):
    monkeypatch.setenv(env_name, value)
    with pytest.raises(ValueError, match=env_name):
        resolve_limits("gemini", "gemini-2.5-flash-lite")


def test_integer_strings_and_small_cloud_limits():
    assert resolve_limits("openai", "gpt-4o-mini", " 8192 ", "+4096") == ModelLimits(8192, 4096)
    assert resolve_limits("openai", "unknown", 1, 1) == ModelLimits(1, 1)


def test_local_defaults_and_unverified_maxima():
    assert model_capabilities("local", "LiquidAI/LFM2.5-2.6B-GGUF") is None
    assert resolve_limits("local", "LiquidAI/LFM2.5-2.6B-GGUF", local=True) == ModelLimits(4096, 2048)
    assert resolve_limits("local", "custom.gguf") == ModelLimits(4096, 2048)
    # A local model bearing a cloud ID still has user-controlled limits.
    assert resolve_limits("openai", "gpt-4o-mini", 256_000, 32_768, local=True) == ModelLimits(256_000, 32_768)
    assert resolve_limits("local", "custom.gguf", 512, 1, local=True) == ModelLimits(512, 1)


def test_local_environment_and_large_output_are_not_capped(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "32768")
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "8192")
    assert resolve_limits("local", "custom.gguf", local=True) == ModelLimits(32768, 8192)
    assert resolve_limits("local", "custom.gguf", 65536, 16384, local=True) == ModelLimits(65536, 16384)


@pytest.mark.parametrize("context", [0, 1, 511, "511", False, 512.0])
def test_local_minimum_context_stays_512(context):
    with pytest.raises(ValueError, match="integer of at least 512"):
        resolve_limits("local", "custom.gguf", context_size=context, local=True)


def test_local_invalid_environment_remains_actionable(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "511")
    with pytest.raises(ValueError, match="integer of at least 512"):
        resolve_limits("local", "custom.gguf", local=True)
    assert resolve_limits("local", "custom.gguf", context_size=8192, local=True) == ModelLimits(8192, 2048)


def test_sonnet_synchronous_output_never_uses_batch_beta_limit():
    with pytest.warns(UserWarning, match="128000"):
        limits = resolve_limits("anthropic", "claude-sonnet-4-6", max_output_tokens=300_000)
    assert limits == ModelLimits(1_000_000, 128_000)
