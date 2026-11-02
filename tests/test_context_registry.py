"""Behavioral coverage for advisory model context resolution and engine reuse."""

from dataclasses import FrozenInstanceError
import io
import json
import sys
from types import SimpleNamespace
import urllib.request
import warnings

import pytest

from nexus_agent_ai.providers.capabilities import (
    ContextProvenance,
    ContextQualityWarning,
    ModelLimits,
    resolve_context_capabilities,
    resolve_limits,
)
from nexus_agent_ai.providers.local_provider import LocalProvider


@pytest.fixture(autouse=True)
def clean_context_environment(monkeypatch):
    for name in (
        "NEXUS_CONTEXT_SIZE", "NEXUS_MAX_OUTPUT_TOKENS",
        "NEXUS_AGENT_MODEL_REPO", "NEXUS_AGENT_MODEL_FILENAME",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Context registry must not make a network request")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    return forbidden


def test_model_report_precedes_table_per_field_without_changing_execution_limits(forbid_network):
    report = {"safe_context": 16_384, "max_context": 65_536, "kv_warn_bytes": 900_000}
    capabilities = resolve_context_capabilities("openai", "gpt-4o-mini", metadata=report)
    assert capabilities.safe_context == 16_384
    assert capabilities.max_context == 65_536
    assert capabilities.kv_warn_bytes == 900_000
    assert capabilities.provenance == ContextProvenance(
        "model-reported", "model-reported", "model-reported",
    )
    # Registry metadata is advisory, never a new default or execution clamp.
    assert resolve_limits("openai", "gpt-4o-mini", metadata=report) == ModelLimits(128_000, 16_384)
    with pytest.warns(ContextQualityWarning, match="QUALITY \\+ RAM"):
        limits = resolve_limits("openai", "gpt-4o-mini", 32_768, metadata=report)
    assert limits.context_size == 32_768
    assert report == {"safe_context": 16_384, "max_context": 65_536, "kv_warn_bytes": 900_000}


def test_partial_report_falls_back_independently_to_exact_table(forbid_network):
    capabilities = resolve_context_capabilities(
        " OpenAI ", "gpt-4o-mini", metadata={"context_length": "65536"},
    )
    assert capabilities.safe_context == 128_000
    assert capabilities.max_context == 65_536
    assert capabilities.kv_warn_bytes is None
    assert capabilities.provenance == ContextProvenance("explicit-table", "model-reported", None)
    with pytest.raises(FrozenInstanceError):
        capabilities.max_context = 1
    with pytest.raises(FrozenInstanceError):
        capabilities.provenance.safe_context = "assumed"


def test_explicit_table_precedes_assumed_fallback_without_family_inference(forbid_network):
    known = resolve_context_capabilities("openai", "gpt-4o-mini")
    unknown = resolve_context_capabilities("openai", "gpt-4o-mini-unverified")
    assert (known.safe_context, known.max_context) == (128_000, 128_000)
    assert known.provenance == ContextProvenance("explicit-table", "explicit-table", None)
    assert unknown.safe_context == 32_768
    assert unknown.max_context is None
    assert unknown.kv_warn_bytes is None
    assert unknown.provenance == ContextProvenance("assumed", None, None)


@pytest.mark.parametrize("report", [
    {"general.architecture": "lfm2", "lfm2.context_length": "32768"},
    {"llama.context_length": 32_768},
    {"n_ctx_train": 32_768},
    {"max_position_embeddings": 32_768},
    {"input_token_limit": 32_768},
])
def test_existing_model_information_reports_max_not_quality_or_kv(report, forbid_network):
    capabilities = resolve_context_capabilities("local", "custom.gguf", metadata=report)
    assert capabilities.safe_context == 4096
    assert capabilities.max_context == 32_768
    assert capabilities.kv_warn_bytes is None
    assert capabilities.provenance == ContextProvenance("assumed", "model-reported", None)


@pytest.mark.parametrize("report", [
    {"n_ctx": 8192, "num_ctx": 8192, "model_max_length": 10**30},
    {"max_context": True, "safe_context": 0, "kv_warn_bytes": -1},
    {"max_context": 8192.0, "safe_context": "invalid", "kv_warn_bytes": "unknown"},
    {"one.context_length": 4096, "two.context_length": 8192},
])
def test_unavailable_invalid_or_ambiguous_information_stays_unknown(report, forbid_network):
    capabilities = resolve_context_capabilities("local", "custom.gguf", metadata=report)
    assert capabilities.safe_context == 4096
    assert capabilities.max_context is None
    assert capabilities.kv_warn_bytes is None
    assert capabilities.provenance == ContextProvenance("assumed", None, None)


@pytest.mark.parametrize("use_environment", [False, True])
@pytest.mark.parametrize("requested", [4095, 4096, 4097])
def test_safe_boundary_warns_only_above_and_never_clamps(monkeypatch, forbid_network, use_environment, requested):
    kwargs = {}
    if use_environment:
        monkeypatch.setenv("NEXUS_CONTEXT_SIZE", str(requested))
    else:
        kwargs["context_size"] = requested
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = LocalProvider(**kwargs)
    assert provider.context_size == requested
    assert provider.max_output_tokens == 512
    assert provider.context_capabilities.safe_context == 4096
    assert provider.context_capabilities.provenance.safe_context == "assumed"
    if requested > 4096:
        assert len(caught) == 1
        assert caught[0].category is ContextQualityWarning
        message = str(caught[0].message)
        assert "QUALITY + RAM WARNING" in message
        assert "safe_context=4096" in message
        assert "assumed" in message
        assert "response quality" in message
        assert "KV-cache RAM" in message
    else:
        assert caught == []


def test_explicit_override_beats_larger_environment_for_warning(monkeypatch, forbid_network):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "65536")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = LocalProvider(context_size=4096)
    assert provider.context_size == 4096
    assert caught == []
    # Environment is not a source of model capabilities.
    assert provider.context_capabilities.max_context is None


def test_lfm_default_and_output_reserve_do_not_change_or_initialize_engine(forbid_network):
    provider = LocalProvider()
    assert provider.model_id == "LiquidAI/LFM2.5-2.6B-GGUF"
    assert (provider.context_size, provider.max_output_tokens) == (4096, 512)
    assert provider._model_instance is None
    assert provider._llama is None
    assert provider.context_capabilities.provenance == ContextProvenance("assumed", None, None)
    assert provider.context_capabilities.max_context is None
    assert provider.context_capabilities.kv_warn_bytes is None


@pytest.mark.parametrize("local_endpoint", [False, True])
def test_current_cloud_provider_exposes_pure_registry_and_subclasses_keep_identity(
    monkeypatch, forbid_network, local_endpoint,
):
    from nexus_agent_ai.providers.openai_provider import OpenAIProvider

    clients = []

    def client(**kwargs):
        clients.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    class ApplicationProvider(OpenAIProvider):
        pass

    provider = ApplicationProvider(
        api_key="fixture",
        base_url="http://127.0.0.1:12345/v1" if local_endpoint else None,
    )
    capabilities = provider.context_capabilities
    assert len(clients) == 1
    assert capabilities.safe_context == (4096 if local_endpoint else 128_000)
    assert capabilities.max_context == (None if local_endpoint else 128_000)
    assert capabilities.kv_warn_bytes is None
    assert capabilities.provenance.safe_context == ("assumed" if local_endpoint else "explicit-table")
    assert provider.context_size == capabilities.safe_context
    assert provider.max_output_tokens == (512 if local_endpoint else 16_384)


def test_fallback_registry_tracks_active_provider_without_loading_a_model(monkeypatch, forbid_network):
    from nexus_agent_ai.providers import fallback_provider
    from nexus_agent_ai.providers.openai_provider import OpenAIProvider

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: object()))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cloud = OpenAIProvider(api_key="fixture")
    local = LocalProvider()
    monkeypatch.setattr(fallback_provider, "FALLBACK_CHAIN", [("openai", lambda **kwargs: cloud)])
    provider = fallback_provider.FallbackProvider(start_provider="openai")
    assert provider.context_capabilities.max_context == 128_000
    assert provider.context_capabilities.provenance.max_context == "explicit-table"
    provider._activate("local", local)
    assert provider.context_capabilities.max_context is None
    assert provider.context_capabilities.provenance.safe_context == "assumed"
    assert provider.context_capabilities.safe_context == 4096
    assert local._model_instance is None


def test_loaded_llama_metadata_reaches_public_provider_without_new_load_or_probe(monkeypatch, tmp_path, forbid_network):
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"engine fixture")
    loads = []

    def load_llama(**kwargs):
        loads.append(kwargs)
        return SimpleNamespace(metadata={"general.architecture": "lfm2", "lfm2.context_length": "32768"})

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=load_llama))
    provider = LocalProvider()
    monkeypatch.setattr(provider, "setup_model", lambda: str(path))
    provider._ensure_loaded()
    provider._ensure_loaded()
    assert len(loads) == 1
    assert loads[0]["n_ctx"] == 4096
    assert provider.context_capabilities.max_context == 32_768
    assert provider.context_capabilities.provenance.max_context == "model-reported"
    assert provider.context_capabilities.provenance.safe_context == "assumed"
    assert (provider.context_size, provider.max_output_tokens) == (4096, 512)


@pytest.mark.parametrize("reported_safe", [2048, 8192])
def test_loaded_report_can_refine_advisory_without_reconfiguring_context(
    monkeypatch, tmp_path, forbid_network, reported_safe,
):
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"engine fixture")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        Llama=lambda **kwargs: SimpleNamespace(metadata={"safe_context": reported_safe}),
    ))
    provider = LocalProvider(context_size=4096)
    monkeypatch.setattr(provider, "setup_model", lambda: str(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider._ensure_loaded()
        provider._ensure_loaded()
    assert provider.context_capabilities.safe_context == reported_safe
    assert provider.context_capabilities.provenance.safe_context == "model-reported"
    assert (provider.context_size, provider.max_output_tokens) == (4096, 512)
    assert len(caught) == (1 if reported_safe < 4096 else 0)
    if caught:
        assert caught[0].category is ContextQualityWarning
        assert "model-reported" in str(caught[0].message)


def test_loaded_transformers_config_reports_capacity_without_a_metadata_probe(monkeypatch, forbid_network):
    loads = []
    config = SimpleNamespace(max_position_embeddings=32_768)

    def load_model(*args, **kwargs):
        loads.append((args, kwargs))
        return SimpleNamespace(config=config)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda index: "fixture GPU",
    )))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
    ))
    provider = LocalProvider()
    monkeypatch.setattr(provider, "setup_model", lambda: "fixture.gguf")
    provider._ensure_loaded()
    assert len(loads) == 1
    assert provider.context_capabilities.max_context == 32_768
    assert provider.context_capabilities.provenance.max_context == "model-reported"
    assert (provider.context_size, provider.max_output_tokens) == (4096, 512)


@pytest.mark.parametrize("selected", [False, True])
def test_server_reuses_only_owned_identity_metadata_with_no_extra_request(monkeypatch, selected):
    provider = LocalProvider()
    provider._server_proc = SimpleNamespace(poll=lambda: None)
    provider._server_port = 12345
    requests = []
    model = {
        "id": provider._server_alias if selected else "someone-elses-model",
        "meta": {"n_ctx_train": 32_768},
    }

    def urlopen(url, timeout):
        requests.append(url)
        payload = {"data": [model]} if url.endswith("/v1/models") else {"status": "ok"}
        response = io.BytesIO(json.dumps(payload).encode())
        response.status = 200
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    try:
        assert provider._is_server_alive() is selected
        assert requests == ["http://127.0.0.1:12345/health", "http://127.0.0.1:12345/v1/models"]
        assert provider.context_capabilities.max_context == (32_768 if selected else None)
        assert provider.context_capabilities.kv_warn_bytes is None
        assert (provider.context_size, provider.max_output_tokens) == (4096, 512)
    finally:
        # Fixture process is not an actual child and must not reach stop().
        provider._server_proc = None
