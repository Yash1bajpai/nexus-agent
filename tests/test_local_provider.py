import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from nexus_agent_ai.providers.base import Tool
from nexus_agent_ai.providers.local_provider import LocalProvider


def test_local_provider_init():
    prov = LocalProvider()
    assert prov.model_id == "LiquidAI/LFM2.5-2.6B-GGUF"
    assert prov.filename == "LFM2.5-2.6B-Q6_K.gguf"


def test_local_provider_env_overrides(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
    prov = LocalProvider()
    assert prov.model_id == "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    assert prov.filename == "qwen2.5-coder-3b-instruct-q4_k_m.gguf"


def test_local_provider_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "env/repo")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "env-file.gguf")
    prov = LocalProvider(model_id="explicit/repo", filename="explicit-file.gguf")
    assert prov.model_id == "explicit/repo"
    assert prov.filename == "explicit-file.gguf"


def test_local_provider_convert_tools():
    prov = LocalProvider()
    sample_tool = Tool(
        name="test_tool",
        description="A dummy test tool.",
        input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        execute=lambda arg1: arg1
    )
    converted = prov._convert_tools([sample_tool])
    assert len(converted) == 1
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "test_tool"
    assert converted[0]["function"]["parameters"]["type"] == "object"


def test_local_provider_setup_model(monkeypatch, capsys):
    prov = LocalProvider()
    mock_hub_download = MagicMock(return_value="/mock/path/to/liquid-gguf")
    mock_hf = MagicMock()
    mock_hf.hf_hub_download = mock_hub_download

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
        path = prov.setup_model()
        assert path == "/mock/path/to/liquid-gguf"
        # Cache-first probe (no network), then full download if missing
        assert mock_hub_download.call_count == 2
        assert mock_hub_download.call_args_list[0].kwargs["local_files_only"] is True
        assert mock_hub_download.call_args_list[1].kwargs["local_files_only"] is False
        
    captured = capsys.readouterr()
    assert "Initializing nexus-agent..." in captured.out
    assert "Local Liquid LFM engine" in captured.out
    assert "Core engine ready!" in captured.out


def test_local_provider_format_tool_result_message():
    prov = LocalProvider()
    res = prov.format_tool_result_message("call_xyz", "Result from tool")
    assert res["role"] == "tool"
    assert res["tool_call_id"] == "call_xyz"
    assert res["content"] == "Result from tool"


def test_recommended_model_config_tiers(monkeypatch):
    from nexus_agent_ai.cli import onboarding
    monkeypatch.setattr(onboarding, "_is_android", lambda: False)

    # Desktop tiers: higher quants than phone for same RAM (swap, no OOM-killer)
    assert onboarding.recommended_model_config({"ram_gb": 2})["filename"] == "LFM2.5-2.6B-Q4_0.gguf"
    assert onboarding.recommended_model_config({"ram_gb": 6})["filename"] == "LFM2.5-2.6B-Q5_K_M.gguf"
    # College-PC case: 8GB desktop must NOT get the phone's Q4_K_M
    assert onboarding.recommended_model_config({"ram_gb": 8})["filename"] == "LFM2.5-2.6B-Q5_K_M.gguf"
    assert onboarding.recommended_model_config({"ram_gb": 12})["filename"] == "LFM2.5-2.6B-Q6_K.gguf"
    cfg16 = onboarding.recommended_model_config({"ram_gb": 32})
    assert cfg16["filename"] == "LFM2.5-2.6B-Q6_K.gguf"
    assert "pull-model" in cfg16["alt_cmd"]
    # Unknown RAM on desktop: assume middle tier, not worst case
    assert onboarding.recommended_model_config({"ram_gb": -1})["filename"] == "LFM2.5-2.6B-Q5_K_M.gguf"

    gpu_cfg = onboarding.recommended_model_config({"ram_gb": 8, "gpu": "RTX 4060 (8GB VRAM)"})
    assert "Qwen" in gpu_cfg["alt_cmd"] or "Mistral" in gpu_cfg["alt_cmd"]
    # 7B suggestion requires enough cores (CPU-only slow machines skip it)
    assert onboarding.recommended_model_config({"ram_gb": 12, "cpu_cores": 4})["alt_cmd"] == ""

    # Android tiers stay conservative (OOM-killer safety)
    monkeypatch.setattr(onboarding, "_is_android", lambda: True)
    assert onboarding.recommended_model_config({"ram_gb": 4})["filename"] == "LFM2.5-2.6B-Q4_0.gguf"
    assert onboarding.recommended_model_config({"ram_gb": 5.3})["filename"] == "LFM2.5-2.6B-Q4_K_M.gguf"
    assert onboarding.recommended_model_config({})["filename"] == "LFM2.5-2.6B-Q4_K_M.gguf"


def test_pull_model_arbitrary_repo(monkeypatch):
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app as cli_app
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)

    captured = {}

    class FakeProv:
        def __init__(self, model_id=None, filename=None):
            captured["repo"] = model_id
            captured["file"] = filename

        def setup_model(self, verify_download=False):
            return "/mock/model.gguf"

    monkeypatch.setattr("nexus_agent_ai.providers.local_provider.LocalProvider", FakeProv)
    runner = CliRunner()
    result = runner.invoke(cli_app, [
        "pull-model",
        "--repo", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "--file", "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    ])
    assert result.exit_code == 0, result.output
    assert captured["repo"] == "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    assert captured["file"] == "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    assert "NEXUS_AGENT_MODEL_REPO=Qwen/Qwen2.5-Coder-3B-Instruct-GGUF" in result.output


def test_pull_model_env_fallback(monkeypatch):
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app as cli_app
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "env/repo")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "env-file.gguf")

    captured = {}

    class FakeProv:
        def __init__(self, model_id=None, filename=None):
            captured["repo"] = model_id
            captured["file"] = filename

        def setup_model(self, verify_download=False):
            return "/mock/model.gguf"

    monkeypatch.setattr("nexus_agent_ai.providers.local_provider.LocalProvider", FakeProv)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["pull-model"])
    assert result.exit_code == 0, result.output
    assert captured["repo"] == "env/repo"
    assert captured["file"] == "env-file.gguf"
