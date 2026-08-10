import pytest
from unittest.mock import patch, MagicMock
from nexus_agent_ai.providers.base import Tool
from nexus_agent_ai.providers.local_provider import LocalQwenProvider


def test_local_qwen_provider_init():
    prov = LocalQwenProvider()
    assert prov.model_id == "LiquidAI/LFM2.5-2.6B-GGUF"
    assert prov.filename == "LFM2.5-2.6B-Q6_K.gguf"


def test_local_qwen_provider_convert_tools():
    prov = LocalQwenProvider()
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


def test_local_qwen_provider_setup_model(monkeypatch, capsys):
    prov = LocalQwenProvider()
    mock_hub_download = MagicMock(return_value="/mock/path/to/liquid-gguf")
    mock_hf = MagicMock()
    mock_hf.hf_hub_download = mock_hub_download
    
    with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
        path = prov.setup_model()
        assert path == "/mock/path/to/liquid-gguf"
        mock_hub_download.assert_called_once_with(
            repo_id="LiquidAI/LFM2.5-2.6B-GGUF",
            filename="LFM2.5-2.6B-Q6_K.gguf",
            local_files_only=False
        )
        
    captured = capsys.readouterr()
    assert "Initializing nexus-agent..." in captured.out
    assert "Local Liquid LFM engine" in captured.out
    assert "Core engine ready!" in captured.out


def test_local_qwen_format_tool_result_message():
    prov = LocalQwenProvider()
    res = prov.format_tool_result_message("call_xyz", "Result from tool")
    assert res["role"] == "tool"
    assert res["tool_call_id"] == "call_xyz"
    assert res["content"] == "Result from tool"
