import pytest
from unittest.mock import patch, MagicMock
from src.providers.base import Tool
from src.providers.local_provider import LocalQwenProvider


def test_local_qwen_provider_init():
    prov = LocalQwenProvider()
    assert prov.model_id == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert prov.model == "qwen2.5-7b-instruct-awq"


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
    mock_snapshot = MagicMock(return_value="/mock/path/to/qwen-awq")
    
    with patch("huggingface_hub.snapshot_download", mock_snapshot):
        path = prov.setup_model()
        assert path == "/mock/path/to/qwen-awq"
        mock_snapshot.assert_called_once_with(
            repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
            local_files_only=False
        )
        
    captured = capsys.readouterr()
    assert "Initializing nexus-agent..." in captured.out
    assert "Downloading core reasoning engine (~4.5 GB)..." in captured.out
    assert "Core engine ready!" in captured.out


def test_local_qwen_format_tool_result_message():
    prov = LocalQwenProvider()
    res = prov.format_tool_result_message("call_xyz", "Result from tool")
    assert res["role"] == "tool"
    assert res["tool_call_id"] == "call_xyz"
    assert res["content"] == "Result from tool"
