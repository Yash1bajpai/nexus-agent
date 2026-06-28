import pytest
from unittest.mock import MagicMock, patch
# pyrefly: ignore [missing-import]
from src.providers.base import Tool
# pyrefly: ignore [missing-import]
from src.providers.anthropic_provider import AnthropicProvider
# pyrefly: ignore [missing-import]
from src.providers.openai_provider import OpenAIProvider
# pyrefly: ignore [missing-import]
from src.providers.gemini_provider import GeminiProvider

@pytest.fixture(autouse=True)
def mock_api_clients(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")
    with patch("anthropic.Anthropic", return_value=MagicMock()), \
         patch("openai.OpenAI", return_value=MagicMock()), \
         patch("google.genai.Client", return_value=MagicMock()):
        yield

sample_tool = Tool(
    name="test_tool",
    description="A dummy test tool.",
    input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
    execute=lambda arg1: arg1
)

def test_anthropic_provider_schema():
    prov = AnthropicProvider(model="claude-3-5-sonnet-20241022")
    converted = prov._convert_tools([sample_tool])
    assert len(converted) == 1
    assert converted[0]["name"] == "test_tool"
    assert converted[0]["input_schema"]["type"] == "object"

def test_openai_provider_schema():
    prov = OpenAIProvider(model="gpt-4o")
    converted = prov._convert_tools([sample_tool])
    assert len(converted) == 1
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "test_tool"

def test_gemini_provider_schema():
    prov = GeminiProvider(model="gemini-2.5-flash")
    converted = prov._convert_tools([sample_tool])
    assert len(converted) == 1
    # Check tool object created by google-genai
    assert converted[0].function_declarations[0].name == "test_tool"

def test_provider_tool_result_format():
    anthropic = AnthropicProvider()
    res_a = anthropic.format_tool_result_message("call_1", "ok result")
    assert res_a["role"] == "user"

    openai_p = OpenAIProvider()
    res_o = openai_p.format_tool_result_message("call_2", "ok result")
    assert res_o["role"] == "tool"
    assert res_o["tool_call_id"] == "call_2"

    gemini = GeminiProvider()
    res_g = gemini.format_tool_result_message("call_3", "ok result")
    assert res_g["role"] == "tool"
