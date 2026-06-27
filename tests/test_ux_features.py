import os
import pytest
from typing import Any
from src.agent.core import parse_at_mentions, Agent
from src.agent.memory import ConversationMemory
from src.providers.base import BaseProvider, ProviderResponse
from src.cli import display

class DummyProvider(BaseProvider):
    @property
    def model(self) -> str:
        return "dummy-model"

    def complete(self, messages, tools=None, system="") -> ProviderResponse:
        return ProviderResponse(text="Hello", input_tokens=10, output_tokens=10)

    def stream(self, messages, tools=None, system="") -> Any:
        yield "Hello"

    def format_tool_result_message(self, tool_call_id: str, output: str) -> dict:
        return {"role": "user", "content": output}

def test_parse_at_mentions(tmp_path):
    # Create a temporary file
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('hello world')", encoding="utf-8")
    
    # Test valid mention
    prompt = f"Please review @{test_file} carefully."
    result = parse_at_mentions(prompt)
    assert "Please review carefully." in result
    assert "[Context attached from @" in result
    assert "print('hello world')" in result

    # Test missing file mention
    missing_prompt = "Check @non_existent_file_123.py please"
    missing_result = parse_at_mentions(missing_prompt)
    assert "[Warning: Mentioned file @non_existent_file_123.py does not exist]" in missing_result

def test_smart_startup_project_mode(tmp_path, monkeypatch):
    # Change cwd to tmp_path with a dummy pyproject.toml
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    
    provider = DummyProvider()
    memory = ConversationMemory()
    agent = Agent(provider=provider, memory=memory)
    
    assert "Project Mode" in agent.mode_str
    assert f"currently working inside the project directory: {tmp_path.name}" in agent.system

def test_status_spinner_helpers():
    status = display.create_status("Testing...")
    assert status is not None
    display.update_status(status, "Updated status...")
    display.stop_status(status)
