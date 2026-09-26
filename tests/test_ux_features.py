import os
import re
import pytest
from typing import Any
from nexus_agent_ai.agent.core import parse_at_mentions, Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse
from nexus_agent_ai.cli import display

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
    # Create temporary files
    test_file = tmp_path / "sample.py"
    test_file.write_text("print('hello world')", encoding="utf-8")
    test_file2 = tmp_path / "config.txt"
    test_file2.write_text("debug=true", encoding="utf-8")
    
    # Test valid mention with space after @
    prompt = f"Please review @ {test_file} and @{test_file2} carefully."
    result = parse_at_mentions(prompt)
    assert "Please review and carefully." in result or "Please review and carefully." in re.sub(r'\s+', ' ', result)
    assert "print('hello world')" in result
    assert "debug=true" in result

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

def test_sqlite_memory(tmp_path):
    db_file = tmp_path / "test_history.db"
    mem = SQLiteMemory(db_path=db_file, session_id="test_sess")
    mem.add("user", "Hello SQLite")
    mem.add("assistant", "Hi there")

    msgs = mem.get()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Hello SQLite"

    mem.clear()
    assert len(mem.get()) == 0

    # Explicitly close all SQLite connections before temp dir cleanup.
    # On Windows, open file handles block directory deletion (PermissionError).
    import sqlite3
    conn = sqlite3.connect(db_file)
    conn.close()
    del mem  # drop reference so SQLite releases any internal handles

def test_repl_completer_slash_and_at_mentions(tmp_path, monkeypatch):
    from nexus_agent_ai.cli.app import _build_repl_completer
    from prompt_toolkit.document import Document

    completer = _build_repl_completer()
    assert completer is not None

    # Test slash command completion
    doc_slash = Document("/cl", cursor_position=3)
    completions = list(completer.get_completions(doc_slash, None))
    text_matches = [c.text for c in completions]
    assert "/clear" in text_matches

    # Test @ mention completion
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello_script.py").write_text("print(1)")
    doc_at = Document("check @hel", cursor_position=10)
    at_completions = list(completer.get_completions(doc_at, None))
    at_texts = [c.text for c in at_completions]
    assert "@hello_script.py" in at_texts

def test_display_print_info():
    from nexus_agent_ai.cli.display import print_info
    # Should execute without error
    print_info("Testing info output")

def test_repl_slash_commands_loop(monkeypatch):
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app as cli_app

    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr("nexus_agent_ai.cli.app.get_provider_instance", lambda p: (DummyProvider(), "dummy"))

    runner = CliRunner()
    # Send /help, /clear, then /exit
    inputs = "/help\n/clear\n/exit\n"
    result = runner.invoke(cli_app, ["repl"], input=inputs)
    assert result.exit_code == 0
    assert "Available REPL commands:" in result.output
    assert "Conversation memory cleared." in result.output
    assert "Ending session. Goodbye!" in result.output

