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
    # Send /help, /sessions, /clear, then /exit
    inputs = "/help\n/sessions\n/clear\n/exit\n"
    result = runner.invoke(cli_app, ["repl"], input=inputs)
    assert result.exit_code == 0
    assert "Available REPL commands:" in result.output
    assert "Conversation memory cleared." in result.output
    assert "Ending session. Goodbye!" in result.output

def test_chat_command_with_persist(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app as cli_app
    from nexus_agent_ai.agent.persistence import SQLiteMemory

    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr("nexus_agent_ai.cli.app.get_provider_instance", lambda p: (DummyProvider(), "dummy"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli_app, ["chat", "Hello test agent", "--persist", "--session", "chat_test_sess", "--no-stream"])
    assert result.exit_code == 0

    db_path = tmp_path / ".nexus-agent" / "history.db"
    assert db_path.exists()
    sessions = SQLiteMemory.list_sessions(db_path=db_path)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "chat_test_sess"

def test_workspace_session_id(tmp_path):
    from nexus_agent_ai.agent.persistence import get_workspace_session_id
    dir1 = tmp_path / "project_a"
    dir2 = tmp_path / "project_b"
    dir1.mkdir()
    dir2.mkdir()

    sess1 = get_workspace_session_id(dir1)
    sess2 = get_workspace_session_id(dir2)

    assert sess1 != sess2
    assert sess1.startswith("project_a_")
    assert sess2.startswith("project_b_")
    # Idempotence check
    assert sess1 == get_workspace_session_id(dir1)

def test_sqlite_sessions_list_and_delete(tmp_path):
    from nexus_agent_ai.agent.persistence import SQLiteMemory
    db_file = tmp_path / "sessions_test.db"

    mem_a = SQLiteMemory(db_path=db_file, session_id="session_alpha")
    mem_a.add("user", "Alpha message 1")
    mem_a.add("assistant", "Alpha message 2")

    mem_b = SQLiteMemory(db_path=db_file, session_id="session_beta")
    mem_b.add("user", "Beta message 1")

    sessions = SQLiteMemory.list_sessions(db_path=db_file)
    assert len(sessions) == 2
    session_ids = [s["session_id"] for s in sessions]
    assert "session_alpha" in session_ids
    assert "session_beta" in session_ids

    alpha_entry = next(s for s in sessions if s["session_id"] == "session_alpha")
    assert alpha_entry["message_count"] == 2

    # Delete session_alpha
    SQLiteMemory.delete_session("session_alpha", db_path=db_file)
    remaining = SQLiteMemory.list_sessions(db_path=db_file)
    assert len(remaining) == 1
    assert remaining[0]["session_id"] == "session_beta"

def test_sessions_cli_command(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app as cli_app
    from nexus_agent_ai.agent.persistence import SQLiteMemory

    nexus_dir = tmp_path / ".nexus-agent"
    nexus_dir.mkdir()
    db_file = nexus_dir / "history.db"
    # Populate dummy sessions
    mem1 = SQLiteMemory(db_path=db_file, session_id="my_custom_session")
    mem1.add("user", "Hello custom")
    del mem1

    # Point home dir or default db_path to tmp_path
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    runner = CliRunner()
    # List sessions
    result = runner.invoke(cli_app, ["sessions"])
    assert result.exit_code == 0
    assert "Persistent Sessions" in result.output
    assert "my_custom_session" in result.output

    # Delete session
    del_result = runner.invoke(cli_app, ["sessions", "--delete", "my_custom_session"])
    assert del_result.exit_code == 0
    assert "Deleted session 'my_custom_session'." in del_result.output

    # Check empty list
    empty_result = runner.invoke(cli_app, ["sessions"])
    assert empty_result.exit_code == 0
    assert "No persistent sessions found" in empty_result.output

def test_package_version():
    from nexus_agent_ai.utils.config import get_package_version
    # Source metadata and installed distribution must agree with the release.
    ver = get_package_version()
    assert ver == "2.8.1"

def test_workspace_session_id_posix_case_sensitivity(monkeypatch, tmp_path):
    from nexus_agent_ai.agent.persistence import get_workspace_session_id
    import os
    # Simulate POSIX environment
    monkeypatch.setattr(os, "name", "posix")
    path_lower = tmp_path / "my_project"
    path_upper = tmp_path / "My_Project"
    
    sess_lower = get_workspace_session_id(path_lower)
    sess_upper = get_workspace_session_id(path_upper)
    assert sess_lower != sess_upper

    # Simulate Windows environment
    monkeypatch.setattr(os, "name", "nt")
    sess_nt_lower = get_workspace_session_id(path_lower)
    sess_nt_upper = get_workspace_session_id(path_upper)
    # On Windows, both resolve to the same normalized hash
    assert sess_nt_lower.split("_")[-1] == sess_nt_upper.split("_")[-1]



