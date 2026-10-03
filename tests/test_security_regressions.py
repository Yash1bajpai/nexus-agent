"""Regression coverage for the September 30 review. Only fake secrets/canaries."""
import io
import json
import urllib.error
from pathlib import Path

import pytest

from nexus_agent_ai.agent.core import Agent, parse_at_mentions
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.tools import (
    execute_read_file, execute_run_tests, execute_git_commit,
    get_all_tools, get_readonly_tools,
)
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.local_provider import LocalProvider

class MaliciousProvider(BaseProvider):
    def __init__(self, name, args):
        self.name, self.args, self.calls = name, args, 0
    def complete(self, messages, tools, system):
        self.calls += 1
        if self.calls == 1:
            return ProviderResponse(tool_calls=[ToolCall("canary", self.name, self.args)],
                                    raw_assistant_message={"role": "assistant", "content": "tool call"})
        return ProviderResponse(text="Done")
    def stream(self, **kwargs):
        yield self.complete(**kwargs)
    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}

@pytest.mark.parametrize("name,args", [
    ("write_file", {"path": "canary.txt", "content": "bad"}),
    ("patch_file", {"path": "existing.txt", "target": "good", "replacement": "bad"}),
    ("run_tests", {}), ("git_commit", {"message": "bad"}),
])
def test_readonly_rejects_unadvertised_tools(tmp_path, monkeypatch, name, args):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.txt").write_text("good")
    p = MaliciousProvider(name, args)
    agent = Agent(p, verbose=False, tools=get_readonly_tools())
    agent.run("Review code")
    assert not (tmp_path / "canary.txt").exists()
    assert (tmp_path / "existing.txt").read_text() == "good"
    assert "not allowed" in agent.memory.get()[2]["content"]

@pytest.mark.parametrize("name", ["credentials.json", ".env.local", "private.key", "token.txt", "id_ed25519"])
def test_mentions_share_secret_policy(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    (tmp_path / name).write_text("DUMMY_SECRET_DO_NOT_SEND")
    assert "Security Blocked" in execute_read_file(name)
    prompt = parse_at_mentions(f"Explain @{name}")
    assert "DUMMY_SECRET_DO_NOT_SEND" not in prompt
    assert "Security Blocked" in prompt

def test_sensitive_symlink_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    secret = tmp_path / "credentials.json"
    secret.write_text("FAKE_VALUE")
    (tmp_path / "innocent.txt").symlink_to(secret)
    assert "FAKE_VALUE" not in parse_at_mentions("Read @innocent.txt")

def test_project_execution_default_denied(tmp_path, monkeypatch):
    monkeypatch.delenv("NEXUS_ALLOW_PROJECT_EXECUTION", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_canary.py").write_text("from pathlib import Path\nPath('canary').write_text('bad')\n")
    assert "disabled" in execute_run_tests()
    assert "run_tests" not in {t.name for t in get_all_tools()}
    assert not (tmp_path / "canary").exists()

def test_project_execution_strips_fake_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    monkeypatch.setenv("NEXUS_FAKE_SECRET", "dummy")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--bad-option")
    (tmp_path / "test_safe.py").write_text("import os\ndef test_env():\n assert 'NEXUS_FAKE_SECRET' not in os.environ\n assert os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'\n")
    assert "PASSED" in execute_run_tests("test_safe.py", "-q")

def test_git_mutation_default_denied(monkeypatch):
    monkeypatch.delenv("NEXUS_ALLOW_GIT_COMMIT", raising=False)
    assert "disabled" in execute_git_commit("bad")
    assert "git_commit" not in {t.name for t in get_all_tools()}

@pytest.mark.parametrize("result", [
    {"role": "tool", "tool_call_id": "1", "content": "result"},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "result"}]},
    {"role": "user", "parts": [{"function_response": {"name": "read", "response": "result"}}]},
])
def test_sqlite_keeps_active_turn_like_memory(tmp_path, result):
    sqlite = SQLiteMemory(tmp_path / "memory.db", "test", max_messages=2)
    ram = ConversationMemory(max_messages=2)
    for msg in [{"role": "user", "content": "original request"},
                {"role": "assistant", "content": "tool call"}, result]:
        sqlite.add_raw(msg); ram.add_raw(msg)
    assert sqlite.get() == ram.get()
    assert sqlite.get()[0]["content"] == "original request"

def test_local_cost_zero():
    p = LocalProvider()
    agent = Agent(p, verbose=False)
    agent.total_input_tokens, agent.total_output_tokens = 1000, 1000
    assert agent.estimated_cost == 0

def test_context_config(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "8192")
    assert LocalProvider().context_size == 8192
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "128")
    with pytest.raises(ValueError): LocalProvider()

def test_local_http_error_has_context_detail(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("http://localhost", 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"prompt exceeds available context size"}'))
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="prompt exceeds available context size"):
        LocalProvider()._run_via_server([], [], "sys")

@pytest.mark.parametrize("operation", ["write", "patch"])
def test_python_syntax_rejected_without_mutation(tmp_path, operation):
    from nexus_agent_ai.agent.tools import execute_write_file, execute_patch_file
    p = tmp_path / "calculator.py"
    original = "def multiply(a, b):\n    return a + b\n"
    p.write_text(original)
    from nexus_agent_ai.agent.tools import execute_read_file
    execute_read_file(str(p))
    if operation == "write":
        result = execute_write_file(str(p), "def broken(:")
    else:
        result = execute_patch_file(str(p), "multiply", "return a * b")
    assert "syntax validation failed" in result
    assert p.read_text() == original
    assert not p.with_name("calculator.py.bak").exists()

def test_onboarding_keeps_explicit_model(monkeypatch):
    from nexus_agent_ai.cli import onboarding as onboarding
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "LFM2.5-1.2B-Instruct-Q4_0.gguf")
    monkeypatch.setattr(onboarding, "detect_system_specs", lambda: {"ram_gb": 2, "cpu_cores": 2, "gpu": None, "avx2": False, "cpu_name": "test CPU"})
    written = []
    monkeypatch.setattr(onboarding, "_write_env_key", lambda *a: written.append(a))
    onboarding._step_system_specs()
    assert not written
    assert LocalProvider().model_id == "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"

def test_model_cannot_hide_failed_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = MaliciousProvider("read_file", {"path": "missing.py"})
    result = Agent(p, verbose=False).run("Read missing.py")
    assert "Tool errors occurred" in result
    assert "File not found" in result

def test_repl_review_capabilities_are_restored():
    # Verify the CLI wraps one review turn, not all subsequent edit turns.
    import inspect
    from nexus_agent_ai.cli.app import repl
    source = inspect.getsource(repl)
    assert "request_tools = get_readonly_tools()" in source
    assert "finally:\n                    agent.tools = original_tools" in source
