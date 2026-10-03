"""Context-window options must reach the engine without changing process config."""
import importlib
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse
from nexus_agent_ai.utils.config import ConfigError

cli = importlib.import_module("nexus_agent_ai.cli.app")

class DummyProvider(BaseProvider):
    model = "test"
    context_size = 8192
    def complete(self, messages, tools, system):
        return ProviderResponse(text="done")
    def stream(self, **kwargs):
        yield self.complete(**kwargs)
    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}

@pytest.fixture(autouse=True)
def no_onboarding(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)

@pytest.mark.parametrize("command", ["chat", "repl"])
def test_option_reaches_factory(command, monkeypatch):
    calls = []
    def factory(name, context_size=None):
        calls.append((name, context_size))
        return DummyProvider(), "local"
    monkeypatch.setattr(cli, "get_provider_instance", factory)
    args = [command] + (["hello"] if command == "chat" else [])
    result = CliRunner().invoke(cli.app, args + ["--provider", "local", "--context-size", "8192"], input="/context\n/exit\n")
    assert result.exit_code == 0, result.output
    assert calls == [("local", 8192)]
    if command == "repl":
        assert "8192 tokens" in result.output
        assert "/compact" in result.output and "context:" in result.output

@pytest.mark.parametrize("command", ["chat", "repl"])
@pytest.mark.parametrize("value", ["511", "0", "-1", "abc", "8192.5"])
def test_bad_cli_value_rejected_before_provider(command, value, monkeypatch):
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: pytest.fail("provider should not start"))
    args = [command] + (["hello"] if command == "chat" else [])
    result = CliRunner().invoke(cli.app, args + ["--context-size", value])
    assert result.exit_code == 2

@pytest.mark.parametrize("provider", ["openai", "gemini", "anthropic", "ollama", "auto", "unknown"])
def test_nonlocal_explicit_context_rejected(provider):
    with pytest.raises(ConfigError, match="requires --provider local"):
        cli.get_provider_instance(provider, context_size=8192)

@pytest.mark.parametrize("alias", ["local", "liquid", "lfm", "default", "demo"])
def test_alias_passes_context_to_provider(alias, monkeypatch):
    monkeypatch.setattr(cli, "_make_local_provider", lambda context_size=None: (context_size, "local"))
    assert cli.get_provider_instance(alias, context_size=8192) == (8192, "local")

def test_precedence_and_no_env_mutation(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "16384")
    assert LocalProvider(context_size=8192).context_size == 8192
    assert LocalProvider().context_size == 16384
    monkeypatch.delenv("NEXUS_CONTEXT_SIZE")
    assert LocalProvider().context_size == 4096
    assert LocalProvider(context_size=512).context_size == 512

@pytest.mark.parametrize("value", [True, False, 511, -1, "bad", 8192.5])
def test_bad_provider_value(value):
    with pytest.raises(ValueError, match="integer of at least 512"):
        LocalProvider(context_size=value)

def test_invalid_env_is_actionable(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "bad")
    with pytest.raises(ValueError, match="integer of at least 512"):
        LocalProvider()
    assert LocalProvider(context_size=8192).context_size == 8192

def test_llama_cpp_receives_window(tmp_path, monkeypatch):
    model = tmp_path / "fake.gguf"
    model.write_text("not a model")
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=lambda **kw: calls.append(kw) or object()))
    p = LocalProvider(context_size=8192)
    monkeypatch.setattr(p, "setup_model", lambda: str(model))
    p._ensure_loaded()
    assert calls[0]["n_ctx"] == 8192

def test_existing_server_not_silently_reused(monkeypatch):
    p = LocalProvider(context_size=8192)
    monkeypatch.setattr(p, "_is_server_alive", lambda: True)
    with pytest.raises(RuntimeError, match="context size is unverified"):
        p._start_llama_server("unused.gguf")

def test_unsupported_ollama_fallback_rejected(monkeypatch):
    p = LocalProvider(context_size=8192)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr(p, "setup_model", lambda: "not-a-file")
    with pytest.raises(RuntimeError, match="cannot be applied to the Ollama fallback"):
        p._ensure_loaded()

def test_gpu_not_silently_ignoring_option(monkeypatch):
    p = LocalProvider(context_size=8192)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(p, "setup_model", lambda: "not-a-file")
    with pytest.raises(RuntimeError, match="not the Transformers engine"):
        p._ensure_loaded()

def test_help_documents_option():
    for cmd in ["chat", "repl"]:
        result = CliRunner().invoke(cli.app, [cmd, "--help"])
        assert result.exit_code == 0
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)  # CI renders Rich help with ANSI codes
        assert "--context-size" in plain

def test_server_launch_receives_window(tmp_path, monkeypatch):
    mod = importlib.import_module("nexus_agent_ai.providers.local_provider")
    monkeypatch.setattr(mod, "_SERVER_DIR", tmp_path)
    p = LocalProvider(context_size=8192)
    checks = iter([False, True])
    monkeypatch.setattr(p, "_is_server_alive", lambda: next(checks))
    monkeypatch.setattr(p, "_download_llama_server", lambda: "fake-server")
    monkeypatch.setattr("time.sleep", lambda _: None)
    commands = []
    class Proc:
        pid = 12345
        def poll(self): return 0
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: commands.append(cmd) or Proc())
    model = tmp_path / "model.gguf"
    model.write_text("test")
    p._start_llama_server(str(model))
    assert commands[0][commands[0].index("-c") + 1] == "8192"
    assert commands[0][commands[0].index("--parallel") + 1] == "1"
    p.stop()
