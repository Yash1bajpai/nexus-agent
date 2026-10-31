"""Context-window options must reach the engine without changing process config."""
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.providers.local_provider import LocalProvider

cli = importlib.import_module("nexus_agent_ai.cli.app")


@pytest.fixture(autouse=True)
def no_onboarding(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)


@pytest.mark.parametrize("command", ["chat", "repl"])
@pytest.mark.parametrize("value", ["511", "0", "-1", "abc", "8192.5"])
def test_bad_cli_value_rejected_before_provider(command, value, monkeypatch):
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: pytest.fail("provider should not start"))
    args = [command] + (["hello"] if command == "chat" else [])
    result = CliRunner().invoke(cli.app, args + ["--context-size", value])
    assert result.exit_code == 2


def test_precedence_and_no_env_mutation(monkeypatch):
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "16384")
    assert LocalProvider(context_size=8192).context_size == 8192
    assert LocalProvider().context_size == 16384
    monkeypatch.delenv("NEXUS_CONTEXT_SIZE")
    assert LocalProvider().context_size == 4096
    assert LocalProvider(context_size=512).context_size == 512

def test_local_output_cap_precedence_and_no_env_mutation(monkeypatch):
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "3072")
    assert LocalProvider().max_output_tokens == 3072
    assert LocalProvider(max_output_tokens=37).max_output_tokens == 37
    assert os.environ["NEXUS_MAX_OUTPUT_TOKENS"] == "3072"
    monkeypatch.delenv("NEXUS_MAX_OUTPUT_TOKENS")
    assert LocalProvider().max_output_tokens == 2048


@pytest.mark.parametrize("command", ["chat", "repl"])
@pytest.mark.parametrize("value", ["0", "-1", "abc", "37.5"])
def test_bad_output_cap_rejected_before_startup(command, value, monkeypatch):
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: pytest.fail("provider should not start"))
    args = [command] + (["hello"] if command == "chat" else [])
    result = CliRunner().invoke(cli.app, args + ["--max-output-tokens", value])
    assert result.exit_code == 2


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

def test_own_verified_server_accepts_explicit_context(tmp_path, monkeypatch):
    p = LocalProvider(context_size=8192)
    model = tmp_path / "model.gguf"
    p._server_model_path = str(model.resolve())
    p._server_context_size = 8192
    monkeypatch.setattr(p, "_is_server_alive", lambda: True)
    monkeypatch.setattr(p, "_download_llama_server", lambda: pytest.fail("must reuse owned matching server"))
    p._start_llama_server(str(model))
    assert p._model_instance == "llama_server"

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


def test_server_launch_receives_window(tmp_path, monkeypatch):
    mod = importlib.import_module("nexus_agent_ai.providers.local_provider")
    monkeypatch.setattr(mod, "_SERVER_DIR", tmp_path)
    p = LocalProvider(context_size=8192)
    monkeypatch.setattr(p, "_is_server_alive", lambda: True)
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cutoff", [False, True])
def test_local_default_budget_finishes_short_edit_without_overcommitting_output(tmp_path, monkeypatch, stream, cutoff):
    from nexus_agent_ai.agent import context
    from nexus_agent_ai.agent.core import Agent

    monkeypatch.delenv("NEXUS_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("NEXUS_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    path = tmp_path / "f.py"
    path.write_text("x = 1\n")
    requests = []
    reasoning = "Preserving the surrounding file contents while applying this small change. " * 10

    def completion(**request):
        requests.append(request)
        index = len(requests)
        if index == 2:
            call = {"id": "read", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "f.py"}),
            }}
        else:
            call = {"id": f"patch_{index}", "type": "function", "function": {
                "name": "patch_file",
                "arguments": json.dumps({
                    "path": "f.py", "target": "x = 0" if index in (1, 3) else "x = 1",
                    "replacement": "x = 2",
                }),
            }}
        truncated = cutoff and index == 4
        if truncated:
            call["function"]["arguments"] = '{"path":"f.py","replacement":"unfinished'
        return {
            "choices": [{"message": {
                "content": reasoning if index <= 4 else "Changed f.py.",
                "tool_calls": [call] if index <= 4 else [],
            }, "finish_reason": "length" if truncated else "tool_calls" if index <= 4 else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    provider = LocalProvider()
    provider._llama = SimpleNamespace(create_chat_completion=completion)
    agent = Agent(provider, verbose=False)
    agent.run("Change x to 2 in f.py", stream=stream)

    assert (provider.context_size, provider.max_output_tokens) == (4096, 2048)
    assert provider.request_output_cap == 2048
    if cutoff:
        assert agent.response_incomplete
        assert path.read_text() == "x = 1\n"
        assert agent.changed_files == set()
    else:
        assert path.read_text() == "x = 2\n"
        assert not agent.unresolved_errors
        assert not agent.edit_not_applied and not agent.response_incomplete
    for request in requests:
        input_tokens = (context.estimate_tokens(request["messages"][0]["content"])
                        + context.tools_tokens(agent.tools)
                        + context.history_tokens(request["messages"][1:]))
        assert input_tokens + request["max_tokens"] <= 4096
        assert request["max_tokens"] > 0


def test_local_adaptation_still_refuses_an_unfit_structured_turn(monkeypatch):
    from nexus_agent_ai.agent import context
    from nexus_agent_ai.agent.core import Agent

    monkeypatch.delenv("NEXUS_CONTEXT_SIZE", raising=False)
    monkeypatch.delenv("NEXUS_MAX_OUTPUT_TOKENS", raising=False)
    provider = LocalProvider()
    agent = Agent(provider, tools=[], verbose=False)
    arguments = json.dumps({"path": "large.py", "content": "x" * 20000})
    messages = [
        {"role": "user", "content": "Write large.py"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "large", "type": "function",
            "function": {"name": "write_file", "arguments": arguments},
        }]},
        {"role": "tool", "tool_call_id": "large", "content": "Written"},
    ]
    with pytest.raises(context.ContextBudgetError):
        agent._fit(messages, [])
