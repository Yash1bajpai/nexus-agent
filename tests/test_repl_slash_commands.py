"""REPL slash commands that call typer commands must not leak typer OptionInfo defaults (2.10.0 bug)."""
import importlib
import subprocess

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse

cli = importlib.import_module("nexus_agent_ai.cli.app")


class MsgProvider(BaseProvider):
    model = "test"
    context_size = 4096
    def complete(self, messages, tools, system):
        return ProviderResponse(text="feat(demo): add change")
    def stream(self, **kwargs):
        yield self.complete(**kwargs)
    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (MsgProvider(), "local"))


def test_repl_pull_model_no_args_and_with_args(monkeypatch):
    seen = []

    class FakeLocal:
        def __init__(self, model_id, filename):
            seen.append((model_id, filename))
        def setup_model(self, verify_download=False):
            return "/fake/path.gguf"

    monkeypatch.setattr("nexus_agent_ai.providers.local_provider.LocalProvider", FakeLocal)
    monkeypatch.setattr("nexus_agent_ai.providers.local_provider.ensure_llama_server_binary", lambda: None)
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "env/repo")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "env.gguf")
    r = CliRunner().invoke(cli.app, ["repl", "--no-persist"],
                           input="/pull-model\n/pull-model --repo a/b --file c.gguf\n/pull-model --bogus\n/exit\n")
    assert r.exit_code == 0, r.output
    assert "OptionInfo" not in r.output
    assert seen == [("env/repo", "env.gguf"), ("a/b", "c.gguf")]
    assert "Usage: /pull-model" in r.output


def test_repl_commit_flow_cancel_and_confirm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    run("add", ".")
    run("commit", "-qm", "init")
    (tmp_path / "a.txt").write_text("two\n")
    r = CliRunner().invoke(cli.app, ["repl", "--no-persist"], input="/commit\nn\n/exit\n")
    assert r.exit_code == 0, r.output
    assert "OptionInfo" not in r.output
    assert "Generated message: feat(demo): add change" in r.output
    assert "Commit aborted" in r.output
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert n == "1"
    r = CliRunner().invoke(cli.app, ["repl", "--no-persist"], input="/commit\ny\n/exit\n")
    assert "OptionInfo" not in r.output
    n = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert n == "2"
