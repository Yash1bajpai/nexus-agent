import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


class ScriptedProvider(BaseProvider):
    model = "scripted-cloud"

    def __init__(self, steps=(), text="Done."):
        self.steps = iter(steps)
        self.text = text
        self.counter = 0

    def complete(self, messages, tools, system):
        step = next(self.steps, None)
        if step is None:
            return ProviderResponse(text=self.text)
        name, args = step
        self.counter += 1
        call_id = f"call_{self.counter}"
        return ProviderResponse(
            tool_calls=[ToolCall(call_id, name, args)],
            raw_assistant_message={
                "role": "assistant", "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                }}],
            },
        )

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


@pytest.fixture
def prepare(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    (tmp_path / "calc.py").write_text("def multiply(a, b):\n    return a + b\n", encoding="utf-8")

    def use(provider):
        monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (provider, "scripted"))
        return CliRunner()

    return use


@pytest.mark.parametrize("args", [
    ["chat", "fix calc.py"],
    ["debug", "calc.py", "--error", "multiply adds"],
    ["generate", "multiply the arguments", "--output", "calc.py"],
])
def test_unapplied_edit_claim_fails_every_edit_command(prepare, tmp_path, args):
    original = (tmp_path / "calc.py").read_bytes()
    result = prepare(ScriptedProvider(text="I fixed the bug.")).invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert (tmp_path / "calc.py").read_bytes() == original


@pytest.mark.parametrize("stream", [False, True])
def test_failed_patch_and_unapplied_claim_exit_nonzero(prepare, tmp_path, stream, monkeypatch):
    original = (tmp_path / "calc.py").read_bytes()
    instances = []
    agent_type = cli.Agent

    def capture_agent(*args, **kwargs):
        agent = agent_type(*args, **kwargs)
        instances.append(agent)
        return agent

    monkeypatch.setattr(cli, "Agent", capture_agent)
    provider = ScriptedProvider([
        ("read_file", {"path": "calc.py"}),
        ("patch_file", {"path": "calc.py", "target": "return a - b", "replacement": "return a * b"}),
    ], text="I fixed the bug.")
    provider.is_local = True
    args = ["chat", "fix calc.py", "-p", "local", "--max-iterations", "3"]
    if not stream:
        args.append("--no-stream")
    result = prepare(provider).invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert (tmp_path / "calc.py").read_bytes() == original
    assert instances[0].unresolved_errors
    assert instances[0].claimed_unapplied_edit


def test_existing_output_is_not_proof_of_generation(prepare, tmp_path):
    original = (tmp_path / "calc.py").read_bytes()
    result = prepare(ScriptedProvider(text="The file is already correct; no changes are needed.")).invoke(
        cli.app, ["generate", "produce a multiply function", "--output", "calc.py", "--no-stream"]
    )
    assert result.exit_code == 1, result.output
    assert (tmp_path / "calc.py").read_bytes() == original


@pytest.mark.parametrize("existing", [False, True])
def test_generation_requires_a_real_target_write(prepare, tmp_path, existing):
    name = "calc.py" if existing else "new.py"
    content = "def multiply(a, b):\n    return a * b\n"
    steps = [("read_file", {"path": name})] if existing else []
    steps.append(("write_file", {"path": name, "content": content}))
    result = prepare(ScriptedProvider(steps)).invoke(
        cli.app, ["generate", "multiply the arguments", "--output", name, "--no-stream"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / name).read_text(encoding="utf-8") == content
    assert ("updated" if existing else "created") in result.output


def test_writing_an_unrelated_file_does_not_satisfy_generation(prepare, tmp_path):
    provider = ScriptedProvider([("write_file", {"path": "unrelated.txt", "content": "changed"})])
    result = prepare(provider).invoke(cli.app, ["generate", "produce multiply", "-o", "calc.py", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert (tmp_path / "unrelated.txt").read_text() == "changed"
    assert "return a + b" in (tmp_path / "calc.py").read_text()


def test_debug_succeeds_after_applying_the_actual_fix(prepare, tmp_path):
    provider = ScriptedProvider([
        ("read_file", {"path": "calc.py"}),
        ("patch_file", {"path": "calc.py", "target": "return a + b", "replacement": "return a * b"}),
    ])
    result = prepare(provider).invoke(cli.app, ["debug", "calc.py", "--error", "multiply adds", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "return a * b" in (tmp_path / "calc.py").read_text()


def test_verification_only_false_test_claim_fails_in_streaming_chat(prepare):
    result = prepare(ScriptedProvider(text="All tests pass.")).invoke(
        cli.app, ["chat", "Check whether the test suite passes."]
    )
    assert result.exit_code == 1, result.output
    assert "unverified" in result.output.lower()


def test_failed_readonly_review_is_not_a_success(prepare, tmp_path):
    provider = ScriptedProvider([("write_file", {"path": "forbidden.txt", "content": "bad"})])
    result = prepare(provider).invoke(cli.app, ["review", "calc.py", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert not (tmp_path / "forbidden.txt").exists()


def test_successful_readonly_review_does_not_require_a_file_edit(prepare, tmp_path):
    original = (tmp_path / "calc.py").read_bytes()
    provider = ScriptedProvider(
        [("read_file", {"path": "calc.py"})],
        text="The multiply function incorrectly adds its arguments.",
    )
    result = prepare(provider).invoke(cli.app, ["review", "calc.py", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "calc.py").read_bytes() == original
    assert "no file was changed" not in result.output.lower()


def test_saved_provider_is_resolved_after_first_run(monkeypatch):
    from nexus_agent_ai.providers import openai_provider

    runner = CliRunner()
    monkeypatch.setenv("DEFAULT_PROVIDER", "local")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time",
                        lambda: monkeypatch.setenv("DEFAULT_PROVIDER", "openai"))
    monkeypatch.setattr(openai_provider, "OpenAIProvider", lambda: ScriptedProvider(text="Hello."))
    monkeypatch.setattr(cli, "_make_local_provider", lambda **kw: pytest.fail("stale local default was used"))
    result = runner.invoke(cli.app, ["chat", "hello", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "Hello." in result.output


def test_explicit_provider_overrides_wizard_selection(monkeypatch):
    from nexus_agent_ai.providers import openai_provider

    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time",
                        lambda: monkeypatch.setenv("DEFAULT_PROVIDER", "local"))
    monkeypatch.setattr(openai_provider, "OpenAIProvider", lambda: ScriptedProvider(text="Explicit choice."))
    monkeypatch.setattr(cli, "_make_local_provider", lambda **kw: pytest.fail("explicit choice was ignored"))
    result = CliRunner().invoke(cli.app, ["chat", "hello", "--provider", "openai", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "Explicit choice." in result.output


def test_safe_cached_quantization_is_reused_before_another_download(monkeypatch):
    from nexus_agent_ai.providers.local_provider import LocalProvider
    from nexus_agent_ai.utils import model_download, model_select

    cached = model_select.BIG_MODELS[1]
    path = model_download.local_path(cached["repo"], cached["filename"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached model fixture")
    monkeypatch.setattr(model_select, "available_ram_gb", lambda: 4.0)
    monkeypatch.setattr(model_select, "confirm_download",
                        lambda *a, **kw: pytest.fail("a fitting cached model was ignored"))
    provider = cli._fit_local_model(LocalProvider(), None, {})
    assert (provider.model_id, provider.filename) == (cached["repo"], cached["filename"])


def test_cached_large_model_does_not_bypass_low_ram_selection(monkeypatch):
    from nexus_agent_ai.providers.local_provider import LocalProvider
    from nexus_agent_ai.utils import model_download, model_select
    import typer

    cached = model_select.BIG_MODELS[0]
    path = model_download.local_path(cached["repo"], cached["filename"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached model fixture")
    monkeypatch.setattr(model_select, "available_ram_gb", lambda: 2.6)
    monkeypatch.setattr(model_select, "confirm_download", lambda *a, **kw: None)
    with pytest.raises(typer.Exit):
        cli._fit_local_model(LocalProvider(), None, {})


def test_accepting_smaller_download_does_not_load_oversized_cached_model(monkeypatch):
    from nexus_agent_ai.providers.local_provider import LocalProvider
    from nexus_agent_ai.utils import model_download, model_select

    cached = model_select.BIG_MODELS[0]
    path = model_download.local_path(cached["repo"], cached["filename"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"cached model fixture")
    monkeypatch.setattr(model_select, "available_ram_gb", lambda: 2.6)
    monkeypatch.setattr(model_select, "confirm_download", lambda choice, *a, **kw: choice)
    selected = cli._fit_local_model(LocalProvider(), None, {})
    assert selected.filename == model_select.SMALL_MODEL["filename"]
    assert model_select.required_ram_gb(model_select.SMALL_MODEL) <= 2.6


def test_home_root_guard_contains_recovery_guidance(monkeypatch):
    from nexus_agent_ai.agent.tools import _validate_workspace_path
    home = Path.home().resolve()
    monkeypatch.chdir(home)
    err = _validate_workspace_path(home)
    assert isinstance(err, str)
    assert "create a project folder (mkdir my-project, cd my-project) and relaunch" in err


def test_chat_startup_home_notice_only_in_global_mode(tmp_path, monkeypatch):
    home = Path.home().resolve()
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (ScriptedProvider(text="Answer"), "stub"))
    runner = CliRunner()

    # Global mode: cwd == home
    monkeypatch.chdir(home)
    res_global = runner.invoke(cli.app, ["chat", "hello", "--no-stream"])
    out_global = " ".join(res_global.output.split())
    assert "create a project folder (mkdir my-project, cd my-project) and relaunch" in out_global

    # Project mode: cwd has .git
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / ".git").mkdir()
    monkeypatch.chdir(proj)
    res_proj = runner.invoke(cli.app, ["chat", "hello", "--no-stream"])
    out_proj = " ".join(res_proj.output.split())
    assert "create a project folder (mkdir my-project, cd my-project) and relaunch" not in out_proj


def test_banner_resume_instruction_and_root_continue_passthrough(monkeypatch):
    home = Path.home().resolve()
    monkeypatch.chdir(home)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (ScriptedProvider(text="Answer"), "stub"))
    runner = CliRunner()

    # Check banner text in repl (sent /exit)
    res_repl = runner.invoke(cli.app, ["repl"], input="/exit\n")
    out_repl = " ".join(res_repl.output.split())
    assert "Resume later with: nexus-agent-ai chat --continue" in out_repl

    # Check root command --continue / -c routes to chat --continue
    invoked = []
    monkeypatch.setattr(cli, "chat", lambda *a, **k: invoked.append(k))
    r_long = runner.invoke(cli.app, ["--continue"])
    assert r_long.exit_code == 0
    assert invoked and invoked[-1].get("continue_last") is True

    invoked.clear()
    r_short = runner.invoke(cli.app, ["-c"])
    assert r_short.exit_code == 0
    assert invoked and invoked[-1].get("continue_last") is True


def _git_repo(tmp_path, track_calc=True):
    import subprocess
    def g(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True)
    g("init", "-q")
    if track_calc and (tmp_path / "calc.py").is_file():
        g("add", "calc.py")
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "chore: add calc")
    else:
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "chore: init")
    return g


def test_commit_subcommand_decline_does_not_commit(prepare, tmp_path):
    g = _git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def multiply(a, b):\n    return a * b * 2\n", encoding="utf-8")
    provider = ScriptedProvider(text="fix: correct multiply implementation")
    result = prepare(provider).invoke(cli.app, ["commit", "--no-stream"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Commit aborted" in result.output
    log = g("log", "--oneline").stdout
    assert "correct multiply" not in log


def test_commit_subcommand_confirm_commits_with_generated_message(prepare, tmp_path):
    g = _git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def multiply(a, b):\n    return a * b * 2\n", encoding="utf-8")
    provider = ScriptedProvider(text="fix: correct multiply implementation")
    result = prepare(provider).invoke(cli.app, ["commit", "--no-stream"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "fix: correct multiply implementation" in g("log", "--oneline").stdout


def test_commit_subcommand_no_changes_exits_nonzero(prepare, tmp_path):
    _git_repo(tmp_path)
    result = prepare(ScriptedProvider(text="unused")).invoke(cli.app, ["commit", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert "No staged or unstaged changes" in result.output


def test_commit_subcommand_untracked_only_fails_fast_without_generating(prepare, tmp_path):
    _git_repo(tmp_path, track_calc=False)
    provider = ScriptedProvider(text="unused")
    result = prepare(provider).invoke(cli.app, ["commit", "--no-stream"], input="y\n")
    assert result.exit_code == 1, result.output
    assert "Only untracked files" in result.output
    assert "Commit with this message" not in result.output
    assert provider.counter == 0


def test_commit_subcommand_outside_repo_exits_nonzero(prepare, tmp_path):
    result = prepare(ScriptedProvider(text="unused")).invoke(cli.app, ["commit", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert "Not inside a git repository" in result.output
