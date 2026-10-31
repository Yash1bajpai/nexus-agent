"""Actual file edits must not turn a recognized assignment request into false success."""
import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.cli import app as cli
from tests.test_real_user_failures import Transcript

PROMPT = "Change x to 2 in s2target.py"
CLAIM = "The file has been updated with x set to 2."


@pytest.mark.parametrize("tool,candidate,stream,complete", [
    ("write_file", "2\n", False, False),
    ("patch_file", "2\n", True, False),
    ("write_file", "x = 2\n", False, True),
    ("patch_file", "x = 2\n", True, True),
    ("write_file", "x = 2\nx = 3\n", False, False),
    ("patch_file", "x = 2\nx = 3\n", False, False),
])
def test_assignment_completion_checks_actual_final_file(tmp_path, tool, candidate, stream, complete):
    path = tmp_path / "s2target.py"
    original = "x = 1\n"
    path.write_text(original, encoding="utf-8")
    args = {"path": path.name}
    if tool == "write_file":
        args["content"] = candidate
    else:
        args.update(target=original, replacement=candidate)
    provider = Transcript([(tool, args), CLAIM])
    events = []
    agent = Agent(provider, max_iterations=2, verbose=False, event_callback=events.append)

    final = agent.run(PROMPT, stream=stream)

    assert path.read_text(encoding="utf-8") == candidate
    assert path.with_name(path.name + ".bak").read_text(encoding="utf-8") == original
    assert not provider.turns  # Checking must neither repair nor ask for another model turn.
    assert agent.changed_files == {str(path.resolve())}
    assert not agent.unresolved_errors
    assert bool(agent.requirements_incomplete) is not complete
    assert cli._check_run_outcome(agent, exit_on_failure=False) is complete
    if not complete:
        assert "incomplete" in final.lower() and "s2target.py" in final
        assert "x" in " ".join(agent.requirements_incomplete)
        if stream:
            visible = "".join(e["content"] for e in events if e["type"] == "stream_chunk")
            assert "incomplete" in visible.lower()
            assert visible.count(CLAIM) == 1


@pytest.mark.parametrize("stream_flags", [["--no-stream"], []], ids=["plain", "stream"])
def test_destructive_assignment_edit_is_nonzero_in_one_shot_cli(tmp_path, monkeypatch, stream_flags):
    path = tmp_path / "s2target.py"
    path.write_text("x = 1\n", encoding="utf-8")
    provider = Transcript([
        ("write_file", {"path": path.name, "content": "2\n"}),
        CLAIM,
    ])
    monkeypatch.setattr(cli, "get_provider_instance", lambda *args, **kwargs: (provider, "offline-transcript"))

    result = CliRunner().invoke(cli.app, ["chat", "-p", "local", PROMPT, *stream_flags])

    assert result.exit_code == 1
    assert "incomplete" in result.output.lower() and "s2target.py" in result.output
    assert "function names" not in result.output  # Assignment errors must not be mislabeled.
    assert path.read_text(encoding="utf-8") == "2\n"
    assert path.with_name(path.name + ".bak").read_text(encoding="utf-8") == "x = 1\n"


def test_assignment_requirement_does_not_survive_next_request(tmp_path):
    path = tmp_path / "s2target.py"
    path.write_text("x = 1\n", encoding="utf-8")
    agent = Agent(Transcript([
        ("write_file", {"path": path.name, "content": "2\n"}), CLAIM, "Hello.",
    ]), max_iterations=2, verbose=False)
    agent.run(PROMPT)
    assert agent.requirements_incomplete

    assert agent.run("Hello") == "Hello."
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("prompt", [
    'Explain this example: "Change x to 2 in s2target.py"',
    "```\nChange x to 2 in s2target.py\n```",
    "Change x to 2 in s2target.py or other.py",
    "Change x to 2 in s2target.py and change y to 3 in other.py",
    "Explain how to change x to 2 in s2target.py",
])
def test_unrecognized_assignment_phrasing_gets_no_invented_check(tmp_path, prompt):
    (tmp_path / "s2target.py").write_text("2\n", encoding="utf-8")
    agent = Agent(Transcript(["This is an example, not a verification claim."]), tools=[], verbose=False)

    final = agent.run(prompt)

    assert not agent.requirements_incomplete
    assert final == "This is an example, not a verification claim."


def test_matching_unrelated_identifier_does_not_satisfy_request(tmp_path):
    (tmp_path / "s2target.py").write_text("x = 2\n", encoding="utf-8")
    agent = Agent(Transcript(["No changes are needed."]), tools=[], verbose=False)

    agent.run("Change y to 2 in s2target.py")

    assert agent.requirements_incomplete
    assert "y" in " ".join(agent.requirements_incomplete)
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("file_state", ["missing", "directory", "syntax-error"])
def test_uncheckable_assignment_source_is_unverified(tmp_path, file_state):
    path = tmp_path / "s2target.py"
    if file_state == "directory":
        path.mkdir()
    elif file_state == "syntax-error":
        path.write_text("x = (\n", encoding="utf-8")
    agent = Agent(Transcript(["No changes are needed."]), tools=[], verbose=False)

    final = agent.run(PROMPT)

    assert agent.requirements_incomplete
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert "incomplete" in final.lower() and "s2target.py" in final


def test_bare_retry_keeps_assignment_verification_despite_attached_source(tmp_path):
    path = tmp_path / "s2target.py"
    path.write_text("x = 1\n", encoding="utf-8")
    agent = Agent(Transcript([
        ("write_file", {"path": path.name, "content": "2\n"}), CLAIM,
        ("write_file", {"path": path.name, "content": "3\n"}), CLAIM,
    ]), max_iterations=2, verbose=False)
    agent.run(PROMPT)
    assert agent.requirements_incomplete

    agent.run("try again")

    assert path.read_text(encoding="utf-8") == "3\n"
    assert agent.requirements_incomplete
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
