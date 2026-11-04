"""Offline scripted completion checks; tool execution is real, model decisions are not."""

import ast
from collections import deque

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


ORIGINAL_E08_REQUEST = "add a subtract function to calculator.py"
ADD_SOURCE = "def add(a, b):\n    return a + b\n"
CORRUPTED_ADD_SOURCE = "def add(a, b):\n    return a - b\n"
SUBTRACT_SOURCE = "def subtract(a, b):\n    return a - b"
CLAIM = "I added the subtract function to calculator.py."
PRESERVED_SOURCE = (
    "# café: preserve this comment and its CRLF bytes.\r\n"
    "def add(a, b):\r\n"
    "    # Existing addition must remain byte-for-byte unchanged.\r\n"
    "    return a + b\r\n"
    "\r\n"
    "# Fin de fichier — keep this trailing comment.\r\n"
).encode("utf-8")


class ScriptedCompletionProvider(BaseProvider):
    model = "offline-c1-transcript"
    is_local = True

    def __init__(self, turns):
        self.turns = deque(turns)
        self.request_count = 0

    def complete(self, messages, tools, system):
        self.request_count += 1
        assert self.turns, "Completion checking made an unexpected provider request"
        turn = self.turns.popleft()
        if isinstance(turn, tuple):
            name, args = turn
            return ProviderResponse(tool_calls=[
                ToolCall(f"c1-{self.request_count}", name, args),
            ])
        return ProviderResponse(text=turn)

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": result}


@pytest.fixture(autouse=True)
def explicit_tool_execution(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    monkeypatch.setenv("NEXUS_AGENT_AUTO_CHECK", "0")


def corrupting_transcript():
    return [
        ("read_file", {"path": "calculator.py"}),
        ("patch_file", {
            "path": "calculator.py",
            "target": "return a + b",
            "replacement": "return a - b",
        }),
        CLAIM,
    ]


def additive_transcript():
    return [
        ("read_file", {"path": "calculator.py"}),
        ("add_function", {
            "path": "calculator.py",
            "function_name": "subtract",
            "new_body": SUBTRACT_SOURCE,
        }),
        CLAIM,
    ]


def assert_real_execution(agent, provider, events, path, mutation_tool):
    assert provider.request_count == 3
    assert not provider.turns
    assert agent.changed_files == {str(path.resolve())}
    assert not agent.unresolved_errors
    assert not agent.claimed_unapplied_edit
    assert not agent.claim_unverified
    assert not agent.unfulfilled_action
    if events is not None:
        assert [event["name"] for event in events if event["type"] == "action"] == [
            "read_file", mutation_tool,
        ]
        observations = [event for event in events if event["type"] == "observe"]
        assert [event["name"] for event in observations] == ["read_file", mutation_tool]
        assert not any(event["result"].startswith("ERROR:") for event in observations)
    messages = agent.memory.get()
    assert any(
        message.get("role") == "user" and message.get("content") == ORIGINAL_E08_REQUEST
        for message in messages
    )
    receipts = [message for message in messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in receipts] == ["c1-1", "c1-2"]
    assert [message["metadata"]["executed"] for message in receipts] == [True, True]
    mutations = agent.memory.get_session_metadata("MUTATION_RECORD", [])
    assert len(mutations) == 1
    assert mutations[0]["path"] == str(path.resolve())
    assert mutations[0]["action"] == mutation_tool
    assert mutations[0]["status"] == "changed"
    assert mutations[0]["success"] is True
    assert mutations[0]["failed"] is False
    # Exact action/receipt lists above exclude fabricated run_tests/run_code evidence.


def assert_names_only_disclaimer(text):
    rendered = " ".join(text.split())
    assert "checks names only" in rendered
    assert "not verified" in rendered


def invoke_cli(monkeypatch, provider, stream):
    instances = []

    def capture_agent(*args, **kwargs):
        agent = Agent(*args, **kwargs)
        instances.append(agent)
        return agent

    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **kw: (provider, "local"))
    monkeypatch.setattr(cli, "Agent", capture_agent)
    args = ["chat", ORIGINAL_E08_REQUEST, "--max-iterations", "4"]
    if not stream:
        args.append("--no-stream")
    result = CliRunner().invoke(cli.app, args)
    assert len(instances) == 1, result.output
    return result, instances[0]


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_original_e08_corrupting_patch_remains_but_completion_is_incomplete(tmp_path, stream):
    path = tmp_path / "calculator.py"
    path.write_bytes(ADD_SOURCE.encode("utf-8"))
    events = []
    provider = ScriptedCompletionProvider(corrupting_transcript())
    agent = Agent(provider, max_iterations=4, verbose=False, event_callback=events.append)

    final = agent.run(ORIGINAL_E08_REQUEST, stream=stream)

    # C1 must expose both defects honestly, not repair/prevent the C2 wrong edit.
    assert path.read_bytes() == CORRUPTED_ADD_SOURCE.encode("utf-8")
    assert [node.name for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)] == ["add"]
    assert_real_execution(agent, provider, events, path, "patch_file")
    assert agent.requirements_incomplete
    assert "calculator.py" in " ".join(agent.requirements_incomplete)
    assert "subtract" in " ".join(agent.requirements_incomplete)
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert final.startswith(CLAIM)
    correction = final[len(CLAIM):]
    assert "Request incomplete:" in correction
    assert "subtract" in correction and "calculator.py" in correction
    assert "partial edits remain in place; no automatic repair was attempted" in correction
    assert "not verification of arbitrary program behavior or every instruction" in correction
    assert "Completion check:" not in final
    if stream:
        rendered = "".join(event["content"] for event in events if event["type"] == "stream_chunk")
        assert rendered.count(CLAIM) == 1
        assert correction.strip() in rendered


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_original_e08_corrupting_patch_exits_nonzero_in_cli(tmp_path, monkeypatch, stream):
    path = tmp_path / "calculator.py"
    path.write_bytes(ADD_SOURCE.encode("utf-8"))
    provider = ScriptedCompletionProvider(corrupting_transcript())

    result, agent = invoke_cli(monkeypatch, provider, stream)

    assert result.exit_code == 1, result.output
    assert path.read_bytes() == CORRUPTED_ADD_SOURCE.encode("utf-8")
    assert agent.requirements_incomplete
    assert_real_execution(agent, provider, None, path, "patch_file")
    rendered = " ".join(result.output.split())
    assert "Request incomplete:" in rendered
    assert "subtract" in rendered and "calculator.py" in rendered
    assert "partial edits remain in place" in rendered
    assert "Completion check:" not in rendered


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_original_e08_real_add_function_preserves_bytes_and_reports_names_only(tmp_path, stream):
    path = tmp_path / "calculator.py"
    path.write_bytes(PRESERVED_SOURCE)
    events = []
    provider = ScriptedCompletionProvider(additive_transcript())
    agent = Agent(provider, max_iterations=4, verbose=False, event_callback=events.append)

    final = agent.run(ORIGINAL_E08_REQUEST, stream=stream)

    assert path.read_bytes() == PRESERVED_SOURCE + (SUBTRACT_SOURCE + "\n").encode("utf-8")
    assert [node.name for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.FunctionDef)] == ["add", "subtract"]
    assert_real_execution(agent, provider, events, path, "add_function")
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)
    assert final.startswith(CLAIM)
    assert_names_only_disclaimer(final)
    if stream:
        rendered = "".join(event["content"] for event in events if event["type"] == "stream_chunk")
        assert rendered.count(CLAIM) == 1
        assert_names_only_disclaimer(rendered)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_original_e08_real_add_function_cli_success_is_names_only(tmp_path, monkeypatch, stream):
    path = tmp_path / "calculator.py"
    path.write_bytes(PRESERVED_SOURCE)
    provider = ScriptedCompletionProvider(additive_transcript())

    result, agent = invoke_cli(monkeypatch, provider, stream)

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == PRESERVED_SOURCE + (SUBTRACT_SOURCE + "\n").encode("utf-8")
    assert_real_execution(agent, provider, None, path, "add_function")
    assert not agent.requirements_incomplete
    assert_names_only_disclaimer(result.output)


def test_original_e08_existing_name_does_not_verify_signature_or_arithmetic(tmp_path):
    path = tmp_path / "calculator.py"
    original = ADD_SOURCE + "\ndef subtract(unrelated):\n    return 'not arithmetic'\n"
    path.write_bytes(original.encode("utf-8"))
    provider = ScriptedCompletionProvider(["The subtract function is already present."])
    events = []
    agent = Agent(provider, tools=[], verbose=False, event_callback=events.append)

    final = agent.run(ORIGINAL_E08_REQUEST)

    assert path.read_bytes() == original.encode("utf-8")
    assert provider.request_count == 1 and not provider.turns
    assert not agent.changed_files
    assert not any(event["type"] in {"action", "observe"} for event in events)
    assert not agent.requirements_incomplete
    assert not agent.claim_unverified
    assert cli._check_run_outcome(agent, exit_on_failure=False)
    assert_names_only_disclaimer(final)
