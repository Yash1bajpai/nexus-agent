"""Exact-fit dispatch guards preserve effects, pairing and duplicate truth."""
import copy
import hashlib
import io
import json
import math
import re
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console

from nexus_agent_ai.agent import context, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import RefusalClass, execution_metadata, execution_result, refusal
from nexus_agent_ai.cli import display
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.messages import assistant_message


class Offline(BaseProvider):
    model = "offline-r8-fixture"
    context_size = 4096
    max_output_tokens = 512
    is_local = False

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []
        self.offered = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        self.offered.append([tool.name for tool in tools])
        return next(self.steps)

    def stream(self, **kwargs):
        yield self.complete(**kwargs)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def call(name, args, identity="actual"):
    return ProviderResponse(tool_calls=[ToolCall(identity, name, args)])


def budget_agent(provider, offered, *, memory=None, verbose=False):
    agent = Agent(provider, memory=memory, tools=offered, verbose=verbose)
    system_tokens = 4096 - 512 - 1573 - context.tools_tokens(offered)
    assert system_tokens > 0
    agent.system = "s" * math.ceil((system_tokens - 1) * context.CHARS_PER_TOKEN)
    assert 4096 - 512 - context.estimate_tokens(agent.system) - context.tools_tokens(offered) == 1573
    return agent


@pytest.fixture(autouse=True)
def no_attachments(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")


def test_10495_token_stat_estimate_is_refused_without_read_or_fatal(tmp_path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_text("x" * (10495 * 4))
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("snapshot must not read rejected file"))
    monkeypatch.setattr(tools.READ_FILE_TOOL, "execute", lambda **kwargs: pytest.fail("handler must not read rejected file"))
    provider = Offline([
        call("read_file", {"path": "large.txt"}),
        ProviderResponse(text="The file was not read because it is too large."),
    ])
    agent = budget_agent(provider, [tools.READ_FILE_TOOL])
    agent.run("Read large.txt")
    assert len(provider.seen) == 2
    results = [message for message in agent.memory.get() if message["role"] == "tool"]
    assert len(results) == 1 and results[0]["tool_call_id"] == "actual"
    text = results[0]["content"]
    assert "~10495 tokens" in text and "the file was not read" in text
    available = int(re.search(r"~(\d+)-token usable context budget", text).group(1))
    assert 0 <= available < 1573
    assert str(path) not in tools._FILE_STATE
    assert agent.memory.get()[-1]["metadata"]["executor_closed"] is True
    assert results[0]["metadata"]["refusal_class"] == "capacity"
    assert results[0]["metadata"]["executed"] is False
    assert agent.unresolved_errors


def test_exact_serialized_boundary_preserves_whole_file_result(tmp_path):
    text = "x" * 1000
    (tmp_path / "small.txt").write_text(text)
    request = "Read small.txt"
    tool_call = ToolCall("boundary", "read_file", {"path": "small.txt"})
    observed = tools.execute_read_file("small.txt")
    expected = [
        {"role": "user", "content": request}, assistant_message("", [tool_call]),
        {"role": "tool", "tool_call_id": "boundary", "content": text,
         "metadata": execution_metadata(observed)},
    ]
    provider = Offline([ProviderResponse(tool_calls=[tool_call]), ProviderResponse(text="Read complete.")])
    agent = Agent(provider, tools=[tools.READ_FILE_TOOL], verbose=False)
    system_tokens = 4096 - 512 - context.tools_tokens(agent.tools) - context.history_tokens(expected)
    agent.system = "s" * math.ceil((system_tokens - 1) * context.CHARS_PER_TOKEN)
    agent.run(request)
    sent = provider.seen[1]
    assert sent == expected
    assert (context.estimate_tokens(agent.system) + context.tools_tokens(agent.tools)
            + context.history_tokens(sent) + 512) == 4096
    assert "[COMPACTED:" not in sent[-1]["content"]


def test_sandbox_block_then_identical_retry_reaches_real_sandboxed_execution(tmp_path, monkeypatch):
    actual_check = tools._sandbox_check
    checked = []

    def temporarily_block(code):
        checked.append(code)
        if len(checked) == 1:
            return refusal(RefusalClass.PERMISSION, "SANDBOX BLOCK: synthetic temporary refusal")
        return actual_check(code)

    monkeypatch.setattr(tools, "_sandbox_check", temporarily_block)
    args = {"code": "print(6 * 7)"}
    provider = Offline([
        call("run_code", args, "blocked"), call("run_code", args, "retry"),
        ProviderResponse(text="42"),
    ])
    agent = Agent(provider, tools=[tools.RUN_CODE_TOOL], verbose=False)
    agent.run("Calculate 6 times 7 using run_code")
    results = [message for message in agent.memory.get() if message["role"] == "tool"]
    assert len(checked) == 2
    assert results[0]["content"].startswith("SANDBOX BLOCK:")
    assert results[1]["content"] == "STDOUT:\n42"
    assert "Suppressed" not in results[1]["content"]
    assert agent.unresolved_errors == []


def test_genuine_directory_duplicate_suppresses_without_reexecution_and_forces_final(monkeypatch):
    executed = []
    original = tools.LIST_DIRECTORY_TOOL.execute

    def list_once(**kwargs):
        executed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(tools.LIST_DIRECTORY_TOOL, "execute", list_once)
    output = io.StringIO()
    monkeypatch.setattr(display, "console", Console(file=output, force_terminal=False, width=120))
    provider = Offline([
        call("list_directory", {"path": "."}, "one"),
        call("list_directory", {"path": "."}, "two"),
        call("list_directory", {"path": "."}, "three"),
        ProviderResponse(text="Directory inspected."),
    ])
    agent = Agent(provider, tools=[tools.LIST_DIRECTORY_TOOL], verbose=True)
    agent.run("List the directory")
    assert len(executed) == 1
    assert provider.offered[-1] == []
    results = [message for message in agent.memory.get() if message["role"] == "tool"]
    assert len(results) == 3
    assert all("Suppressed (duplicate)" in message["content"] for message in results[1:])
    rendered = output.getvalue()
    assert rendered.count("[OBSERVE] Done (") == 1
    assert rendered.count("[OBSERVE] Suppressed (duplicate) (") == 2


def test_executed_failing_pytest_output_is_retained_without_false_nonexecution(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    full = "=== Pytest Results: FAILED (exit code 1) ===\n" + "test detail\n" * 2000
    marker = tmp_path / "executed.marker"

    def pytest_handler(**kwargs):
        marker.write_text("handler executed")
        return execution_result(refusal(RefusalClass.OTHER, full), executed=True,
                                outcome="run_tests FAILED (exit code 1); 3 of 17 failed.")

    monkeypatch.setattr(tools.RUN_TESTS_TOOL, "execute", pytest_handler)
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="output")
    provider = Offline([call("run_tests", {}), ProviderResponse(text="The test run failed.")])
    agent = budget_agent(provider, [tools.RUN_TESTS_TOOL], memory=memory)
    agent.run("Run the tests")
    assert marker.read_text() == "handler executed"
    result = next(message for message in memory.get() if message["role"] == "tool")
    assert "run_tests FAILED (exit code 1); 3 of 17 failed" in result["content"]
    assert "Not executed" not in result["content"] and "retained as artifact" in result["content"]
    identity = hashlib.sha256(("run_tests\0" + full).encode()).hexdigest()
    assert identity in result["content"]
    with sqlite3.connect(memory.db_path) as conn:
        content, protected = conn.execute(
            "SELECT content, protected FROM context_artifacts WHERE session_id=? AND artifact_id=?",
            (memory.session_id, identity),
        ).fetchone()
    assert content == full and protected == 1
    assert context.history_tokens(provider.seen[1]) <= 1573


def test_opt_in_auto_check_output_guard_keeps_successful_write_truth(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    monkeypatch.setenv("NEXUS_AGENT_AUTO_CHECK", "1")
    full = "=== Pytest Results: FAILED (exit code 1) ===\n" + "test failure\n" * 2000
    monkeypatch.setattr(tools, "execute_run_tests", lambda: execution_result(
        refusal(RefusalClass.OTHER, full), executed=True, outcome="run_tests FAILED (exit code 1)."))
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="auto")
    provider = Offline([
        call("write_file", {"path": "created.txt", "content": "actual write"}),
        ProviderResponse(text="The file was created; its test run failed."),
    ])
    agent = budget_agent(provider, [tools.WRITE_FILE_TOOL, tools.RUN_TESTS_TOOL], memory=memory)
    agent.run("Create created.txt")
    assert (tmp_path / "created.txt").read_text() == "actual write"
    assert str(tmp_path / "created.txt") in agent.changed_files
    result = next(message for message in memory.get() if message["role"] == "tool")
    assert "Successfully wrote" in result["content"]
    assert "run_tests FAILED" in result["content"] and "retained as artifact" in result["content"]
    assert "no effect occurred" not in result["content"]
    assert context.history_tokens(provider.seen[1]) <= 1573


def test_handler_exception_after_observed_write_never_claims_no_file_changed(tmp_path, monkeypatch):
    def partial_write(path, content):
        Path(path).write_text(content)
        raise RuntimeError("controlled failure after the actual write")

    monkeypatch.setattr(tools.WRITE_FILE_TOOL, "execute", partial_write)
    provider = Offline([
        call("write_file", {"path": "partial.txt", "content": "observed side effect"}),
        ProviderResponse(text="The operation's completion is unknown."),
    ])
    agent = Agent(provider, tools=[tools.WRITE_FILE_TOOL], verbose=False)
    result = agent.run("Create partial.txt", require_edit=True)
    assert (tmp_path / "partial.txt").read_text() == "observed side effect"
    assert str(tmp_path / "partial.txt") in agent.changed_files
    assert agent.unresolved_errors
    assert not agent.edit_not_applied
    assert "No file was changed" not in result and "no file was changed" not in result
    assert "no file tool modified" not in result.lower()
