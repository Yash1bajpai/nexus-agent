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

from nexus_agent_ai.agent import context, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import RefusalClass, RunnerStatus, execution_metadata, execution_result, refusal, runner_result, runner_status
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


class LocalControlledProvider(BaseProvider):
    """Controlled local profile provider with adaptive output budget support."""
    model = "controlled-local-test"
    context_size = 4096
    max_output_tokens = 512
    is_local = True
    _supports_output_budget = True

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []
        self.offered = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        self.offered.append([t.name for t in tools])
        return next(self.steps)

    def stream(self, **kwargs):
        yield self.complete(**kwargs)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def test_search_web_bounded_whole_line_prefix_production_path(tmp_path, monkeypatch):
    """Production path oracle: recorded 1,712-byte search output admits fact-bearing verbatim prefix."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    raw_identity = fixture["identity"]
    assert len(raw_content.encode("utf-8")) == 1712
    assert raw_identity == "5180dcc1137d1226c75d208b8afd71e6d94db21cbc32b540ef51f595acde3275"

    def mock_search(**kwargs):
        return execution_result(raw_content, executed=True, outcome=fixture["outcome"])
    monkeypatch.setattr(tools.SEARCH_WEB_TOOL, "execute", mock_search)
    monkeypatch.setattr(tools, "execute_search_web", mock_search)

    # Seed ALL 14 canonical messages from recorded B run without dropping any
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="search-oracle")
    memory.replace(fixture["canonical_before"])
    assert len(memory.get()) == 14

    # Track retention writes to prove pure fit probes perform zero SQLite writes before admission
    write_operations = []
    real_retain = memory.retain_tool_output
    def tracking_retain(plan):
        write_operations.append(plan["artifact_id"])
        return real_retain(plan)
    monkeypatch.setattr(memory, "retain_tool_output", tracking_retain)

    provider = LocalControlledProvider([
        ProviderResponse(tool_calls=[ToolCall(fixture["call_id"], fixture["tool_name"], fixture["call_args"])]),
        ProviderResponse(text="Neutral observation acknowledging search results."),
    ])

    # 13 tools including RUN_TESTS_TOOL as consented in B
    all_13_tools = list(tools.get_all_tools())
    if "run_tests" not in [t.name for t in all_13_tools]:
        all_13_tools.append(tools.RUN_TESTS_TOOL)
    assert len(all_13_tools) == 13

    agent = Agent(provider, tools=all_13_tools, memory=memory, verbose=False)

    # Source core.py:441-451 Windows header reconstruction
    ws = fixture["workspace"]
    mode_prompt = (
        f"You are Nexus-Agent, currently working inside the project directory: {ws['project_name']}.\n"
        f"Working directory: {ws['resolved_cwd']}"
    )
    agent.system = f"{mode_prompt}\nYou help developers write, debug, review, and understand code.\n\n{core.RULES_PROMPT}"
    result = agent.run(fixture["user_prompt"])
    assert "Neutral observation" in result

    # Verify provider received second request with 13 tools
    assert len(provider.seen) == 2
    assert len(provider.offered[1]) == 13
    assert "run_tests" in provider.offered[1]

    # Emitted tool result must select EXACT recorded call_id, not first role tool
    emitted_messages = provider.seen[1]
    tool_msg = next(m for m in emitted_messages if m.get("tool_call_id") == fixture["call_id"])
    assert tool_msg["role"] == "tool"

    # Oracle assertion: Fact-bearing verbatim prefix in emitted payload
    assert "Releases: ggml-org/llama.cpp - GitHub" in tool_msg["content"]
    assert "https://github.com/ggml-org/llama.cpp/releases" in tool_msg["content"]
    assert "ggml-org/llama.cpp on GitHub | Release Alert" in tool_msg["content"]
    assert "https://releasealert.dev/github/ggml-org/llama.cpp" in tool_msg["content"]
    assert "tokens omitted from context; retained as artifact" in tool_msg["content"]
    assert raw_identity in tool_msg["content"]

    # Parse emitted prefix and verify exact derivation from raw content
    prefix_lines = []
    for line in tool_msg["content"].splitlines():
        if line.startswith("Outcome:") or line.startswith("Search results for:"):
            continue
        if "tokens omitted from context;" in line:
            break
        prefix_lines.append(line)
    prefix_body = "\n".join(prefix_lines).strip()
    assert "Releases: ggml-org/llama.cpp - GitHub" in prefix_body
    assert prefix_body in raw_content

    # Suffix omission check: omitted estimate matches actual excluded suffix tokens
    match = re.search(r"~(\d+)\s+tokens omitted", tool_msg["content"])
    assert match is not None
    omitted_reported = int(match.group(1))
    full_tokens = context.estimate_tokens(raw_content)
    assert 0 < omitted_reported < full_tokens

    # SQLite artifact store verification: exactly 1 write occurred
    assert len(write_operations) == 1
    with sqlite3.connect(memory.db_path) as conn:
        row = conn.execute(
            "SELECT content, byte_len, protected, tool_name FROM context_artifacts WHERE session_id=? AND artifact_id=?",
            (memory.session_id, raw_identity),
        ).fetchone()
    assert row is not None
    stored_content, stored_bytes, stored_protected, stored_tool = row
    assert stored_content == raw_content
    assert stored_bytes == 1712
    assert stored_protected == 1
    assert stored_tool == "search_web"


def test_search_web_store_failure_drops_prefix_and_uses_full_payload_estimate(tmp_path, monkeypatch):
    """When SQLite artifact store is full, drop prefix and use truthful discard notes."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    def mock_search(**kwargs):
        return execution_result(raw_content, executed=True, outcome=fixture["outcome"])
    monkeypatch.setattr(tools.SEARCH_WEB_TOOL, "execute", mock_search)
    monkeypatch.setattr(tools, "execute_search_web", mock_search)

    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="store-fail")
    memory.replace(fixture["canonical_before"])
    store_attempts = 0

    def failing_retain(plan):
        nonlocal store_attempts
        store_attempts += 1
        return False  # Store full

    monkeypatch.setattr(memory, "retain_tool_output", failing_retain)

    provider = LocalControlledProvider([
        ProviderResponse(tool_calls=[ToolCall(fixture["call_id"], fixture["tool_name"], fixture["call_args"])]),
        ProviderResponse(text="Neutral response after discard."),
    ])

    all_13_tools = list(tools.get_all_tools())
    if "run_tests" not in [t.name for t in all_13_tools]:
        all_13_tools.append(tools.RUN_TESTS_TOOL)

    agent = Agent(provider, tools=all_13_tools, memory=memory, verbose=False)
    ws = fixture["workspace"]
    mode_prompt = (
        f"You are Nexus-Agent, currently working inside the project directory: {ws['project_name']}.\n"
        f"Working directory: {ws['resolved_cwd']}"
    )
    agent.system = f"{mode_prompt}\nYou help developers write, debug, review, and understand code.\n\n{core.RULES_PROMPT}"
    agent.run(fixture["user_prompt"])

    # At most 1 store attempt was made (no second store branch fallthrough)
    assert store_attempts == 1

    tool_msg = next(m for m in provider.seen[1] if m.get("tool_call_id") == fixture["call_id"])
    # Prefix is dropped
    assert "Releases: ggml-org/llama.cpp" not in tool_msg["content"]
    # No false retained claim
    assert "retained as artifact" not in tool_msg["content"]
    # Truthful discard note with full payload estimate and actual reason
    full_tokens = context.estimate_tokens(raw_content)
    assert f"~{full_tokens} tokens omitted from context; artifact store full; output discarded." in tool_msg["content"]


def test_search_web_store_exception_drops_prefix_and_uses_full_payload_estimate(tmp_path, monkeypatch):
    """When SQLite artifact store raises an exception, drop prefix and report error reason."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    def mock_search(**kwargs):
        return execution_result(raw_content, executed=True, outcome=fixture["outcome"])
    monkeypatch.setattr(tools.SEARCH_WEB_TOOL, "execute", mock_search)
    monkeypatch.setattr(tools, "execute_search_web", mock_search)

    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="store-error")
    memory.replace(fixture["canonical_before"])
    store_attempts = 0

    def error_retain(plan):
        nonlocal store_attempts
        store_attempts += 1
        raise RuntimeError("database locked")

    monkeypatch.setattr(memory, "retain_tool_output", error_retain)

    provider = LocalControlledProvider([
        ProviderResponse(tool_calls=[ToolCall(fixture["call_id"], fixture["tool_name"], fixture["call_args"])]),
        ProviderResponse(text="Neutral response after error."),
    ])

    all_13_tools = list(tools.get_all_tools())
    if "run_tests" not in [t.name for t in all_13_tools]:
        all_13_tools.append(tools.RUN_TESTS_TOOL)

    agent = Agent(provider, tools=all_13_tools, memory=memory, verbose=False)
    ws = fixture["workspace"]
    mode_prompt = (
        f"You are Nexus-Agent, currently working inside the project directory: {ws['project_name']}.\n"
        f"Working directory: {ws['resolved_cwd']}"
    )
    agent.system = f"{mode_prompt}\nYou help developers write, debug, review, and understand code.\n\n{core.RULES_PROMPT}"
    agent.run(fixture["user_prompt"])
    assert store_attempts == 1

    tool_msg = next(m for m in provider.seen[1] if m.get("tool_call_id") == fixture["call_id"])
    assert "Releases: ggml-org/llama.cpp" not in tool_msg["content"]
    assert "retained as artifact" not in tool_msg["content"]
    full_tokens = context.estimate_tokens(raw_content)
    assert f"~{full_tokens} tokens omitted from context; artifact retention failed; output discarded." in tool_msg["content"]


def test_search_web_store_unavailable_drops_prefix_and_uses_full_payload_estimate(tmp_path):
    """When artifact store is unavailable (retain is None), drop prefix and use truthful discard notes."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    class UnavailableMemory:
        retain_tool_output = None

    from nexus_agent_ai.agent.result_budget import guard_tool_result
    res = execution_result(raw_content, executed=True, outcome=fixture["outcome"])

    full_tokens = context.estimate_tokens(raw_content)
    expected_discard = (
        f"Outcome: {fixture['outcome']}\n"
        f"~{full_tokens} tokens omitted from context; artifact store unavailable; output discarded."
    )

    result = guard_tool_result("search_web", res, memory=UnavailableMemory(), fits=lambda c: str(c) == expected_discard)
    assert str(result) == expected_discard
    assert "retained as artifact" not in str(result)
    assert "Releases:" not in str(result)
    assert result.executed is True
    assert result.outcome == fixture["outcome"]


def test_search_web_no_excerpt_fits_falls_back_to_flat_receipt(tmp_path, monkeypatch):
    """When prefix candidates cannot fit the prospective budget but flat receipt fits, fall back cleanly."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="flat-fallback")

    from nexus_agent_ai.agent.result_budget import guard_tool_result
    res = execution_result(raw_content, executed=True, outcome=fixture["outcome"])

    full_tokens = context.estimate_tokens(raw_content)
    expected_flat = (
        f"Outcome: {fixture['outcome']}\n"
        f"~{full_tokens} tokens omitted from context; retained as artifact {fixture['identity']}."
    )

    def exact_flat_only_callback(candidate):
        # Reject raw and all prefix candidates (even header-only ones); accept only exact flat receipt
        return str(candidate) == expected_flat

    write_operations = []
    real_retain = memory.retain_tool_output
    def tracking_retain(plan):
        write_operations.append(plan)
        return real_retain(plan)
    monkeypatch.setattr(memory, "retain_tool_output", tracking_retain)

    result = guard_tool_result("search_web", res, memory=memory, fits=exact_flat_only_callback)
    assert str(result) == expected_flat
    assert "Search results for:" not in str(result)
    assert "Releases:" not in str(result)
    assert len(write_operations) == 1
    assert write_operations[0]["artifact_id"] == fixture["identity"]
    assert write_operations[0]["byte_len"] == 1712
    with sqlite3.connect(memory.db_path) as conn:
        row = conn.execute(
            "SELECT content, byte_len, protected, tool_name FROM context_artifacts WHERE session_id=? AND artifact_id=?",
            (memory.session_id, fixture["identity"]),
        ).fetchone()
    assert row is not None
    assert row[0] == raw_content
    assert row[1] == 1712
    assert row[2] == 1


def test_search_web_prefix_constraints_data_over_2048():
    """Prefix candidates on data > 2048 chars are contiguous, <= 8 whole lines, <= 2048 chars, descending."""
    from nexus_agent_ai.agent.result_budget import guard_tool_result

    # Create 10 lines of 300 chars: 9-char header + 290 'X's + 1 newline = 300 chars. Total 3,000 chars (> 2,048)
    sample_lines = [f"Line {i:02d}: {'X' * 290}\n" for i in range(10)]
    sample_content = "".join(sample_lines)
    assert len(sample_content) == 3000
    evaluated_prefixes = []

    def inspect_fits(candidate):
        c_str = str(candidate)
        # Filter for prefix candidate structure (outcome status header + omission suffix)
        if c_str.startswith("Outcome: ok\n") and "\n\n~" in c_str and "retained as artifact" in c_str:
            prefix_part = c_str[len("Outcome: ok\n"):c_str.find("\n\n~")]
            # Assert starts with ENTIRE original first line including newline before append
            assert prefix_part.startswith(sample_lines[0])
            evaluated_prefixes.append(prefix_part)
        return False

    class DummyMemory:
        def retain_tool_output(self, plan): return True

    res = execution_result(sample_content, executed=True, outcome="ok")
    with pytest.raises(context.ContextBudgetError):
        guard_tool_result("search_web", res, memory=DummyMemory(), fits=inspect_fits)

    # 10 lines of 300 chars: k=8 (2400 chars) and k=7 (2100 chars) exceed 2048 chars.
    # Exactly 6 eligible prefixes evaluated: k=6 (1800), k=5 (1500), k=4 (1200), k=3 (900), k=2 (600), k=1 (300).
    expected_prefixes = ["".join(sample_lines[:k]) for k in range(6, 0, -1)]
    assert evaluated_prefixes == expected_prefixes
    assert len(evaluated_prefixes) == 6
    prev_len = float("inf")
    for prefix in evaluated_prefixes:
        # Must be contiguous from start of raw content
        assert prefix == sample_content[:len(prefix)]
        # Must not exceed 2048 characters
        assert len(prefix) <= 2048
        # Must end with newline (whole line, no partial cut in text/URL)
        assert prefix.endswith("\n")
        # Must be evaluated in descending length order
        assert len(prefix) < prev_len
        prev_len = len(prefix)


def test_search_web_oversize_first_line_yields_no_eligible_prefix():
    """When line 0 alone exceeds 2048 chars, exactly zero prefix candidates are eligible."""
    from nexus_agent_ai.agent.result_budget import guard_tool_result

    oversize_lines = ["Line 0: " + ("a" * 2500) + "\n", "Line 1: bbb\n"]
    oversize_content = "".join(oversize_lines)
    evaluated_prefixes = []

    def inspect_fits(candidate):
        c_str = str(candidate)
        if c_str.startswith("Outcome: ok\n") and "\n\n~" in c_str and "retained as artifact" in c_str:
            prefix_part = c_str[len("Outcome: ok\n"):c_str.find("\n\n~")]
            if prefix_part:
                assert prefix_part.startswith(oversize_lines[0])
                evaluated_prefixes.append(prefix_part)
        return False

    class DummyMemory:
        def retain_tool_output(self, plan): return True

    res = execution_result(oversize_content, executed=True, outcome="ok")
    with pytest.raises(context.ContextBudgetError):
        guard_tool_result("search_web", res, memory=DummyMemory(), fits=inspect_fits)

    # Zero prefix candidates were evaluated because line 0 exceeds 2048 chars
    assert evaluated_prefixes == []


def test_search_web_short_lines_evaluates_exactly_max_8():
    """When text has > 8 short lines, exactly 8 prefix candidates (k=8..1) are evaluated."""
    from nexus_agent_ai.agent.result_budget import guard_tool_result

    short_lines = [f"Line {i:02d}\n" for i in range(15)]
    short_lines_content = "".join(short_lines)
    evaluated_prefixes = []

    def inspect_fits(candidate):
        c_str = str(candidate)
        if c_str.startswith("Outcome: ok\n") and "\n\n~" in c_str and "retained as artifact" in c_str:
            prefix_part = c_str[len("Outcome: ok\n"):c_str.find("\n\n~")]
            # Assert starts with entire original first line including newline before append
            assert prefix_part.startswith(short_lines[0])
            evaluated_prefixes.append(prefix_part)
        return False

    class DummyMemory:
        def retain_tool_output(self, plan): return True

    res = execution_result(short_lines_content, executed=True, outcome="ok")
    with pytest.raises(context.ContextBudgetError):
        guard_tool_result("search_web", res, memory=DummyMemory(), fits=inspect_fits)

    # Exactly 8 prefix candidates evaluated: k=8 down to 1
    expected_prefixes = ["".join(short_lines[:k]) for k in range(8, 0, -1)]
    assert evaluated_prefixes == expected_prefixes
    assert len(evaluated_prefixes) == 8
    prev_len = float("inf")
    for prefix in evaluated_prefixes:
        assert prefix == short_lines_content[:len(prefix)]
        assert prefix.endswith("\n")
        assert len(prefix) < prev_len
        prev_len = len(prefix)

def test_non_search_tools_preserves_metadata_and_runner_status(tmp_path):
    """Non-search tools (run_code, run_tests, read_file) preserve typed RunnerStatus and execution metadata."""
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="non-search-meta")
    from nexus_agent_ai.agent.result_budget import guard_tool_result

    # 1. run_code completed outcome with runner_result tagging
    res_code_raw = execution_result("print('hello')", executed=True, outcome="Fixture Python process completed.")
    res_code = runner_result(res_code_raw, tool="run_code", status=RunnerStatus.COMPLETED)
    assert runner_status(res_code, "run_code") is RunnerStatus.COMPLETED
    assert execution_metadata(res_code)["executed"] is True
    assert execution_metadata(res_code)["outcome"] == "Fixture Python process completed."

    # Fit predicate rejects raw and accepts exact flat receipt
    code_id = hashlib.sha256(("run_code\0" + str(res_code)).encode("utf-8")).hexdigest()
    expected_code_flat = (
        f"Outcome: Fixture Python process completed.\n"
        f"~{context.estimate_tokens(str(res_code))} tokens omitted from context; retained as artifact {code_id}."
    )
    result_code = guard_tool_result("run_code", res_code, memory=memory, fits=lambda c: str(c) == expected_code_flat)
    assert str(result_code) == expected_code_flat
    assert result_code.executed is True
    assert result_code.outcome == "Fixture Python process completed."
    assert runner_status(result_code, "run_code") is RunnerStatus.COMPLETED
    assert execution_metadata(result_code)["executed"] is True
    assert execution_metadata(result_code)["outcome"] == "Fixture Python process completed."

    # 2. run_tests failed outcome with runner_result tagging
    res_tests_raw = execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Tests failed (1 failed).")
    res_tests = runner_result(res_tests_raw, tool="run_tests", status=RunnerStatus.FAILED)
    assert runner_status(res_tests, "run_tests") is RunnerStatus.FAILED
    assert execution_metadata(res_tests)["executed"] is True
    assert execution_metadata(res_tests)["outcome"] == "Tests failed (1 failed)."

    tests_id = hashlib.sha256(("run_tests\0" + str(res_tests)).encode("utf-8")).hexdigest()
    expected_tests_flat = (
        f"Outcome: Tests failed (1 failed).\n"
        f"~{context.estimate_tokens(str(res_tests))} tokens omitted from context; retained as artifact {tests_id}."
    )
    result_tests = guard_tool_result("run_tests", res_tests, memory=memory, fits=lambda c: str(c) == expected_tests_flat)
    assert str(result_tests) == expected_tests_flat
    assert result_tests.executed is True
    assert result_tests.outcome == "Tests failed (1 failed)."
    assert runner_status(result_tests, "run_tests") is RunnerStatus.FAILED
    assert execution_metadata(result_tests)["executed"] is True
    assert execution_metadata(result_tests)["outcome"] == "Tests failed (1 failed)."

    # 3. read_file refusal/metadata (refusal takes only kind, message)
    res_read = refusal(RefusalClass.TARGETING, "path outside sandbox")
    assert res_read.executed is False
    assert res_read.outcome == "Not executed."
    assert res_read.refusal_class is RefusalClass.TARGETING
    assert execution_metadata(res_read)["executed"] is False
    assert execution_metadata(res_read)["refusal_class"] == RefusalClass.TARGETING.value

    read_id = hashlib.sha256(("read_file\0" + str(res_read)).encode("utf-8")).hexdigest()
    expected_read_flat = (
        f"Outcome: Not executed.\n"
        f"~{context.estimate_tokens(str(res_read))} tokens omitted from context; retained as artifact {read_id}."
    )
    result_read = guard_tool_result("read_file", res_read, memory=memory, fits=lambda c: str(c) == expected_read_flat)
    assert str(result_read) == expected_read_flat
    assert result_read.executed is False
    assert result_read.outcome == "Not executed."
    assert getattr(result_read, "refusal_class", None) is RefusalClass.TARGETING
    assert execution_metadata(result_read)["executed"] is False
    assert execution_metadata(result_read)["refusal_class"] == RefusalClass.TARGETING.value


def test_search_web_postcapture_accepted_stored_payload_exact_fidelity():
    """Post-capture verification of stored accepted emission payload from historical smoke log (NOT a fresh execution)."""
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    captured = fixture["accepted_captured_tool_message"]

    # Verify origin smoke log hash
    assert fixture["accepted_smoke_log_hash"] == "9da6a8365657d9812651d83e238461206be8d471724d515e96900c01a4a69da2"

    # Known status prefix
    status_header = f"Outcome: {fixture['outcome']}\n"
    assert captured.startswith(status_header)
    after_status = captured[len(status_header):]

    # Exact suffix notice delimiter
    match = re.search(r"\n\n~(\d+) tokens omitted from context; retained as artifact ([0-9a-f]{64})\.", after_status)
    assert match is not None
    reported_n = int(match.group(1))
    reported_hash = match.group(2)
    assert reported_hash == fixture["identity"]

    # Emitted prefix preserves all original newline and query header characters without .strip()
    suffix_notice = match.group(0)
    assert after_status.endswith(suffix_notice)
    emitted_prefix = after_status[:-len(suffix_notice)]

    # 1. Exact prefix equality
    assert emitted_prefix == raw_content[:len(emitted_prefix)]
    assert emitted_prefix.endswith("\n")

    # 2. Exact suffix token estimate equality
    actual_suffix = raw_content[len(emitted_prefix):]
    calculated_suffix_tokens = context.estimate_tokens(actual_suffix)
    assert calculated_suffix_tokens == reported_n
    assert reported_n == 376


def test_search_web_fits_probe_observes_zero_artifacts_row_count_during_selection(tmp_path):
    """Passive observation confirms zero context_artifacts row count during candidate probe callbacks before admission.

    NOTE: This test verifies artifact row count visibility (COUNT(*) == 0) only, and does not claim
    full SQLite engine trace proofs, write-delete cycle detection, or real planner call-count guarantees.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "q1_search_recorded_1712.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    raw_content = fixture["content"]
    db_path = tmp_path / "trace_history.db"
    memory = SQLiteMemory(db_path, session_id="probe-trace")

    from nexus_agent_ai.agent.result_budget import guard_tool_result
    res = execution_result(raw_content, executed=True, outcome=fixture["outcome"])

    def check_artifacts_count():
        with sqlite3.connect(db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM context_artifacts").fetchone()[0]

    # Pre-probe artifact count must be 0
    assert check_artifacts_count() == 0

    callback_entries = 0
    probe_modifying_observed = False

    def probe_fits(candidate):
        nonlocal callback_entries, probe_modifying_observed
        callback_entries += 1
        # Passive state verification: during candidate probing callbacks, context_artifacts count in DB is strictly 0
        if check_artifacts_count() != 0:
            probe_modifying_observed = True
        # Raw check (callback 1) and first 2 prefixes (callbacks 2 and 3) rejected; third prefix accepted on callback 4
        if callback_entries < 4:
            return False
        return True

    result = guard_tool_result("search_web", res, memory=memory, fits=probe_fits)

    # 1. During rejected candidate probe window: zero artifact rows existed in DB
    assert not probe_modifying_observed
    assert callback_entries == 4

    # 2. After candidate was admitted outside probe callback, row count is exactly 1
    assert check_artifacts_count() == 1
    assert "tokens omitted from context; retained as artifact" in str(result)
