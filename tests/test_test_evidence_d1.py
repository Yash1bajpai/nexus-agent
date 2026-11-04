"""Offline consumer regressions for bounded runner evidence, not test semantics."""
import copy
import importlib.util
import json
import subprocess

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import consent, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.guardrails import LoopGuard
from nexus_agent_ai.agent.refusals import (
    RunnerStatus, execution_metadata, execution_result, runner_result,
)
from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.messages import assistant_message


E12_CODE = (
    "def add(a, b):\n    return a + b\n\n"
    "def subtract(a, b):\n    return a - b\n\ncalculator = {}"
)
E12_REPLY = "I've verified the test runs correctly. The calculator functions are working as expected."
E12_REQUEST = "write a test for calculator.py and run it"


class Scripted(BaseProvider):
    model = "offline-d1-evidence"
    context_size = 4096
    max_output_tokens = 1024

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        assert self.turns, "an unverified claim must not invent another provider request"
        turn = self.turns.pop(0)
        if isinstance(turn, tuple):
            call = ToolCall(f"call_{len(self.seen)}", turn[0], turn[1])
            return ProviderResponse(tool_calls=[call], raw_assistant_message=assistant_message("", [call]))
        return ProviderResponse(text=turn, raw_assistant_message=assistant_message(turn, []))

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": str(result)}


@pytest.fixture(autouse=True)
def deterministic_evidence_environment(monkeypatch):
    monkeypatch.setenv("NEXUS_GUARDRAILS", "1")
    monkeypatch.setenv("NEXUS_AGENT_AUTO_CHECK", "0")
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")


def assert_unverified(guard, answer):
    verdict, reviewed = guard.review_final(answer)
    assert verdict == "ok"
    assert guard.claim_unverified
    assert reviewed.startswith(answer)
    assert "unverified" in reviewed[len(answer):].lower()
    return reviewed


def assert_verified(guard, answer):
    verdict, reviewed = guard.review_final(answer)
    assert verdict == "ok" and reviewed == answer
    assert not guard.claim_unverified


def assert_e12_canonical(agent, provider, original, path):
    assert len(provider.seen) == 2 and not provider.turns
    assert agent.claim_unverified and agent.claim_unverified_error
    assert not agent.changed_files
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.glob("*.py")) == ["calculator.py"]
    messages = agent.memory.get()
    bindings = [message for message in messages if message.get("tool_calls")]
    assert len(bindings) == 1
    calls = bindings[0]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run_code"
    assert json.loads(calls[0]["function"]["arguments"]) == {"code": E12_CODE}
    results = [message for message in messages if message["role"] == "tool"]
    assert results == [{
        "role": "tool", "tool_call_id": calls[0]["id"],
        "content": "[Code executed successfully with no output]",
        "metadata": {"executed": True, "outcome": "Code completed (exit code 0)."},
    }]
    assert provider.seen[1][-1] == results[0]
    assert messages[-1] == {"role": "assistant", "content": E12_REPLY}
    assert not any(message.get("metadata", {}).get("internal") for message in messages)


@pytest.mark.parametrize("stream", [False, True])
def test_exact_e12_agent_replay_preserves_original_and_qualifies_without_retry(tmp_path, stream):
    path = tmp_path / "calculator.py"
    original = b"def add(a, b):\n    return a + b\n"
    path.write_bytes(original)
    provider = Scripted([("run_code", {"code": E12_CODE}), E12_REPLY])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    final = agent.run(E12_REQUEST, stream=stream)
    assert final.startswith(E12_REPLY)
    assert "unverified" in final[len(E12_REPLY):].lower()
    assert_e12_canonical(agent, provider, original, path)
    assert [event["name"] for event in events if event["type"] == "action"] == ["run_code"]
    if stream:
        rendered = "".join(event["content"] for event in events if event["type"] == "stream_chunk")
        assert rendered.startswith(E12_REPLY)
        assert rendered.count(E12_REPLY) == 1
        assert "unverified" in rendered[len(E12_REPLY):].lower()


@pytest.mark.parametrize("stream", [False, True])
def test_exact_e12_cli_exits_nonzero_without_inventing_tests_or_mutation(tmp_path, monkeypatch, stream):
    path = tmp_path / "calculator.py"
    original = b"def add(a, b):\n    return a + b\n"
    path.write_bytes(original)
    provider = Scripted([("run_code", {"code": E12_CODE}), E12_REPLY])
    instances = []
    agent_type = cli.Agent

    def capture_agent(*args, **kwargs):
        agent = agent_type(*args, **kwargs)
        instances.append(agent)
        return agent

    monkeypatch.setattr(cli, "Agent", capture_agent)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *args, **kwargs: (provider, "scripted"))
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    args = ["chat", E12_REQUEST]
    if not stream:
        args.append("--no-stream")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert "unverified" in result.output.lower()
    assert len(instances) == 1
    assert_e12_canonical(instances[0], provider, original, path)


@pytest.mark.parametrize("code", [
    "pass",
    E12_CODE,
    "assert 1 == 1",
    "if False:\n    assert False",
    "def never_called():\n    assert False",
    'print("Pytest Results: PASSED")',
    "assert False",
    "1 / 0",
])
def test_actual_code_result_never_becomes_project_or_assertion_proof(code):
    source = tools.execute_run_code(code)
    guard = LoopGuard(tools.execute_list_directory)
    wrapped = guard.after_tool("run_code", {"code": code}, source)
    assert str(wrapped) == str(source)
    assert execution_metadata(wrapped) == execution_metadata(source)
    for answer in ("All tests passed.", "All assertions passed.", "I ran a test to confirm the fix.", E12_REPLY):
        assert_unverified(guard, answer)


def test_caught_assertion_is_rejected_by_actual_unchanged_sandbox():
    code = "try:\n    assert False\nexcept AssertionError:\n    pass"
    result = tools.execute_run_code(code)
    assert result.executed is False
    guard = LoopGuard(tools.execute_list_directory)
    guard.after_tool("run_code", {"code": code}, result)
    assert_unverified(guard, "All assertions passed.")
    assert_unverified(guard, "The snippet completed successfully.")


@pytest.mark.parametrize("mode", ["stderr_failure", "timeout", "sandbox", "unknown"])
def test_noncompletion_cannot_support_successful_snippet_claim(mode, monkeypatch):
    if mode in ("timeout", "unknown"):
        def incomplete(*args, **kwargs):
            if mode == "timeout":
                raise subprocess.TimeoutExpired("python", tools.CODE_EXECUTION_TIMEOUT)
            raise RuntimeError("fixture runner outcome unavailable")
        monkeypatch.setattr(tools.subprocess, "run", incomplete)
    code = "assert False" if mode == "stderr_failure" else "import os" if mode == "sandbox" else "pass"
    result = tools.execute_run_code(code)
    if mode == "stderr_failure":
        assert result.startswith("STDERR:") and not result.startswith("ERROR:")
        assert execution_metadata(result) == {
            "executed": True, "outcome": "Code FAILED (exit code 1).", "refusal_class": "other",
        }
    guard = LoopGuard(tools.execute_list_directory)
    guard.after_tool("run_code", {"code": code}, result)
    assert_unverified(guard, "The snippet completed successfully.")
    assert_unverified(guard, "All tests passed.")


@pytest.mark.parametrize("stream", [False, True])
def test_real_completed_snippet_truthful_reply_is_unchanged(stream):
    answer = "The snippet completed successfully and printed 42. This is detached computation, not project test evidence."
    provider = Scripted([("run_code", {"code": "print(6 * 7)"}), answer])
    agent = Agent(provider, tools=[tools.RUN_CODE_TOOL], verbose=False)
    assert agent.run("Compute six times seven", stream=stream) == answer
    assert not agent.claim_unverified and not agent.unresolved_errors
    assert len(provider.seen) == 2
    result = next(message for message in agent.memory.get() if message["role"] == "tool")
    assert result["content"] == "STDOUT:\n42"
    assert result["metadata"] == {"executed": True, "outcome": "Code completed (exit code 0)."}


@pytest.mark.parametrize("code, answer", [
    ("pass", "All tests passed."),
    ("assert False", "The snippet completed successfully."),
    ("if False:\n    assert False", "All assertions passed."),
])
@pytest.mark.parametrize("stream", [False, True])
def test_new_unverified_code_paths_do_not_add_provider_calls(code, answer, stream):
    provider = Scripted([("run_code", {"code": code}), answer])
    agent = Agent(provider, tools=[tools.RUN_CODE_TOOL], verbose=False)
    final = agent.run("Report the observed execution", stream=stream)
    assert agent.claim_unverified
    assert final.startswith(answer) and "unverified" in final[len(answer):].lower()
    assert len(provider.seen) == 2 and not provider.turns
    assert agent.memory.get()[-1] == {"role": "assistant", "content": answer}


@pytest.mark.parametrize("name, raw", [
    ("run_code", "STDOUT:\nok"),
    ("run_code", "[Code executed successfully with no output]"),
    ("run_code", "STDERR:\nAssertionError"),
    ("run_tests", "Pytest Results: PASSED\n1 passed"),
    ("run_tests", "Pytest Results: FAILED\nassert 1 == 2"),
])
def test_legacy_untyped_runner_text_stays_unknown_without_guessing_success_or_retry(name, raw):
    guard = LoopGuard(tools.execute_list_directory)
    guard.after_tool(name, {}, raw)
    assert_unverified(guard, "All tests passed.")
    assert_unverified(guard, "The snippet completed successfully.")


@pytest.mark.parametrize("name, raw", [
    ("run_code", "STDOUT:\nok"),
    ("run_tests", "Pytest Results: PASSED\n1 passed"),
    ("run_tests", "Pytest Results: FAILED\nassert 1 == 2"),
])
def test_unknown_raw_agent_results_do_not_create_new_honesty_retries(name, raw, monkeypatch):
    monkeypatch.setattr(core, "execute_tool", lambda *args, **kwargs: raw)
    offered = tools.RUN_CODE_TOOL if name == "run_code" else tools.RUN_TESTS_TOOL
    provider = Scripted([(name, {}), "All tests passed."])
    agent = Agent(provider, tools=[offered], verbose=False)
    final = agent.run("Report the observed test result")
    assert agent.claim_unverified and "unverified" in final.lower()
    assert len(provider.seen) == 2 and not provider.turns


def observe_suite(state, tmp_path, monkeypatch):
    """Use the real source handler; only availability/consent are injected."""
    path = tmp_path / "test_selected.py"
    path.write_text("def test_selected():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, "fixture permission"))
    if state == "failed":
        path.write_text("def test_selected():\n    assert False\n", encoding="utf-8")
    elif state == "no_tests":
        path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    elif state == "declined":
        monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "User declined project execution."))
    elif state == "unavailable":
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                            None if name == "pytest" else real_find_spec(name, *args, **kwargs))
    guard = LoopGuard(tools.execute_list_directory)
    result = tools.execute_run_tests(str(path), "-q")
    guard.after_tool("run_tests", {"path": str(path), "args": "-q"}, result)
    if state == "stale":
        tools.execute_read_file(str(path))
        args = {"path": str(path), "target": "assert True", "replacement": "assert 1 == 1"}
        changed = tools.execute_patch_file(**args)
        assert not changed.startswith("ERROR:")
        guard.after_tool("patch_file", args, changed)
    return guard, result


@pytest.mark.parametrize("state", ["fresh", "failed", "no_tests", "declined", "unavailable", "stale"])
def test_suite_authority_survives_later_code_without_being_created_or_refreshed(state, tmp_path, monkeypatch):
    guard, _ = observe_suite(state, tmp_path, monkeypatch)
    guard.after_tool("run_code", {"code": "print(42)"}, tools.execute_run_code("print(42)"))
    answer = "All tests passed."
    if state == "fresh":
        assert_verified(guard, answer)
    elif state == "failed":
        verdict, _ = guard.review_final(answer)
        assert verdict == "retry" and guard.claim_unverified
    else:
        verdict, reviewed = guard.review_final(answer)
        assert verdict == "ok" and guard.claim_unverified and reviewed != answer
    # Neither permission, no-collection nor stale results are relabeled as failed suites.
    if state in ("no_tests", "declined", "unavailable", "stale"):
        assert "run_tests last reported FAILED" not in reviewed
    assert_verified(guard, "The snippet completed successfully and printed 42.")


@pytest.mark.parametrize("answer", [
    "The snippet test passed.",
    "The inline test passed.",
    "The detached test passed.",
    "The scratchpad test passed.",
    "The run_code test passed.",
    "All assertions passed.",
    "The assertion passed.",
    "The snippet completed successfully.",
    "The test suite passed. All assertions passed.",
])
def test_earlier_real_suite_pass_cannot_launder_later_failed_snippet_or_assertions(answer, tmp_path, monkeypatch):
    guard, _ = observe_suite("fresh", tmp_path, monkeypatch)
    failed = tools.execute_run_code("assert False")
    guard.after_tool("run_code", {"code": "assert False"}, failed)
    assert_verified(guard, "All tests passed.")
    assert_unverified(guard, answer)
    # Reviewing a scoped unverified statement must not destroy the valid suite fact.
    assert_verified(guard, "The test suite passed.")


def test_failed_code_before_real_suite_pass_retains_distinct_evidence(tmp_path, monkeypatch):
    failed = tools.execute_run_code("assert False")
    _, suite = observe_suite("fresh", tmp_path, monkeypatch)
    # A separate guard observes the actual results in the opposite order.
    ordered = LoopGuard(tools.execute_list_directory)
    ordered.after_tool("run_code", {"code": "assert False"}, failed)
    ordered.after_tool("run_tests", {}, suite)
    assert_verified(ordered, "All tests passed.")
    assert_unverified(ordered, "All assertions passed.")
    assert_unverified(ordered, "The snippet completed successfully.")


@pytest.mark.parametrize("codes, latest_completed, all_completed", [
    (["print(1)", "print(2)"], True, True),
    (["assert False", "print(2)"], True, False),
    (["print(1)", "assert False"], False, False),
    ([None, "print(2)"], True, False),
    (["print(1)", None], False, False),
])
def test_latest_snippet_completion_does_not_attest_all_earlier_snippets(codes, latest_completed, all_completed):
    guard = LoopGuard(tools.execute_list_directory)
    for code in codes:
        result = "STDOUT:\nunknown legacy result" if code is None else tools.execute_run_code(code)
        guard.after_tool("run_code", {"code": code or "print(0)"}, result)
    for answer, supported in [
        ("The snippet completed successfully.", latest_completed),
        ("All snippets completed successfully.", all_completed),
    ]:
        if supported:
            assert_verified(guard, answer)
        else:
            assert_unverified(guard, answer)
    # Invocation-only wording does not assert successful completion.
    assert_verified(guard, "I ran the snippet.")


@pytest.mark.parametrize("answer", [
    E12_REPLY,
    "I've verified the test runs correctly.",
    "I’ve verified the test runs correctly.",
    "I have verified the tests run correctly.",
    "The test runs properly.",
    "The tests ran correctly.",
    "The test ran properly.",
    "The tests run properly.",
    "The assertion passed.",
    "Assertions passed.",
    "I've verified the test runs correctly. I can run the project suite next.",
    "The tests ran properly. The next step would be to inspect the file.",
    "The test runs correctly; I haven't run the full project suite.",
])
def test_bounded_e12_grammar_flags_affirmative_clause_without_suite_evidence(answer):
    assert_unverified(LoopGuard(tools.execute_list_directory), answer)


@pytest.mark.parametrize("answer", [
    "I have not verified the test runs correctly.",
    "I haven't verified the test runs correctly.",
    "I haven’t verified the test runs correctly.",
    "I can't verify the test runs correctly.",
    "I can’t verify the test runs correctly.",
    "The test does not run correctly.",
    "The tests didn't run properly.",
    "The tests didn’t run properly.",
    "No assertions passed.",
    "The assertion hasn't passed.",
    "I can verify the test runs correctly.",
    "I'll verify the test runs correctly.",
    "I’ll verify the test runs correctly.",
    "If the tests run correctly, inspect the result.",
    "Let me check whether the tests ran properly.",
    "For example: the test runs correctly.",
    "Example: the tests ran properly.",
    "Sample reply: I've verified the test runs correctly.",
    'The earlier model said "I\'ve verified the test runs correctly."',
    "The earlier model said “I’ve verified the test runs correctly.”",
    "The example says 'the test runs correctly'.",
    "The earlier reply was 'I've verified the test runs correctly.'",
    "`I've verified the test runs correctly.`",
    "```text\nThe tests ran properly.\n```",
    "> The test runs correctly.\nThat is an unsupported quotation.",
    "The calculator functions are working as expected.",
    "The calculation is fixed.",
])
def test_bounded_e12_grammar_respects_clause_exclusions_without_broad_working_detector(answer):
    assert_verified(LoopGuard(tools.execute_list_directory), answer)


@pytest.mark.parametrize("answer", [
    "The snippet test passed.", "The inline test passed.", "All assertions passed.",
])
def test_typed_completed_code_never_strengthens_scoped_test_or_assertion_claim(answer):
    guard = LoopGuard(tools.execute_list_directory)
    # These are intentional trusted fixture facts, not inferred from the code/output.
    guard.after_tool("run_code", {"code": "assert 1 == 1"}, runner_result(
        execution_result("STDOUT:\nok", executed=True, outcome="Fixture Python process completed."),
        tool="run_code", status=RunnerStatus.COMPLETED,
    ))
    assert_unverified(guard, answer)
