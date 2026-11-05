"""Private runner evidence follows execution facts, never printed success claims."""
import copy
import importlib.util
import subprocess

import pytest

from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent import consent, tools as T
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.agent.refusals import (
    RefusalClass, RunnerStatus as S, execution_metadata, execution_result,
    refusal, replace_result_text, runner_result, runner_status,
)
from nexus_agent_ai.agent.result_budget import guard_tool_result
from nexus_agent_ai.cli.app import _check_run_outcome


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.reset_file_tracking()
    monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "No grant."))
    yield
    T.reset_file_tracking()


@pytest.fixture
def approved_project(monkeypatch):
    # Explicit fixture consent applies only to this isolated temporary workspace.
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, "Explicit test fixture consent."))


@pytest.mark.parametrize("code", [
    "pass", "assert 1 == 1", "if False:\n    assert False",
    "print('PASSED')",
])
def test_actual_code_exit_zero_is_completion_not_suite_proof(code):
    result = T.execute_tool("run_code", {"code": code})
    assert runner_status(result, "run_code") is S.COMPLETED
    assert runner_status(result, "run_tests") is S.UNKNOWN
    assert execution_metadata(result) == {"executed": True, "outcome": "Code completed (exit code 0)."}
    assert str(result) == ("STDOUT:\nPASSED" if code.startswith("print") else "[Code executed successfully with no output]")


def test_actual_assertion_failure_does_not_need_error_prefix():
    result = T.execute_run_code("assert False")
    assert not result.startswith("ERROR")
    assert "AssertionError" in result
    assert runner_status(result, "run_code") is S.FAILED
    assert execution_metadata(result) == {
        "executed": True, "outcome": "Code FAILED (exit code 1).", "refusal_class": "other",
    }


@pytest.mark.parametrize("code,language", [
    ("pass", "ruby"), ("import os", "python"),
    ("try:\n    assert False\nexcept AssertionError:\n    pass", "python"),
])
def test_code_refusals_are_not_executed(code, language):
    result = T.execute_run_code(code, language)
    assert runner_status(result, "run_code") is S.NOT_EXECUTED
    assert execution_metadata(result)["executed"] is False


@pytest.mark.parametrize("tool,exc,status,executed", [
    ("run_code", subprocess.TimeoutExpired("python", 1), S.TIMED_OUT, True),
    ("run_tests", subprocess.TimeoutExpired("pytest", 45, output=b"PASSED", stderr=b"tail"), S.TIMED_OUT, True),
    ("run_code", PermissionError("blocked"), S.NOT_EXECUTED, False),
    ("run_tests", PermissionError("blocked"), S.NOT_EXECUTED, False),
    ("run_code", RuntimeError("unknown"), S.UNKNOWN, True),
    ("run_tests", RuntimeError("unknown"), S.UNKNOWN, True),
])
def test_runner_exception_facts(monkeypatch, approved_project, tool, exc, status, executed):
    def fail(*args, **kwargs):
        raise exc
    monkeypatch.setattr(T.subprocess, "run", fail)
    result = T.execute_tool(tool, {"code": "pass"} if tool == "run_code" else {})
    assert runner_status(result, tool) is status
    assert execution_metadata(result)["executed"] is executed
    if tool == "run_tests" and status is S.TIMED_OUT:
        assert "--- stdout tail ---\nPASSED\n--- end stdout tail ---" in result
        assert result.outcome == "Pytest timed out after 45 seconds; completion unknown."


@pytest.mark.parametrize("source,status,exit_code", [
    ("def test_example():\n    assert True\n", S.PASSED, 0),
    ("def test_example():\n    print('PASSED')\n    assert False\n", S.FAILED, 1),
    ("value = 1\n", S.NO_TESTS, 5),
])
def test_actual_consented_suite_status(tmp_path, approved_project, source, status, exit_code):
    (tmp_path / "test_example.py").write_text(source, encoding="utf-8")
    result = T.execute_tool("run_tests", {"path": "test_example.py", "args": "-q"})
    assert runner_status(result, "run_tests") is status
    assert runner_status(result, "run_code") is S.UNKNOWN
    label = "NO TESTS COLLECTED" if exit_code == 5 else "PASSED" if exit_code == 0 else "FAILED"
    expected = {"executed": True, "outcome": f"Pytest {label} (exit code {exit_code})."}
    if exit_code:
        expected["refusal_class"] = "other"
    assert execution_metadata(result) == expected
    assert result.startswith(f"=== Pytest Results: {label}")
    assert "--- stdout tail (" in result and "--- stderr tail (" in result


@pytest.mark.parametrize("why", ["Declined.", "No grant.", "PASSED arbitrary explanation"])
def test_consent_not_approved_without_reason_parsing(monkeypatch, why):
    monkeypatch.setattr(consent, "execution_allowed", lambda: (False, why))
    result = T.execute_run_tests()
    assert runner_status(result, "run_tests") is S.NOT_APPROVED
    assert str(result) == ("ERROR: Project test execution is not approved. Pytest imports arbitrary project code "
                           "with your user permissions; it is NOT sandboxed. " + why)
    assert execution_metadata(result) == {"executed": False, "outcome": "Not executed.", "refusal_class": "permission"}


def test_unavailable_runner(monkeypatch, approved_project):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    result = T.execute_run_tests()
    assert runner_status(result, "run_tests") is S.UNAVAILABLE
    assert result.executed is False


@pytest.mark.parametrize("args", [
    {"path": "missing.py"}, {"path": "::node"}, {"path": "../outside.py"},
    {"args": "--bad"}, {"args": "-k 'x;bad'"}, {"args": "-k=x;bad"},
    {"args": "--maxfail=0"}, {"args": "--maxfail nope"},
])
def test_preexecution_rejections(approved_project, args):
    result = T.execute_run_tests(**args)
    assert runner_status(result, "run_tests") is S.NOT_EXECUTED
    assert result.executed is False


def test_transport_preserves_private_facts_and_public_receipts():
    source = execution_result(refusal(RefusalClass.OTHER, "PASSED " * 4000), executed=True,
                              outcome="Pytest FAILED (exit code 1).")
    source.recovery_hint = "retained"
    tagged = runner_result(source, tool="run_tests", status=S.FAILED)
    assert str(tagged) == str(source)
    assert execution_metadata(tagged) == execution_metadata(source)
    assert tagged.recovery_hint == "retained"
    changed = replace_result_text(tagged, "replacement")
    changed = execution_result(changed, executed=True, outcome=tagged.outcome)
    admitted = guard_tool_result("run_tests", changed, memory=None, fits=lambda text: str(text) != "replacement")
    assert admitted.startswith("Outcome: Pytest FAILED (exit code 1).\n")
    assert "output discarded" in admitted
    assert runner_status(admitted, "run_tests") is S.FAILED
    assert admitted.recovery_hint == "retained"
    assert execution_metadata(admitted) == execution_metadata(source)
    assert set(execution_metadata(admitted)) == {"executed", "outcome", "refusal_class"}
    assert runner_status(str(admitted), "run_tests") is S.UNKNOWN


@pytest.mark.parametrize("text", ["STDOUT:\nok", "PASSED", "[Code executed successfully with no output]"])
def test_untyped_text_never_establishes_runner_proof(text):
    assert runner_status(text, "run_code") is S.UNKNOWN
    assert runner_status(text, "run_tests") is S.UNKNOWN


def test_malformed_or_inconsistent_metadata_is_unknown():
    result = runner_result(execution_result("ok", executed=True, outcome="unchanged"), tool="run_tests", status=S.PASSED)
    assert runner_status(result, "run_tests") is S.PASSED
    result.executed = False
    assert runner_status(result, "run_tests") is S.UNKNOWN
    result.executed = 1
    assert runner_status(result, "run_tests") is S.UNKNOWN
    result.executed = True
    result._runner_status = "passed"
    assert runner_status(result, "run_tests") is S.UNKNOWN
    result._runner_status = S.COMPLETED
    assert runner_status(result, "run_tests") is S.UNKNOWN
    result._runner_status = S.PASSED
    result._runner_tool = "run_code"
    assert runner_status(result, "run_tests") is S.UNKNOWN
    assert runner_status(result, "run_code") is S.UNKNOWN


@pytest.mark.parametrize("result,tool,status,error", [
    (1, "run_code", S.COMPLETED, TypeError),
    ("ok", "other", S.COMPLETED, ValueError),
    ("ok", "run_code", "completed", TypeError),
    ("ok", "run_code", S.PASSED, ValueError),
    ("ok", "run_tests", S.COMPLETED, ValueError),
])
def test_trusted_helper_rejects_malformed_inputs(result, tool, status, error):
    with pytest.raises(error):
        runner_result(result, tool=tool, status=status)


@pytest.mark.parametrize("tool,exit_code,status", [
    ("run_code", 1, S.FAILED), ("run_code", 0, S.COMPLETED),
    ("run_tests", 1, S.FAILED), ("run_tests", 5, S.NO_TESTS),
    ("run_tests", 0, S.PASSED),
])
def test_source_returncode_not_stdout_authority(monkeypatch, approved_project, tool, exit_code, status):
    from types import SimpleNamespace
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=exit_code, stdout="PASSED\n", stderr=""))
    result = T.execute_tool(tool, {"code": "pass"} if tool == "run_code" else {})
    assert runner_status(result, tool) is status
    assert result.executed is True
    if tool == "run_code":
        assert str(result) == "STDOUT:\nPASSED"


class ScriptedProvider(BaseProvider):
    model = "offline-runner-fixture"
    context_size = 32768
    max_output_tokens = 512
    is_local = False

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        step = next(self.steps)
        return step() if callable(step) else step

    def stream(self, **kwargs):
        yield self.complete(**kwargs)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


@pytest.mark.parametrize("same_tool_success", [False, True])
def test_run_code_name_error_same_turn_chain_and_clear(same_tool_success, tmp_path):
    prompt = "Please review calculator.py for bugs. Explain any incorrect behavior you find; do not edit files."
    calculator = tmp_path / "calculator.py"
    seed = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a + b\n"
    calculator.write_text(seed, encoding="utf-8")
    call_1 = ToolCall("call_1", "run_code", {"code": "calculator.py"})
    resp_1 = ProviderResponse(tool_calls=[call_1])
    call_2 = ToolCall("call_2", "read_file", {"path": "calculator.py"})
    resp_2 = ProviderResponse(tool_calls=[call_2])

    if same_tool_success:
        call_3 = ToolCall("call_3", "run_code", {"code": "print(1 + 1)"})
        resp_3 = ProviderResponse(tool_calls=[call_3])
        resp_final = ProviderResponse(text="Reviewed and computed 1 + 1 = 2.")
        steps = [resp_1, resp_2, resp_3, resp_final]
    else:
        resp_final = ProviderResponse(text="Reviewed calculator.py.")
        steps = [resp_1, resp_2, resp_final]

    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_CODE_TOOL, T.READ_FILE_TOOL], verbose=False)

    agent.run(prompt)
    assert calculator.read_text() == seed
    assert _check_run_outcome(agent, exit_on_failure=False) is same_tool_success

    if same_tool_success:
        assert agent.unresolved_errors == []
    else:
        assert len(agent.unresolved_errors) == 1
        assert "NameError" in agent.unresolved_errors[0]
        assert "calculator" in agent.unresolved_errors[0]


@pytest.mark.parametrize("pass_tests", [False, True])
def test_colliding_path_write_does_not_clear_failed_run_tests(pass_tests, tmp_path, approved_project):
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_example():\n    assert False\n", encoding="utf-8")

    prompt = "Fix test_sample.py and verify"
    call_1 = ToolCall("t1", "run_tests", {"path": "test_sample.py"})
    resp_1 = ProviderResponse(tool_calls=[call_1])
    call_2 = ToolCall("t2", "read_file", {"path": "test_sample.py"})
    resp_2 = ProviderResponse(tool_calls=[call_2])
    call_3 = ToolCall("t3", "write_file", {
        "path": "test_sample.py",
        "content": "def test_example():\n    assert True\n",
    })
    resp_3 = ProviderResponse(tool_calls=[call_3])

    if pass_tests:
        call_4 = ToolCall("t4", "run_tests", {"path": "test_sample.py"})
        resp_4 = ProviderResponse(tool_calls=[call_4])
        resp_final = ProviderResponse(text="Tests fixed and verified passing.")
        steps = [resp_1, resp_2, resp_3, resp_4, resp_final]
    else:
        resp_final = ProviderResponse(text="Updated test_sample.py without re-running tests.")
        steps = [resp_1, resp_2, resp_3, resp_final]

    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL, T.READ_FILE_TOOL, T.WRITE_FILE_TOOL], verbose=False)

    agent.run(prompt)
    assert _check_run_outcome(agent, exit_on_failure=False) is pass_tests
    assert test_file.read_text() == "def test_example():\n    assert True\n"

    if pass_tests:
        assert agent.unresolved_errors == []
    else:
        assert len(agent.unresolved_errors) == 1
        assert "FAILED" in agent.unresolved_errors[0]


@pytest.mark.parametrize("status,tool,call_args", [
    (S.FAILED, "run_code", {"code": "assert False"}),
    (S.NO_TESTS, "run_tests", {}),
    (S.TIMED_OUT, "run_code", {"code": "pass"}),
    (S.UNKNOWN, "run_code", {"code": "pass"}),
    (S.NOT_EXECUTED, "run_code", {"code": "import os"}),
    (S.NOT_APPROVED, "run_tests", {}),
    (S.UNAVAILABLE, "run_tests", {}),
])
def test_runner_conservative_statuses_remain_unresolved(monkeypatch, status, tool, call_args):
    if status is S.NOT_APPROVED:
        monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "Denied."))
        # Interactive sessions can offer the tool before execution consent is denied.
        monkeypatch.setattr(consent, "offer_run_tests", lambda: True)
    else:
        monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
        monkeypatch.setattr(consent, "execution_allowed", lambda: (True, "Granted."))

    if status is S.UNAVAILABLE:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    elif status in (S.TIMED_OUT, S.UNKNOWN):
        def raise_err(*args, **kwargs):
            if status is S.TIMED_OUT:
                raise subprocess.TimeoutExpired("python", 1)
            raise RuntimeError("unexpected failure")
        monkeypatch.setattr(T.subprocess, "run", raise_err)
    elif status is S.NO_TESTS:
        from types import SimpleNamespace
        monkeypatch.setattr(T.subprocess, "run", lambda *a, **kw: SimpleNamespace(
            returncode=5, stdout="", stderr=""))

    observed_results = []
    original_execute = T.execute_tool

    def capturing_execute(name, args, **kwargs):
        res = original_execute(name, args, **kwargs)
        if name == tool:
            observed_results.append(res)
        return res

    monkeypatch.setattr("nexus_agent_ai.agent.core.execute_tool", capturing_execute)

    tools_list = [T.RUN_CODE_TOOL, T.RUN_TESTS_TOOL]
    provider = ScriptedProvider([
        ProviderResponse(tool_calls=[ToolCall("c1", tool, call_args)]),
        ProviderResponse(text="Done."),
    ])
    agent = Agent(provider, tools=tools_list, verbose=False)
    agent.run(f"Execute {tool}")

    assert len(observed_results) == 1
    assert runner_status(observed_results[0], tool) is status
    assert len(agent.unresolved_errors) == 1
    assert agent.unresolved_errors[0] == str(observed_results[0])
    assert _check_run_outcome(agent, exit_on_failure=False) is False


def test_run_code_guard_tool_result_preserves_typed_receipt():
    result = T.execute_run_code("assert False")
    assert runner_status(result, "run_code") is S.FAILED
    assert execution_metadata(result)["executed"] is True
    assert execution_metadata(result)["outcome"] == "Code FAILED (exit code 1)."

    admitted = guard_tool_result("run_code", result, memory=None, fits=lambda t: True)
    assert runner_status(admitted, "run_code") is S.FAILED
    assert execution_metadata(admitted) == execution_metadata(result)

    shortened = guard_tool_result(
        "run_code", result, memory=None,
        fits=lambda text: str(text) != str(result) and len(str(text)) < len(str(result)),
    )
    assert str(shortened) != str(result)
    assert runner_status(shortened, "run_code") is S.FAILED
    assert execution_metadata(shortened) == execution_metadata(result)
    assert execution_metadata(shortened)["executed"] is True
    assert execution_metadata(shortened)["outcome"] == "Code FAILED (exit code 1)."
    assert runner_status(str(shortened), "run_code") is S.UNKNOWN


@pytest.mark.parametrize("initial_args,followup_args", [
    ({"path": "."}, {"path": "./"}),
    ({"path": "./"}, {"path": "."}),
    ({"path": "."}, {"path": "__CWD__"}),
    ({"path": "__CWD__"}, {"path": "."}),
    ({}, {"path": ""}),
    ({"path": ""}, {}),
    ({}, {"path": "   "}),
])
def test_run_tests_scope_normalization_clears_proven_target_aliases(initial_args, followup_args, monkeypatch):
    from pathlib import Path
    from nexus_agent_ai.agent import core
    if initial_args.get("path") == "__CWD__":
        initial_args = {"path": str(Path.cwd().resolve())}
    if followup_args.get("path") == "__CWD__":
        followup_args = {"path": str(Path.cwd().resolve())}
    results = iter([
        runner_result(
            execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.FAILED,
        ),
        runner_result(
            execution_result("Pytest Results: PASSED", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.PASSED,
        ),
    ])
    monkeypatch.setattr(core, "execute_tool", lambda name, args, **kwargs: next(results))
    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", initial_args)]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", followup_args)]),
        ProviderResponse(text="All tests pass."),
    ]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")
    assert agent.unresolved_errors == []
    assert _check_run_outcome(agent, exit_on_failure=False) is True


@pytest.mark.parametrize("initial_path,followup_path", [
    ("test_sample.py::test_one", "./test_sample.py::test_one"),
    ("./test_sample.py::test_one", "test_sample.py::test_one"),
])
def test_run_tests_scope_normalization_clears_proven_file_selector_aliases(tmp_path, initial_path, followup_path, monkeypatch):
    from nexus_agent_ai.agent import core
    (tmp_path / "test_sample.py").write_text("def test_one(): pass\n")
    results = iter([
        runner_result(
            execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.FAILED,
        ),
        runner_result(
            execution_result("Pytest Results: PASSED", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.PASSED,
        ),
    ])
    monkeypatch.setattr(core, "execute_tool", lambda name, args, **kwargs: next(results))
    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", {"path": initial_path})]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", {"path": followup_path})]),
        ProviderResponse(text="All tests pass."),
    ]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")
    assert agent.unresolved_errors == []
    assert _check_run_outcome(agent, exit_on_failure=False) is True


@pytest.mark.parametrize("initial_args,followup_args", [
    ({}, {"path": "."}),
    ({"path": "."}, {}),
    ({}, {"path": "./"}),
    ({}, {"path": "tests"}),
    ({"path": "tests"}, {}),
    ({"path": "test_sample.py::test_one"}, {"path": "test_sample.py::test_two"}),
    ({}, {"args": "-k filter"}),
    ({"args": "-k filter"}, {}),
    ({"path": 123}, {"path": "123"}),
    ({"path": "../outside"}, {"path": "."}),
])
def test_run_tests_scope_normalization_preserves_unrelated_target_and_selection_failures(initial_args, followup_args, monkeypatch):
    from nexus_agent_ai.agent import core
    results = iter([
        runner_result(
            execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.FAILED,
        ),
        runner_result(
            execution_result("Pytest Results: PASSED", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.PASSED,
        ),
    ])
    monkeypatch.setattr(core, "execute_tool", lambda name, args, **kwargs: next(results))
    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", initial_args)]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", followup_args)]),
        ProviderResponse(text="All tests pass."),
    ]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")
    assert len(agent.unresolved_errors) == 1
    assert "FAILED" in agent.unresolved_errors[0]
    assert _check_run_outcome(agent, exit_on_failure=False) is False


def test_configured_testpaths_default_discovery_vs_cwd_regression(tmp_path, approved_project, monkeypatch):
    """Configured testpaths default discovery and explicit cwd are distinct scopes."""
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_inside.py").write_text("def test_inside(): pass\n", encoding="utf-8")
    (tmp_path / "test_outside.py").write_text("def test_outside(): assert False\n", encoding="utf-8")

    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", {"path": "."})]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", {})]),
        ProviderResponse(text="All tests pass."),
    ]
    provider = ScriptedProvider(steps)
    dispatched = []
    real_execute = T.RUN_TESTS_TOOL.execute

    def capturing_execute(**kwargs):
        res = real_execute(**kwargs)
        dispatched.append(res)
        return res

    monkeypatch.setattr(T.RUN_TESTS_TOOL, "execute", capturing_execute)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")

    assert len(dispatched) == 2
    assert runner_status(dispatched[0], "run_tests") is S.FAILED
    assert runner_status(dispatched[1], "run_tests") is S.PASSED
    assert len(agent.unresolved_errors) == 1
    assert "test_outside.py" in agent.unresolved_errors[0]
    assert _check_run_outcome(agent, exit_on_failure=False) is False

def test_run_tests_scope_key_grounding():
    import json
    from nexus_agent_ai.agent.core import _run_tests_scope_key, _DEFAULT_DISCOVERY

    assert _run_tests_scope_key({}) == ("run_tests", _DEFAULT_DISCOVERY, "", "")
    assert _run_tests_scope_key({"path": ""}) == ("run_tests", _DEFAULT_DISCOVERY, "", "")
    assert _run_tests_scope_key({"path": "   "}) == ("run_tests", _DEFAULT_DISCOVERY, "", "")

    cwd_key = _run_tests_scope_key({"path": "."})
    assert cwd_key == _run_tests_scope_key({"path": "./"})
    assert cwd_key[1] != _DEFAULT_DISCOVERY

    int_key = _run_tests_scope_key({"path": 123})
    str_key = _run_tests_scope_key({"path": "123"})
    assert int_key != str_key
    assert int_key[1][0] == "OPAQUE_INVALID"
    assert int_key[1][1] == "int"

    list_args_key = _run_tests_scope_key({"args": ["-v"]})
    assert list_args_key[3][0] == "OPAQUE_INVALID"
    assert list_args_key[3][1] == "list"
    assert json.loads(list_args_key[3][2]) == ["-v"]
    assert isinstance(hash(list_args_key), int)

    refused_key = _run_tests_scope_key({"path": "../outside"})
    assert refused_key[1][0] == "OPAQUE_INVALID"
    assert isinstance(hash(refused_key), int)


@pytest.mark.parametrize("path_val", [[], {}, [1], None, False, 0])
@pytest.mark.parametrize("args_val", [[], {}, {"x": 1}, None, False, 0])
def test_run_tests_scope_key_hashability_and_malformed_separation(path_val, args_val):
    from nexus_agent_ai.agent.core import _run_tests_scope_key
    key = _run_tests_scope_key({"path": path_val, "args": args_val})
    assert isinstance(hash(key), int)
    assert key[1][0] == "OPAQUE_INVALID"
    assert key[1][1] == type(path_val).__name__
    assert key[3][0] == "OPAQUE_INVALID"
    assert key[3][1] == type(args_val).__name__


def test_run_tests_scope_key_deterministic_dict_order():
    from nexus_agent_ai.agent.core import _run_tests_scope_key
    key1 = _run_tests_scope_key({"args": {"b": 2, "a": 1}})
    key2 = _run_tests_scope_key({"args": {"a": 1, "b": 2}})
    assert key1 == key2
    assert hash(key1) == hash(key2)

@pytest.mark.parametrize("malformed_args", [
    {"path": []},
    {"path": {}},
    {"path": [1]},
    {"args": []},
    {"args": {}},
    {"args": {"x": 1}},
    {"path": [1], "args": {"x": 1}},
])
def test_agent_malformed_run_tests_call_does_not_raise_type_error_and_preserves_valid_failure(malformed_args, monkeypatch):
    from nexus_agent_ai.agent import core
    valid_failure = runner_result(
        execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Fixture test process completed."),
        tool="run_tests", status=S.FAILED,
    )
    malformed_result = runner_result(
        execution_result("Pytest Results: FAILED\nmalformed arguments", executed=False, outcome="Execution failed."),
        tool="run_tests", status=S.NOT_EXECUTED,
    )
    results = iter([valid_failure, malformed_result])
    monkeypatch.setattr(core, "execute_tool", lambda name, args, **kwargs: next(results))

    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", {})]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", malformed_args)]),
        ProviderResponse(text="All tests pass."),
    ]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")

    assert len(agent.unresolved_errors) == 2
    assert any("assert 1 == 2" in err for err in agent.unresolved_errors)
    assert _check_run_outcome(agent, exit_on_failure=False) is False

@pytest.mark.parametrize("status", [S.FAILED, S.UNKNOWN, S.TIMED_OUT])
def test_run_tests_scope_normalization_conservative_for_non_passed(status, monkeypatch):
    from nexus_agent_ai.agent import core
    results = iter([
        runner_result(
            execution_result("Pytest Results: FAILED\nassert 1 == 2", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=S.FAILED,
        ),
        runner_result(
            execution_result("Pytest Results: non-passed", executed=True, outcome="Fixture test process completed."),
            tool="run_tests", status=status,
        ),
    ])
    monkeypatch.setattr(core, "execute_tool", lambda name, args, **kwargs: next(results))
    steps = [
        ProviderResponse(tool_calls=[ToolCall("c1", "run_tests", {})]),
        ProviderResponse(tool_calls=[ToolCall("c2", "run_tests", {"path": ""})]),
        ProviderResponse(text="All tests pass."),
        ProviderResponse(text="The test outcome remains unresolved."),
    ]
    provider = ScriptedProvider(steps)
    agent = Agent(provider, tools=[T.RUN_TESTS_TOOL], verbose=False)
    agent.run("Run the tests")
    assert len(agent.unresolved_errors) == 1
    assert _check_run_outcome(agent, exit_on_failure=False) is False
