"""Private runner evidence follows execution facts, never printed success claims."""
import importlib.util
import subprocess

import pytest

from nexus_agent_ai.agent import consent, tools as T
from nexus_agent_ai.agent.refusals import (
    RefusalClass, RunnerStatus as S, execution_metadata, execution_result,
    refusal, replace_result_text, runner_result, runner_status,
)
from nexus_agent_ai.agent.result_budget import guard_tool_result


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
