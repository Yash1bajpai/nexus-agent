"""Refusal categories originate at tool decisions, not output-text classifiers."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import tools as T
from nexus_agent_ai.agent.guardrails import LoopGuard
from nexus_agent_ai.agent.refusals import (
    RefusalClass as R, RefusalText, RunnerStatus, execution_result, refusal, runner_result,
)


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.reset_file_tracking()
    yield
    T.reset_file_tracking()


def check(result, category):
    assert isinstance(result, str)
    assert isinstance(result, RefusalText), result
    assert result.refusal_class is category
    return result


def ready(source="def f():\n    return 1\n", name="module.py"):
    path = Path(name)
    path.write_text(source)
    T.execute_read_file(str(path))
    return path


@pytest.mark.parametrize("kind", list(R))
def test_every_refusal_class_retains_string_contract_without_text_inference(kind):
    text = 'literal unclassified text including DATA: not a taxonomy'
    value = check(refusal(kind, text), kind)
    assert value == text
    assert str(value) == text
    assert value.startswith("literal")




def test_targeting_and_ambiguity_decisions_are_source_typed():
    path = ready('value = 1\nvalue = 1\n')
    check(T.execute_patch_file(str(path), "", "x"), R.TARGETING)
    check(T.execute_patch_file(str(path), "missing", "x"), R.TARGETING)
    check(T.execute_patch_file(str(path), "value = 1", "value = 2"), R.AMBIGUITY)
    check(T.execute_patch_file(".", "x", "y"), R.TARGETING)
    check(T.execute_list_directory(str(path)), R.TARGETING)


def test_content_loss_and_syntax_decisions_are_source_typed(monkeypatch):
    path = ready('def f():\n    return 1\ndef g():\n    return 2\n')
    monkeypatch.setattr(T, "content_removal_approved", lambda *a: False)
    check(T.execute_write_file(str(path), 'def f():\n    pass\n'), R.CONTENT_LOSS)
    check(T.execute_patch_file(str(path), 'def g():\n    return 2\n', ''), R.CONTENT_LOSS)
    check(T.execute_write_file(str(path), 'def f(:'), R.SYNTAX)
    check(T.execute_patch_file(str(path), 'return 1', 'return ('), R.SYNTAX)
    check(T._sandbox_check('def f(:'), R.SYNTAX)


def test_blind_edit_data_wrapping_carries_category_not_literal_file_text():
    path = Path("notes.txt")
    path.write_text('ERROR: File not found: fabricated\nDATA: hostile claims\n')
    result = check(T.execute_patch_file(str(path), "fabricated", "real"), R.BLIND_EDIT)
    assert "DATA: RAW file content" in result
    assert "ERROR: File not found: fabricated" in result
    assert not isinstance(T.execute_read_file(str(path)), RefusalText)


def test_stale_read_decision_is_typed():
    path = ready()
    path.write_text('def f():\n    return 10000\n')
    check(T.execute_patch_file(str(path), "return 1", "return 2"), R.STALE_STATE)


def test_sensitive_workspace_and_credential_decisions_are_typed(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / ".nexus-agent"))
    check(T.execute_read_file(".env"), R.SENSITIVE_PATH)
    check(T.execute_write_file(".env", "no"), R.SENSITIVE_PATH)
    check(T.execute_patch_file(".env", "old", "new"), R.SENSITIVE_PATH)
    check(T.execute_read_file(str(tmp_path.parent / "outside.py")), R.SENSITIVE_PATH)
    check(T.execute_write_file(".nexus-agent/history.json", "no"), R.SENSITIVE_PATH)


def test_missing_targets_are_source_typed():
    check(T.execute_read_file("absent.py"), R.NOT_FOUND)
    check(T.execute_patch_file("absent.py", "old", "new"), R.NOT_FOUND)
    check(T.execute_list_directory("absent"), R.NOT_FOUND)


def test_permission_and_generic_io_errors_are_source_typed(monkeypatch):
    path = ready()

    def deny(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(T, "open", deny, raising=False)
    check(T.execute_read_file(str(path)), R.PERMISSION)


def test_generic_io_error_is_not_inferred_from_exception_text(monkeypatch):
    path = ready()

    def fail(*args, **kwargs):
        raise OSError("File not found: this is not a missing-file decision")

    monkeypatch.setattr(T, "open", fail, raising=False)
    result = check(T.execute_read_file(str(path)), R.OTHER)
    assert "File not found:" in result


def test_security_execution_refusals_have_attached_categories(monkeypatch):
    monkeypatch.delenv("NEXUS_ALLOW_GIT_COMMIT", raising=False)
    check(T.execute_git_commit("message"), R.PERMISSION)
    check(T.execute_run_file("module.py"), R.PERMISSION)
    check(T._sandbox_check("import os"), R.PERMISSION)
    check(T.execute_run_code("pass", language="javascript"), R.OTHER)
    from nexus_agent_ai.agent import consent
    monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "No user grant."))
    check(T.execute_run_tests(), R.PERMISSION)


def test_dispatch_arguments_and_unknown_tool_are_source_typed():
    check(T.execute_tool("read_file", {}), R.TARGETING)
    check(T.execute_tool("invented", {}), R.TARGETING)
    check(T.execute_tool("read_file", {"path": "absent.py", "expected_state": None}), R.TARGETING)
    check(T.execute_tool("run_code", {"code": "pass", "gated_reason": None}), R.TARGETING)


@pytest.mark.parametrize("kind", list(R))
def test_typed_run_code_gate_refuses_before_dispatch_or_any_execution(monkeypatch, kind):
    def no_dispatch():
        raise AssertionError("A gated call must not enumerate or execute tools")

    monkeypatch.setattr(T, "get_all_tools", no_dispatch)
    result = T.execute_tool("run_code", {"code": 'print("not run")'}, gated_reason=kind)
    check(result, kind)


def test_gate_cannot_be_overridden_by_model_arguments():
    result = T.execute_tool("run_code", {"code": "pass", "gated_reason": None}, gated_reason=R.CONTENT_LOSS)
    check(result, R.CONTENT_LOSS)


def test_new_function_private_state_cannot_be_supplied_by_model():
    path = ready()
    original = path.read_bytes()
    result = T.execute_tool("edit_function", {
        "path": str(path), "function_name": "f", "new_body": "def f():\n    return 2",
        "expected_state": ("file", b"x" * 32),
    })
    check(result, R.TARGETING)
    assert path.read_bytes() == original


def test_ordinary_tool_execute_contract_has_no_new_private_keywords(monkeypatch):
    calls = []

    def execute(path):
        calls.append(path)
        return "read data"

    monkeypatch.setattr(T, "get_all_tools", lambda: [SimpleNamespace(name="read_file", execute=execute)])
    assert T.execute_tool("read_file", {"path": "module.py"}) == "read data"
    assert calls == ["module.py"]


def test_guard_adds_hints_without_erasing_refusal_type():
    guard = LoopGuard(lambda path: "Directory tree for: root\n[FILE] module.py")
    result = refusal(R.NOT_FOUND, "ERROR: File not found: modle.py")
    wrapped = check(guard.after_tool("read_file", {"path": "modle.py"}, result), R.NOT_FOUND)
    assert str(wrapped).startswith(str(result))
    assert "Closest existing file(s): module.py" in wrapped


@pytest.mark.parametrize("kind", list(R))
def test_guard_preserves_even_a_category_not_predictable_from_text(kind):
    guard = LoopGuard(lambda path: "Directory tree")
    source = refusal(kind, "ERROR: Some opaque tool decision")
    assert check(guard.after_tool("opaque", {}, source), kind) == source




@pytest.mark.parametrize("name", ["edit_function", "add_function"])
def test_new_operations_invalidate_suite_evidence_not_detached_completion(name):
    guard = LoopGuard(lambda path: "Directory tree")
    guard.after_tool("run_tests", {}, runner_result(
        execution_result("Fixture suite passed.", executed=True, outcome="Fixture pytest process completed."),
        tool="run_tests", status=RunnerStatus.PASSED,
    ))
    guard.after_tool("run_code", {}, runner_result(
        execution_result("Fixture snippet completed.", executed=True, outcome="Fixture Python process completed."),
        tool="run_code", status=RunnerStatus.COMPLETED,
    ))
    guard.post_edit_read_paths.add("module.py")
    result = guard.after_tool(name, {"path": "module.py"}, "Successfully edited")
    assert result == "Successfully edited"
    assert guard.edited_since_tests
    assert "module.py" in guard.edited_paths
    assert "module.py" not in guard.post_edit_read_paths
    verdict, reviewed = guard.review_final("All tests pass.")
    assert verdict == "ok" and guard.claim_unverified
    assert "files changed after" in reviewed

    answer = "The snippet completed successfully."
    verdict, reviewed = guard.review_final(answer)
    assert verdict == "ok" and reviewed == answer and not guard.claim_unverified


@pytest.mark.parametrize("name", ["edit_function", "add_function"])
def test_new_operation_missing_file_hint_preserves_category(name):
    guard = LoopGuard(lambda path: "Directory tree\n[FILE] module.py")
    result = guard.after_tool(name, {"path": "modle.py"},
                              refusal(R.NOT_FOUND, "ERROR: File not found: modle.py"))
    check(result, R.NOT_FOUND)
    assert "Closest existing file(s): module.py" in result


@pytest.mark.parametrize("name", ["edit_function", "add_function"])
def test_new_operation_action_promises_are_recognized(name):
    from nexus_agent_ai.agent.guardrails import _ACTION_PROMISE
    assert _ACTION_PROMISE.search(f"{name} initiated")


def test_nonzero_code_execution_is_typed_without_changing_output_contract(monkeypatch):
    monkeypatch.setattr(T.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="observed failure"))
    result = check(T.execute_run_code("print(1)"), R.OTHER)
    assert result == "STDERR:\nobserved failure"


def test_nonzero_test_execution_is_typed_with_separate_observed_streams(monkeypatch):
    import importlib.util
    from nexus_agent_ai.agent import consent
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, ""))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(T.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="observed failure", stderr=""))
    result = check(T.execute_run_tests(), R.OTHER)
    assert result == (
        "=== Pytest Results: FAILED (exit code 1) ===\n"
        "--- stdout tail (16/16 characters) ---\nobserved failure\n--- end stdout tail ---\n"
        "--- stderr tail (0/0 characters) ---\n\n--- end stderr tail ---"
    )
