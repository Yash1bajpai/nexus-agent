"""Execution receipts describe handler facts, never infer success from raw output."""
import builtins
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import consent, tools as T
from nexus_agent_ai.agent.refusals import (
    RefusalClass, RefusalText, ToolResult, execution_ran, execution_result, refusal,
)
from nexus_agent_ai.providers.base import Tool


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.reset_file_tracking()
    yield
    T.reset_file_tracking()


def receipt(result, *, executed, outcome=None):
    assert isinstance(result, str)
    assert result.executed is executed
    assert execution_ran(result) is executed
    if outcome is not None:
        assert result.outcome == outcome
    assert isinstance(result.outcome, str)
    assert len(result.outcome) < 200
    return result


def test_optional_size_hook_is_private_and_read_schema_stays_unchanged():
    plain = Tool("plain", "description", {"type": "object"}, lambda: "ok")
    assert plain.estimate_result_tokens is None
    read = next(tool for tool in T.get_all_tools() if tool.name == "read_file")
    assert read.estimate_result_tokens is T.estimate_read_file_tokens
    assert read.input_schema == {
        "type": "object",
        "properties": {"path": {
            "type": "string", "description": "Relative or absolute file path to read.",
        }},
        "required": ["path"],
    }


def test_read_estimate_stats_bytes_without_opening_or_remembering(monkeypatch):
    path = Path("large.txt")
    path.write_bytes(b"a" * 17)

    def forbidden(*args, **kwargs):
        raise AssertionError("estimate must not open or remember file contents")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(T, "_remember_file", forbidden)
    assert T.READ_FILE_TOOL.estimate_result_tokens(path="large.txt") == 5
    assert T._FILE_STATE == {}
    assert T._FILE_DIGESTS == {}
    assert T._AUTO_READ_FILES == set()


@pytest.mark.parametrize("path", [".env", "private.key", "secret-data.txt", ".ssh/id_rsa"])
def test_estimate_sensitive_paths_defer_without_stat_or_read(path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sensitive paths must be rejected before file stat")

    # Isolate the size probe from workspace validation's own metadata checks.
    monkeypatch.setattr(T, "_validate_workspace_path", lambda value: Path(value))
    monkeypatch.setattr(Path, "stat", forbidden)
    assert T.estimate_read_file_tokens(path) is None
    assert not T._FILE_STATE


def test_estimate_workspace_state_and_symlink_boundaries(tmp_path, monkeypatch):
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("outside")
    Path("alias.txt").symlink_to(outside)
    state = tmp_path / "state"
    state.mkdir()
    (state / "history.txt").write_text("state")
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(state))
    assert T.estimate_read_file_tokens(str(outside)) is None
    assert T.estimate_read_file_tokens("alias.txt") is None
    assert T.estimate_read_file_tokens("state/history.txt") is None
    assert not T._FILE_STATE


def test_estimate_home_protection_is_identical_to_read(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    Path(".bashrc").write_text("protected")
    assert T.estimate_read_file_tokens(".bashrc") is None
    assert T.estimate_read_file_tokens(".") is None
    assert T.execute_read_file(".bashrc").refusal_class is RefusalClass.SENSITIVE_PATH


@pytest.mark.parametrize("kwargs", [{}, {"path": None}, {"path": []}, {"path": "x", "memory": None}])
def test_bad_estimate_arguments_defer_to_typed_dispatch(kwargs):
    assert T.estimate_read_file_tokens(**kwargs) is None
    result = receipt(T.execute_tool("read_file", kwargs), executed=False)
    assert isinstance(result, RefusalText)
    assert result.refusal_class is RefusalClass.TARGETING


def test_estimate_missing_directory_and_unreadable_paths_preserve_read_errors(monkeypatch):
    Path("folder").mkdir()
    Path("data.txt").write_text("abcd")
    assert T.estimate_read_file_tokens("absent.txt") is None
    assert T.estimate_read_file_tokens("folder") is None
    monkeypatch.setattr(T.os, "access", lambda *a: False)
    assert T.estimate_read_file_tokens("data.txt") is None
    missing = receipt(T.execute_tool("read_file", {"path": "absent.txt"}), executed=False)
    assert missing.refusal_class is RefusalClass.NOT_FOUND
    assert missing == "ERROR: File not found: absent.txt"


def test_estimate_stat_permission_error_does_not_become_budget_refusal(monkeypatch):
    Path("data.txt").write_text("abcd")

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "stat", denied)
    assert T.estimate_read_file_tokens("data.txt") is None
    monkeypatch.setattr(T, "open", denied, raising=False)
    result = receipt(T.execute_read_file("data.txt"), executed=False)
    assert result.refusal_class is RefusalClass.PERMISSION


def test_execution_result_preserves_refusal_enum_and_c1_recovery_metadata():
    source = refusal(RefusalClass.BLIND_EDIT, "unchanged voice")
    source.recovery_read = ("file.py", b"digest", "private source")
    source._private_fact = object()
    result = receipt(execution_result(source, executed=True, outcome="Completed."), executed=True)
    assert isinstance(result, RefusalText)
    assert result == source == "unchanged voice"
    assert result.refusal_class is RefusalClass.BLIND_EDIT
    assert result.recovery_read is source.recovery_read
    assert result._private_fact is source._private_fact
    assert source.executed is False


def test_execution_result_preserves_other_string_subclasses():
    class CustomText(str):
        pass

    source = CustomText("text")
    source._marker = object()
    result = execution_result(source, executed=True, outcome="Completed.")
    assert isinstance(result, CustomText)
    assert result._marker is source._marker
    assert execution_ran(result)


@pytest.mark.parametrize("text", [
    "ERROR: failed", " SANDBOX BLOCK: blocked", "[SYSTEM] duplicate",
    "[SYSTEM WARNING: Duplicate Tool Call Detected]", "Not executed: aborted",
    "Suppressed duplicate", "[Suppressed: duplicate]", "Tool call suppressed: duplicate",
])
def test_legacy_execution_fallback_is_conservative(text):
    assert execution_ran(text) is False
    assert execution_ran(execution_result(text, executed=True, outcome="Completed.")) is True


def test_real_read_starting_with_error_has_positive_receipt_and_tracking():
    Path("data.txt").write_text("ERROR: literal file content")
    result = receipt(T.execute_tool("read_file", {"path": "data.txt"}), executed=True)
    assert isinstance(result, ToolResult)
    assert not isinstance(result, RefusalText)
    assert result == "ERROR: literal file content"
    assert str(Path("data.txt").resolve()) in T._FILE_STATE


@pytest.mark.parametrize("code", [0, 1, -9])
def test_run_code_receipts_report_real_exit_without_copying_stdout(code, monkeypatch):
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=code, stdout="observed stdout", stderr="observed stderr",
    ))
    result = receipt(T.execute_run_code("print(1)"), executed=True)
    assert result == "STDOUT:\nobserved stdout\nSTDERR:\nobserved stderr"
    assert f"exit code {code}" in result.outcome
    assert "observed" not in result.outcome
    assert isinstance(result, RefusalText) is (code != 0)


def test_run_code_no_output_failure_never_claims_success(monkeypatch):
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="",
    ))
    result = receipt(T.execute_run_code("pass"), executed=True)
    assert result == "[Code execution failed with exit code 1 and no output]"
    assert "success" not in result


def test_run_code_timeout_is_execution_not_refusal_to_start(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", T.CODE_EXECUTION_TIMEOUT)

    monkeypatch.setattr(T.subprocess, "run", timeout)
    result = receipt(T.execute_run_code("while True: pass"), executed=True)
    assert isinstance(result, RefusalText)
    assert "timed out" in result.outcome
    assert "completion unknown" in result.outcome
    assert "no effect" not in result.outcome


def test_sandbox_refusal_then_corrected_handler_can_execute(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="1\n", stderr="")

    monkeypatch.setattr(T.subprocess, "run", run)
    for _ in range(2):
        blocked = receipt(T.execute_tool("run_code", {"code": "import os"}), executed=False)
        assert blocked.refusal_class is RefusalClass.PERMISSION
    assert calls == []
    corrected = receipt(T.execute_tool("run_code", {"code": "print(1)"}), executed=True)
    assert corrected == "STDOUT:\n1"
    assert len(calls) == 1


def test_unsupported_code_and_unapproved_tests_do_not_execute(monkeypatch):
    monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "No grant."))
    receipt(T.execute_run_code("pass", "javascript"), executed=False)
    receipt(T.execute_run_tests(), executed=False)
    receipt(T.execute_run_file("file.py"), executed=False)


@pytest.fixture
def approved_tests(monkeypatch):
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, ""))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())


@pytest.mark.parametrize("code,status", [(0, "PASSED"), (1, "FAILED"), (5, "NO TESTS COLLECTED"), (2, "FAILED")])
def test_pytest_receipt_uses_exit_status_and_preserves_exact_stream_tails(code, status, approved_tests, monkeypatch):
    stdout, stderr = " \n" + "o" * 3100 + "\n ", "err\r\n "
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=code, stdout=stdout, stderr=stderr,
    ))
    result = receipt(T.execute_run_tests(), executed=True,
                     outcome=f"Pytest {status} (exit code {code}).")
    assert f"--- stdout tail (3000/{len(stdout)} characters) ---\n" + stdout[-3000:] + "\n--- end stdout tail ---" in result
    assert f"--- stderr tail ({len(stderr)}/{len(stderr)} characters) ---\n" + stderr + "\n--- end stderr tail ---" in result
    assert isinstance(result, RefusalText) is (code != 0)
    assert "err" not in result.outcome
    if code != 5:
        assert "collected" not in result.outcome


def test_pytest_timeout_keeps_observed_byte_tails_and_unknown_completion(approved_tests, monkeypatch):
    stdout, stderr = b" \nstdout\n ", b"stderr\xff\n"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("pytest", 45, output=stdout, stderr=stderr)

    monkeypatch.setattr(T.subprocess, "run", timeout)
    result = receipt(T.execute_run_tests(), executed=True)
    assert stdout.decode() in result
    assert stderr.decode("utf-8", errors="replace") in result
    assert "completion unknown" in result.outcome
    assert "no effect" not in result.outcome


@pytest.mark.parametrize("args", [{}, {"path": "x", "memory": None}, {"path": "x", "extra": True}, None])
def test_invalid_dispatch_arguments_never_start_handler(args, monkeypatch):
    calls = []

    def handler(path):
        calls.append(path)
        return "data"

    monkeypatch.setattr(T, "get_all_tools", lambda: [SimpleNamespace(name="read_file", execute=handler)])
    result = receipt(T.execute_tool("read_file", args), executed=False)
    assert result.refusal_class is RefusalClass.TARGETING
    assert calls == []


@pytest.mark.parametrize("error", [TypeError("internal bug"), RuntimeError("internal bug")])
def test_handler_exception_after_start_is_unknown_possible_effects(error, monkeypatch):
    calls = []

    def handler(path):
        calls.append(path)
        raise error

    monkeypatch.setattr(T, "get_all_tools", lambda: [SimpleNamespace(name="read_file", execute=handler)])
    result = receipt(T.execute_tool("read_file", {"path": "x"}), executed=True)
    assert calls == ["x"]
    assert result.refusal_class is RefusalClass.OTHER
    assert "completion unknown" in result.outcome
    assert "no effect" not in result.outcome


@pytest.mark.parametrize("text,ran", [("ordinary result", True), ("ERROR: legacy refusal", False)])
def test_dispatch_wraps_legacy_returns_without_changing_public_text(text, ran, monkeypatch):
    monkeypatch.setattr(T, "get_all_tools", lambda: [SimpleNamespace(name="legacy", execute=lambda: text)])
    result = receipt(T.execute_tool("legacy", {}), executed=ran)
    assert result == text


@pytest.mark.parametrize("code", [0, 1, 128])
def test_git_diff_receipt_reports_observed_exit_without_raw_diff(code, tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_git_root", lambda: tmp_path)
    monkeypatch.setattr(T, "_git_diff_paths", lambda *a: (["module.py"], False, True))
    monkeypatch.setattr(T, "_safe_git_run", lambda *a, **kw: SimpleNamespace(
        returncode=code, stdout="diff --git a/module.py b/module.py\n+raw source", stderr="raw error",
    ))
    result = receipt(T.execute_git_diff(), executed=True)
    assert f"exit code {code}" in result.outcome
    assert "raw" not in result.outcome
    if code:
        assert result.refusal_class is RefusalClass.OTHER
        assert "FAILED" in result.outcome
    else:
        assert "+raw source" in result


def test_git_diff_clean_receipt_is_not_invented_patch_success(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_git_root", lambda: tmp_path)
    monkeypatch.setattr(T, "_git_diff_paths", lambda *a: ([], False, False))
    monkeypatch.setattr(T, "_git_status_entries", lambda *a: ([], False, False))
    monkeypatch.setattr(T, "_safe_git_run", lambda *a, **kw: SimpleNamespace(returncode=0))
    result = receipt(T.execute_git_diff(), executed=True)
    assert result == "No changes to commit. Working tree is clean and nothing is staged."
    assert "working tree clean" in result.outcome


def test_estimate_does_not_grant_read_before_edit_authority():
    Path("module.py").write_text("value = 1\n")
    assert T.estimate_read_file_tokens("module.py") == 3
    result = receipt(T.execute_patch_file("module.py", "value = 1", "value = 2"), executed=False)
    assert result.refusal_class is RefusalClass.BLIND_EDIT
    assert hasattr(result, "recovery_read")
    assert Path("module.py").read_text() == "value = 1\n"


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "x", "expected_state": None}),
    ("run_code", {"code": "pass", "gated_reason": None}),
    ("history_read", {"query": "x", "memory": None}),
])
def test_private_dispatch_state_cannot_be_supplied_by_model(name, args, monkeypatch):
    def forbidden():
        raise AssertionError("private arguments must be rejected before dispatch")

    monkeypatch.setattr(T, "get_all_tools", forbidden)
    result = receipt(T.execute_tool(name, args), executed=False)
    assert result.refusal_class is RefusalClass.TARGETING


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": []}),
    ("run_code", {"code": "pass", "language": None}),
    ("patch_file", {"path": "x", "target": "x", "replacement": "y", "allow_multiple": 1}),
    ("history_read", {"query": "x", "limit": True}),
])
def test_schema_type_errors_are_targeting_before_handler_start(name, args):
    result = receipt(T.execute_tool(name, args), executed=False)
    assert result.refusal_class is RefusalClass.TARGETING


@pytest.mark.parametrize("name", ["run_code", "run_tests"])
def test_subprocess_permission_denial_is_not_execution(name, approved_tests, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("not permitted")

    monkeypatch.setattr(T.subprocess, "run", denied)
    result = T.execute_run_code("pass") if name == "run_code" else T.execute_run_tests()
    receipt(result, executed=False)
    assert result.refusal_class is RefusalClass.PERMISSION


def test_mutator_exception_after_write_does_not_claim_no_effect(monkeypatch):
    def failed_tracking(*args, **kwargs):
        raise RuntimeError("tracking failed after write")

    monkeypatch.setattr(T, "_remember_file", failed_tracking)
    result = receipt(T.execute_write_file("new.txt", "real contents"), executed=True)
    assert Path("new.txt").read_text() == "real contents"
    assert result.refusal_class is RefusalClass.OTHER
    assert "completion unknown" in result.outcome
    assert "no effect" not in result.outcome


@pytest.mark.parametrize("redacted", [False, True])
def test_history_lookup_receipt_preserves_exact_record_schema(redacted, tmp_path, monkeypatch):
    import json
    from nexus_agent_ai.agent.persistence import SQLiteMemory

    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="receipts")
    memory.replace([{"role": "user", "content": "history"}])
    entry = {
        "entry_id": 1, "role": "tool" if redacted else "assistant",
        "content": "private source" if redacted else "literal history",
        "timestamp": "2026-11-03T00:00:00Z", "truncated": False,
        "paths": [".env"] if redacted else [],
        "content_kind": "tool_result" if redacted else "conversation",
        "provenance_known": True,
    }
    monkeypatch.setattr(memory, "history_entries", lambda *a: [entry])
    result = receipt(
        T.execute_history_read("query", memory=memory), executed=True,
        outcome="history_read query completed with redactions" if redacted else "history_read query completed",
    )
    records = json.loads(result[len(T._HISTORY_READ_HEADER) + 1:])
    assert set(records[0]) == {"entry_id", "role", "timestamp", "text", "truncated"}
    assert records[0]["entry_id"] == 1
    assert records[0]["truncated"] is False
    assert not T._FILE_STATE
    if redacted:
        assert isinstance(result, RefusalText)
        assert result.refusal_class is RefusalClass.SENSITIVE_PATH
        assert "private source" not in result
    else:
        assert records[0]["text"] == "literal history"


def test_history_unavailable_and_permission_denied_lookup_are_not_execution(tmp_path, monkeypatch):
    from nexus_agent_ai.agent.persistence import SQLiteMemory

    receipt(T.execute_history_read("query", memory=None), executed=False)
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="receipts")
    memory.replace([{"role": "user", "content": "history"}])

    def denied(*args, **kwargs):
        raise PermissionError("private reason")

    monkeypatch.setattr(memory, "history_entries", denied)
    result = receipt(T.execute_history_read("query", memory=memory), executed=False)
    assert result.refusal_class is RefusalClass.PERMISSION
    assert "private reason" not in result
