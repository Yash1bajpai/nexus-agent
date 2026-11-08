"""Tests for execute_run_code subprocess returncode handling and error formatting.

Covers:
1. rc=0 byte-identical formatting across empty, stdout-only, stderr-only, and combined streams.
2. Positive nonzero rc starts with 'ERROR:', contains exact numeric code, preserves streams, and omits success fallback.
3. Negative nonzero rc starts with 'ERROR:', contains exact numeric code, preserves streams, and omits success fallback.
4. Temporary file cleanup across rc=0, nonzero rc, TimeoutExpired, and generic subprocess exceptions.
5. Pre-launch guards: AST sandbox violations and syntax errors abort before subprocess launch.
6. Unsupported language rejection aborts before subprocess launch.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch
import pytest

from nexus_agent_ai.agent import tools
from nexus_agent_ai.utils.config import CODE_EXECUTION_TIMEOUT


# -----------------------------------------------------------------------------
# 1. Return code 0: Byte-identical preservation
# -----------------------------------------------------------------------------

def test_run_code_rc0_empty_output():
    proc = CompletedProcess(["python", "snippet.py"], 0, "", "")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1 + 1")
    assert result == "[Code executed successfully with no output]"


def test_run_code_rc0_stdout_only():
    proc = CompletedProcess(["python", "snippet.py"], 0, "42\n", "")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("print(42)")
    assert result == "STDOUT:\n42"


def test_run_code_rc0_stderr_only():
    proc = CompletedProcess(["python", "snippet.py"], 0, "", "DeprecationWarning: test\n")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1")
    assert result == "STDERR:\nDeprecationWarning: test"


def test_run_code_rc0_both_stdout_and_stderr():
    proc = CompletedProcess(["python", "snippet.py"], 0, "computed value\n", "debug info\n")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("print('computed value')")
    assert result == "STDOUT:\ncomputed value\nSTDERR:\ndebug info"


# -----------------------------------------------------------------------------
# 2. Positive nonzero return code: Starts with ERROR:, reports exact numeric rc
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("rc", [1, 2, 127])
def test_run_code_positive_nonzero_empty_streams(rc: int):
    proc = CompletedProcess(["python", "snippet.py"], rc, "", "")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1")
    assert result.startswith("ERROR:")
    assert f"return code {rc}" in result
    assert "[Code executed successfully with no output]" not in result


def test_run_code_positive_nonzero_stdout_only():
    proc = CompletedProcess(["python", "snippet.py"], 1, "partial result\n", "")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("print('partial result')")
    assert result.startswith("ERROR:")
    assert "return code 1" in result
    assert "STDOUT:\npartial result" in result
    assert "STDERR:" not in result
    assert "[Code executed successfully with no output]" not in result


def test_run_code_positive_nonzero_stderr_only():
    proc = CompletedProcess(["python", "snippet.py"], 1, "", "ZeroDivisionError: division by zero\n")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1")
    assert result.startswith("ERROR:")
    assert "return code 1" in result
    assert "STDOUT:" not in result
    assert "STDERR:\nZeroDivisionError: division by zero" in result
    assert "[Code executed successfully with no output]" not in result


def test_run_code_positive_nonzero_both_streams():
    proc = CompletedProcess(["python", "snippet.py"], 1, "partial output\n", "runtime trace\n")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("print('partial output')")
    assert result.startswith("ERROR:")
    assert "return code 1" in result
    assert "STDOUT:\npartial output" in result
    assert "STDERR:\nruntime trace" in result
    assert "[Code executed successfully with no output]" not in result


# -----------------------------------------------------------------------------
# 3. Negative nonzero return code (e.g. killed by signal): Exact numeric code
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("rc", [-9, -15])
def test_run_code_negative_nonzero_empty_streams(rc: int):
    proc = CompletedProcess(["python", "snippet.py"], rc, "", "")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1")
    assert result.startswith("ERROR:")
    assert f"return code {rc}" in result
    assert "[Code executed successfully with no output]" not in result


def test_run_code_negative_nonzero_with_streams():
    proc = CompletedProcess(["python", "snippet.py"], -9, "pre-kill buffer\n", "Killed\n")
    with patch.object(tools.subprocess, "run", return_value=proc):
        result = tools.execute_run_code("x = 1")
    assert result.startswith("ERROR:")
    assert "return code -9" in result
    assert "STDOUT:\npre-kill buffer" in result
    assert "STDERR:\nKilled" in result
    assert "[Code executed successfully with no output]" not in result


# -----------------------------------------------------------------------------
# 4. Temporary file cleanup: File unlinked across all lifecycle paths
# -----------------------------------------------------------------------------

def test_run_code_tempfile_cleanup_on_rc0():
    observed_temp = None

    def fake_run(cmd, **kwargs):
        nonlocal observed_temp
        observed_temp = Path(cmd[1])
        assert observed_temp.exists(), "Temporary script must exist while subprocess is active"
        return CompletedProcess(cmd, 0, "ok", "")

    with patch.object(tools.subprocess, "run", side_effect=fake_run):
        res = tools.execute_run_code("x = 1")
    assert res == "STDOUT:\nok"
    assert observed_temp is not None
    assert not observed_temp.exists(), "Temporary script must be unlinked after rc=0 execution"


def test_run_code_tempfile_cleanup_on_nonzero_rc():
    observed_temp = None

    def fake_run(cmd, **kwargs):
        nonlocal observed_temp
        observed_temp = Path(cmd[1])
        assert observed_temp.exists(), "Temporary script must exist while subprocess is active"
        return CompletedProcess(cmd, 1, "", "fault")

    with patch.object(tools.subprocess, "run", side_effect=fake_run):
        res = tools.execute_run_code("x = 1")
    assert res.startswith("ERROR:")
    assert observed_temp is not None
    assert not observed_temp.exists(), "Temporary script must be unlinked after nonzero rc execution"


def test_run_code_tempfile_cleanup_on_timeout():
    observed_temp = None

    def fake_run(cmd, **kwargs):
        nonlocal observed_temp
        observed_temp = Path(cmd[1])
        assert observed_temp.exists(), "Temporary script must exist while subprocess is active"
        raise subprocess.TimeoutExpired(cmd, CODE_EXECUTION_TIMEOUT)

    with patch.object(tools.subprocess, "run", side_effect=fake_run):
        res = tools.execute_run_code("x = 1")
    assert res == f"ERROR: Code execution timed out after {CODE_EXECUTION_TIMEOUT} seconds."
    assert observed_temp is not None
    assert not observed_temp.exists(), "Temporary script must be unlinked after timeout"


def test_run_code_tempfile_cleanup_on_subprocess_exception():
    observed_temp = None

    def fake_run(cmd, **kwargs):
        nonlocal observed_temp
        observed_temp = Path(cmd[1])
        assert observed_temp.exists(), "Temporary script must exist while subprocess is active"
        raise OSError("Subprocess spawn denied")

    with patch.object(tools.subprocess, "run", side_effect=fake_run):
        res = tools.execute_run_code("x = 1")
    assert res == "ERROR: Code execution failed: Subprocess spawn denied"
    assert observed_temp is not None
    assert not observed_temp.exists(), "Temporary script must be unlinked after unexpected spawn error"


# -----------------------------------------------------------------------------
# 5. Pre-launch checks: Blocked constructs never launch a subprocess
# -----------------------------------------------------------------------------

def test_run_code_ast_violation_aborts_before_subprocess():
    with patch.object(tools.subprocess, "run") as mock_run:
        res = tools.execute_run_code("import os")
    assert "SANDBOX BLOCK:" in res
    mock_run.assert_not_called()


def test_run_code_syntax_error_aborts_before_subprocess():
    with patch.object(tools.subprocess, "run") as mock_run:
        res = tools.execute_run_code("def (")
    assert res.startswith("ERROR: Code has a syntax error:")
    mock_run.assert_not_called()


def test_run_code_unsupported_language_aborts_before_subprocess():
    with patch.object(tools.subprocess, "run") as mock_run:
        res = tools.execute_run_code("puts 'hello'", language="ruby")
    assert res == "ERROR: Running language 'ruby' is not supported yet. Only Python is supported."
    mock_run.assert_not_called()
