"""Exercise real pytest targeting and bounded, factual no-tests diagnostics."""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import consent, tools as T
from nexus_agent_ai.agent.refusals import RefusalClass, RefusalText


@pytest.fixture
def approved(monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")


def failed(result, kind=RefusalClass.OTHER):
    assert isinstance(result, RefusalText), result
    assert result.refusal_class is kind
    return result


def observed_runner(monkeypatch, *, exit_code=5, stdout="", stderr=""):
    calls = []
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, "fixture permission"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=exit_code, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(T.subprocess, "run", run)
    return calls


def test_real_pytest_accepts_explicit_nonstandard_filename_and_exact_node_id(tmp_path, approved):
    source = tmp_path / "arithmetic_checks.py"
    source.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('number', [2], ids=['case with spaces'])\n"
        "def test_double(number):\n    assert number * 2 == 4\n\n"
        "def test_not_selected():\n    assert False\n"
    )
    result = T.execute_tool("run_tests", {
        "path": str(source) + "::test_double[case with spaces]", "args": "-q",
    })
    assert not isinstance(result, RefusalText), result
    assert "=== Pytest Results: PASSED ===" in result
    assert "1 passed" in result
    assert "FAILED" not in result


def test_real_pytest_collects_tests_from_explicit_non_test_filename(tmp_path, approved):
    source = tmp_path / "checks.py"
    source.write_text("def test_behavior():\n    assert sum([2, 3]) == 5\n")
    result = T.execute_run_tests(str(source), "-q")
    assert "PASSED" in result and "1 passed" in result
    assert not isinstance(result, RefusalText)


def test_real_pytest_source_without_tests_reports_observed_zero_not_success(tmp_path, approved):
    source = tmp_path / "calculator.py"
    source.write_text("def add(a, b):\n    return a + b\n")
    result = failed(T.execute_tool("run_tests", {"path": str(source), "args": "-q"}))
    assert "NO TESTS COLLECTED (exit code 5)" in result
    assert f"Target: {source}" in result
    assert "Observed target kind: file; collected: 0 tests." in result
    assert "Write an actual test" in result
    assert "no tests ran" in result
    assert "PASSED" not in result
    assert "--- stdout tail" in result and "--- stderr tail" in result


@pytest.mark.parametrize("exit_code", [0, 1, 2, 5])
def test_result_tails_preserve_exact_whitespace_and_separate_streams(tmp_path, monkeypatch, exit_code):
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    stdout = " \n\t leading stdout\n\n  trailing stdout \t\n"
    stderr = "\n  stderr diagnostic\t \n\n"
    observed_runner(monkeypatch, exit_code=exit_code, stdout=stdout, stderr=stderr)
    result = T.execute_run_tests(str(source))
    expected = (f"--- stdout tail ({len(stdout)}/{len(stdout)} characters) ---\n" + stdout
                + "\n--- end stdout tail ---\n"
                + f"--- stderr tail ({len(stderr)}/{len(stderr)} characters) ---\n" + stderr
                + "\n--- end stderr tail ---")
    assert result.endswith(expected)
    if exit_code:
        failed(result)
    else:
        assert not isinstance(result, RefusalText)


def test_failure_keeps_actual_tail_of_each_stream_not_a_shared_head_budget(monkeypatch):
    stdout = "DISCARDED_STDOUT_HEAD" + "x" * 4000 + " \nactual stdout tail \t\n"
    stderr = "DISCARDED_STDERR_HEAD" + "y" * 4000 + "\nactual stderr tail  \n"
    observed_runner(monkeypatch, exit_code=1, stdout=stdout, stderr=stderr)
    result = failed(T.execute_run_tests())
    assert "DISCARDED_STDOUT_HEAD" not in result
    assert "DISCARDED_STDERR_HEAD" not in result
    assert stdout[-3000:] + "\n--- end stdout tail ---" in result
    assert stderr[-3000:] + "\n--- end stderr tail ---" in result
    assert f"stdout tail (3000/{len(stdout)} characters)" in result
    assert f"stderr tail (3000/{len(stderr)} characters)" in result


def test_exit_five_names_default_directory_and_zero_collection(tmp_path, monkeypatch):
    observed_runner(monkeypatch)
    result = failed(T.execute_run_tests())
    assert f"Target: {tmp_path}" in result
    assert "Observed target kind: directory; collected: 0 tests." in result


def test_node_id_retains_selector_and_existing_execution_guards(tmp_path, monkeypatch):
    source = tmp_path / "checks.py"
    source.write_text("pass\n")
    monkeypatch.setenv("FAKE_PRIVATE_CREDENTIAL", "must never be inherited")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unsafe-from-environment")
    monkeypatch.setenv("PYTHONPATH", "do-not-inherit")
    calls = observed_runner(monkeypatch, exit_code=0)
    suffix = "::TestGroup::test_case[hyphen-space value]"
    result = T.execute_run_tests(str(source) + suffix, "-q --maxfail=2 -k 'case and not other'")
    assert "PASSED" in result
    command, options = calls[0]
    assert command == [sys.executable, "-m", "pytest", "-o", "addopts=", str(source) + suffix,
                       "-q", "--maxfail=2", "-k", "case and not other"]
    assert options["timeout"] == 45.0
    assert options["cwd"] == str(tmp_path)
    assert options["stdin"] == subprocess.DEVNULL
    assert options["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "FAKE_PRIVATE_CREDENTIAL" not in options["env"]
    assert "PYTEST_ADDOPTS" not in options["env"]
    assert "PYTHONPATH" not in options["env"]
    assert "must never be inherited" not in result


@pytest.mark.parametrize("target_kind", ["outside", "sensitive", "state", "alias", "home_root", "home_dotfile"])
def test_node_id_cannot_bypass_workspace_sensitive_or_home_guards(tmp_path, monkeypatch, target_kind):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    calls = observed_runner(monkeypatch)
    if target_kind == "outside":
        target = tmp_path / "outside.py"
    elif target_kind == "sensitive":
        target = work / "credentials.py"
    elif target_kind == "state":
        state = work / "agent-state"
        monkeypatch.setenv("NEXUS_AGENT_HOME", str(state))
        target = state / "checks.py"
    elif target_kind == "alias":
        target = work / "checks.py"
        target.symlink_to(tmp_path / "outside.py")
    else:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: work))
        target = work if target_kind == "home_root" else work / ".bashrc"
    result = failed(T.execute_run_tests(str(target) + "::test_example"), RefusalClass.SENSITIVE_PATH)
    assert not calls
    assert "NO TESTS COLLECTED" not in result


def test_node_id_still_requires_existing_path_and_file_selector(tmp_path, monkeypatch):
    calls = observed_runner(monkeypatch)
    failed(T.execute_run_tests("missing.py::test_example"), RefusalClass.NOT_FOUND)
    failed(T.execute_run_tests(str(tmp_path) + "::test_example"), RefusalClass.TARGETING)
    failed(T.execute_run_tests("::test_example"), RefusalClass.TARGETING)
    source = tmp_path / "checks.py"
    source.write_text("pass\n")
    failed(T.execute_run_tests(str(source) + "::"), RefusalClass.TARGETING)
    assert not calls


@pytest.mark.parametrize("args", ["--override-ini=addopts=bad", "-p injected_plugin", "--rootdir=/", "--collect-only", "-k 'x;unsafe'", "--maxfail=0"])
def test_unsafe_flags_do_not_execute_after_valid_node_target(tmp_path, monkeypatch, args):
    source = tmp_path / "checks.py"
    source.write_text("pass\n")
    calls = observed_runner(monkeypatch)
    failed(T.execute_run_tests(str(source) + "::test_example", args), RefusalClass.TARGETING)
    assert not calls


def test_permission_and_missing_pytest_remain_typed_before_execution(monkeypatch):
    calls = observed_runner(monkeypatch)
    monkeypatch.setattr(consent, "execution_allowed", lambda: (False, "No grant"))
    failed(T.execute_run_tests("source.py::test_example"), RefusalClass.PERMISSION)
    monkeypatch.setattr(consent, "execution_allowed", lambda: (True, ""))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert "pytest is not installed" in failed(T.execute_run_tests())
    assert not calls


def test_timeout_keeps_exact_available_stream_tails(monkeypatch):
    observed_runner(monkeypatch)
    stdout = b" \npartial stdout \t\n"
    stderr = b"\n partial stderr  \n"
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 45.0, output=stdout, stderr=stderr)
    monkeypatch.setattr(T.subprocess, "run", timeout)
    result = failed(T.execute_run_tests())
    assert "timed out after 45 seconds" in result
    assert "--- stdout tail ---\n" + stdout.decode() + "\n--- end stdout tail ---" in result
    assert "--- stderr tail ---\n" + stderr.decode() + "\n--- end stderr tail ---" in result
    assert "PASSED" not in result
