"""Installed-wheel gate, test staging, and premodel audit tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / ".github/qualification/runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("q1_runner_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_actual_worker_exec_imports_are_installed(tmp_path):
    """Exact critical native node required in frozen contract manifest."""
    import nexus_agent_ai
    from nexus_agent_ai.agent import text_count_worker as worker
    package = Path(nexus_agent_ai.__file__).resolve().parent
    worker_script = package / "agent/text_count_worker.py"
    assert worker_script.is_file(), "text_count_worker.py must exist in installed package"
    assert not worker_script.is_symlink()

    for name in ("_workspace_paths.py", "text_count_scan.py", "text_count_limits.py", "refusals.py"):
        sibling = worker_script.with_name(name)
        assert sibling.is_file(), f"{name} must exist as sibling"
        assert not sibling.is_symlink()

    # 1. Direct -I -S -B execution of worker script with empty stdin must fail closed with exit 2
    direct = subprocess.run([sys.executable, "-I", "-S", "-B", str(worker_script)],
                            input=b"", capture_output=True, timeout=10)
    assert direct.returncode == 2, f"Direct execution must fail-closed on empty input, got {direct.returncode}"

    # 2. End-to-end child execution and receipt verification inside isolated workspace
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha beta gamma")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = worker.inspect_text("sample.txt", "beta", timeout=10)
    finally:
        os.chdir(old_cwd)
    assert result["executed"] is True
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["count"] == 1


def test_stage_tests_creates_isolated_config(tmp_path):
    runner = load_runner()
    # Create synthetic git repo to verify stage_tests without checkout or .git dependency
    synthetic_source = tmp_path / "synthetic_source"
    synthetic_source.mkdir()
    subprocess.run(["git", "init", str(synthetic_source)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Q1 Gate"], cwd=synthetic_source, check=True)
    subprocess.run(["git", "config", "user.email", "gate@test.local"], cwd=synthetic_source, check=True)

    (synthetic_source / "tests").mkdir()
    (synthetic_source / "tests/test_dummy.py").write_text("def test_dummy(): pass\n", encoding="utf-8")
    (synthetic_source / ".github/e2e").mkdir(parents=True)
    (synthetic_source / ".github/e2e/driver.py").write_text("# driver\n", encoding="utf-8")
    for helper in runner.HELPERS:
        hp = synthetic_source / helper
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text("# helper\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=synthetic_source, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=synthetic_source, check=True)

    contract_data = {"critical_native": []}
    dest = tmp_path / "staged-tests"
    staged = runner.stage_tests(synthetic_source, dest, contract_data)

    assert dest.is_dir()
    assert (dest / "isolated-pytest.ini").is_file()
    assert not (dest / "src").exists(), "source tree must not be staged"

    ini_content = (dest / "isolated-pytest.ini").read_text(encoding="utf-8")
    assert "pythonpath =" in ini_content
    assert "addopts =" in ini_content

    assert "tests/test_dummy.py" in staged["files"]
    assert ".github/e2e/driver.py" in staged["files"]

def test_critical_outcomes_evaluates_exact_node_and_platform():
    runner = load_runner()
    critical = [
        {"nodeid": "tests/test_sample.py::test_a", "platform": "all", "count": 1, "match": "exact"},
        {"nodeid": "tests/test_sample.py::test_param", "platform": "all", "count": 2, "match": "parameterized"},
        {"nodeid": "tests/test_win.py::test_native", "platform": "win32", "count": 1, "match": "exact"},
    ]

    outcomes_pass = {
        "tests/test_sample.py::test_a": "passed",
        "tests/test_sample.py::test_param[0]": "passed",
        "tests/test_sample.py::test_param[1]": "passed",
        "tests/test_win.py::test_native": "passed",
    }
    result_win = runner.critical_outcomes(outcomes_pass, critical, "win32")
    assert result_win["qualified"] is True

    result_linux = runner.critical_outcomes(outcomes_pass, critical, "linux")
    assert result_linux["qualified"] is True

    outcomes_skip = dict(outcomes_pass, **{"tests/test_win.py::test_native": "skipped"})
    result_win_skip = runner.critical_outcomes(outcomes_skip, critical, "win32")
    assert result_win_skip["qualified"] is False, "A skipped required native node must fail gate"

    outcomes_missing = dict(outcomes_pass)
    del outcomes_missing["tests/test_sample.py::test_a"]
    result_missing = runner.critical_outcomes(outcomes_missing, critical, "win32")
    assert result_missing["qualified"] is False


def test_premodel_audit_blocks_network_and_fetch_processes():
    runner = load_runner()
    import nexus_agent_ai
    package = Path(nexus_agent_ai.__file__).resolve().parent
    package_py = {p.relative_to(package).as_posix(): runner.digest(p) for p in package.rglob("*.py")}

    audit = runner.PremodelAudit(package, package_py)

    with pytest.raises(OSError, match="nonloopback_network_attempt"):
        audit("socket.connect", (None, ("example.com", 443)))

    with pytest.raises(OSError, match="fetch_or_model_process"):
        audit("subprocess.Popen", ("curl", ["curl.exe", "https://example.com"]))

    with pytest.raises(OSError, match="fetch_or_model_process"):
        audit("subprocess.Popen", ("llama-server", ["llama-server.exe", "-m", "foo"]))

    with pytest.raises(OSError, match="remote_git_attempt"):
        audit("subprocess.Popen", ("git", ["git", "fetch", "origin"]))

    # Loopback connect is allowed
    audit("socket.connect", (None, ("127.0.0.1", 8080)))
    audit("socket.connect", (None, ("::1", 8080)))


def test_deadline_from_file_validates_budget(tmp_path):
    runner = load_runner()
    deadline_file = tmp_path / "deadline.json"

    valid_deadline = {
        "work_deadline_utc": time.time() + 3600,
        "job_limit_minutes": 120,
        "reserve_minutes": 10,
    }
    deadline_file.write_text(json.dumps(valid_deadline))
    dl = runner.deadline_from_file(deadline_file)
    assert dl > time.monotonic()

    invalid_job = dict(valid_deadline, job_limit_minutes=60)
    deadline_file.write_text(json.dumps(invalid_job))
    with pytest.raises(runner.GateError, match="Wrong overall/teardown schedule"):
        runner.deadline_from_file(deadline_file)

    expired = dict(valid_deadline, work_deadline_utc=time.time() - 10)
    deadline_file.write_text(json.dumps(expired))
    with pytest.raises(runner.GateError, match="exhausted or invalid"):
        runner.deadline_from_file(deadline_file)
