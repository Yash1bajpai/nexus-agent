"""Runner tests use tiny files and subprocess fakes; no model or network execution."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid
import zipfile

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / ".github/diagnostics/budget_diagnostic_runner.py"
spec = importlib.util.spec_from_file_location("budget_diagnostic_runner_under_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def scope_factory():
    def create(root):
        root = root.resolve()
        root.mkdir(exist_ok=False)
        state = root / "state"
        state.mkdir()
        home, workspace, output = state / "home", state / "workspace", root / "output"
        for path in (home, workspace, output):
            path.mkdir()
        scope = {"schema_version": 1, "root": str(root), "state": str(state),
                 "home": str(home), "workspace": str(workspace), "output": str(output),
                 "scope_path": str(root / "scope.json"), "session_id": runner.SESSION_ID,
                 "db_path": str(home / ".nexus-agent/history.db"),
                 "run_id": str(uuid.uuid4()), "max_capture_bytes": runner.CAPTURE_CAP}
        Path(scope["scope_path"]).write_text(json.dumps(scope))
        return scope
    return create


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    path = tmp_path / "explicit.gguf"
    path.write_bytes(b"synthetic model fixture, never loaded")
    monkeypatch.setattr(runner, "MODEL_SHA256", runner.sha256_file(path))
    return path


def snapshot_pair(scope, row, *, kind="planner", preceding=True, coverage=True):
    directory = Path(scope["output"]) / row
    directory.mkdir(exist_ok=True)
    common = {"schema_version": 1, "row": row, "session_id": runner.SESSION_ID,
              "run_id": scope["run_id"], "replay_qualified": True, "truncated": False}
    success = {**common, "id": 2, "input_hash": "success-hash", "kind": "planner",
               "stage": "outbound_initial_fit", "actual_fit_completed": True,
               "result": {"success": True}}
    reference = {"id": 2, "file": "latest_success.json", "input_hash": "success-hash"} if preceding else None
    failure = {**common, "id": 3, "input_hash": "failure-hash", "kind": kind,
               "previous_success": reference, "result": {"success": False, "error": {
                   "class": "ContextBudgetError", "text": (
                       "Historical omission coverage gap: synthetic" if coverage else
                       "The tool's truthful outcome and output-omission receipt cannot fit the current context.")}}}
    manifest = {**common, "latest_success": reference, "failure": {
        "id": 3, "file": "failure.json", "input_hash": "failure-hash"}}
    for name, value in (("latest_success.json", success), ("failure.json", failure),
                        ("manifest.json", manifest)):
        runner.write_json(directory / name, value)
    return directory, success, failure


def exact_replay(snapshot):
    return {"status": "reproduced", "error_match": not snapshot["result"]["success"],
            "token_identical": snapshot["result"]["success"], "classification": "unresolved"}


def test_frozen_prompts_and_model_identity():
    assert runner.MODEL_REPO == "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    assert runner.MODEL_FILENAME == "LFM2.5-1.2B-Instruct-Q4_0.gguf"
    assert runner.MODEL_SHA256 == "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
    assert runner.ROWS == ("seed", "filler-1", "filler-2", "filler-3", "filler-4", "filler-5", "filler-6")
    prefix = ["chat", "-p", "local", "--no-stream", "--persist", "--session", "r11-budget-synthetic"]
    assert runner.frozen_argv("seed") == prefix + [
        "My name is Zephyr-QX7. Please read the file nexus_cfg_v9.yaml with your read_file tool and tell me its owner."]
    for index in range(1, 7):
        assert runner.frozen_argv(f"filler-{index}") == prefix + [
            f"Read evict_filler.txt with your read_file tool and count how many times the word alpha appears in section {index}."]
    with pytest.raises(ValueError):
        runner.frozen_argv("query")


def test_real_scope_api_is_exclusive_and_frozen_argv_agrees(tmp_path):
    from nexus_agent_ai.agent import budget_diagnostics
    root = tmp_path / "real-scope"
    scope = budget_diagnostics.create_scope(root)
    for row in runner.ROWS:
        assert budget_diagnostics.frozen_argv(row) == runner.frozen_argv(row)
    assert Path(scope["home"]) == root / "state/home"
    assert Path(scope["workspace"]) == root / "state/workspace"
    assert Path(scope["output"]) == root / "output"
    assert Path(scope["db_path"]) == root / "state/home/.nexus-agent/history.db"
    assert not Path(scope["home"]).is_relative_to(Path(scope["output"]))
    with pytest.raises(FileExistsError):
        budget_diagnostics.create_scope(root)


def test_stage_copies_only_verified_model_not_history_or_auth(tmp_path, tiny_model, scope_factory):
    scope = scope_factory(tmp_path / "scope")
    (tiny_model.parent / "token").write_text("secret")
    (tiny_model.parent / "history.db").write_text("private")
    home = Path(scope["home"])
    provenance = runner.stage_model(tiny_model, home)
    files = [path.relative_to(home) for path in home.rglob("*") if path.is_file()]
    assert files == [Path(".nexus-agent/models/LiquidAI--LFM2.5-1.2B-Instruct-GGUF") / runner.MODEL_FILENAME]
    staged = home / files[0]
    assert staged.read_bytes() == tiny_model.read_bytes()
    tiny_model.write_bytes(b"mutated source")
    assert staged.read_bytes() != tiny_model.read_bytes()
    assert provenance["sha256"] == runner.sha256_file(staged)
    with pytest.raises(ValueError, match="SHA256"):
        runner.stage_model(tiny_model, tmp_path / "different-home")
    assert not (tmp_path / "different-home").exists()


def test_child_environment_drops_host_secrets_and_configuration(tmp_path, scope_factory, monkeypatch):
    scope = scope_factory(tmp_path / "scope")
    for key in ("HF_TOKEN", "OPENAI_API_KEY", "HTTPS_PROXY", "PYTHONPATH", "NEXUS_CONTEXT_SIZE",
                "NEXUS_AGENT_ASSUME_YES", "NEXUS_ALLOW_PROJECT_EXECUTION", "NEXUS_AGENT_HOME"):
        monkeypatch.setenv(key, "sensitive-parent-value")
    git = tmp_path / "trusted-git" / ("git.exe" if os.name == "nt" else "git")
    git.parent.mkdir()
    git.write_bytes(b"synthetic executable, never run")
    untrusted_directory = tmp_path / "untrusted-parent-path"
    monkeypatch.setenv("PATH", str(untrusted_directory) + os.pathsep + str(git.parent))
    monkeypatch.setattr(runner.shutil, "which", lambda executable: str(git))
    environment = runner.child_environment(scope)
    assert environment["HOME"] == environment["USERPROFILE"] == scope["home"]
    assert environment["NEXUS_AGENT_HOME"] == str(Path(scope["home"]) / ".nexus-agent")
    assert not any(value == "sensitive-parent-value" for value in environment.values())
    assert set(environment) <= {
        "HOME", "USERPROFILE", "NEXUS_AGENT_HOME", "NEXUS_AGENT_MODEL_REPO", "NEXUS_AGENT_MODEL_FILENAME",
        "PATH", "TMPDIR", "TMP", "TEMP", "LANG", "PYTHONUTF8", "PYTHONNOUSERSITE", "SYSTEMROOT", "WINDIR"}
    assert Path(environment["TMP"]).parent == Path(scope["state"])
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert str(git.parent.resolve()) in environment["PATH"].split(os.pathsep)
    assert str(untrusted_directory) not in environment["PATH"].split(os.pathsep)
    with pytest.raises(FileExistsError):
        runner.child_environment(scope)


class FakeProcess:
    pid = 987654321

    def __init__(self, *, timeout=False):
        self.stdout = io.BytesIO(b"x" * (runner.STREAM_CAP * 3))
        self.stderr = io.BytesIO(b"synthetic stderr")
        self.returncode = None
        self.timeout = timeout
        self.waits = []
        self.killed = False

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.timeout and len(self.waits) == 1:
            raise subprocess.TimeoutExpired("synthetic", timeout)
        self.returncode = -9 if self.timeout else 0
        return self.returncode

    def kill(self):
        self.killed = True

    def poll(self):
        return self.returncode


def test_process_argv_cwd_timeout_and_stream_caps(tmp_path, scope_factory, monkeypatch):
    scope = scope_factory(tmp_path / "scope")
    environment = runner.child_environment(scope)
    seen = []
    process = FakeProcess()
    monkeypatch.setattr(runner.os, "killpg", lambda *args: None, raising=False)
    def popen(argv, **options):
        seen.append((argv, options))
        return process
    result = runner.execute_row(scope, "filler-2", environment, popen=popen)
    assert len(seen) == 1
    argv, options = seen[0]
    assert argv == [sys.executable, "-I", "-m", "nexus_agent_ai.diagnostic_cli",
                    "--scope", scope["scope_path"], "--row", "filler-2", "--", *runner.frozen_argv("filler-2")]
    assert options["cwd"] == scope["workspace"]
    assert options["env"] is environment
    assert options["stdin"] == subprocess.DEVNULL
    assert process.waits == [420]
    stdout = Path(scope["output"]) / "filler-2/stdout.txt"
    assert stdout.stat().st_size <= runner.STREAM_CAP
    assert stdout.read_bytes().endswith(runner.OUTPUT_MARKER)
    assert result["stdout"]["bytes_seen"] == runner.STREAM_CAP * 3
    assert result["stdout"]["truncated"] is True
    assert result["stderr"]["truncated"] is False
    assert result["output_capture_complete"] is True


def test_windows_timeout_kills_process_tree_without_retry(tmp_path, scope_factory, monkeypatch):
    scope = scope_factory(tmp_path / "scope")
    environment = runner.child_environment(scope)
    environment["SYSTEMROOT"] = str(tmp_path / "Windows")
    process = FakeProcess(timeout=True)
    calls = []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(runner.subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)))
    result = runner.execute_row(scope, "seed", environment, popen=lambda *args, **kwargs: process)
    assert result["timed_out"] is True
    assert process.waits == [420, 10]
    assert process.killed is True
    assert len(calls) == 1
    assert calls[0][0] == [str(Path(environment["SYSTEMROOT"]) / "System32/taskkill.exe"),
                           "/PID", str(process.pid), "/T", "/F"]
    assert calls[0][1]["env"] is environment
    assert calls[0][1]["timeout"] == 10


def test_oversized_json_is_valid_explicit_unreplayable_marker(tmp_path):
    path = tmp_path / "report.json"
    assert runner.write_json(path, {"huge": "x" * runner.JSON_CAP}) is False
    assert path.stat().st_size <= runner.JSON_CAP
    assert json.loads(path.read_text()) == {"replay_qualified": False, "truncated": True,
                                          "reason": "runner JSON byte cap exceeded"}
    assert not path.with_suffix(".json.tmp").exists()
    path.write_bytes(b"x" * (runner.CAPTURE_CAP + 1))
    assert runner.read_json(path) is None
    path.write_text("{broken")
    assert runner.read_json(path) is None


def test_stop_requires_real_coverage_and_token_identical_linked_actual_success(tmp_path, scope_factory):
    scope = scope_factory(tmp_path / "scope")
    directory, success, failure = snapshot_pair(scope, "seed")
    result = runner.inspect_row(scope, "seed", exact_replay)
    assert result["status"] == "reproduced"
    assert result["stop"] is True
    assert result["preceding_success"] == "token_identical"
    assert (directory / "failure-replay.json").is_file()
    assert (directory / "success-replay.json").is_file()
    success["run_id"] = "different-invocation"
    runner.write_json(directory / "latest_success.json", success)
    assert runner.inspect_row(scope, "seed", exact_replay)["stop"] is False


@pytest.mark.parametrize("mutation", ["missing", "hash", "id", "stage", "not_actual", "mismatch", "oversized"])
def test_no_invented_preceding_success(tmp_path, scope_factory, mutation):
    scope = scope_factory(tmp_path / "scope")
    directory, success, failure = snapshot_pair(scope, "seed", preceding=mutation != "missing")
    if mutation == "hash":
        success["input_hash"] = "not-linked"
    if mutation == "id":
        success["id"] = failure["id"] + 1
    if mutation == "stage":
        success["stage"] = "tool_result_fit_probe"
    if mutation == "not_actual":
        success["actual_fit_completed"] = False
    runner.write_json(directory / "latest_success.json", success)
    def replay(value):
        if value["result"]["success"] and mutation == "mismatch":
            return {"status": "reproduced", "token_identical": False}
        if value["result"]["success"] and mutation == "oversized":
            return {**exact_replay(value), "excess": "x" * runner.JSON_CAP}
        return exact_replay(value)
    result = runner.inspect_row(scope, "seed", replay)
    assert result["stop"] is False
    assert result["preceding_success"] != "token_identical"


def test_receipt_only_and_mismatching_failure_do_not_stop(tmp_path, scope_factory):
    scope = scope_factory(tmp_path / "scope")
    snapshot_pair(scope, "seed", kind="receipt", coverage=False)
    result = runner.inspect_row(scope, "seed", exact_replay)
    assert result["status"] == "reproduced"
    assert result["failure_kind"] == "receipt"
    assert result["stop"] is False
    assert runner.inspect_row(scope, "seed", lambda value: {"status": "mismatch"})["status"] == "unreplayable"


@pytest.mark.parametrize("mode,expected_rows,expected_status", [
    ("coverage", 1, "reproduced"), ("missing-success", 7, "reproduced"),
    ("receipt", 7, "reproduced"), ("none", 7, "not_reproduced"),
    ("timeout", 7, "unreplayable"), ("replay-error", 7, "unreplayable")])
def test_bundle_exact_fixtures_seven_rows_and_honest_stop(tmp_path, tiny_model, scope_factory, mode, expected_rows, expected_status):
    calls = []
    def execute(scope, row, environment):
        calls.append(row)
        workspace = Path(scope["workspace"])
        assert (workspace / "nexus_cfg_v9.yaml").read_text() == "owner: zephyr-labs\nrevision: 9\nmode: quarantine\n"
        assert (workspace / "evict_filler.txt").read_text() == (
            "Filler record alpha beta gamma delta epsilon zeta. " * 120 + "\n") * 6
        assert environment["HOME"] == environment["USERPROFILE"] == scope["home"]
        snapshot_pair(scope, row, kind="receipt" if mode == "receipt" else "planner",
                      preceding=mode != "missing-success", coverage=mode != "receipt")
        if mode == "none":
            manifest_path = Path(scope["output"]) / row / "manifest.json"
            manifest = runner.read_json(manifest_path)
            manifest["failure"] = None
            runner.write_json(manifest_path, manifest)
        return {"row": row, "exit_code": 0, "timed_out": mode == "timeout", "output_capture_complete": True}
    def replay(value):
        if mode == "replay-error":
            raise RuntimeError("sensitive exception text must not persist")
        return exact_replay(value)
    root = tmp_path / "bundle"
    result = runner.run_bundle(root, tiny_model, {"commit": "a" * 40}, create_scope=scope_factory,
                               replay_snapshot=replay, executor=execute)
    assert calls == list(runner.ROWS[:expected_rows])
    assert result["status"] == expected_status
    assert result["stopped_early"] == (mode == "coverage")
    assert result["model_readiness_claim"] is False
    assert "sensitive exception text" not in (root / "output/summary.json").read_text()
    assert not list((root / "output").rglob("*.gguf"))
    assert not list((root / "output").rglob("history.db"))
    assert (root / "state/home/.nexus-agent/models" / runner.MODEL_REPO.replace("/", "--") / runner.MODEL_FILENAME).is_file()


def test_model_mismatch_prevents_all_commands(tmp_path, tiny_model, scope_factory, monkeypatch):
    monkeypatch.setattr(runner, "MODEL_SHA256", "0" * 64)
    calls = []
    result = runner.run_bundle(tmp_path / "bundle", tiny_model, {}, create_scope=scope_factory,
                               replay_snapshot=exact_replay, executor=lambda *args: calls.append(args))
    assert calls == []
    assert result["status"] == "unreplayable"
    assert (tmp_path / "bundle/output/summary.json").is_file()


def test_provenance_rejects_source_tree_and_wrong_commit_before_import(tmp_path, monkeypatch):
    with monkeypatch.context() as gate_patch:
        gate_patch.setattr(runner.importlib, "import_module", lambda name: pytest.fail("gate must reject before import"))
        with pytest.raises(ValueError, match="commit gate"):
            runner.installed_provenance(tmp_path / "missing.whl", tmp_path, "a" * 40, "b" * 40)
    source = tmp_path / "source"
    source.mkdir()
    package = source / "__init__.py"
    package.write_text("")
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(package)))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=("a" * 40 + "\n").encode()))
    with pytest.raises(ValueError, match="site-packages"):
        runner.installed_provenance(tmp_path / "missing.whl", source, "a" * 40, "a" * 40)


def test_provenance_hashes_wheel_and_installed_payload(tmp_path, monkeypatch):
    prefix = tmp_path / "isolated-venv"
    package = prefix / "Lib/site-packages/nexus_agent_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"# exact wheel payload\n")
    wheel = tmp_path / "fresh.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nexus_agent_ai/__init__.py", b"# exact wheel payload\n")
    monkeypatch.setattr(runner.sys, "prefix", str(prefix))
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "fixture-version")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=("a" * 40 + "\n").encode()))
    report = runner.installed_provenance(wheel, tmp_path / "source", "a" * 40, "a" * 40)
    assert report["wheel_sha256"] == runner.sha256_file(wheel)
    assert report["import_path"] == str(package / "__init__.py")
    assert report["installed_package_files"] == 1
    assert report["commit"] == "a" * 40
    assert report["checkout_sha_verified"] is True
    (package / "__init__.py").write_text("# changed after install")
    with pytest.raises(ValueError, match="differs"):
        runner.installed_provenance(wheel, tmp_path / "source", "a" * 40, "a" * 40)


def test_workflow_is_dispatch_only_gated_windows_and_narrow_upload():
    workflow = (RUNNER_PATH.parents[1] / "workflows/budget-diagnostic.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "  push:" not in workflow and "  pull_request:" not in workflow
    assert "runs-on: windows-latest" in workflow
    assert "timeout-minutes: 25" in workflow
    assert "startsWith(github.ref, 'refs/heads/diagnostic/')" in workflow
    assert "github.sha == inputs.expected_sha" in workflow
    assert workflow.index("Confirm checkout identity") < workflow.index("Build one fresh wheel") < workflow.index("Fetch and hash")
    assert "persist-credentials: false" in workflow
    assert "--retry" not in workflow and "actions/cache" not in workflow
    assert "path: ${{ runner.temp }}/budget-diagnostic-run/output/" in workflow
    assert "DEV" in workflow and "main as a workaround" in workflow


def test_launch_errors_are_not_retried_or_mistaken_for_reproduction(tmp_path, tiny_model, scope_factory):
    calls = []
    def fail_launch(scope, row, environment):
        calls.append(row)
        raise OSError("private parent path must not persist")
    result = runner.run_bundle(tmp_path / "bundle", tiny_model, {}, create_scope=scope_factory,
                               replay_snapshot=exact_replay, executor=fail_launch)
    assert calls == list(runner.ROWS)
    assert result["status"] == "unreplayable"
    assert result["stopped_early"] is False
    assert "private parent path" not in json.dumps(result)


def test_nonzero_exit_without_fatal_capture_is_unreplayable(tmp_path, tiny_model, scope_factory):
    def execute(scope, row, environment):
        directory, _, _ = snapshot_pair(scope, row)
        manifest = runner.read_json(directory / "manifest.json")
        manifest["failure"] = None
        runner.write_json(directory / "manifest.json", manifest)
        return {"row": row, "exit_code": 1, "timed_out": False, "output_capture_complete": True}
    result = runner.run_bundle(tmp_path / "bundle", tiny_model, {}, create_scope=scope_factory,
                               replay_snapshot=exact_replay, executor=execute)
    assert len(result["rows"]) == 7
    assert result["status"] == "unreplayable"
    assert all(item["observation"]["status"] == "unreplayable" for item in result["rows"])


@pytest.mark.parametrize("failure", ["mismatch", "timeout", "nonzero", "missing", "encoding"])
def test_provenance_checks_actual_checkout_with_bounded_local_subprocess(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setenv("GIT_DIR", "/unrelated/checkout")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("HF_TOKEN", "private-token")
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: pytest.fail("checkout gate must precede import"))
    def git(argv, **options):
        calls.append((argv, options))
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 10, output=b"private-error")
        if failure == "nonzero":
            raise subprocess.CalledProcessError(1, argv, output=b"private-error")
        if failure == "missing":
            raise FileNotFoundError("private-error")
        if failure == "encoding":
            return SimpleNamespace(stdout=b"\xff")
        return SimpleNamespace(stdout=("b" * 40 + "\n").encode())
    monkeypatch.setattr(runner.subprocess, "run", git)
    with pytest.raises(ValueError, match="Checkout commit") as error:
        runner.installed_provenance(tmp_path / "fresh.whl", tmp_path / "checkout", "a" * 40, "a" * 40)
    assert "private-error" not in str(error.value)
    assert len(calls) == 1
    argv, options = calls[0]
    assert argv == ["git", "-C", str((tmp_path / "checkout").resolve()), "rev-parse", "HEAD"]
    assert options["timeout"] == 10
    assert options["check"] is True
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"] == subprocess.PIPE
    assert options["stderr"] == subprocess.DEVNULL
    assert "GIT_DIR" not in options["env"]
    assert "GIT_CONFIG_COUNT" not in options["env"]
    assert "HF_TOKEN" not in options["env"]
    assert options["env"]["GIT_CONFIG_NOSYSTEM"] == "1"


def test_windows_child_path_retains_only_interpreter_system_and_git_directories(tmp_path, monkeypatch):
    root = tmp_path / "Windows"
    git = tmp_path / "Git/cmd/git.exe"
    git.parent.mkdir(parents=True)
    git.write_bytes(b"synthetic Git, never executed")
    monkeypatch.setattr(runner, "os", SimpleNamespace(
        name="nt", environ={"SYSTEMROOT": str(root), "PATH": "/untrusted/ambient/bin"},
        pathsep=os.pathsep, defpath="relative-current-directory",
    ))
    monkeypatch.setattr(runner.shutil, "which", lambda executable: str(git))
    path, metadata = runner.trusted_child_path()
    assert path.split(os.pathsep) == [
        str(Path(sys.executable).parent), str(root / "System32"), str(root), str(git.parent.resolve())]
    assert "/untrusted/ambient/bin" not in path
    assert "relative-current-directory" not in path
    assert metadata["git_executable_discovered"] is True
    assert metadata["limitations"] == []
    assert str(git.parent) not in json.dumps(metadata)


def test_missing_git_is_an_explicit_provenance_limitation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda executable: None)
    path, metadata = runner.trusted_child_path()
    assert metadata["git_executable_discovered"] is False
    assert metadata["limitations"] == [
        "Git executable was not discovered; git_status/git_diff may be unavailable"]
    assert path.split(os.pathsep)[0] == str(Path(sys.executable).parent)
