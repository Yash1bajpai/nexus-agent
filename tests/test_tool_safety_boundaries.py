"""Real filesystem and Git regressions for tool confinement and atomic edits."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nexus_agent_ai.agent import consent, tools as T


def _edit(operation, path, content="new\n"):
    if operation == "write":
        return T.execute_write_file(str(path), content)
    return T.execute_patch_file(str(path), "old", content.rstrip("\n"))


@pytest.mark.parametrize("operation", ["write", "patch"])
@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_backup_links_cannot_modify_an_outside_file(tmp_path, operation, link_kind):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside must stay intact\n")
    target = work / "notes.txt"
    target.write_bytes(b"old\n")
    backup = target.with_name("notes.txt.bak")
    try:
        if link_kind == "symlink":
            backup.symlink_to(outside)
        else:
            os.link(outside, backup)
    except OSError as exc:
        pytest.skip(f"Filesystem cannot create {link_kind}: {exc}")
    previous = Path.cwd()
    try:
        os.chdir(work)
        T.execute_read_file(str(target))
        result = _edit(operation, target)
    finally:
        os.chdir(previous)
    assert result.startswith("ERROR:")
    assert target.read_bytes() == b"old\n"
    assert outside.read_bytes() == b"outside must stay intact\n"
    assert backup.read_bytes() == outside.read_bytes()
    assert not list(work.glob(".*.tmp"))


@pytest.mark.parametrize("operation", ["write", "patch"])
def test_atomic_original_replacement_does_not_modify_a_hardlinked_outside_file(
    tmp_path, monkeypatch, operation
):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"old\n")
    work = tmp_path / "work"
    work.mkdir()
    target = work / "notes.txt"
    try:
        os.link(outside, target)
    except OSError as exc:
        pytest.skip(f"Filesystem cannot create hardlinks: {exc}")
    monkeypatch.chdir(work)
    T.execute_read_file(str(target))
    assert _edit(operation, target).startswith("Successfully")
    assert target.read_bytes() == b"new\n"
    assert outside.read_bytes() == b"old\n"


def test_invalid_unicode_new_file_does_not_create_parent_or_temporary_files(tmp_path):
    target = tmp_path / "new-directory" / "notes.txt"
    assert T.execute_write_file(str(target), "bad\ud800").startswith("ERROR:")
    assert not target.parent.exists()


@pytest.mark.parametrize("operation", ["write", "patch"])
def test_surrogate_is_rejected_before_original_or_backup_changes(tmp_path, operation):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"old\r\n")
    backup = target.with_name("notes.txt.bak")
    backup.write_bytes(b"previous backup")
    T.execute_read_file(str(target))
    result = _edit(operation, target, "bad\ud800\n")
    assert result.startswith("ERROR:")
    assert target.read_bytes() == b"old\r\n"
    assert backup.read_bytes() == b"previous backup"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("operation", ["write", "patch"])
@pytest.mark.parametrize("failed_destination", ["original", "backup"])
def test_replace_failure_leaves_original_and_cleans_temporary_files(
    tmp_path, monkeypatch, operation, failed_destination
):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"old\r\n")
    backup = target.with_name("notes.txt.bak")
    backup.write_bytes(b"previous backup")
    T.execute_read_file(str(target))
    real_replace = os.replace

    def failing_replace(source, destination):
        blocked = target if failed_destination == "original" else backup
        if Path(destination) == blocked:
            raise OSError("injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(T.os, "replace", failing_replace)
    result = _edit(operation, target)
    assert result.startswith("ERROR:")
    assert target.read_bytes() == b"old\r\n"
    if failed_destination == "backup":
        assert backup.read_bytes() == b"previous backup"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("operation", ["write", "patch"])
def test_edit_preserves_crlf_and_permissions(tmp_path, operation):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"old\r\nsecond\r\n")
    target.chmod(0o640)
    mode = stat.S_IMODE(target.stat().st_mode)
    T.execute_read_file(str(target))
    result = _edit(operation, target, "new\nsecond\n" if operation == "write" else "new\n")
    assert result.startswith("Successfully")
    assert target.read_bytes() == b"new\r\nsecond\r\n"
    assert target.with_name("notes.txt.bak").read_bytes() == b"old\r\nsecond\r\n"
    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("operation", ["write", "patch"])
def test_no_op_is_not_reported_as_an_edit(tmp_path, operation):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"old\n")
    T.execute_read_file(str(target))
    before = target.stat()
    result = _edit(operation, target, "old\n")
    assert result.startswith("No changes:")
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert not target.with_name("notes.txt.bak").exists()


@pytest.mark.parametrize("state_kind", ["default", "environment"])
def test_overlapping_state_is_unreadable_uneditable_and_cannot_self_grant(
    tmp_path, monkeypatch, state_kind
):
    monkeypatch.delenv("NEXUS_ALLOW_PROJECT_EXECUTION", raising=False)
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    consent._session_allowed.clear()
    consent._session_denied.clear()
    if state_kind == "default":
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.delenv("NEXUS_AGENT_HOME", raising=False)
        state = tmp_path / ".nexus-agent"
    else:
        state = tmp_path / "nested" / "agent-state"
        monkeypatch.setenv("NEXUS_AGENT_HOME", str(state))
    state.mkdir(parents=True)
    target = state / "trusted_folders.json"
    target.write_text('{"folders": []}', encoding="utf-8")
    payload = json.dumps({"folders": [consent.folder_key()]})
    for result in (
        T.execute_read_file(str(target)),
        T.execute_write_file(str(target), payload),
        T.execute_patch_file(str(target), "[]", json.dumps([consent.folder_key()])),
        T.execute_list_directory(str(state)),
    ):
        assert result.startswith("ERROR:") and "state directories are protected" in result
    assert target.read_text(encoding="utf-8") == '{"folders": []}'
    assert state.name not in T.execute_list_directory()
    assert not consent.execution_allowed_without_asking()
    assert consent.execution_allowed()[0] is False


def test_unavailable_home_keeps_workspace_tools_and_explicit_state_protection(tmp_path, monkeypatch):
    state = tmp_path / "agent-state"
    state.mkdir()
    (state / "entry.json").write_text("private state", encoding="utf-8")
    (tmp_path / "normal.txt").write_text("workspace content", encoding="utf-8")
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(state))

    def missing_home(cls):
        raise RuntimeError("Home directory unavailable")

    monkeypatch.setattr(Path, "home", classmethod(missing_home))
    assert T.execute_read_file("normal.txt") == "workspace content"
    assert "state directories are protected" in T.execute_read_file(str(state / "entry.json"))
    assert T.execute_write_file(str(state / "new.json"), "blocked").startswith("ERROR:")
    assert not (state / "new.json").exists()


def test_saved_exact_folder_grant_still_works_without_a_terminal(monkeypatch, tmp_path):
    monkeypatch.delenv("NEXUS_ALLOW_PROJECT_EXECUTION", raising=False)
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    consent._session_allowed.clear()
    consent._session_denied.clear()
    consent.trust()
    assert consent.execution_allowed() == (True, "remembered for this folder")
    assert consent.offer_run_tests()
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    assert consent.execution_allowed()[0] is False
    assert not consent.offer_run_tests()


def _git(root, *arguments):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.run(
        ["git", "-C", str(root), *arguments], env=env, check=True,
        capture_output=True, text=True, encoding="utf-8",
    )


@pytest.fixture
def git_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Safety Test")
    _git(tmp_path, "config", "user.email", "safety@example.invalid")
    return tmp_path


def _commit_all(root):
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "test fixture")


def test_staged_secret_is_excluded_even_on_unborn_head(git_repo):
    (git_repo / ".env").write_text("FAKE_STAGED_VALUE_DO_NOT_DISCLOSE\n")
    (git_repo / "safe.txt").write_text("public change\n")
    _git(git_repo, "add", "-A")
    result = T.execute_git_diff()
    assert "public change" in result
    assert ".env" not in result and "FAKE_STAGED_VALUE" not in result
    status = T.execute_git_status()
    assert "safe.txt" in status and ".env" not in status


@pytest.mark.parametrize("old,new", [(".env", "safe.txt"), ("safe.txt", ".env")])
def test_rename_filters_both_sensitive_sides(git_repo, old, new):
    (git_repo / old).write_text("FAKE_RENAME_VALUE_DO_NOT_DISCLOSE\n")
    _commit_all(git_repo)
    _git(git_repo, "mv", old, new)
    result = T.execute_git_diff()
    assert "omitted" in result
    assert "FAKE_RENAME_VALUE" not in result and ".env" not in result
    assert ".env" not in T.execute_git_status()


@pytest.mark.parametrize("direction", ["inbound", "outbound"])
def test_git_rename_cannot_disclose_paths_outside_current_workspace(git_repo, monkeypatch, direction):
    work = git_repo / "workspace"
    work.mkdir()
    inside = work / "visible.txt"
    outside = git_repo / "outside-private.txt"
    source, destination = (outside, inside) if direction == "inbound" else (inside, outside)
    source.write_text("OUTSIDE_RENAME_CONTENT\n")
    _commit_all(git_repo)
    _git(git_repo, "mv", str(source), str(destination))
    monkeypatch.chdir(work)
    result = T.execute_git_diff()
    assert "omitted" in result
    assert "outside-private.txt" not in result and "OUTSIDE_RENAME_CONTENT" not in result
    assert "outside-private.txt" not in T.execute_git_status()


def test_git_diff_filters_other_project_paths(git_repo, monkeypatch):
    work = git_repo / "workspace"
    work.mkdir()
    (work / "inside.txt").write_text("inside original\n")
    (git_repo / "outside.txt").write_text("outside original\n")
    _commit_all(git_repo)
    (work / "inside.txt").write_text("inside changed\n")
    (git_repo / "outside.txt").write_text("OUTSIDE_DO_NOT_DISCLOSE\n")
    monkeypatch.chdir(work)
    result = T.execute_git_diff()
    assert "inside changed" in result
    assert "outside.txt" not in result and "OUTSIDE_DO_NOT_DISCLOSE" not in result


@pytest.mark.parametrize("helper", ["external_env", "external_config", "textconv", "fsmonitor", "clean", "process"])
def test_git_inspection_never_executes_repository_helpers(git_repo, monkeypatch, helper):
    tracked = git_repo / "tracked.txt"
    tracked.write_text("before\n")
    (git_repo / ".gitattributes").write_text("tracked.txt diff=canary filter=canary\n")
    _commit_all(git_repo)
    tracked.write_text("after\n")
    canary = git_repo / "canary-executed"
    script = git_repo / "helper.py"
    script.write_text(f"from pathlib import Path\nPath({str(canary)!r}).write_text('executed')\n")
    command = f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"'
    if helper == "external_env":
        monkeypatch.setenv("GIT_EXTERNAL_DIFF", command)
    elif helper == "external_config":
        _git(git_repo, "config", "diff.external", command)
    elif helper == "textconv":
        _git(git_repo, "config", "diff.canary.textconv", command)
    elif helper == "fsmonitor":
        _git(git_repo, "config", "core.fsmonitor", command)
    else:
        _git(git_repo, "config", f"filter.canary.{helper}", command)
        _git(git_repo, "config", "filter.canary.required", "true")
    assert "after" in T.execute_git_diff()
    assert not T.execute_git_status().startswith("ERROR:")
    assert not canary.exists()


def test_git_diff_non_repository_is_an_error():
    result = T.execute_git_diff()
    assert result.startswith("ERROR:") and "clean" not in result.lower()


@pytest.mark.parametrize("with_head", [False, True])
def test_untracked_only_is_not_called_clean(git_repo, with_head):
    if with_head:
        (git_repo / "tracked.txt").write_text("initial\n")
        _commit_all(git_repo)
    (git_repo / "new.txt").write_text("untracked\n")
    result = T.execute_git_diff()
    assert "Untracked files exist" in result and "new.txt" in result
    assert "clean" not in result.lower()


def test_diff_command_errors_are_not_silently_reported_clean(git_repo, monkeypatch):
    original = T._safe_git_run

    def fail_diff(args, **kwargs):
        if "diff" in args:
            return subprocess.CompletedProcess(args, 128, "", "injected failure")
        return original(args, **kwargs)

    monkeypatch.setattr(T, "_safe_git_run", fail_diff)
    result = T.execute_git_diff()
    assert result.startswith("ERROR:") and "clean" not in result.lower()


def test_git_inspection_strips_secrets_and_inherited_git_overrides(git_repo, monkeypatch):
    monkeypatch.setenv("NEXUS_FAKE_SECRET", "must-not-enter-git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "must-not-execute")
    real_run = T.subprocess.run
    environments = []

    def capture_run(*args, **kwargs):
        environments.append(kwargs.get("env"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(T.subprocess, "run", capture_run)
    assert not T.execute_git_status().startswith("ERROR:")
    assert environments
    for environment in environments:
        assert environment is not None
        assert "NEXUS_FAKE_SECRET" not in environment
        assert "GIT_CONFIG_COUNT" not in environment
        assert "GIT_CONFIG_KEY_0" not in environment
        assert "GIT_CONFIG_VALUE_0" not in environment
