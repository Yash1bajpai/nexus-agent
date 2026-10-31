"""Run the macOS qualification log-capture step itself, without a YAML dependency."""

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/macos.yml"
CAPTURE_STEP = "Capture first-run engine server logs"


def _capture_script():
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"      - name: {CAPTURE_STEP}") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("      - ")),
        len(lines),
    )
    run = lines.index("        run: |", start, end)
    # Extract the actual literal run block, not a Python reimplementation of capture.
    return textwrap.dedent("\n".join(lines[run + 1 : end])) + "\n"


@pytest.fixture
def run_capture(tmp_path):
    pwsh = os.environ.get("NEXUS_TEST_PWSH") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell is required; set NEXUS_TEST_PWSH or put pwsh on PATH")
    script = tmp_path / "capture.ps1"
    script.write_text("$ErrorActionPreference = 'Stop'\n" + _capture_script(), encoding="utf-8")

    def run(home, runner_temp):
        environment = os.environ.copy()
        environment.update(HOME=str(home), USERPROFILE=str(home), RUNNER_TEMP=str(runner_temp))
        result = subprocess.run(
            [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


def test_capture_copies_only_server_logs_without_changing_originals(tmp_path, run_capture):
    home = tmp_path / "isolated home"
    server = home / ".nexus-agent" / "llama-server"
    server.mkdir(parents=True)
    logs = {
        "server-nexus-100.log": b"build: actual-engine-build\nggml_metal: selected Metal backend\n",
        "server-nexus-200.log": b"server startup failed\n\x00unchanged bytes\r\n",
    }
    for name, content in logs.items():
        (server / name).write_bytes(content)
    (server / "credentials.json").write_text("private engine credentials", encoding="utf-8")
    (server / "server-other.log").write_text("unrelated server output", encoding="utf-8")
    (server / "server-nexus-300.log.bak").write_text("not a log", encoding="utf-8")
    nested = server / "server-nexus-directory.log"
    nested.mkdir()
    (nested / "server-nexus-private.log").write_text("not a top-level log", encoding="utf-8")
    (home / ".nexus-agent" / "config.json").write_text("private provider key", encoding="utf-8")
    originals = {path.relative_to(home): path.read_bytes() for path in home.rglob("*") if path.is_file()}

    runner_temp = tmp_path / "runner temp"
    artifact = runner_temp / "nexus-first-run"
    partial_state = {
        "install.log": b"installation completed\n",
        "s2/chat.log": b"[ERROR] requirement incomplete\nTokens: 42 Time: 1s\n",
        "s2/s2target.py": b"x = 1\n",
        "s2/s2target.py.bak": b"x = 1\n",
    }
    for name, content in partial_state.items():
        target = artifact / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    run_capture(home, runner_temp)

    captured = {path.relative_to(artifact).as_posix(): path.read_bytes()
                for path in artifact.rglob("*") if path.is_file()}
    assert captured == {**partial_state, **{f"llama-server/{name}": content for name, content in logs.items()}}
    for relative, content in originals.items():
        assert (home / relative).read_bytes() == content
    assert {path.name for path in runner_temp.iterdir()} == {"nexus-first-run"}


@pytest.mark.parametrize("engine_directory_exists", [False, True])
@pytest.mark.parametrize("artifact_exists", [False, True])
def test_capture_succeeds_without_server_logs(
    tmp_path, run_capture, engine_directory_exists, artifact_exists
):
    home = tmp_path / "home"
    home.mkdir()
    if engine_directory_exists:
        server = home / ".nexus-agent" / "llama-server"
        server.mkdir(parents=True)
        (server / "credentials.json").write_text("do not archive", encoding="utf-8")
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    artifact = runner_temp / "nexus-first-run"
    if artifact_exists:
        artifact.mkdir()
        (artifact / "install.log").write_bytes(b"installation failed before engine startup\n")

    run_capture(home, runner_temp)

    assert artifact.is_dir()
    captured = {path.relative_to(artifact).as_posix(): path.read_bytes()
                for path in artifact.rglob("*") if path.is_file()}
    assert captured == ({"install.log": b"installation failed before engine startup\n"}
                        if artifact_exists else {})
    assert {path.name for path in runner_temp.iterdir()} == {"nexus-first-run"}
    if engine_directory_exists:
        assert (server / "credentials.json").read_text(encoding="utf-8") == "do not archive"
