"""Seven frozen synthetic commands; never a readiness or full-E2E driver.

Only output/ is publishable. State, HOME, workspace, and the verified model stay
outside it. This script intentionally has no download, retry, or prompt override.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile

MODEL_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
MODEL_FILENAME = "LFM2.5-1.2B-Instruct-Q4_0.gguf"
MODEL_SHA256 = "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
SESSION_ID = "r11-budget-synthetic"
ROWS = ("seed", *(f"filler-{i}" for i in range(1, 7)))
COMMAND_TIMEOUT = 420
STREAM_CAP = 64 * 1024
JSON_CAP = 128 * 1024
CAPTURE_CAP = 512 * 1024
OUTPUT_MARKER = b"\n[diagnostic output truncated: byte cap exceeded]\n"
OWNER_YAML = "owner: zephyr-labs\nrevision: 9\nmode: quarantine\n"
FILLER_TEXT = ("Filler record alpha beta gamma delta epsilon zeta. " * 120 + "\n") * 6


def frozen_argv(row: str) -> list[str]:
    if row not in ROWS:
        raise ValueError("Unknown synthetic row")
    prompt = (
        "My name is Zephyr-QX7. Please read the file nexus_cfg_v9.yaml with "
        "your read_file tool and tell me its owner."
        if row == "seed" else
        "Read evict_filler.txt with your read_file tool and count how "
        f"many times the word alpha appears in section {row.split('-')[1]}."
    )
    return ["chat", "-p", "local", "--no-stream", "--persist", "--session", SESSION_ID, prompt]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict, cap: int = JSON_CAP) -> bool:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8")
    qualified = len(data) <= cap
    if not qualified:
        data = json.dumps({"replay_qualified": False, "truncated": True,
                           "reason": "runner JSON byte cap exceeded"}).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return qualified


def read_json(path: Path, cap: int = CAPTURE_CAP) -> dict | None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        return None
    try:
        with path.open("rb") as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def stage_model(source: Path, home: Path) -> dict:
    """Copy precisely one verified GGUF, never a cache/home directory or credentials."""
    source = source.resolve(strict=True)
    if not source.is_file() or sha256_file(source) != MODEL_SHA256:
        raise ValueError("Explicit model source does not match the required SHA256")
    destination = home / ".nexus-agent" / "models" / MODEL_REPO.replace("/", "--") / MODEL_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=False)
    # A private copy avoids subsequent model-source mutations changing the staged file.
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    if sha256_file(destination) != MODEL_SHA256:
        destination.unlink()
        raise ValueError("Staged model SHA256 mismatch")
    return {"repo": MODEL_REPO, "filename": MODEL_FILENAME,
            "sha256": MODEL_SHA256, "bytes": destination.stat().st_size,
            "staging": "verified model-only copy"}


def trusted_child_path() -> tuple[str, dict]:
    """Preserve tool discovery without forwarding the ambient executable search path."""
    directories = [str(Path(sys.executable).parent)]
    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
        if system_root:
            root = Path(system_root)
            if root.is_absolute():
                directories.extend((str(root / "System32"), str(root)))
    else:
        directories.extend(part for part in os.defpath.split(os.pathsep)
                           if part and Path(part).is_absolute())
    git = shutil.which("git")
    discovered = bool(git and Path(git).is_absolute() and Path(git).is_file())
    if discovered:
        directories.append(str(Path(git).resolve().parent))
    metadata = {
        "git_executable_discovered": discovered,
        "limitations": ([] if discovered else
                        ["Git executable was not discovered; git_status/git_diff may be unavailable"]),
        "path_policy": "interpreter directory, OS system directories, discovered Git directory only",
    }
    return os.pathsep.join(dict.fromkeys(directories)), metadata


def child_environment(scope: dict) -> dict[str, str]:
    home = Path(scope["home"])
    temporary = Path(scope["state"]) / "tmp"
    temporary.mkdir(exist_ok=False)
    environment = {
        "HOME": str(home), "USERPROFILE": str(home),
        "NEXUS_AGENT_HOME": str(home / ".nexus-agent"),
        "NEXUS_AGENT_MODEL_REPO": MODEL_REPO,
        "NEXUS_AGENT_MODEL_FILENAME": MODEL_FILENAME,
        "PATH": trusted_child_path()[0],
        "TMPDIR": str(temporary), "TMP": str(temporary), "TEMP": str(temporary),
        "LANG": "C.UTF-8", "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1",
    }
    # Python/native process launch needs these OS locations on Windows; nothing
    # resembling credentials, user configuration, proxies or model overrides survives.
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                environment[key] = os.environ[key]
    return environment


def _drain(stream, path: Path, result: dict) -> None:
    """Drain pipes continuously without ever buffering or persisting unbounded output."""
    kept = 0
    total = 0
    truncated = False
    try:
        with path.open("xb") as target:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                available = STREAM_CAP - len(OUTPUT_MARKER) - kept
                if available > 0:
                    piece = chunk[:available]
                    target.write(piece)
                    kept += len(piece)
                if total > STREAM_CAP - len(OUTPUT_MARKER):
                    truncated = True
            if truncated:
                target.write(OUTPUT_MARKER)
        result.update(bytes_seen=total, truncated=truncated, file=path.name)
    except Exception:
        result.update(truncated=True, file=path.name, error="output capture failed")
    finally:
        stream.close()


def _windows_job_factory():
    """Load only the reviewed sibling, including when this script runs under -I."""
    cached = getattr(_windows_job_factory, "factory", None)
    if cached is None:
        path = Path(__file__).resolve().with_name("windows_job.py")
        spec = importlib.util.spec_from_file_location("_budget_diagnostic_windows_job", path)
        if spec is None or spec.loader is None:
            raise ImportError("Owned Windows lifecycle module is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cached = module.WindowsJobProcess
        _windows_job_factory.factory = cached
    return cached


def _cleanup_posix_process(process, timeout=10):
    errors = []
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as error:
        errors.append("process_group_" + type(error).__name__)
    try:
        if process.poll() is None:
            process.wait(timeout=timeout)
    except Exception as error:
        errors.append("owned_process_wait_" + type(error).__name__)
    return {"ownership": "owned_posix_process_group", "confirmed": not errors, "errors": errors}


def execute_row(scope: dict, row: str, environment: dict, *, popen=None, windows_factory=None) -> dict:
    row_output = Path(scope["output"]) / row
    row_output.mkdir(exist_ok=True)
    argv = [sys.executable, "-I", "-m", "nexus_agent_ai.diagnostic_cli",
            "--scope", scope["scope_path"], "--row", row, "--", *frozen_argv(row)]
    options = {"cwd": scope["workspace"], "env": environment,
               "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
               "stderr": subprocess.PIPE}
    if os.name == "nt":
        factory = windows_factory or _windows_job_factory()
    else:
        options["start_new_session"] = True
        factory = popen or subprocess.Popen
    started = time.monotonic()
    process = None
    stdout, stderr = {}, {}
    threads = []
    started_streams = []
    timed_out = False
    failure = {}
    cleanup = {"ownership": "not_started", "confirmed": False, "errors": ["launch_failed"]}

    def capture_stream(stream, path, result):
        try:
            _drain(stream, path, result)
        except Exception:
            # Closing a reader can fail after _drain recorded its payload.
            result.update(error="output capture failed")

    try:
        process = factory(argv, **options)
        for stream, name, result in ((process.stdout, "stdout.txt", stdout),
                                     (process.stderr, "stderr.txt", stderr)):
            thread = threading.Thread(target=capture_stream, args=(stream, row_output / name, result), daemon=True)
            thread.start()
            threads.append(thread)
            started_streams.append(stream)
        try:
            process.wait(timeout=COMMAND_TIMEOUT)
        except subprocess.TimeoutExpired:
            timed_out = True
    except Exception as error:
        failure["launch_error" if process is None else "execution_error"] = type(error).__name__
        if process is None:
            cleanup = getattr(error, "cleanup", cleanup)
    finally:
        teardown_started = time.monotonic()
        deadline = teardown_started + 10
        if process is not None:
            try:
                remaining = max(0, deadline - time.monotonic())
                cleanup = (process.cleanup(timeout=remaining) if os.name == "nt"
                           else _cleanup_posix_process(process, timeout=remaining))
            except Exception as error:
                cleanup = {"ownership": "owned_process", "confirmed": False,
                           "errors": ["cleanup_" + type(error).__name__]}
        # Descendants must release inherited write ends before readers are joined.
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and stream not in started_streams:
                    try:
                        stream.close()
                    except Exception as error:
                        failure["stream_close_error"] = type(error).__name__
        teardown_duration = round(time.monotonic() - teardown_started, 3)
    capture_complete = (len(threads) == 2 and not any(thread.is_alive() for thread in threads)
                        and "error" not in stdout and "error" not in stderr)
    return {"row": row, "argv": argv,
            "exit_code": process.returncode if process is not None else None,
            "timed_out": timed_out, "duration_seconds": round(time.monotonic() - started, 3),
            "teardown_duration_seconds": teardown_duration,
            "stdout": stdout, "stderr": stderr, "cleanup": cleanup, **failure,
            "lifecycle_safe": (cleanup.get("confirmed") is True and not cleanup.get("errors")
                               and capture_complete and not failure),
            "output_capture_complete": capture_complete}


def installed_provenance(wheel: Path, source_root: Path, commit: str,
                         expected_commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit) or commit != expected_commit:
        raise ValueError("Exact expected commit gate failed")
    git_environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                git_environment[key] = os.environ[key]
    try:
        checkout = subprocess.run(
            ["git", "-C", str(source_root.resolve()), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=10, check=True,
            env=git_environment,
        )
        checkout_commit = checkout.stdout.decode("ascii").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise ValueError("Checkout commit verification failed") from None
    if checkout_commit != expected_commit:
        raise ValueError("Checkout commit does not match expected SHA")
    package = importlib.import_module("nexus_agent_ai")
    origin = Path(package.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    if (sys.prefix == sys.base_prefix or not origin.is_relative_to(prefix)
            or "site-packages" not in origin.parts or origin.is_relative_to(source_root.resolve())):
        raise ValueError("Diagnostic must import a fresh wheel from isolated venv site-packages")
    installed_root = origin.parent
    package_hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith("nexus_agent_ai/"):
                continue
            relative = Path(entry.filename).relative_to("nexus_agent_ai")
            destination = (installed_root / relative).resolve()
            if not destination.is_relative_to(installed_root):
                raise ValueError("Unsafe wheel member")
            expected_hash = hashlib.sha256(archive.read(entry)).hexdigest()
            if not destination.is_file() or sha256_file(destination) != expected_hash:
                raise ValueError("Installed package differs from explicit wheel")
            package_hashes[entry.filename] = expected_hash
    if "nexus_agent_ai/__init__.py" not in package_hashes:
        raise ValueError("Explicit wheel does not contain the installed package")
    return {"commit": checkout_commit, "expected_commit": expected_commit,
            "checkout_sha_verified": True,
            "wheel": wheel.name, "wheel_sha256": sha256_file(wheel),
            "version": importlib.metadata.version("nexus-agent-ai"),
            "interpreter": sys.executable, "python_version": sys.version,
            "prefix": str(prefix), "import_path": str(origin),
            "installed_package_files": len(package_hashes),
            "installed_package_hash": hashlib.sha256(json.dumps(package_hashes, sort_keys=True).encode()).hexdigest(),
            "child_executables": trusted_child_path()[1],
            "source_tree_import_rejected": True}


def _qualified(snapshot: dict | None, scope: dict, row: str) -> bool:
    return bool(snapshot and snapshot.get("schema_version") == 1
                and snapshot.get("replay_qualified") is True
                and snapshot.get("truncated") is False
                and snapshot.get("row") == row
                and snapshot.get("session_id") == SESSION_ID
                and snapshot.get("run_id") == scope["run_id"])


def inspect_row(scope: dict, row: str, replay_snapshot) -> dict:
    directory = Path(scope["output"]) / row
    manifest = read_json(directory / "manifest.json")
    result = {"status": "not_reproduced", "stop": False,
              "preceding_success": "not_applicable"}
    if not manifest:
        return {**result, "status": "unreplayable", "reason": "missing or oversized capture manifest"}
    if (manifest.get("row") != row or manifest.get("run_id") != scope["run_id"]
            or manifest.get("session_id") != SESSION_ID):
        return {**result, "status": "unreplayable", "reason": "capture manifest identity mismatch"}
    failure_ref = manifest.get("failure")
    if not failure_ref:
        if manifest.get("truncated"):
            return {**result, "status": "unreplayable", "reason": "capture truncated"}
        return result
    failure = read_json(directory / "failure.json")
    if (not _qualified(failure, scope, row) or not isinstance(failure_ref, dict)
            or failure_ref.get("file") != "failure.json"
            or failure_ref.get("id") != failure.get("id")
            or failure_ref.get("input_hash") != failure.get("input_hash")):
        return {**result, "status": "unreplayable", "reason": "fatal capture is not replay qualified"}
    if failure.get("result", {}).get("success") is not False:
        return {**result, "status": "unreplayable", "reason": "capture is not a real fatal result"}
    replay = replay_snapshot(failure)
    report_saved = write_json(directory / "failure-replay.json", replay)
    reproduced = report_saved and replay.get("status") == "reproduced" and replay.get("error_match") is True
    error = failure.get("result", {}).get("error", {})
    coverage = (failure.get("kind") == "planner" and error.get("class") == "ContextBudgetError"
                and str(error.get("text", "")).startswith("Historical omission coverage gap:"))
    result.update(status="reproduced" if reproduced else "unreplayable",
                  failure_kind=failure.get("kind"), coverage_failure=coverage,
                  classification=replay.get("classification", "unresolved"),
                  preceding_success="missing_same_invocation")
    reference = failure.get("previous_success")
    success = read_json(directory / "latest_success.json")
    if (isinstance(reference, dict) and reference.get("file") == "latest_success.json"
            and _qualified(success, scope, row) and success.get("kind") == "planner"
            and success.get("stage") in ("outbound_initial_fit", "outbound_post_result_fit")
            and success.get("actual_fit_completed") is True
            and success.get("result", {}).get("success") is True
            and reference.get("id") == success.get("id")
            and reference.get("input_hash") == success.get("input_hash")
            and isinstance(success.get("id"), int) and isinstance(failure.get("id"), int)
            and success["id"] < failure["id"]):
        success_report = replay_snapshot(success)
        saved = write_json(directory / "success-replay.json", success_report)
        sufficient = saved and success_report.get("status") == "reproduced" and success_report.get("token_identical") is True
        result["preceding_success"] = "token_identical" if sufficient else "replay_mismatch_or_unreplayable"
        result["stop"] = bool(reproduced and coverage and sufficient)
    return result


def run_bundle(root: Path, model_source: Path, provenance: dict, *,
               create_scope=None, replay_snapshot=None, executor=None) -> dict:
    if create_scope is None:
        create_scope = importlib.import_module("nexus_agent_ai.agent.budget_diagnostics").create_scope
    if replay_snapshot is None:
        replay_snapshot = importlib.import_module("nexus_agent_ai.budget_replay").replay_snapshot
    scope = create_scope(root)
    output = Path(scope["output"])
    summary = {"schema_version": 1, "run_id": scope["run_id"], "session_id": SESSION_ID,
               "status": "not_reproduced", "rows": [], "stopped_early": False,
               "model_readiness_claim": False,
               "limits": {"per_command_seconds": COMMAND_TIMEOUT, "stream_bytes": STREAM_CAP,
                          "capture_bytes_per_invocation": CAPTURE_CAP, "json_bytes": JSON_CAP}}
    try:
        provenance = {**provenance, "model": stage_model(model_source, Path(scope["home"]))}
        if not write_json(output / "provenance.json", provenance):
            raise ValueError("Provenance exceeds byte cap")
        workspace = Path(scope["workspace"])
        (workspace / "nexus_cfg_v9.yaml").write_text(OWNER_YAML, encoding="utf-8")
        (workspace / "evict_filler.txt").write_text(FILLER_TEXT, encoding="utf-8")
        environment = child_environment(scope)
        for row in ROWS:
            try:
                command = (executor or execute_row)(scope, row, environment)
            except Exception as error:
                command = {"row": row, "exit_code": None, "timed_out": False,
                           "output_capture_complete": False, "lifecycle_safe": False,
                           "launch_error": type(error).__name__}
            (output / row).mkdir(exist_ok=True)
            cleanup = command.get("cleanup", {})
            lifecycle_failed = (command.get("lifecycle_safe") is False
                                or not command.get("output_capture_complete", True)
                                or (cleanup and (cleanup.get("confirmed") is not True
                                                 or cleanup.get("errors"))))
            if lifecycle_failed:
                observation = {"status": "unreplayable", "stop": False,
                               "reason": "owned lifecycle or output capture could not be confirmed"}
            else:
                try:
                    observation = inspect_row(scope, row, replay_snapshot)
                except Exception as error:
                    observation = {"status": "unreplayable", "stop": False,
                                   "reason": "offline replay raised " + type(error).__name__}
            if not lifecycle_failed and command.get("timed_out"):
                observation.update(status="unreplayable", stop=False, reason="command or output capture did not complete")
            elif command.get("exit_code") != 0 and observation["status"] == "not_reproduced":
                observation.update(status="unreplayable", stop=False, reason="nonzero command without a replay-qualified fatal")
            command["observation"] = observation
            summary["rows"].append(command)
            write_json(output / row / "command.json", command)
            if lifecycle_failed:
                summary.update(status="unreplayable", stopped_early=True,
                               reason="fail-closed lifecycle stop; no subsequent command launched")
                break
            if observation["stop"]:
                summary.update(status="reproduced", stopped_early=True,
                               reason="coverage failure reproduced with token-identical same-invocation preceding success")
                break
        else:
            observations = [row["observation"] for row in summary["rows"]]
            if any(item["status"] == "reproduced" for item in observations):
                summary.update(status="reproduced", reason="fatal replay reproduced; no sufficient coverage transition stop pair")
            elif any(item["status"] == "unreplayable" for item in observations):
                summary.update(status="unreplayable", reason="capture or replay incomplete; no qualified coverage transition")
            else:
                summary["reason"] = "seven frozen commands completed without a captured real fatal"
    except Exception as error:
        # Never persist exception text that could contain inherited host details.
        summary.update(status="unreplayable", reason="runner setup/execution failed: " + type(error).__name__)
    finally:
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "model-source", "wheel", "source-root", "commit", "expected-commit"):
        parser.add_argument("--" + option, required=True)
    arguments = parser.parse_args(argv)
    provenance = installed_provenance(Path(arguments.wheel), Path(arguments.source_root),
                                      arguments.commit, arguments.expected_commit)
    summary = run_bundle(Path(arguments.root), Path(arguments.model_source), provenance)
    print(json.dumps({"status": summary["status"], "rows": len(summary["rows"]),
                      "stopped_early": summary["stopped_early"], "model_readiness_claim": False}))
    return 2 if summary["status"] == "unreplayable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
