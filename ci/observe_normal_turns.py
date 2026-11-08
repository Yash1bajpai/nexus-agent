"""Minimal, bounded observation runner for two normal CLI chat turns.

Executes two unmodified user prompts in a persistent session on the baseline model:
  Turn 1: "Fix multiply in calc.py so it multiplies."
  Turn 2: "Now make it return 0 if either input is None."

Observations captured (Console and Job Summary only; NO artifact uploads):
  - Raw CLI exit codes, execution times, and direct-to-file stdout/stderr.
  - Before and after bytes for calc.py; unified text diffs (NEVER executed).
  - Parent process lifetime only; descendant cleanup remains UNKNOWN.
  - Read-only SQLite inspection verifying exact user prompt strings in sequence.
  - Model status remains strictly UNVERIFIED for manual reviewer decision.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import stat


INITIAL_CALC_PY = b"def multiply(a,b):\n    return a+b\n"

PROMPT_1 = "Fix multiply in calc.py so it multiplies."
PROMPT_2 = "Now make it return 0 if either input is None."

MAX_STREAM_BYTES = 1024 * 1024  # 1 MiB stream file quota
TURN_TIMEOUT_SECONDS = 480.0    # 8 minutes per turn


def write_file_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read_regular_bytes(path: Path, limit: int) -> bytes:
    """Fail closed on links/reparse points, nonregular files, races or oversize."""
    def identity(st):
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def cross_identity(st):
        if os.name != "nt":
            return identity(st)
        # CPython 3.12 Windows pathname ctime is birthtime while descriptor
        # ctime is ChangeTime. Never equate these cross-API meanings.
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                getattr(st, "st_birthtime_ns", None))

    before = path.lstat()
    for parent in path.absolute().parents:
        pst = parent.lstat()
        if stat.S_ISLNK(pst.st_mode) or getattr(pst, "st_file_attributes", 0) & 0x400:
            raise ValueError("Reparse/symlink ancestor")
    if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & 0x400):
        raise ValueError("Not a regular non-reparse file: " + str(path))
    if before.st_size > limit:
        raise ValueError("File exceeds read limit: " + str(path))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if cross_identity(before) != cross_identity(opened):
            raise ValueError("File identity changed before read")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(limit + 1)
        after = os.fstat(fd)
        current = path.lstat()
        if (len(data) > limit or identity(opened) != identity(after)
                or identity(before) != identity(current)
                or cross_identity(after) != cross_identity(current)
                or getattr(current, "st_file_attributes", 0) & 0x400):
            raise ValueError("File changed during read or exceeded limit")
        return data
    finally:
        os.close(fd)


def snapshot_calc(path: Path) -> Tuple[bytes, Optional[str]]:
    try:
        return read_regular_bytes(path, 64 * 1024), None
    except Exception as e:
        return b"", str(e)


def stream_excerpt(path: Path, limit: int = 64 * 1024) -> Dict[str, Any]:
    """Strictly bound reads/print; disk capture quota is best-effort, not hard."""
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode)
                or getattr(before, "st_file_attributes", 0) & 0x400):
            raise ValueError("Stream file is not regular")
        total = before.st_size
        if total <= limit:
            data = read_regular_bytes(path, limit)
        else:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                opened = os.fstat(fd)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError("Stream identity changed")
                with os.fdopen(fd, "rb", closefd=False) as f:
                    head = f.read(limit // 2)
                    f.seek(max(0, total - limit // 2))
                    tail = f.read(limit // 2)
                after = os.fstat(fd)
                current = path.lstat()
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino):
                    raise ValueError("Stream changed during excerpt read")
                data = head + b"\n[TRUNCATED: middle omitted]\n" + tail
            finally:
                os.close(fd)
        return {"total_bytes": total, "truncated": total > limit,
                "text": data.decode("utf-8", errors="replace")}
    except Exception as e:
        return {"total_bytes": None, "truncated": False, "text": "[unavailable: " + str(e) + "]", "error": str(e)}


def read_file_tail(path: Path, max_bytes: int = 64 * 1024) -> str:
    ev = stream_excerpt(path, max_bytes)
    return f"total_bytes={ev['total_bytes']} TRUNCATED={ev['truncated']}\n" + ev["text"]


def compute_unified_diff(original: bytes, modified: bytes, filename: str) -> str:
    orig_lines = original.decode("utf-8", errors="replace").splitlines(keepends=True)
    mod_lines = modified.decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


def run_observed_turn(cmd: List[str], cwd: Path, stdout_file: Path,
                      stderr_file: Path, timeout: float = TURN_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Best-effort monitored disk capture. Descendant cleanup stays UNKNOWN."""
    info: Dict[str, Any] = {"command": cmd, "exit_code": None, "duration_seconds": 0.0,
        "timed_out": False, "cleanup_status": "DESCENDANTS_UNKNOWN",
        "quota_exceeded": False, "parent_exited": False}
    start = time.monotonic()
    proc = None
    try:
        with open(stdout_file, "wb") as f_out, open(stderr_file, "wb") as f_err:
            try:
                proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=f_out, stderr=f_err)
                deadline = start + timeout
                while True:
                    rc = proc.poll()
                    if (stdout_file.stat().st_size > MAX_STREAM_BYTES
                            or stderr_file.stat().st_size > MAX_STREAM_BYTES):
                        info["quota_exceeded"] = True
                    if rc is not None:
                        break
                    if time.monotonic() >= deadline:
                        info["timed_out"] = True
                        break
                    if info["quota_exceeded"]:
                        break
                    time.sleep(0.1)
            finally:
                # Applies to timeout, quota, exception, and normal exit. Only parent
                # lifetime is established, never descendant-tree containment.
                if proc is not None:
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=3.0)
                        except Exception:
                            try:
                                proc.kill()
                                proc.wait(timeout=2.0)
                            except Exception as e:
                                info["cleanup_error"] = str(e)
                    info["exit_code"] = proc.poll()
                    info["parent_exited"] = info["exit_code"] is not None
    except Exception as e:
        info["error"] = str(e)
    finally:
        info["duration_seconds"] = round(time.monotonic() - start, 2)
        for label, path in [("stdout", stdout_file), ("stderr", stderr_file)]:
            ev = stream_excerpt(path)
            info[label + "_bytes"] = ev["total_bytes"]
            info[label + "_truncated"] = ev["truncated"]
            if ev.get("error"):
                info[label + "_error"] = ev["error"]
            if (ev["total_bytes"] or 0) > MAX_STREAM_BYTES:
                info["quota_exceeded"] = True
        info["incomplete"] = bool(info["timed_out"] or info["quota_exceeded"]
            or info.get("error") or not info["parent_exited"]
            or info["stdout_truncated"] or info["stderr_truncated"]
            or info.get("stdout_error") or info.get("stderr_error"))
    return info


def inspect_sqlite_history(session_id: str) -> Dict[str, Any]:
    """Inspect history.db using read-only mode=ro and verify exact prompt sequence."""
    home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home())
    db_path = home / ".nexus-agent" / "history.db"

    ev: Dict[str, Any] = {
        "db_found": db_path.is_file(),
        "db_path": str(db_path),
        "session_id": session_id,
        "total_rows": 0,
        "user_messages": [],
        "prompt1_exact_match": False,
        "prompt2_exact_match": False,
        "sequence_exact_match": False,
    }

    if not db_path.is_file():
        return ev

    conn = None
    try:
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        count, max_bytes = conn.execute(
            "SELECT count(*),coalesce(max(length(CAST(message_json AS BLOB))),0) "
            "FROM conversation_history WHERE session_id = ?", (session_id,)).fetchone()
        ev["total_rows"] = count
        if count > 100 or max_bytes > 64 * 1024:
            raise ValueError("History exceeds bounded row/message limits")
        cur = conn.execute(
            "SELECT id, role, message_json FROM conversation_history "
            "WHERE session_id = ? AND length(CAST(message_json AS BLOB)) <= ? "
            "ORDER BY id ASC LIMIT 101", (session_id, 64 * 1024))
        user_prompts = []
        for row_id, role, msg_raw in cur:
            if role == "user":
                payload = json.loads(msg_raw)
                if not isinstance(payload, dict) or payload.get("role") != "user":
                    raise ValueError("User role mismatch in history")
                user_prompts.append(payload.get("content"))
        ev["user_messages"] = user_prompts
        ev["prompt1_exact_match"] = bool(user_prompts and user_prompts[0] == PROMPT_1)
        ev["prompt2_exact_match"] = bool(len(user_prompts) == 2 and user_prompts[1] == PROMPT_2)
        ev["sequence_exact_match"] = user_prompts == [PROMPT_1, PROMPT_2]

    except Exception as e:
        ev["error"] = str(e)
    finally:
        if conn is not None:
            conn.close()

    return ev


def inspect_server_logs() -> List[Dict[str, str]]:
    """Capture tails of owned server-*.log files from isolated home."""
    home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home())
    server_dir = home / ".nexus-agent" / "llama-server"
    logs = []
    if server_dir.is_dir():
        for p in sorted(server_dir.glob("server-*.log"))[:4]:
            logs.append({
                "filename": p.name,
                "excerpt": stream_excerpt(p, limit=64 * 1024),
            })
    return logs


def main():
    parser = argparse.ArgumentParser(description="Nexus Two-Turn Normal Observation Runner")
    parser.add_argument("--workspace", required=True, type=Path, help="Scratch directory for code editing")
    parser.add_argument("--evidence-dir", required=True, type=Path, help="Directory to save raw observation logs")
    parser.add_argument("--cli-exe", required=True, help="Path to installed nexus-agent.exe")
    parser.add_argument("--session-id", default="obs-session-001", help="Persistent session ID")
    args = parser.parse_args()

    ws: Path = args.workspace.resolve()
    ev_dir: Path = args.evidence_dir.resolve()
    ws.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)

    calc_file = ws / "calc.py"
    write_file_bytes(calc_file, INITIAL_CALC_PY)
    write_file_bytes(ev_dir / "calc_initial.py", INITIAL_CALC_PY)

    print("=================================================================")
    print("NEXUS NORMAL TWO-TURN OBSERVATION RUN")
    print(f"Workspace: {ws}")
    print(f"CLI Executable: {args.cli_exe}")
    print(f"Dedicated Session: {args.session_id}")
    print("=================================================================")

    # -------------------------------------------------------------------------
    # Turn 1: "Fix multiply in calc.py so it multiplies."
    # -------------------------------------------------------------------------
    print(f"\n[Turn 1] Executing Prompt: {PROMPT_1}")
    cmd1 = [
        args.cli_exe,
        "chat",
        "-p", "local",
        "--verbose",
        "--no-stream",
        "--persist",
        "--session", args.session_id,
        PROMPT_1,
    ]

    t1_stdout = ev_dir / "turn1_stdout.log"
    t1_stderr = ev_dir / "turn1_stderr.log"
    t1_info = run_observed_turn(cmd1, cwd=ws, stdout_file=t1_stdout, stderr_file=t1_stderr)

    calc_t1_bytes, calc_t1_error = snapshot_calc(calc_file)
    write_file_bytes(ev_dir / "calc_after_turn1.py", calc_t1_bytes)
    t1_changed = None if calc_t1_error else calc_t1_bytes != INITIAL_CALC_PY
    t1_diff = compute_unified_diff(INITIAL_CALC_PY, calc_t1_bytes, "calc.py") if calc_t1_error is None else "# SNAPSHOT UNAVAILABLE: no deletion inferred"
    write_file_bytes(ev_dir / "turn1_diff.patch", t1_diff.encode("utf-8"))

    print(f"Turn 1 Exit Code: {t1_info['exit_code']} (Duration: {t1_info['duration_seconds']}s)")
    print(f"Turn 1 Cleanup: {t1_info['cleanup_status']}")
    print(f"calc.py modified in Turn 1: {t1_changed}")

    # -------------------------------------------------------------------------
    # Turn 2: "Now make it return 0 if either input is None."
    # -------------------------------------------------------------------------
    t2_info: Dict[str, Any] = {"skipped": True, "reason": "Turn 1 process failed or did not exit"}
    calc_t2_bytes = calc_t1_bytes
    t2_diff = ""
    calc_t2_error = None
    t2_changed = None

    if t1_info["parent_exited"] and not t1_info["incomplete"] and calc_t1_error is None:
        print(f"\n[Turn 2] Executing Prompt: {PROMPT_2}")
        cmd2 = [
            args.cli_exe,
            "chat",
            "-p", "local",
            "--verbose",
            "--no-stream",
            "--persist",
            "--session", args.session_id,
            PROMPT_2,
        ]

        t2_stdout = ev_dir / "turn2_stdout.log"
        t2_stderr = ev_dir / "turn2_stderr.log"
        t2_info = run_observed_turn(cmd2, cwd=ws, stdout_file=t2_stdout, stderr_file=t2_stderr)
        t2_info["skipped"] = False

        calc_t2_bytes, calc_t2_error = snapshot_calc(calc_file)
        write_file_bytes(ev_dir / "calc_after_turn2.py", calc_t2_bytes)
        t2_changed = None if calc_t2_error else calc_t2_bytes != calc_t1_bytes
        t2_diff = compute_unified_diff(calc_t1_bytes, calc_t2_bytes, "calc.py") if calc_t2_error is None else "# SNAPSHOT UNAVAILABLE: no deletion inferred"
        write_file_bytes(ev_dir / "turn2_diff.patch", t2_diff.encode("utf-8"))

        print(f"Turn 2 Exit Code: {t2_info['exit_code']} (Duration: {t2_info['duration_seconds']}s)")
        print(f"Turn 2 Cleanup: {t2_info['cleanup_status']}")
        print(f"calc.py modified in Turn 2: {t2_changed}")
    else:
        print("\n[Turn 2] SKIPPED: Turn 1 process did not exit normally; avoiding overlapping processes.")

    # -------------------------------------------------------------------------
    # Evidence Inspection (SQLite, Server Logs)
    # -------------------------------------------------------------------------
    print("Turn 1 raw observation:", json.dumps(t1_info, ensure_ascii=True))
    print("Turn 2 raw observation:", json.dumps(t2_info, ensure_ascii=True))
    print("Snapshot errors:", calc_t1_error, calc_t2_error)
    history_ev = inspect_sqlite_history(args.session_id)
    print("Bounded session history:", json.dumps(history_ev, ensure_ascii=True))
    server_logs = inspect_server_logs()

    # -------------------------------------------------------------------------
    # Console-Only Raw Evidence Display
    # -------------------------------------------------------------------------
    print("\n=================================================================")
    print("RAW EVIDENCE (CONSOLE EXCERPTS)")
    print("=================================================================")

    print("\n--- calc.py (Initial) ---")
    print(INITIAL_CALC_PY.decode("utf-8"))

    print("--- calc.py (After Turn 1) ---")
    print(calc_t1_bytes.decode("utf-8", errors="replace"))

    print("--- Turn 1 Unified Diff ---")
    print(t1_diff if t1_diff else "[No changes made in Turn 1]")

    print("--- calc.py (After Turn 2) ---")
    print(calc_t2_bytes.decode("utf-8", errors="replace"))

    print("--- Turn 2 Unified Diff ---")
    print(t2_diff if t2_diff else "[No changes made in Turn 2]")

    print("--- Turn 1 Stdout bounded head/tail ---")
    print(read_file_tail(ev_dir / "turn1_stdout.log"))

    print("--- Turn 1 Stderr bounded head/tail ---")
    print(read_file_tail(ev_dir / "turn1_stderr.log"))

    if not t2_info.get("skipped"):
        print("--- Turn 2 Stdout bounded head/tail ---")
        print(read_file_tail(ev_dir / "turn2_stdout.log"))

        print("--- Turn 2 Stderr bounded head/tail ---")
        print(read_file_tail(ev_dir / "turn2_stderr.log"))

    for slog in server_logs:
        print(f"--- Server Log ({slog['filename']}) Tail ---")
        print(json.dumps(slog["excerpt"], ensure_ascii=True))

    # -------------------------------------------------------------------------
    # Markdown Summary Generation (Step Summary Only)
    # -------------------------------------------------------------------------
    incomplete = bool(t1_info["incomplete"] or t2_info.get("incomplete") or t2_info.get("skipped")
        or calc_t1_error or calc_t2_error or not history_ev["sequence_exact_match"]
        or any(x["excerpt"].get("truncated") or x["excerpt"].get("error") for x in server_logs))
    summary_lines = [
        "# Nexus Normal Two-Turn Observation Summary",
        "",
        f"- **Observation Capture:** {'INCOMPLETE / UNVERIFIED' if incomplete else 'OBSERVED / UNVERIFIED'} (manual file-diff review required)",
        "- **Capture Limits:** Disk quota monitoring is best-effort, not a hard storage bound. Printed streams are bounded; TRUNCATED means INCOMPLETE.",
        "- **Engine Evidence:** Pull-model exit 0 is not engine-load proof. Server log tails and CLI output are raw observations only; fallback/readiness are not inferred.",
        "- **Model Status:** `UNVERIFIED` (Evidence observation only; not a qualification verdict)",
        "- **Model Target:** `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (Baseline small model)",
        "- **Persistence Session:** `" + args.session_id + "`",
        "",
        "## Turn Executions",
        "| Turn | Prompt | Exit Code | Duration | File Changed | Cleanup Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| 1 | `{PROMPT_1}` | {t1_info.get('exit_code')} | {t1_info.get('duration_seconds')}s | {t1_changed} | {t1_info.get('cleanup_status')} |",
    ]

    if t2_info.get("skipped"):
        summary_lines.append(f"| 2 | `{PROMPT_2}` | SKIPPED | N/A | UNKNOWN | N/A |")
    else:
        summary_lines.append(
            f"| 2 | `{PROMPT_2}` | {t2_info.get('exit_code')} | {t2_info.get('duration_seconds')}s | {t2_changed} | {t2_info.get('cleanup_status')} |"
        )

    summary_lines.extend([
        "",
        "## SQLite History Integrity",
        f"- **Database Found:** {history_ev['db_found']}",
        f"- **Total Rows for Session:** {history_ev['total_rows']}",
        f"- **Prompt 1 Exact Match:** {history_ev['prompt1_exact_match']}",
        f"- **Prompt 2 Exact Match:** {history_ev['prompt2_exact_match']}",
        f"- **Sequence Exact Order:** {history_ev['sequence_exact_match']}",
        "",
        "## Observed Code Diffs",
        "### Turn 1 Diff",
        "```diff",
        t1_diff if t1_diff else "# No diff produced in Turn 1",
        "```",
        "### Turn 2 Diff",
        "```diff",
        t2_diff if t2_diff else "# No diff produced in Turn 2",
        "```",
        "",
        "---",
        "> **Notice:** Generated code was NOT imported, compiled, or executed. Reviewers inspect diffs directly above.",
    ])

    summary_text = "\n".join(summary_lines) + "\n"
    (ev_dir / "summary.md").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
