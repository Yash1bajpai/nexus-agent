"""Q1 Windows interactive candidate driver: ConPTY session execution and cleanup.

Drives installed nexus_agent_ai repl through real pseudo-console (pywinpty ConPTY).
Executes two fixed sessions: 'tasks' and 'compact'.
Verifies A native gate, installed provenance, TTY/Backspace probes, attributed SQLite
canonical history, context token math, calculator fix behavior, and process cleanup.
Bounded evidence output under EVIDENCE_CAP (64MB).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import winpty
    from winpty import PtyProcess
except ImportError:
    winpty = None
    PtyProcess = None

try:
    from . import interactive_assess as assess
except (ImportError, ValueError):
    _assess_path = Path(__file__).resolve().with_name("interactive_assess.py")
    _spec = importlib.util.spec_from_file_location("interactive_assess", _assess_path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"Cannot load interactive_assess from {_assess_path}")
    assess = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(assess)

RUNNER_SHA256 = "6bf1c0f1eff306eec2b73f6b77a9661e4b077d9846f200e0f692472491db97db"
MODEL_SHA256 = "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
MODEL_REL = Path(".nexus-agent/models/LiquidAI--LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf")
EVIDENCE_CAP = 64 * 1024 * 1024
MANIFEST_RESERVE = 1024 * 1024
_CAP_LOCK = threading.Lock()
STARTUP_TIMEOUT = 900.0   # 15 minutes for startup (covers model download if needed)
ROW_TIMEOUT = 420.0       # 7 minutes per model turn
SLASH_TIMEOUT = 30.0      # 30 seconds for slash commands
PROCESS_EXIT_TIMEOUT = 15.0

BRIEF_TEXT = (
    "Harbor Desk Deployment Summary - Release Cycle 2026-Q1\n"
    "The infrastructure migration to containerized worker nodes has completed across all four primary clusters.\n"
    "Database latency decreased by 18% following the implementation of connection pooling and read-replica routing.\n"
    "During the staging soak test, three edge cases were detected in the background sync worker:\n"
    "1. Token expiration during long-running batch jobs resulted in transient HTTP 401 retries.\n"
    "2. File descriptors exceeded the soft limit under concurrent payload downloads exceeding 50 MB.\n"
    "3. Cache invalidation events occasionally lagged behind replica write confirmations by 150 milliseconds.\n"
    "All three anomalies were mitigated by adjusting timeout thresholds and pool limits.\n"
)

CALCULATOR_INITIAL = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a + b\n"
SLASH_BUG_INITIAL = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a + b\n"
README_INITIAL = "Project Harbor Desk\nOwner Mira Chen\n"

CONTROLLER_CODE = (
    "import sys\n"
    "import os\n"
    "import runpy\n"
    "try:\n"
    "    from nexus_agent_ai.utils.interactive import stdin_is_interactive, _is_windows_console\n"
    "    interactive = stdin_is_interactive()\n"
    "    win_console = _is_windows_console(sys.stdin)\n"
    "except Exception:\n"
    "    interactive = False\n"
    "    win_console = False\n"
    "isatty = getattr(sys.stdin, 'isatty', lambda: False)()\n"
    "print(f'__NEXUS_TTY_PROBE__: isatty={isatty} win_console={win_console} interactive={interactive}', flush=True)\n"
    "sys.argv = ['nexus-agent-ai'] + sys.argv[1:]\n"
    "runpy.run_module('nexus_agent_ai', run_name='__main__')\n"
)

# Note on row-03: Predeclared natural-review prompt uses 'Please review...' rather than
# leading 'Review ...' because app.py:596 intercepts leading 'review ' as a shortcut
# and extracts the suffix as a filename. 'Please review...' preserves natural model evaluation.
PROTOCOL_TASKS = [
    {"name": "row-01-readme", "kind": "model", "prompt": "Read readme.txt and tell me the project name and owner.", "backspace": True, "timeout": ROW_TIMEOUT},
    {"name": "row-02-greeting", "kind": "model", "prompt": "Create greeting.txt containing exactly Hello Harbor Desk followed by one newline.", "backspace": False, "timeout": ROW_TIMEOUT},
    {"name": "row-03-calculator-review", "kind": "model", "prompt": "Please review calculator.py for bugs. Explain any incorrect behavior you find; do not edit files.", "backspace": False, "timeout": ROW_TIMEOUT},
    {"name": "row-04-slash-review", "kind": "model", "prompt": "/review slash_bug.py", "backspace": False, "timeout": ROW_TIMEOUT},
    {"name": "row-05-calculator-fix", "kind": "model", "prompt": "Find and fix the bug in calculator.py so multiply works correctly. Preserve add and all unrelated code.", "backspace": False, "timeout": ROW_TIMEOUT},
    {"name": "row-06-web-search", "kind": "model", "prompt": "Search the web for the latest llama.cpp release and give its tag and official release URL.", "backspace": False, "timeout": ROW_TIMEOUT},
    {"name": "row-07-exit", "kind": "exit", "prompt": "/exit", "backspace": False, "timeout": PROCESS_EXIT_TIMEOUT},
]

PROTOCOL_COMPACT = [
    {"name": "compact-turn-01-remember", "kind": "model", "prompt": "Remember that the release codename is Amber-Kite-731. Reply only Noted.", "timeout": ROW_TIMEOUT},
    {"name": "compact-turn-02-brief", "kind": "model", "prompt": "Read brief.txt and summarize it in four short bullet points.", "timeout": ROW_TIMEOUT},
    {"name": "compact-turn-03-path-vs-name", "kind": "model", "prompt": "Explain the difference between a file path and a filename in two sentences.", "timeout": ROW_TIMEOUT},
    {"name": "compact-turn-04-backup-reason", "kind": "model", "prompt": "Give one reason to keep a backup before editing a file, in one sentence.", "timeout": ROW_TIMEOUT},
    {"name": "compact-context-before", "kind": "slash", "prompt": "/context", "timeout": SLASH_TIMEOUT},
    {"name": "compact-row-06-slash-compact", "kind": "slash", "prompt": "/compact", "timeout": SLASH_TIMEOUT},
    {"name": "compact-context-after", "kind": "slash", "prompt": "/context", "timeout": SLASH_TIMEOUT},
    {"name": "compact-row-08-recall", "kind": "model", "prompt": "What release codename did I ask you to remember earlier?", "timeout": ROW_TIMEOUT},
    {"name": "compact-turn-09-exit", "kind": "exit", "prompt": "/exit", "timeout": PROCESS_EXIT_TIMEOUT},
]


def digest(path: Path | str) -> str:
    """Compute sha256 hex digest of file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path | str, name: str) -> Any:
    """Load python module dynamically from file path."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deadline_from_file(path: Path | str) -> float:
    """Parse and validate work deadline from deadline.json, returning monotonic deadline."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("job_limit_minutes") != 120 or data.get("reserve_minutes") != 10:
        raise ValueError("Invalid overall/teardown schedule in deadline file")
    remaining = data["work_deadline_utc"] - time.time()
    if not (0 < remaining <= 110 * 60):
        raise ValueError(f"Work deadline exhausted or invalid: {remaining}s remaining")
    return time.monotonic() + remaining


def child_environment(home: Path, temporary: Path, workspace: Path) -> Dict[str, str]:
    """Construct isolated child process environment with exact existing pins."""
    directories = [str(Path(sys.executable).parent)]
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            directories.extend([str(Path(os.environ[key]) / "System32"), os.environ[key]])
    if os.name != "nt":
        directories.extend(os.defpath.split(os.pathsep))
    if git_exe := shutil.which("git"):
        directories.append(str(Path(git_exe).resolve().parent))
    home, temporary, workspace = Path(home), Path(temporary), Path(workspace)
    home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    return {
        **{k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ},
        "HOME": str(home),
        "USERPROFILE": str(home),
        "NEXUS_AGENT_HOME": str(home / ".nexus-agent"),
        "NEXUS_AGENT_MODEL_REPO": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "NEXUS_AGENT_MODEL_FILENAME": "LFM2.5-1.2B-Instruct-Q4_0.gguf",
        "NEXUS_E2E_WORKDIR": str(workspace),
        "PATH": os.pathsep.join(dict.fromkeys(directories)),
        "TMP": str(temporary), "TEMP": str(temporary), "TMPDIR": str(temporary),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PROMPT_TOOLKIT_NO_CPR": "1",
    }


def _charge(tracker, size, cap=EVIDENCE_CAP):
    with _CAP_LOCK:
        if tracker[0] + size > cap - MANIFEST_RESERVE:
            raise RuntimeError("Aggregate session evidence cap exceeded")
        tracker[0] += size


def charge_and_write(path: Path, data: bytes, tracker: List[int], cap: int = EVIDENCE_CAP) -> None:
    """Write data while charging against shared session evidence budget."""
    _charge(tracker, len(data), cap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def check_evidence_cap(output_dir: Path, cap: int = EVIDENCE_CAP, reserve: int = MANIFEST_RESERVE) -> int:
    """Calculate regular file size in output_dir and verify cap."""
    total = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file() and not p.is_symlink()) if output_dir.exists() else 0
    if total + reserve > cap:
        raise RuntimeError(f"Evidence output cap exceeded: {total} + {reserve} > {cap}")
    return total


def record_owned_process(root: Path, pid: int) -> None:
    """Record owned process metadata for cleanup."""
    output_dir = Path(root) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    owned_file = output_dir / "owned_processes.json"
    owned = []
    if owned_file.exists():
        try:
            owned = json.loads(owned_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    create_time = 0.0
    exe, cmdline = "", []
    try:
        import psutil
        proc = psutil.Process(pid)
        create_time = proc.create_time()
        exe, cmdline = proc.exe(), proc.cmdline()
    except Exception:
        pass
    if create_time > 0 and not any(e.get("pid") == pid for e in owned):
        owned.append({"pid": pid, "create_time": create_time, "exe": exe, "cmdline": cmdline})
        owned_file.write_text(json.dumps(owned, indent=2), encoding="utf-8")


def scan_and_record_descendants(root: Path, parent_pid: int) -> None:
    """Scan and record child processes of parent PID."""
    try:
        import psutil
        for child in psutil.Process(parent_pid).children(recursive=True):
            record_owned_process(root, child.pid)
    except Exception:
        pass


def fetch_sqlite_messages(db_path: Path, session_id: str) -> List[Dict[str, Any]]:
    """Query synthetic session messages from history.db in read-only mode, extracting tool bindings."""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = []
            for r in conn.execute("SELECT id, session_id, role, message_json, timestamp FROM conversation_history WHERE session_id = ? ORDER BY id ASC", (session_id,)):
                payload = json.loads(r["message_json"]) if r["message_json"] else {}
                if not isinstance(payload, dict):
                    payload = {}
                rows.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": payload.get("content"),
                    "tool_calls": payload.get("tool_calls"),
                    "tool_call_id": payload.get("tool_call_id"),
                    "name": payload.get("name"),
                    "message_json": r["message_json"],
                    "timestamp": str(r["timestamp"]),
                })
            return rows
    except Exception:
        return []


def canonical_diff(canonical_before: List[Dict[str, Any]], canonical_after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute newly added or rewritten messages stably via content-prefix matching."""
    if not canonical_after:
        return []
    if not canonical_before:
        return list(canonical_after)
    s_before: Dict[str, List[Dict[str, Any]]] = {}
    for r in canonical_before:
        s_before.setdefault(r["session_id"], []).append(r)
    s_after: Dict[str, List[Dict[str, Any]]] = {}
    for r in canonical_after:
        s_after.setdefault(r["session_id"], []).append(r)
    delta = []
    for sid, after_list in s_after.items():
        before_list = s_before.get(sid, [])
        if not before_list:
            delta.extend(after_list)
            continue
        prefix_len = 0
        while (prefix_len < len(before_list) and prefix_len < len(after_list) and
               before_list[prefix_len]["role"] == after_list[prefix_len]["role"] and
               before_list[prefix_len]["message_json"] == after_list[prefix_len]["message_json"]):
            prefix_len += 1
        if prefix_len == len(before_list):
            delta.extend(after_list[prefix_len:])
        else:
            before_multiset = [b["message_json"] for b in before_list]
            for r in after_list:
                if r["message_json"] in before_multiset:
                    before_multiset.remove(r["message_json"])
                else:
                    delta.append(r)
    return delta


def snapshot_workspace_files(workspace: Path) -> Dict[str, str]:
    """Capture snapshot text of known files in workspace."""
    res = {}
    for fname in ("calculator.py", "slash_bug.py", "readme.txt", "brief.txt", "greeting.txt"):
        p = workspace / fname
        if p.is_file():
            res[fname] = p.read_text(encoding="utf-8", errors="replace")
    return res


def verify_calculator_behavior(workspace: Path, before_src: str) -> Tuple[bool, bool]:
    """Test multiply and add behavior on calculator.py and verify add AST preservation."""
    calc_path = workspace / "calculator.py"
    if not calc_path.is_file():
        return False, False
    code = ("import sys, json; sys.path.insert(0, '.'); import calculator; "
            "print(json.dumps({'mult_3_4': calculator.multiply(3, 4), 'mult_neg': calculator.multiply(-2, 5), "
            "'mult_zero': calculator.multiply(0, 7), 'add_3_4': calculator.add(3, 4)}))")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(workspace), capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return False, False
        data = json.loads(proc.stdout.strip())
        behavior_ok = (data.get("mult_3_4") == 12 and data.get("mult_neg") == -10 and data.get("mult_zero") == 0 and data.get("add_3_4") == 7)
    except Exception:
        return False, False

    try:
        tree_old = ast.parse(before_src)
        tree_new = ast.parse(calc_path.read_text(encoding="utf-8"))
        add_old = next((n for n in tree_old.body if isinstance(n, ast.FunctionDef) and n.name == "add"), None)
        add_new = next((n for n in tree_new.body if isinstance(n, ast.FunctionDef) and n.name == "add"), None)
        add_preserved = bool(add_old and add_new and ast.dump(add_old, include_attributes=False) == ast.dump(add_new, include_attributes=False))
    except Exception:
        add_preserved = False

    return behavior_ok and add_preserved, add_preserved


class PtySession:
    """Manages an interactive session driven via pywinpty ConPTY with a bounded finite reader."""

    def __init__(self, argv: List[str], cwd: Path, env: Dict[str, str], root: Path,
                 byte_tracker: Optional[List[int]] = None, pty_proc: Any = None):
        self.root = Path(root)
        if pty_proc is not None:
            self.pty = pty_proc
        else:
            if PtyProcess is None:
                raise RuntimeError("pywinpty is required to spawn PtySession")
            self.pty = PtyProcess.spawn(argv, cwd=str(cwd), env=env, dimensions=(30, 120))
        self.pid = getattr(self.pty, "pid", 0)
        self.out_queue: queue.Queue[str] = queue.Queue(maxsize=1024)
        self.stop_event = threading.Event()
        self.all_output = ""
        self.byte_tracker = byte_tracker if byte_tracker is not None else [0]
        self.error_flag: Optional[str] = None
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        if self.pid > 0:
            record_owned_process(self.root, self.pid)

    def _reader_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.pty.read(1024)
                if chunk:
                    chunk_bytes = len(chunk.encode("utf-8", errors="replace"))
                    try:
                        _charge(self.byte_tracker, chunk_bytes)
                    except RuntimeError:
                        self.error_flag = "output_cap_exceeded"
                        break
                    try:
                        self.out_queue.put(chunk, timeout=0.5)
                    except queue.Full:
                        self.error_flag = "queue_full"
                        break
                else:
                    if hasattr(self.pty, "isalive") and not self.pty.isalive():
                        break
                    time.sleep(0.01)
            except EOFError:
                break
            except Exception as e:
                self.error_flag = f"read_error: {type(e).__name__}"
                break

    def wait_for_ready_prompt(self, timeout: float, deadline: float, prompt_pattern: str = r"^>>\s*You:?\s*$") -> Tuple[str, bool]:
        start = time.monotonic()
        buf = ""
        target_deadline = min(deadline, start + timeout)
        while time.monotonic() < target_deadline:
            if self.error_flag:
                return buf, False
            if hasattr(self.pty, "isalive") and not self.pty.isalive() and self.out_queue.empty():
                break
            try:
                chunk = self.out_queue.get(timeout=0.05)
                buf += chunk
                self.all_output += chunk
                lines = [l.strip() for l in assess.strip_ansi(buf).splitlines() if l.strip()]
                if lines and re.match(prompt_pattern, lines[-1]):
                    if self.pid > 0:
                        scan_and_record_descendants(self.root, self.pid)
                    return buf, True
            except queue.Empty:
                if self.pid > 0:
                    scan_and_record_descendants(self.root, self.pid)
                continue
        lines = [l.strip() for l in assess.strip_ansi(buf).splitlines() if l.strip()]
        found = bool(lines and re.match(prompt_pattern, lines[-1])) and not bool(self.error_flag)
        if self.pid > 0:
            scan_and_record_descendants(self.root, self.pid)
        return buf, found

    def wait_for_exit(self, timeout: float = PROCESS_EXIT_TIMEOUT, deadline: Optional[float] = None) -> Tuple[str, bool]:
        start = time.monotonic()
        buf = ""
        target_timeout = timeout
        if deadline is not None:
            target_timeout = min(timeout, max(0.0, deadline - time.monotonic()))
        while time.monotonic() - start < target_timeout and not self.error_flag:
            if hasattr(self.pty, "isalive") and not self.pty.isalive() and self.out_queue.empty():
                return buf, True
            try:
                chunk = self.out_queue.get(timeout=0.05)
                buf += chunk
                self.all_output += chunk
            except queue.Empty:
                continue
        alive = getattr(self.pty, "isalive", lambda: False)()
        return buf, not alive and not self.error_flag

    def send_input(self, text: str) -> None:
        self.pty.write(text)

    def close(self) -> None:
        self.stop_event.set()
        try:
            if hasattr(self.pty, "isalive") and self.pty.isalive():
                self.pty.close()
        except Exception:
            pass


def cleanup_root(root: Path | str) -> Dict[str, Any]:
    """Clean up owned processes recorded under root. Never startup model or kill unrelated trees."""
    root = Path(root).resolve()
    output_dir = root / "output"
    owned_file = output_dir / "owned_processes.json"
    killed, errors, remaining, remaining_llama = [], [], [], []
    try:
        owned = json.loads(owned_file.read_text()) if owned_file.exists() else []
    except (OSError, ValueError):
        owned = []
        errors.append("owned metadata unreadable")
    try:
        import psutil
    except ImportError:
        psutil = None
        errors.append("psutil unavailable")
    if os.name != "nt":
        errors.append("Windows cleanup not exercised on this platform")

    def same_process(process, entry):
        return process.create_time() == entry.get("create_time") and entry.get("create_time", 0) > 0

    def server_under_root(process):
        if (process.info.get("name") or "").lower() not in ("llama-server", "llama-server.exe"):
            return False
        args = process.info.get("cmdline") or []
        return any(str(root).lower() in arg.lower() for arg in args)

    if psutil is not None and os.name == "nt":
        for entry in owned:
            try:
                process = psutil.Process(entry["pid"])
                if same_process(process, entry):
                    result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                            capture_output=True, timeout=10)
                    killed.append({"pid": process.pid, "returncode": result.returncode})
            except psutil.NoSuchProcess:
                pass
            except Exception as exc:
                errors.append("owned tree cleanup: " + type(exc).__name__)
        # Server arguments point at this fresh HOME; never target a name alone.
        try:
            for process in psutil.process_iter(["pid", "name", "cmdline"]):
                if server_under_root(process):
                    result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                            capture_output=True, timeout=10)
                    killed.append({"pid": process.pid, "returncode": result.returncode})
            time.sleep(0.5)
            for entry in owned:
                try:
                    process = psutil.Process(entry["pid"])
                    if same_process(process, entry) and process.is_running():
                        remaining.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
            for process in psutil.process_iter(["pid", "name", "cmdline"]):
                if server_under_root(process):
                    remaining_llama.append(process.pid)
        except Exception as exc:
            errors.append("owned process final check: " + type(exc).__name__)
    result = {"confirmed": not errors and not remaining and not remaining_llama,
              "owned_count": len(owned), "killed_pids": killed, "remaining_owned": remaining,
              "remaining_llama_servers": remaining_llama, "errors": errors}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleanup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _execute_protocol_session(protocol: List[Dict[str, Any]], session_id: str, pty: PtySession,
                              workspace: Path, home: Path, history_db: Path, deadline: float,
                              is_tasks: bool, output_dir: Path, target_rows: List[Dict[str, Any]],
                              byte_tracker: List[int], on_after_first_real_task: Any = None) -> bool:
    """Execute a declared protocol table in-place, preserving row IDs and stopping on first timeout."""
    session_stopped = False
    first_real_checked = False
    context_metrics_before: Dict[str, Any] = {}

    for idx, step in enumerate(protocol):
        name, kind, prompt, timeout = step["name"], step["kind"], step["prompt"], step["timeout"]
        if session_stopped or time.monotonic() >= deadline:
            target_rows[idx] = {"name": name, "task": {"status": assess.NOT_RUN, "evidence": ["Session stopped or deadline exhausted"]}}
            continue

        before_db = fetch_sqlite_messages(history_db, session_id)
        before_files = snapshot_workspace_files(workspace)

        # Four completed genuine turns, with the seed outside the last two.
        if name == "compact-row-06-slash-compact":
            starts = [i for i, m in enumerate(before_db) if m.get("role") == "user"
                      and not str(m.get("content", "")).startswith("[Earlier conversation compacted")]
            completed = sum(any(m.get("role") == "assistant" and not m.get("tool_calls")
                                for m in before_db[start:ends])
                            for start, ends in zip(starts, starts[1:] + [len(before_db)]))
            prefix = before_db[:starts[-2]] if len(starts) >= 2 else []
            seed_in_prefix = any(m.get("role") == "user" and "Amber-Kite-731" in str(m.get("content", "")) for m in prefix)
            if completed < 4 or not seed_in_prefix:
                target_rows[idx] = {"name": name, "task": {"status": assess.HARNESS_ERROR,
                    "evidence": [{"completed_turns": completed, "seed_in_prefix": seed_in_prefix}]}}
                session_stopped = True
                continue

        pty.send_input(prompt + ("x\x08\r" if step.get("backspace") else "\r"))
        if kind == "exit":
            _, exited = pty.wait_for_exit(timeout=PROCESS_EXIT_TIMEOUT, deadline=deadline)
            target_rows[idx] = {"name": name, "executor": {"status": assess.PASS if exited else assess.INCOMPLETE, "evidence": [{"clean_exit": exited}]}}
            break

        rem = deadline - time.monotonic()
        if rem <= 0:
            target_rows[idx] = {"name": name, "task": {"status": assess.INCOMPLETE, "evidence": ["Deadline exhausted during step"]}}
            session_stopped = True
            continue

        raw_chunk, ready = pty.wait_for_ready_prompt(timeout=min(timeout, rem), deadline=deadline)
        after_db = fetch_sqlite_messages(history_db, session_id)
        after_files = snapshot_workspace_files(workspace)
        delta_msgs = canonical_diff(before_db, after_db)
        assistant_ans = assess.extract_assistant_turn_answer(delta_msgs)
        user_msg = next((m for m in delta_msgs if m.get("role") == "user"), None)

        if not ready or (kind == "model" and not assistant_ans):
            target_rows[idx] = {"name": name, "task": {"status": assess.INCOMPLETE, "evidence": ["Timeout, incomplete turn or reader failure", pty.error_flag]},
                               "raw_chunk": raw_chunk, "canonical_before": before_db, "canonical_after": after_db,
                               "known_files_before": before_files, "known_files_after": after_files}
            charge_and_write(output_dir / "rows" / f"{session_id}_{idx:02d}_{name}.json",
                             json.dumps(target_rows[idx], indent=2).encode(), byte_tracker)
            session_stopped = True
            continue

        if name == "row-01-readme":
            row_res = assess.assess_task_readme(assistant_ans, delta_msgs, user_msg, prompt)
        elif name == "row-02-greeting":
            greeting_bytes = (workspace / "greeting.txt").read_bytes() if (workspace / "greeting.txt").exists() else None
            row_res = assess.assess_task_greeting(greeting_bytes, delta_msgs)
        elif name == "row-03-calculator-review":
            row_res = assess.assess_task_calculator_review(assistant_ans, before_files.get("calculator.py", ""), after_files.get("calculator.py", ""), delta_msgs, is_slash=False)
        elif name == "row-04-slash-review":
            row_res = assess.assess_task_calculator_review(assistant_ans, before_files.get("slash_bug.py", ""), after_files.get("slash_bug.py", ""), delta_msgs, is_slash=True)
        elif name == "row-05-calculator-fix":
            beh_ok, add_ok = verify_calculator_behavior(workspace, before_files.get("calculator.py", ""))
            row_res = assess.assess_task_calculator_fix(beh_ok, add_ok, before_files.get("calculator.py", ""), after_files.get("calculator.py", ""), delta_msgs)
        elif name == "row-06-web-search":
            row_res = assess.assess_task_web_search(assistant_ans, delta_msgs)
        elif name == "compact-context-before":
            context_metrics_before = assess.parse_context_output(raw_chunk)
            row_res = assess.assess_context_command(context_metrics_before, "before")
        elif name == "compact-row-06-slash-compact":
            row_res = assess.assess_compact_slash(assess.parse_compact_output(raw_chunk), after_db)
        elif name == "compact-context-after":
            context_metrics_after = assess.parse_context_output(raw_chunk)
            row_res = assess.assess_context_command(context_metrics_after, "after")
            if context_metrics_before.get("history") and context_metrics_after.get("history") and context_metrics_after["history"] >= context_metrics_before["history"]:
                row_res["task"] = {"status": assess.FAIL, "evidence": ["Context history tokens did not decrease after /compact"]}
        elif name == "compact-row-08-recall":
            row_res = assess.assess_compact_recall(assistant_ans, expected_codename="Amber-Kite-731")
        else:
            row_res = {"name": name, "task": {"status": assess.PASS, "evidence": ["Step completed"]}}

        row_res["raw_chunk"] = raw_chunk
        row_res["canonical_before"] = before_db
        row_res["canonical_after"] = after_db
        row_res["canonical_delta"] = delta_msgs
        row_res["known_files_before"] = before_files
        row_res["known_files_after"] = after_files
        target_rows[idx] = row_res

        # Persist row record to disk before next step
        row_bytes = json.dumps(row_res, indent=2).encode("utf-8")
        charge_and_write(output_dir / "rows" / f"{session_id}_{idx:02d}_{name}.json", row_bytes, byte_tracker)

        if is_tasks and not first_real_checked and on_after_first_real_task:
            first_real_checked = True
            if not on_after_first_real_task():
                session_stopped = True

    return not session_stopped


def run_driver(args: argparse.Namespace) -> int:
    """Execute the full interactive qualification protocol."""
    source_root = Path(args.source_root).resolve(strict=True)
    root = Path(args.root).resolve()
    wheel = Path(args.wheel).resolve(strict=True)
    commit = args.commit
    deadline_file = Path(args.deadline_file).resolve(strict=True)
    native_evidence_path = Path(args.native_evidence).resolve(strict=True)

    # Hash-pin runner before import
    runner_path = source_root / ".github/qualification/runner.py"
    if digest(runner_path) != RUNNER_SHA256:
        raise RuntimeError(f"Runner hash mismatch: {digest(runner_path)}")

    # Enforce Windows platform
    if os.name != "nt":
        raise RuntimeError("Windows platform required for interactive ConPTY qualification")

    verifier = load_module(source_root / ".github/qualification/verify_windows_interactive.py", "verify_windows_interactive")
    native_gate = verifier.check_native_evidence(native_evidence_path)

    runner_mod = load_module(runner_path, "q1_runner")
    installed_provenance = runner_mod.verify_installed(wheel, source_root, commit)

    if installed_provenance["package_py"] != native_gate["provenance"]["package_py"]:
        raise RuntimeError("Installed runtime package bytes differ from A native gate package_py")

    deadline = deadline_from_file(deadline_file)
    workspace, home, temporary = root / "workspace", root / "home", root / "tmp"
    output_dir = root / "output"
    effects_dir, transcripts_dir = output_dir / "effects", output_dir / "transcripts"

    if any(d.exists() for d in (workspace, home, temporary)):
        raise RuntimeError("Fresh workspace, HOME and temporary directories required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Fresh dedicated output required")

    for d in (workspace, home, temporary, output_dir, effects_dir, transcripts_dir, output_dir / "rows"):
        d.mkdir(parents=True, exist_ok=True)

    (workspace / "calculator.py").write_text(CALCULATOR_INITIAL, encoding="utf-8")
    (workspace / "slash_bug.py").write_text(SLASH_BUG_INITIAL, encoding="utf-8")
    (workspace / "readme.txt").write_text(README_INITIAL, encoding="utf-8")
    (workspace / "brief.txt").write_text(BRIEF_TEXT, encoding="utf-8")

    (home / ".nexus_agent_initialized").write_text("Nexus-Agent initialized.\n", encoding="utf-8")
    env = child_environment(home, temporary, workspace)
    history_db = home / ".nexus-agent/history.db"

    tasks_rows = [{"name": r["name"], "task": {"status": assess.NOT_RUN, "evidence": ["Not reached"]}} for r in PROTOCOL_TASKS]
    compact_rows = [{"name": r["name"], "task": {"status": assess.NOT_RUN, "evidence": ["Not reached"]}} for r in PROTOCOL_COMPACT]
    model_verified_initial, model_verified_final = False, False
    tty_probe_tasks, tty_probe_compact = {"observed": False}, {"observed": False}
    cleanup_facts = {}
    byte_tracker = [0]

    def verify_initial_model() -> bool:
        nonlocal model_verified_initial
        m = home / MODEL_REL
        model_verified_initial = m.is_file() and digest(m) == MODEL_SHA256
        return model_verified_initial

    try:
        # Session 1: tasks
        s1_id = "interactive-tasks-001"
        argv1 = [sys.executable, "-I", "-X", "utf8", "-c", CONTROLLER_CODE,
                 "repl", "--provider", "local", "--context-size", "4096", "--max-output-tokens", "512",
                 "--max-iterations", "10", "--persist", "--session", s1_id, "--no-stream"]
        pty1 = PtySession(argv1, cwd=workspace, env=env, root=root, byte_tracker=byte_tracker)
        s1_completed = False
        try:
            init_out1, ready1 = pty1.wait_for_ready_prompt(timeout=STARTUP_TIMEOUT, deadline=deadline)
            tty_probe_tasks = assess.parse_tty_probe(init_out1)
            if ready1:
                s1_completed = _execute_protocol_session(
                    PROTOCOL_TASKS, s1_id, pty1, workspace, home, history_db,
                    deadline=deadline, is_tasks=True, output_dir=output_dir, target_rows=tasks_rows,
                    byte_tracker=byte_tracker, on_after_first_real_task=verify_initial_model
                )
            else:
                tasks_rows[0] = {"name": PROTOCOL_TASKS[0]["name"], "task": {"status": assess.INCOMPLETE, "evidence": ["Startup timeout"]}}
        finally:
            try:
                charge_and_write(transcripts_dir / "tasks.log", pty1.all_output.encode("utf-8", errors="replace"), byte_tracker)
            finally:
                pty1.close()

        # Session 2: compact (only if tasks session completed all turns and model verified)
        if s1_completed and model_verified_initial and time.monotonic() < deadline:
            s2_id = "interactive-compact-001"
            argv2 = [sys.executable, "-I", "-X", "utf8", "-c", CONTROLLER_CODE,
                     "repl", "--provider", "local", "--context-size", "4096", "--max-output-tokens", "512",
                     "--max-iterations", "10", "--persist", "--session", s2_id, "--no-stream"]
            pty2 = PtySession(argv2, cwd=workspace, env=env, root=root, byte_tracker=byte_tracker)
            try:
                init_out2, ready2 = pty2.wait_for_ready_prompt(timeout=STARTUP_TIMEOUT, deadline=deadline)
                tty_probe_compact = assess.parse_tty_probe(init_out2)
                if ready2:
                    _execute_protocol_session(
                        PROTOCOL_COMPACT, s2_id, pty2, workspace, home, history_db,
                        deadline=deadline, is_tasks=False, output_dir=output_dir, target_rows=compact_rows,
                        byte_tracker=byte_tracker
                    )
                else:
                    compact_rows[0] = {"name": PROTOCOL_COMPACT[0]["name"], "task": {"status": assess.INCOMPLETE, "evidence": ["Startup timeout"]}}
            finally:
                try:
                    charge_and_write(transcripts_dir / "compact.log", pty2.all_output.encode("utf-8", errors="replace"), byte_tracker)
                finally:
                    pty2.close()

        m_fin = home / MODEL_REL
        model_verified_final = m_fin.is_file() and digest(m_fin) == MODEL_SHA256

    finally:
        cleanup_facts = cleanup_root(root)

        owned_meta = json.loads((output_dir / "owned_processes.json").read_text(encoding="utf-8")) if (output_dir / "owned_processes.json").exists() else []
        cleanup_axis = assess.assess_cleanup(owned_meta, cleanup_facts.get("killed_pids", []),
                                             cleanup_facts.get("remaining_llama_servers", []),
                                             cleanup_facts.get("confirmed", False), cleanup_facts.get("errors", []))

        both_tty_ok = bool(tty_probe_tasks.get("interactive") and tty_probe_compact.get("interactive"))
        tty_axis = {"status": assess.PASS if both_tty_ok else assess.FAIL, "evidence": [tty_probe_tasks, tty_probe_compact]}
        budget_expired = time.monotonic() >= deadline
        all_rows = tasks_rows + compact_rows
        summary = assess.summarize_axes(all_rows, cleanup_axis, tty_axis, model_verified_initial, model_verified_final, budget_expired)

        complete = summary.get("automated_core_pass", False)

        results = {
            "commit": commit, "wheel": wheel.name, "native_evidence_verified": True,
            "native_gate": {
                "commit": native_gate.get("commit"), "platform": native_gate.get("platform"),
                "success": native_gate.get("success"), "qualified": native_gate.get("qualified"),
                "untraced_native": native_gate.get("untraced_native"),
            },
            "installed_provenance": installed_provenance,
            "model_sha256": MODEL_SHA256,
            "model_sha256_observations": {
                "initial_verified": model_verified_initial,
                "final_verified": model_verified_final,
                "expected": MODEL_SHA256,
            },
            "tty_probes": {"tasks": tty_probe_tasks, "compact": tty_probe_compact},
            "sessions": {key: [{k: v for k, v in r.items() if k not in
                         {"raw_chunk", "canonical_before", "canonical_after", "canonical_delta", "known_files_before", "known_files_after"}}
                         for r in rows] for key, rows in (("tasks", tasks_rows), ("compact", compact_rows))},
            "assessment": summary, "cleanup": cleanup_facts,
            "automated_core_pass": summary.get("automated_core_pass", False),
            "independent_review_required": summary.get("independent_review_required", True),
            "complete": complete,
        }

        results_bytes = json.dumps(results, indent=2).encode("utf-8")
        if len(results_bytes) > MANIFEST_RESERVE or check_evidence_cap(output_dir, reserve=len(results_bytes)) + len(results_bytes) > EVIDENCE_CAP:
            raise RuntimeError("Results exceed reserved evidence space")
        (output_dir / "results.json").write_bytes(results_bytes)

    return 0 if complete else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run interactive qualification sessions")
    run_parser.add_argument("--source-root", required=True)
    run_parser.add_argument("--root", required=True)
    run_parser.add_argument("--wheel", required=True)
    run_parser.add_argument("--commit", required=True)
    run_parser.add_argument("--deadline-file", required=True)
    run_parser.add_argument("--native-evidence", required=True)

    cleanup_parser = sub.add_parser("cleanup", help="Clean up owned processes for root")
    cleanup_parser.add_argument("--root", required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            return run_driver(args)
        except Exception as e:
            print(f"RUN DRIVER FAILED: {e}", file=sys.stderr)
            return 1
    elif args.command == "cleanup":
        result = cleanup_root(args.root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("confirmed", False) else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
