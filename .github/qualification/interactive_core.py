"""Q1 Windows interactive candidate driver: ConPTY session execution and cleanup.

Drives installed nexus_agent_ai repl through real pseudo-console (pywinpty ConPTY).
Executes two fixed sessions: 'tasks' and 'compact'.
Verifies A native gate, installed provenance, TTY/Backspace probes, attributed SQLite
canonical history, context token math, calculator fix behavior, and process cleanup.
Bounded evidence output under EVIDENCE_CAP (48MB).
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
EVIDENCE_CAP = 48 * 1024 * 1024
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
        # Probe-only: the AST allowlist is not OS containment and Windows has no
        # rlimits/Job Objects here, so run_code is refused outright (coverage gap).
        "NEXUS_AGENT_DISABLE_RUN_CODE": "1",
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


def record_observation_error(root: Path | str, error_msg: str) -> None:
    """Record observation error into persistent evidence."""
    try:
        output_dir = Path(root) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        err_file = output_dir / "observation_errors.json"
        errs = []
        if err_file.exists():
            try:
                loaded = json.loads(err_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    errs = loaded
                elif loaded:
                    errs = [str(loaded)]
            except Exception:
                errs = []
        errs.append(str(error_msg))
        err_file.write_text(json.dumps(errs, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Observation error (evidence write failed: {type(exc).__name__}): {error_msg}", file=sys.stderr)


def _verify_registered_parent(owned: List[Dict[str, Any]], parent_pid: int) -> Optional[Dict[str, Any]]:
    """Re-verify a registered parent's identity RIGHT NOW through a fresh
    handle: exact recorded ticks and canonical exe must match. Returns the
    registry entry on success, None on any failure (caller fails closed).
    This - not psutil enumeration freshness - is what proves the parent
    relationship at registration time."""
    entry = next((e for e in owned if e.get("pid") == parent_pid), None)
    if entry is None or type(entry.get("create_ticks")) is not int:
        return None
    current = _capture_registration_identity(parent_pid)
    if current is None:
        return None
    if current["create_ticks"] != entry["create_ticks"] or current["exe"] != entry.get("exe"):
        return None
    return entry


def record_owned_process(root: Path, pid: int, expected: Optional[Dict[str, Any]] = None,
                         expected_creator: Optional[int] = None) -> None:
    """Record owned process metadata for cleanup, with source-bound ownership.

    Ownership is established, never assumed: the single-channel registration
    identity (exact ticks + canonical exe + creator pid through ONE handle)
    must exactly match the earliest observed anchor when one was captured
    (spawn time for roots, scan time for descendants), and the creator must
    be the harness itself or a registered parent whose identity re-verifies
    at this moment. Any failure records an observation error and raises
    BEFORE the process is registered - a recycled occupant is never adopted.
    psutil fields are retained as display-only observation anchors. Earliest
    observation is itself post-spawn: this binds registration to the first
    observed identity; it does not reach back to CreateProcessW."""
    if pid <= 0:
        err = f"Invalid PID {pid} to record as owned process"
        record_observation_error(root, err)
        raise ValueError(err)

    output_dir = Path(root) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    owned_file = output_dir / "owned_processes.json"
    owned = []
    if owned_file.exists():
        try:
            loaded = json.loads(owned_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("owned metadata must be a list")
            owned = loaded
        except Exception as exc:
            err = f"owned metadata unreadable: {exc}"
            record_observation_error(root, err)
            raise RuntimeError(err)

    try:
        import psutil
        proc = psutil.Process(pid)
        create_time = proc.create_time()
        if create_time <= 0:
            err = f"Process {pid} returned non-positive create_time: {create_time}"
            record_observation_error(root, err)
            raise RuntimeError(err)
        ps_exe, cmdline = proc.exe(), proc.cmdline()
    except Exception as exc:
        record_observation_error(root, f"psutil inspection failed for PID {pid}: {exc}")
        raise
    if any(e.get("pid") == pid and e.get("create_time") == create_time for e in owned):
        return

    capture_diag: Dict[str, str] = {}
    identity = _capture_registration_identity(pid, capture_diag)
    if identity is None:
        # Best-effort existence hint to interpret the diagnostic gap later.
        # UNVERIFIED: not an ownership record, never an input to termination.
        try:
            import psutil as _psutil
            exists_hint: Any = _psutil.pid_exists(pid)
        except Exception as exc:
            exists_hint = f"unavailable:{type(exc).__name__}"
        err = (f"Registration identity unavailable for PID {pid}: no single-channel "
               f"handle capture (ticks/exe/creator) - refusing to register "
               f"(stage={capture_diag.get('stage', 'unknown')}, "
               f"detail={capture_diag.get('detail', 'unavailable')}; "
               f"UNVERIFIED_IDENTITY_OBSERVATION pid_exists={exists_hint})")
        record_observation_error(root, err)
        raise RuntimeError(err)

    if expected is None:
        err = (f"Required ownership anchor missing for PID {pid}: the earliest "
               f"observation capture failed or was never taken - refusing to "
               f"register from a later capture alone")
        record_observation_error(root, err)
        raise RuntimeError(err)
    if (identity["create_ticks"] != expected.get("create_ticks")
            or identity["exe"] != expected.get("exe")
            or identity["creator_pid"] != expected.get("creator_pid")):
        err = (f"Ownership anchor mismatch for PID {pid}: process replaced between "
               f"earliest observation and registration - refusing to adopt occupant")
        record_observation_error(root, err)
        raise RuntimeError(err)

    harness_pid = os.getpid()
    creator = identity["creator_pid"]
    creator_ticks: Optional[int] = None
    if expected_creator is not None:
        if creator != expected_creator:
            err = (f"Creator mismatch for PID {pid}: expected creator {expected_creator}, "
                   f"observed {creator} - refusing to register")
            record_observation_error(root, err)
            raise RuntimeError(err)
        if creator != harness_pid:
            parent = _verify_registered_parent(owned, creator)
            if parent is None:
                err = (f"Expected creator {creator} of PID {pid} is not a registered, "
                       f"currently verified parent - refusing to register")
                record_observation_error(root, err)
                raise RuntimeError(err)
            creator_ticks = parent["create_ticks"]
    elif creator != harness_pid:
        parent = _verify_registered_parent(owned, creator)
        if parent is None:
            err = (f"Creator {creator} of PID {pid} is neither the harness nor a "
                   f"registered, currently verified parent - refusing to register")
            record_observation_error(root, err)
            raise RuntimeError(err)
        creator_ticks = parent["create_ticks"]
    if creator_ticks is not None and identity["create_ticks"] < creator_ticks:
        err = f"PID {pid} predates its verified parent - refusing to register"
        record_observation_error(root, err)
        raise RuntimeError(err)

    owned.append({"pid": pid, "create_time": create_time,
                  "create_ticks": identity["create_ticks"], "exe": identity["exe"],
                  "creator_pid": creator, "creator_ticks": creator_ticks,
                  "observed_exe": ps_exe, "cmdline": cmdline})
    try:
        owned_file.write_text(json.dumps(owned, indent=2), encoding="utf-8")
    except Exception as exc:
        record_observation_error(root, f"Failed to write owned_processes.json: {exc}")
        raise


def scan_and_record_descendants(root: Path, parent_pid: int) -> None:
    """Scan and record child processes of parent PID.

    The parent must be a registered, currently re-verified owned process
    before any enumeration is trusted; psutil children() freshness alone is
    not proof (enumeration can race). Each discovered child is anchored at
    scan time (single-channel capture) and registered with that exact anchor
    plus its verified creator. A child that fails registration records an
    observation error (failing later cleanup closed) without aborting the
    remaining scan."""
    if parent_pid <= 0:
        err = f"Invalid parent PID {parent_pid} for descendant scan"
        record_observation_error(root, err)
        raise ValueError(err)
    output_dir = Path(root) / "output"
    owned_file = output_dir / "owned_processes.json"
    try:
        owned = json.loads(owned_file.read_text(encoding="utf-8")) if owned_file.exists() else []
    except Exception as exc:
        err = f"owned metadata unreadable for parent verification: {exc}"
        record_observation_error(root, err)
        raise RuntimeError(err)
    if _verify_registered_parent(owned, parent_pid) is None:
        err = f"Parent PID {parent_pid} is not a registered, currently verified owned process"
        record_observation_error(root, err)
        raise RuntimeError(err)
    try:
        import psutil
        frontier = [parent_pid]
        seen = {parent_pid}
        while frontier:
            current = frontier.pop(0)
            try:
                children = psutil.Process(current).children(recursive=False)
            except Exception as exc:
                record_observation_error(root, f"psutil children enumeration failed for PID {current}: {exc}")
                continue
            for child in children:
                if child.pid in seen:
                    continue
                seen.add(child.pid)
                anchor_diag: Dict[str, str] = {}
                anchor = _capture_registration_identity(child.pid, anchor_diag)  # earliest observation
                if anchor is None:
                    record_observation_error(
                        root, f"descendant anchor capture failed for PID {child.pid} "
                              f"(discovered under parent {current}; stage={anchor_diag.get('stage', 'unknown')}, "
                              f"detail={anchor_diag.get('detail', 'unavailable')}); not registered")
                    continue
                try:
                    record_owned_process(root, child.pid, expected=anchor, expected_creator=current)
                except Exception as exc:
                    record_observation_error(root, f"descendant registration failed for PID {child.pid}: {exc}")
                    continue
                frontier.append(child.pid)
    except Exception as exc:
        record_observation_error(root, f"psutil scan children failed for PID {parent_pid}: {exc}")
        raise

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


def verify_calculator_static(workspace: Path, before_src: str) -> Dict[str, bool]:
    """Static-only checks on the model-written calculator.py.

    Never executes, imports or compiles the file: ast.parse is pure syntax
    analysis, so model-written code cannot run on the runner from this path.
    Behavioral coverage is NOT_TESTED by design."""
    calc_path = workspace / "calculator.py"
    facts = {"syntax_ok": False, "add_preserved": False, "multiply_changed": False}
    if not calc_path.is_file():
        return facts
    try:
        tree_old = ast.parse(before_src)
        tree_new = ast.parse(calc_path.read_text(encoding="utf-8"))
        facts["syntax_ok"] = True
    except Exception:
        return facts

    def _fn(tree, name):
        return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)

    add_old, add_new = _fn(tree_old, "add"), _fn(tree_new, "add")
    facts["add_preserved"] = bool(add_old is not None and add_new is not None
                                  and ast.dump(add_old, include_attributes=False) == ast.dump(add_new, include_attributes=False))
    mult_old, mult_new = _fn(tree_old, "multiply"), _fn(tree_new, "multiply")
    facts["multiply_changed"] = bool(mult_old is not None and mult_new is not None
                                     and ast.dump(mult_old, include_attributes=False) != ast.dump(mult_new, include_attributes=False))
    return facts


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
        # Root ownership is established BEFORE the reader starts and before
        # ANY protocol input: the earliest post-spawn identity anchor is
        # captured immediately and registration must exactly match it.
        # Failure sets root_ownership_failed and closes the session, so no
        # prompt can be driven without a cleanup-valid root identity.
        try:
            anchor = _capture_registration_identity(self.pid) if self.pid > 0 else None
            if self.pid > 0 and anchor is None:
                raise RuntimeError(
                    f"root anchor capture failed for PID {self.pid} "
                    f"(single-channel ticks/exe/creator unavailable)")
            record_owned_process(self.root, self.pid, expected=anchor,
                                 expected_creator=os.getpid())
        except Exception as exc:
            self.error_flag = f"root_ownership_failed: {exc}"
            record_observation_error(self.root, self.error_flag)
            self.close()
            return
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

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

    def _scan_descendants(self) -> bool:
        if self.error_flag:
            return False
        if self.pid > 0:
            try:
                scan_and_record_descendants(self.root, self.pid)
            except Exception as exc:
                self.error_flag = f"descendant_observation_error: {exc}"
                return False
        return True

    # The app's full multi-line run_tests trust prompt, matched as a whole block,
    # never a bare suffix. The harness never answers it: consent is human-only,
    # generic PTY text is not authentication, and even a fully spoofed block only
    # causes a fail-safe row coverage gap plus a fresh session - never execution.
    _CONSENT_BLOCK_RE = re.compile(
        r"Nexus wants to run pytest in:[\s\S]*?\[o\] once[\s\S]*?\[a\] always[\s\S]*?\[N\] no[\s\S]*?Allow\?\s*\[o/a/N\]:\s*$")

    def wait_for_ready_prompt(self, timeout: float, deadline: float, prompt_pattern: str = r"^>>\s*You:?\s*$") -> Tuple[str, bool, bool]:
        """Wait for the ready prompt; returns (output chunk, ready, consent_wait).

        consent_wait is True when the app's run_tests trust prompt blocked the
        turn. The harness sends nothing, so the real consent gate is untouched
        and the caller records a coverage gap instead of waiting out the timeout."""
        start = time.monotonic()
        buf = ""
        target_deadline = min(deadline, start + timeout)
        while time.monotonic() < target_deadline:
            if self.error_flag:
                return buf, False, False
            if hasattr(self.pty, "isalive") and not self.pty.isalive() and self.out_queue.empty():
                break
            try:
                chunk = self.out_queue.get(timeout=0.05)
                buf += chunk
                self.all_output += chunk
                stripped = assess.strip_ansi(buf)
                if self._CONSENT_BLOCK_RE.search(stripped):
                    return buf, False, True
                lines = [l.strip() for l in stripped.splitlines() if l.strip()]
                if lines and re.match(prompt_pattern, lines[-1]):
                    if not self._scan_descendants():
                        return buf, False, False
                    return buf, True, False
            except queue.Empty:
                if not self._scan_descendants():
                    return buf, False, False
                continue
        stripped = assess.strip_ansi(buf)
        if self._CONSENT_BLOCK_RE.search(stripped):
            return buf, False, True
        lines = [l.strip() for l in stripped.splitlines() if l.strip()]
        found = bool(lines and re.match(prompt_pattern, lines[-1])) and not bool(self.error_flag)
        if not self._scan_descendants():
            return buf, False, False
        return buf, found, False

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
        if self.error_flag:
            raise RuntimeError(f"Cannot send input after session failure: {self.error_flag}")
        self.pty.write(text)

    def close(self) -> None:
        self.stop_event.set()
        try:
            if hasattr(self.pty, "isalive") and self.pty.isalive():
                self.pty.close()
        except Exception:
            pass


_IS_WINDOWS = os.name == "nt"  # test seam: monkeypatch this, never os.name


def _valid_owned_entry(entry: Any) -> bool:
    """Strict owned-identity shape.

    type(pid) is exactly int (bool rejected) and positive; create_time is a
    non-bool finite positive number; create_ticks is exactly int (bool
    rejected) and positive - the exact handle-derived FILETIME identity
    recorded at registration, never a float approximation; exe is a non-empty
    recorded path; creator_pid is exactly int (bool rejected) and positive;
    creator_ticks is None (harness root) or exactly int positive (verified
    registered parent). Older metadata without create_ticks or creator_pid,
    and anything else less, fails closed with zero kills - a partial
    identity is never targeted."""
    import math
    return (isinstance(entry, dict)
            and type(entry.get("pid")) is int and entry["pid"] > 0
            and type(entry.get("create_time")) in (int, float)
            and math.isfinite(entry["create_time"]) and entry["create_time"] > 0
            and type(entry.get("create_ticks")) is int and entry["create_ticks"] > 0
            and isinstance(entry.get("exe"), str) and bool(entry["exe"].strip())
            and type(entry.get("creator_pid")) is int and entry["creator_pid"] > 0
            and (entry.get("creator_ticks") is None
                 or (type(entry.get("creator_ticks")) is int and entry["creator_ticks"] > 0)))


def _read_strict_owned_metadata(root: Path, result: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Load owned-process metadata fail-closed: missing, unreadable, empty or any
    malformed entry records an error and returns None so nothing is killed."""
    try:
        owned = json.loads((root / "output" / "owned_processes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["errors"].append("owned metadata unreadable: " + type(exc).__name__)
        return None
    if not isinstance(owned, list) or not owned or any(not _valid_owned_entry(e) for e in owned):
        result["errors"].append("owned metadata missing, empty or malformed")
        return None
    return owned


def _observation_errors_present(root: Path, result: Dict[str, Any]) -> bool:
    """Any recorded observation error (or unreadable record) fails closed."""
    obs_file = root / "output" / "observation_errors.json"
    if not obs_file.exists():
        return False
    try:
        loaded = json.loads(obs_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["errors"].append("observation errors unreadable: " + type(exc).__name__)
        return True
    if loaded:
        result["errors"].append("observation errors recorded; failing closed")
        return True
    return False


def _load_kernel32():
    """kernel32 with explicit 64-bit-safe HANDLE/DWORD/BOOL/UINT signatures
    and use_last_error on every call. Raises on non-Windows."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    LPFILETIME = ctypes.POINTER(wintypes.FILETIME)
    LPDWORD = ctypes.POINTER(wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, LPFILETIME, LPFILETIME, LPFILETIME, LPFILETIME]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, LPDWORD]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


def _last_win_error() -> Any:
    """Calling thread's last error code after a use_last_error call. Falls
    back to 0 only where the platform exposes no last-error facility - the
    Win32 paths here are Windows-gated in production, so the fallback is
    reachable solely by non-Windows test rigs."""
    import ctypes
    getter = getattr(ctypes, "get_last_error", None)
    return getter() if getter is not None else 0


def _read_creation_ticks(kernel32: Any, handle: Any, _diag: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Exact integer FILETIME creation ticks for the process pinned by handle,
    or None on OS failure. Integer 100-ns ticks - no float conversion, so an
    exact equality check against the registration record is meaningful.
    _diag, when given, receives the failure stage and native error for
    receipts only; it never changes the fail-closed outcome."""
    import ctypes
    from ctypes import wintypes
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                    ctypes.byref(kernel), ctypes.byref(user)):
        if _diag is not None:
            _diag.update({"stage": "ticks", "detail": f"GetLastError={_last_win_error()}"})
        return None
    return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)


def _load_ntdll():
    """ntdll with an explicit typed NtQueryInformationProcess binding:
    NTSTATUS (signed LONG) result, full argument types, use_last_error."""
    import ctypes
    from ctypes import wintypes
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtQueryInformationProcess.restype = wintypes.LONG
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG)]
    return ntdll


def _process_basic_information_type():
    """PROCESS_BASIC_INFORMATION per the Windows SDK winternl.h layout
    (Reserved1, PebBaseAddress, Reserved2[2], UniqueProcessId,
    InheritedFromUniqueProcessId), using platform-native ctypes alignment -
    no manual packing."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [("Reserved1", wintypes.LPVOID),
                    ("PebBaseAddress", wintypes.LPVOID),
                    ("Reserved2", wintypes.LPVOID * 2),
                    ("UniqueProcessId", wintypes.LPVOID),
                    ("InheritedFromUniqueProcessId", wintypes.LPVOID)]
    return PROCESS_BASIC_INFORMATION


def _pbi_layout_supported(pbi: Any) -> bool:
    """Native sizeof/offset gate for the documented layout: x64 sizeof 48 with
    UniqueProcessId at 32 and InheritedFromUniqueProcessId at 40; x86 sizeof
    24 with offsets 16/20. Any drift fails closed."""
    import ctypes
    import struct
    if struct.calcsize("P") == 8:
        return (ctypes.sizeof(pbi) == 48
                and pbi.UniqueProcessId.offset == 32
                and pbi.InheritedFromUniqueProcessId.offset == 40)
    return (ctypes.sizeof(pbi) == 24
            and pbi.UniqueProcessId.offset == 16
            and pbi.InheritedFromUniqueProcessId.offset == 20)


def _read_creator_pid(ntdll: Any, handle: Any, pid: int, _diag: Optional[Dict[str, str]] = None) -> Optional[int]:
    """Creator (InheritedFromUniqueProcessId) of the process pinned by handle.

    Accepts only STATUS_SUCCESS (0), a full-structure ReturnLength, a
    supported native layout, UniqueProcessId == the opened pid (drift/wrong
    process guard), and a positive creator. Everything else fails closed.
    _diag, when given, records which guard failed for receipts only; it never
    changes the fail-closed outcome."""
    def _fail(detail: str) -> None:
        if _diag is not None:
            _diag.update({"stage": "creator", "detail": detail})
    try:
        import ctypes
        from ctypes import wintypes
        pbi = _process_basic_information_type()
        if not _pbi_layout_supported(pbi):
            _fail("native PROCESS_BASIC_INFORMATION layout unsupported")
            return None
        info = pbi()
        returned = wintypes.ULONG(0)
        status = ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(info),
                                                 ctypes.sizeof(info), ctypes.byref(returned))
        if status != 0:
            _fail(f"NTSTATUS={status} (0x{status & 0xFFFFFFFF:08X})")
            return None
        if returned.value != ctypes.sizeof(info):
            _fail(f"ReturnLength={returned.value} expected={ctypes.sizeof(info)}")
            return None
        unique = int(info.UniqueProcessId or 0)
        creator = int(info.InheritedFromUniqueProcessId or 0)
        if unique != pid:
            _fail(f"UniqueProcessId={unique} != opened pid {pid}")
            return None
        if creator <= 0:
            _fail(f"nonpositive creator={creator}")
            return None
        return creator
    except Exception as exc:
        _fail(f"exception:{type(exc).__name__}")
        return None


def _read_image_path(kernel32: Any, handle: Any, _diag: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Canonical image path of the process pinned by handle, or None.
    _diag, when given, receives the failure stage and native error for
    receipts only; it never changes the fail-closed outcome."""
    import ctypes
    from ctypes import wintypes
    size = wintypes.DWORD(1024)
    buf = ctypes.create_unicode_buffer(size.value)
    if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        if _diag is not None:
            _diag.update({"stage": "image", "detail": f"GetLastError={_last_win_error()}"})
        return None
    return buf.value


def _capture_registration_identity(pid: int, _diag: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Single-channel registration identity: exact integer FILETIME creation
    ticks, the canonical image path, AND the creator pid, all read through
    ONE stable open handle (QUERY_LIMITED_INFORMATION only - no termination
    rights). The handle pins the process object, so ticks, exe and creator
    always describe one real process and can never be mixed across a
    recycled occupant. None on non-Windows, any OS failure, any layout or
    status failure, or any CloseHandle failure (lifecycle unaccounted);
    callers fail closed on None. _diag, when given, records which capture
    stage failed (open/ticks/image/creator/close/exception) with the native
    error for receipts only; it never changes the fail-closed outcome."""
    if not _IS_WINDOWS:
        return None
    try:
        kernel32 = _load_kernel32()
        ntdll = _load_ntdll()
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            if _diag is not None:
                _diag.update({"stage": "open", "detail": f"GetLastError={_last_win_error()}"})
            return None
        close_ok = True
        try:
            ticks = _read_creation_ticks(kernel32, handle, _diag)
            exe = _read_image_path(kernel32, handle, _diag) if ticks is not None else None
            creator = _read_creator_pid(ntdll, handle, pid, _diag) if exe else None
        finally:
            if not kernel32.CloseHandle(handle):
                close_ok = False
                if _diag is not None:
                    _diag.update({"stage": "close", "detail": f"GetLastError={_last_win_error()}"})
        # No identity record unless the handle lifecycle is fully accounted.
        if not close_ok or ticks is None or not exe or creator is None:
            return None
        return {"create_ticks": ticks, "exe": exe, "creator_pid": creator}
    except Exception as exc:
        if _diag is not None:
            _diag.update({"stage": "exception", "detail": f"exception:{type(exc).__name__}"})
        return None


def _terminate_verified(entry: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """Terminate exactly the recorded identity via a stable Windows process handle.

    Opens a handle to the PID FIRST, then verifies identity THROUGH THAT
    HANDLE (which pins the process object): the exact integer FILETIME
    creation ticks recorded at ownership registration must equal the handle's
    creation ticks exactly (no float, no tolerance), and the image path must
    match exactly. Only then is THAT handle terminated. A PID recycled after
    recording never matches, and there is no check-then-kill TOCTOU window
    because identity is bound to the handle, not the number. Only this exact
    identity is terminated - no tree adoption. Fails closed (records an
    error, returns False) whenever a stable handle, handle-based
    verification, or handle cleanup cannot be accounted for - a
    CloseHandle failure means no success claim. A True return is not exit
    proof: TerminateProcess is asynchronous and the outer poll/receipt gate
    supplies verified exit."""
    if not _IS_WINDOWS:
        result["errors"].append("stable process handle unavailable on this platform")
        return False
    try:
        import ctypes
        kernel32 = _load_kernel32()
        handle = kernel32.OpenProcess(0x1000 | 0x0001, False, entry["pid"])  # QUERY_LIMITED_INFORMATION | TERMINATE
        if not handle:
            result["errors"].append(f"OpenProcess failed for pid {entry['pid']}: error {_last_win_error()}")
            return False
        try:
            handle_create_ticks = _read_creation_ticks(kernel32, handle)
            if handle_create_ticks is None:
                result["errors"].append(f"GetProcessTimes failed for pid {entry['pid']}: error {_last_win_error()}")
                return False
            if handle_create_ticks != entry.get("create_ticks"):
                result["errors"].append(f"handle identity mismatch for pid {entry['pid']}: exact creation ticks differ")
                return False
            handle_exe = _read_image_path(kernel32, handle)
            if handle_exe is None:
                result["errors"].append(f"QueryFullProcessImageName failed for pid {entry['pid']}: error {_last_win_error()}")
                return False
            if handle_exe != entry["exe"]:
                result["errors"].append(f"handle identity mismatch for pid {entry['pid']}: image path differs")
                return False
            if not kernel32.TerminateProcess(handle, 1):
                result["errors"].append(f"TerminateProcess failed for pid {entry['pid']}: error {_last_win_error()}")
                return False
        finally:
            # No success claim before the handle lifecycle is accounted for:
            # a CloseHandle failure records diagnostics and fails closed.
            close_error = None
            if not kernel32.CloseHandle(handle):
                close_error = _last_win_error()
                result["errors"].append(f"CloseHandle failed for pid {entry['pid']}: error {close_error}")
        if close_error is not None:
            return False
        # True only means: identity verified THROUGH the handle and termination
        # issued on that exact handle with its lifecycle fully accounted.
        # TerminateProcess is asynchronous - this is NOT exit proof; the outer
        # poll and receipt gate supply verified exit before any handoff.
        return True
    except Exception as exc:
        result["errors"].append("stable handle termination: " + type(exc).__name__)
        return False


def _mark_rows_incomplete(protocol, target_rows, start_idx: int, message: str, detail: Any = None) -> None:
    for r in range(start_idx, len(protocol)):
        if target_rows[r]["task"]["status"] == assess.NOT_RUN:
            evidence: List[Any] = [message] + ([detail] if detail else [])
            target_rows[r] = {"name": protocol[r]["name"], "task": {"status": assess.INCOMPLETE, "evidence": evidence}}


def orchestrate_task_sessions(run_session, terminate_blocked, protocol, target_rows,
                              max_sessions: int = 3, should_stop=None) -> bool:
    """Tasks-protocol restart orchestration.

    run_session(session_num, start_idx) -> (completed, restart_idx|None).
    terminate_blocked(session_num) is invoked ONLY after a consent-blocked
    session, before the fresh one launches, and must return a mapping with
    "terminated" (the persisted receipt). A fresh session launches solely
    after verified termination. Bounded by max_sessions; on cap, stop
    condition or termination failure the remaining rows are marked INCOMPLETE
    (never PASS) in their original order."""
    start_idx = 0
    for session_num in range(1, max_sessions + 1):
        if should_stop is not None and should_stop():
            return False
        if session_num > 1:
            term = terminate_blocked(session_num)
            if not term.get("terminated"):
                _mark_rows_incomplete(protocol, target_rows, start_idx,
                                      "Blocked session owned processes could not be stopped and exit-verified; "
                                      "fresh session not launched over a live uncontrolled process", term)
                return False
        seg_completed, restart_idx = run_session(session_num, start_idx)
        if restart_idx is not None:
            start_idx = restart_idx
            continue
        return bool(seg_completed)
    _mark_rows_incomplete(protocol, target_rows, start_idx, "Consent-prompt session restart limit reached")
    return False


def terminate_owned_processes(
                              root: Path, byte_tracker: List[int], receipt_name: str,
                              pty: Any = None, settle_seconds: float = 10.0) -> Dict[str, Any]:
    """Stop recorded owned processes for root, verify exit, persist a receipt.

    Kills ONLY positively established identities from strictly validated
    root/output/owned_processes.json (exact handle-derived integer creation
    ticks + recorded exe verified THROUGH the kill handle BEFORE each kill). No ambient process discovery or adoption. PID reuse is
    safe: a mismatched occupant is never killed. Missing/malformed/empty
    metadata or any recorded observation error fails closed with zero kills.
    The receipt (kills, mismatches, survivors, PTY closure proof) is persisted
    to evidence before returning, so a fresh session launches only after the
    blocked session's closure and exit are proven on disk. Workspace and
    evidence are preserved."""
    root = Path(root).resolve()
    result: Dict[str, Any] = {"terminated": False, "killed_pids": [], "remaining": [],
                              "skipped_identity_mismatch": [], "pty_closed": None, "errors": []}

    if pty is not None:
        alive = bool(getattr(pty.pty, "isalive", lambda: False)())
        result["pty_closed"] = not alive
        if alive:
            result["errors"].append("blocked session PTY still open")

    owned = _read_strict_owned_metadata(root, result)
    if owned is not None:
        _observation_errors_present(root, result)
    try:
        import psutil
    except ImportError:
        result["errors"].append("psutil unavailable")
        psutil = None
    if not _IS_WINDOWS:
        result["errors"].append("Windows cleanup not exercised on this platform")

    if owned is not None and psutil is not None and _IS_WINDOWS and not result["errors"]:
        for entry in owned:
            try:
                candidate = psutil.Process(entry["pid"])
                if candidate.create_time() != entry["create_time"] or \
                        str(candidate.exe()) != entry["exe"]:
                    # Recorded identity already gone; PID occupant is not ours - never kill it.
                    result["skipped_identity_mismatch"].append(entry["pid"])
                    continue
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                result["errors"].append("pre-kill inspection: " + type(exc).__name__)
                continue
            # Terminate exactly this recorded identity through a stable handle
            # verified after opening: no PID-based taskkill TOCTOU, no /T tree
            # adoption (descendants are recorded identities of their own).
            if _terminate_verified(entry, result):
                result["killed_pids"].append(entry["pid"])

        deadline = time.monotonic() + settle_seconds
        while not result["errors"]:
            remaining: List[int] = []
            verify_failed = False
            for entry in owned:
                try:
                    process = psutil.Process(entry["pid"])
                    if process.create_time() == entry["create_time"] and process.is_running():
                        remaining.append(entry["pid"])
                except psutil.NoSuchProcess:
                    pass
                except Exception:
                    verify_failed = True
            if verify_failed:
                result["errors"].append("exit verification failed")
                break
            if not remaining:
                result["terminated"] = True
                break
            if time.monotonic() >= deadline:
                result["remaining"] = remaining
                break
            time.sleep(0.2)

    receipt = {"receipt": receipt_name, "terminated": result["terminated"],
               "killed_pids": result["killed_pids"],
               "skipped_identity_mismatch": result["skipped_identity_mismatch"],
               "remaining": result["remaining"], "pty_closed": result["pty_closed"],
               "errors": result["errors"], "recorded_at": time.time()}
    try:
        charge_and_write(root / "output" / f"termination_{receipt_name}.json",
                         json.dumps(receipt, indent=2).encode("utf-8"), byte_tracker)
    except Exception as exc:
        result["errors"].append("receipt persist failed: " + type(exc).__name__)
        result["terminated"] = False
    return result


def cleanup_root(root: Path | str) -> Dict[str, Any]:
    """Clean up owned processes recorded under root. Never startup model or kill unrelated trees."""
    root = Path(root).resolve()
    output_dir = root / "output"
    owned_file = output_dir / "owned_processes.json"
    obs_errors_file = output_dir / "observation_errors.json"
    killed, errors, remaining, remaining_llama = [], [], [], []

    if obs_errors_file.exists():
        try:
            loaded_errs = json.loads(obs_errors_file.read_text(encoding="utf-8"))
            if isinstance(loaded_errs, list):
                errors.extend(str(e) for e in loaded_errs)
            elif loaded_errs:
                errors.append(str(loaded_errs))
        except (OSError, ValueError) as exc:
            errors.append(f"observation errors unreadable: {type(exc).__name__}")

    strict: Dict[str, Any] = {"errors": []}
    if not owned_file.exists():
        owned = []
        errors.append("missing owned metadata")
    else:
        loaded = _read_strict_owned_metadata(root, strict)
        if loaded is None:
            owned = []
            errors.extend(strict["errors"])
        else:
            owned = loaded
    try:
        import psutil
    except ImportError:
        psutil = None
        errors.append("psutil unavailable")
    if not _IS_WINDOWS:
        errors.append("Windows cleanup not exercised on this platform")

    if psutil is not None and _IS_WINDOWS:
        # Only positively established recorded identities are killed: exact
        # create_time + recorded exe checked BEFORE every kill. No ambient
        # server discovery or adoption; an unrecorded process is never touched.
        for entry in owned:
            try:
                process = psutil.Process(entry["pid"])
                if process.create_time() != entry["create_time"] or \
                        str(process.exe()) != entry["exe"]:
                    continue  # recorded identity gone; PID occupant is not ours
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                errors.append("pre-kill inspection: " + type(exc).__name__)
                continue
            # Stable-handle termination of exactly this recorded identity: no
            # PID-based taskkill TOCTOU, no /T tree adoption.
            term_result: Dict[str, Any] = {"errors": []}
            if _terminate_verified(entry, term_result):
                killed.append({"pid": entry["pid"], "method": "verified_handle"})
            errors.extend(term_result["errors"])
        try:
            time.sleep(0.5)
            for entry in owned:
                try:
                    process = psutil.Process(entry["pid"])
                    if process.create_time() == entry["create_time"] and process.is_running():
                        remaining.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
        except Exception as exc:
            errors.append("owned process final check: " + type(exc).__name__)
    result = {"confirmed": bool(owned) and not errors and not remaining and not remaining_llama,
              "owned_count": len(owned), "killed_pids": killed, "remaining_owned": remaining,
              "remaining_llama_servers": remaining_llama, "errors": errors}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleanup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _execute_protocol_session(protocol: List[Dict[str, Any]], session_id: str, pty: PtySession,
                              workspace: Path, home: Path, history_db: Path, deadline: float,
                              is_tasks: bool, output_dir: Path, target_rows: List[Dict[str, Any]],
                              byte_tracker: List[int], on_after_first_real_task: Any = None,
                              start_idx: int = 0) -> Tuple[bool, Optional[int]]:
    """Execute a declared protocol table in-place, preserving row IDs and stopping on first timeout.

    Returns (completed, restart_idx). restart_idx is set when a row stopped at
    the app's run_tests trust prompt: the harness records that row as a consent
    coverage gap (never a pass, never answered) and the caller continues the
    remaining rows in a fresh session so one expected prompt cannot consume the
    whole matrix."""
    session_stopped = False
    first_real_checked = False
    context_metrics_before: Dict[str, Any] = {}

    for idx, step in enumerate(protocol):
        if idx < start_idx:
            continue
        name, kind, prompt, timeout = step["name"], step["kind"], step["prompt"], step["timeout"]
        if session_stopped or pty.error_flag or time.monotonic() >= deadline:
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

        raw_chunk, ready, consent_wait = pty.wait_for_ready_prompt(timeout=min(timeout, rem), deadline=deadline)
        after_db = fetch_sqlite_messages(history_db, session_id)
        after_files = snapshot_workspace_files(workspace)
        delta_msgs = canonical_diff(before_db, after_db)
        assistant_ans = assess.extract_assistant_turn_answer(delta_msgs)
        user_msg = next((m for m in delta_msgs if m.get("role") == "user"), None)

        if consent_wait:
            target_rows[idx] = {"name": name,
                                "task": {"status": assess.INCOMPLETE,
                                         "evidence": ["consent_wait_coverage_gap: prompt-shaped output observed (shape only, not an "
                                                      "authenticated request); no approval, grant or any other input was sent. "
                                                      "Session closure and owned-process exit are verified separately and persisted "
                                                      "in output/termination_*.json before any fresh launch", pty.error_flag]},
                                "raw_chunk": raw_chunk, "canonical_before": before_db, "canonical_after": after_db,
                                "known_files_before": before_files, "known_files_after": after_files}
            charge_and_write(output_dir / "rows" / f"{session_id}_{idx:02d}_{name}.json",
                             json.dumps(target_rows[idx], indent=2).encode(), byte_tracker)
            return False, idx + 1

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
            static_facts = verify_calculator_static(workspace, before_files.get("calculator.py", ""))
            row_res = assess.assess_task_calculator_fix(static_facts, before_files.get("calculator.py", ""), after_files.get("calculator.py", ""), delta_msgs)
        elif name == "row-06-web-search":
            row_res = assess.assess_task_web_search(assistant_ans, delta_msgs)
        elif name == "compact-turn-02-brief":
            row_res = assess.assess_task_brief(assistant_ans, delta_msgs)
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
            try:
                if not on_after_first_real_task():
                    session_stopped = True
            except Exception:
                pass

    return not session_stopped, None


def capture_backend_identity(root: Path, output_dir: Path, deadline: float, source_root: Path) -> Dict[str, Any]:
    """Capture bounded runtime backend executable, PID, create_time, hash and version.

    This records facts of the actual running server executable while it is alive,
    distinct from any warmup or model invocation.
    """
    meta_file = output_dir / "owned_processes.json"
    backend_meta = {
        "complete": False,
        "server_found": False,
        "pid": None,
        "create_time": None,
        "exe": None,
        "argv": None,
        "exe_sha256": None,
        "version_output": None,
        "version_returncode": None,
        "error": None,
        "observation_errors": [],
    }
    try:
        if not meta_file.exists():
            backend_meta["error"] = "owned_processes.json missing"
            return backend_meta
        try:
            owned = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(owned, list):
                backend_meta["error"] = "owned_processes.json is not a list"
                return backend_meta
        except Exception as exc:
            backend_meta["error"] = f"Failed to read owned_processes: {exc}"
            return backend_meta

        import psutil

        server_entry = None
        for entry in owned:
            exe = entry.get("exe") or ""
            pid = entry.get("pid")
            if exe.lower().endswith(("llama-server.exe", "llama-server")) and pid:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running() and proc.create_time() == entry.get("create_time"):
                        server_entry = entry
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    backend_meta["observation_errors"].append({"pid": pid, "type": type(exc).__name__})
                    continue
                except Exception as exc:
                    backend_meta["observation_errors"].append({"pid": pid, "type": type(exc).__name__})
                    continue

        if not server_entry:
            backend_meta["error"] = "No active running llama-server process observed"
            return backend_meta

        pid = server_entry.get("pid")
        exe_path = server_entry.get("exe")
        backend_meta.update({
            "server_found": True,
            "pid": pid,
            "create_time": server_entry.get("create_time"),
            "exe": exe_path,
            "argv": server_entry.get("cmdline"),
        })

        if not exe_path or not Path(exe_path).is_file():
            backend_meta["error"] = f"Server executable missing at {exe_path}"
            return backend_meta

        try:
            backend_meta["exe_sha256"] = digest(Path(exe_path))
        except Exception as exc:
            backend_meta["error"] = f"Failed to digest exe: {exc}"
            return backend_meta

        remaining_sec = deadline - time.monotonic()
        if remaining_sec <= 0.0:
            backend_meta["error"] = "Work deadline expired before backend version query"
            return backend_meta

        v_timeout = min(10.0, remaining_sec)
        try:
            runner_path = Path(source_root) / ".github/qualification/runner.py"
            runner_mod = load_module(runner_path, "q1_runner_for_proc")
            owner = runner_mod.owned_module(source_root)
            domain_dir = root / "version-process"
            domain_dir.mkdir(parents=True, exist_ok=True)
            with owner.ProcessDomain(domain_dir, deadline=deadline) as domain:
                v_res = domain.run([exe_path, "--version"], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=v_timeout)
                raw_out = (v_res.stdout or v_res.stderr or "").strip()[:4096]
                backend_meta["version_output"] = raw_out
                backend_meta["version_returncode"] = v_res.returncode
                if v_res.returncode == 0:
                    backend_meta["complete"] = True
                else:
                    backend_meta["error"] = f"Version query returned nonzero code {v_res.returncode}"
        except Exception as exc:
            backend_meta["error"] = f"Version query failed: {type(exc).__name__}: {str(exc)[:256]}"
    except Exception as exc:
        backend_meta["error"] = f"Capture exception: {type(exc).__name__}: {str(exc)[:256]}"
    return backend_meta

def run_driver(args: argparse.Namespace) -> int:
    """Execute the full interactive qualification protocol."""
    source_root = Path(args.source_root).resolve(strict=True)
    root = Path(args.root).resolve()
    wheel = Path(args.wheel).resolve(strict=True)
    commit = args.commit
    tree = getattr(args, "tree", None)
    if not commit or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == "0" * 40:
        raise RuntimeError("Mandatory external approved commit seal is missing, unresolved, or zero; execution rejected")
    if not tree or not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree) or tree == "0" * 40:
        raise RuntimeError("Mandatory external approved tree seal is missing, unresolved, or zero; execution rejected")
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
    native_gate = verifier.check_native_evidence(native_evidence_path, commit=commit, tree=tree, wheel=wheel)

    runner_mod = load_module(runner_path, "q1_runner")
    installed_provenance = runner_mod.verify_installed(wheel, source_root, commit)
    if installed_provenance.get("commit") != commit:
        raise RuntimeError("Installed runtime commit differs from approved commit seal")
    if installed_provenance.get("tree") != tree:
        raise RuntimeError("Installed runtime tree differs from approved tree seal")

    if installed_provenance["package_py"] != native_gate["provenance"]["package_py"]:
        raise RuntimeError("Installed runtime package bytes differ from A native gate package_py")
    if installed_provenance["wheel_sha256"] != native_gate.get("wheel_sha256"):
        raise RuntimeError("Installed wheel SHA256 differs from native gate wheel_sha256")

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
    backend_identity = {"complete": False, "server_found": False, "attempts": [], "errors": []}

    def verify_initial_model() -> bool:
        nonlocal model_verified_initial
        m = home / MODEL_REL
        model_verified_initial = m.is_file() and digest(m) == MODEL_SHA256
        return model_verified_initial

    def on_after_first_real_task_hook() -> bool:
        ok = verify_initial_model()
        if not ok:
            return False
        if not backend_identity.get("complete", False):
            try:
                attempt = capture_backend_identity(root, output_dir, deadline, source_root)
            except Exception as exc:
                attempt = {"complete": False, "server_found": False, "error": f"Capture exception: {type(exc).__name__}: {str(exc)[:256]}", "observation_errors": []}
            backend_identity["attempts"].append(attempt)
            if attempt.get("complete", False):
                backend_identity.update(attempt)
            else:
                backend_identity["errors"].append(attempt.get("error", "Unknown error"))
        return True

    try:
        # Session 1: tasks. A row that stops at the app's run_tests trust prompt
        # is recorded as a consent coverage gap (human-only consent; the harness
        # never answers); the blocked session's owned processes are verifiably
        # terminated (receipt persisted) BEFORE the remaining rows continue in a
        # fresh session, so one expected prompt cannot consume the whole matrix.
        blocked_pty: List[Any] = [None]

        def _run_one_tasks_session(session_num: int, start_idx: int):
            nonlocal tty_probe_tasks
            s1_id = f"interactive-tasks-{session_num:03d}"
            argv1 = [sys.executable, "-I", "-X", "utf8", "-c", CONTROLLER_CODE,
                     "repl", "--provider", "local", "--context-size", "4096", "--max-output-tokens", "512",
                     "--max-iterations", "10", "--persist", "--session", s1_id, "--no-stream"]
            pty1 = PtySession(argv1, cwd=workspace, env=env, root=root, byte_tracker=byte_tracker)
            try:
                init_out1, ready1, _cw1 = pty1.wait_for_ready_prompt(timeout=STARTUP_TIMEOUT, deadline=deadline)
                if session_num == 1:
                    tty_probe_tasks = assess.parse_tty_probe(init_out1)
                if not ready1:
                    tasks_rows[start_idx] = {"name": PROTOCOL_TASKS[start_idx]["name"],
                                             "task": {"status": assess.INCOMPLETE, "evidence": ["Startup timeout"]}}
                    return False, None
                return _execute_protocol_session(
                    PROTOCOL_TASKS, s1_id, pty1, workspace, home, history_db,
                    deadline=deadline, is_tasks=True, output_dir=output_dir, target_rows=tasks_rows,
                    byte_tracker=byte_tracker,
                    on_after_first_real_task=on_after_first_real_task_hook if session_num == 1 else None,
                    start_idx=start_idx
                )
            finally:
                try:
                    log_name = "tasks.log" if session_num == 1 else f"tasks-{session_num}.log"
                    charge_and_write(transcripts_dir / log_name, pty1.all_output.encode("utf-8", errors="replace"), byte_tracker)
                finally:
                    pty1.close()
                    blocked_pty[0] = pty1

        def _terminate_blocked_session(session_num: int):
            return terminate_owned_processes(root, byte_tracker, f"after_session_{session_num - 1:03d}",
                                             pty=blocked_pty[0])

        s1_completed = orchestrate_task_sessions(
            _run_one_tasks_session, _terminate_blocked_session, PROTOCOL_TASKS, tasks_rows,
            max_sessions=3, should_stop=lambda: time.monotonic() >= deadline)

        # Session 2: compact (only if tasks session completed all turns and model verified)
        if s1_completed and model_verified_initial and time.monotonic() < deadline:
            s2_id = "interactive-compact-001"
            argv2 = [sys.executable, "-I", "-X", "utf8", "-c", CONTROLLER_CODE,
                     "repl", "--provider", "local", "--context-size", "4096", "--max-output-tokens", "512",
                     "--max-iterations", "10", "--persist", "--session", s2_id, "--no-stream"]
            pty2 = PtySession(argv2, cwd=workspace, env=env, root=root, byte_tracker=byte_tracker)
            try:
                init_out2, ready2, _cw2 = pty2.wait_for_ready_prompt(timeout=STARTUP_TIMEOUT, deadline=deadline)
                tty_probe_compact = assess.parse_tty_probe(init_out2)
                if ready2:
                    _execute_protocol_session(
                        PROTOCOL_COMPACT, s2_id, pty2, workspace, home, history_db,
                        deadline=deadline, is_tasks=False, output_dir=output_dir, target_rows=compact_rows,
                        byte_tracker=byte_tracker
                    )
                    if not backend_identity.get("complete", False):
                        try:
                            attempt = capture_backend_identity(root, output_dir, deadline, source_root)
                        except Exception as exc:
                            attempt = {"complete": False, "server_found": False, "error": f"Capture exception: {type(exc).__name__}: {str(exc)[:256]}", "observation_errors": []}
                        backend_identity["attempts"].append(attempt)
                        if attempt.get("complete", False):
                            backend_identity["post_exit_capture"] = attempt
                        else:
                            backend_identity["errors"].append(attempt.get("error", "Unknown error"))
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
        try:
            (output_dir / "backend_identity.json").write_text(json.dumps(backend_identity, indent=2), encoding="utf-8")
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

        complete = bool(summary.get("automated_core_pass", False) and backend_identity.get("complete", False))

        coverage_gaps = [
            "run_tests: consent_wait_coverage_gap possible (human-only consent; the harness never answers "
            "and no project-code execution is approved); affected rows record INCOMPLETE, never PASS",
            "row-05 calculator behavior: NOT_TESTED (harness execution/import of model-written files "
            "removed; static syntax/content checks only, best outcome REVIEW)",
            "run_code: refused probe-only via NEXUS_AGENT_DISABLE_RUN_CODE=1 (AST allowlist is not OS "
            "containment and this runner has no rlimits/Job Objects); sandboxed-computation behavior NOT_TESTED",
        ]

        results = {
            "commit": commit, "wheel": wheel.name, "native_evidence_verified": True,
            "coverage_gaps": coverage_gaps,
            "native_gate": {
                "commit": native_gate.get("commit"), "platform": native_gate.get("platform"),
                "success": native_gate.get("success"), "qualified": native_gate.get("qualified"),
                "untraced_native": native_gate.get("untraced_native"),
            },
            "installed_provenance": installed_provenance,
            "backend_identity": backend_identity,
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


def conpty_check(args: argparse.Namespace) -> int:
    """Non-model ConPTY compatibility proof through the production PtySession path.

    Spawns ONE benign child (this interpreter, held on stdin) through the
    production PtySession: the ownership anchor and creator registration must
    succeed BEFORE any I/O, which is only possible when ConPTY creates the
    child in-process (creator == this harness). A winpty-agent fallback or any
    ownership failure closes the session and fails this check. One bounded
    echo round is observed, then the live recorded child is terminated through
    production terminate_owned_processes with the receipt retained in
    root/output. Never a model, never network, never an unrelated process.
    This evidence is NOT interactive qualification and permits no model use."""
    root = Path(args.root).resolve()
    deadline = deadline_from_file(args.deadline_file)
    byte_tracker: List[int] = [0]
    evidence: Dict[str, Any] = {
        "platform": sys.platform,
        "conpty_compatibility_only": True,
        "interactive_qualification": False,
        "final_qualification": False,
        "model_execution_permitted": False,
        "external_execution_approval": False,
        "network_used": False,
        "checks": {},
        "errors": [],
    }

    def finish(code: int) -> int:
        out = root / "output"
        out.mkdir(parents=True, exist_ok=True)
        receipt = out / "termination_conpty_check.json"
        evidence["termination_receipt_sha256"] = (
            hashlib.sha256(receipt.read_bytes()).hexdigest()
            if receipt.is_file() else None)
        charge_and_write(out / "conpty-compat.json",
                         json.dumps(evidence, indent=2).encode("utf-8"), byte_tracker)
        print(json.dumps(evidence, indent=2))
        return code

    if not _IS_WINDOWS:
        evidence["errors"].append("ConPTY compatibility is only verifiable on Windows")
        return finish(1)
    if PtyProcess is None:
        evidence["errors"].append("pywinpty unavailable")
        return finish(1)
    build = sys.getwindowsversion().build  # Windows-only attribute
    evidence["windows_build"] = build
    evidence["conpty_expected_by_build"] = build >= 17763

    argv = [sys.executable, "-c",
            "import sys; print('conpty-echo-ok', flush=True); sys.stdin.readline()"]
    session = PtySession(argv, cwd=root, env=dict(os.environ), root=root,
                         byte_tracker=byte_tracker)
    try:
        if session.error_flag:
            evidence["errors"].append(session.error_flag)
            return finish(1)
        evidence["checks"]["ownership_registered_pre_io"] = True
        evidence["child_pid"] = session.pid
        read_back: Dict[str, Any] = {"errors": []}
        entries = _read_strict_owned_metadata(root, read_back)
        if read_back["errors"]:
            evidence["errors"].extend(read_back["errors"])
            return finish(1)
        entry = next((e for e in (entries or []) if e.get("pid") == session.pid), None)
        evidence["creator_pid"] = entry.get("creator_pid") if entry else None
        evidence["checks"]["creator_is_harness"] = bool(
            entry and entry.get("creator_pid") == os.getpid())
        if not evidence["checks"]["creator_is_harness"]:
            evidence["errors"].append(
                "PTY child creator is not this harness (agent-fallback backend?)")
            return finish(1)

        buf = ""
        echo_deadline = min(time.monotonic() + 60.0, deadline - 5)
        while time.monotonic() < echo_deadline and "conpty-echo-ok" not in buf:
            try:
                buf += session.out_queue.get(timeout=0.5)
            except queue.Empty:
                if session.error_flag:
                    break
        evidence["checks"]["echo_observed"] = "conpty-echo-ok" in buf
        if not evidence["checks"]["echo_observed"]:
            evidence["errors"].append(
                session.error_flag or "bounded echo not observed")
            return finish(1)

        term = terminate_owned_processes(root, byte_tracker, "conpty_check",
                                         settle_seconds=5.0)
        evidence["termination"] = {k: term[k] for k in
                                   ("terminated", "killed_pids",
                                    "skipped_identity_mismatch", "remaining", "errors")}
        evidence["checks"]["terminated"] = term["terminated"] is True
        if not evidence["checks"]["terminated"]:
            return finish(1)
        return finish(0)
    finally:
        session.close()


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
    run_parser.add_argument("--tree", required=True)

    cleanup_parser = sub.add_parser("cleanup", help="Clean up owned processes for root")
    cleanup_parser.add_argument("--root", required=True)

    conpty_parser = sub.add_parser("conpty-check",
                                   help="Non-model ConPTY compatibility proof (not qualification)")
    conpty_parser.add_argument("--root", required=True)
    conpty_parser.add_argument("--deadline-file", required=True)

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
    elif args.command == "conpty-check":
        try:
            return conpty_check(args)
        except Exception as e:
            print(f"CONPTY CHECK FAILED: {e}", file=sys.stderr)
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
