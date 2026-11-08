# proof_runner_probe.py
"""Nexus e2e — Minimal runner-envelope probe harness.

Demonstrates outer-bound runner mechanics before any proof execution:
  --mode timeout:
      Spawns cmd.exe child + ping.exe grandchild with detached process flags
      (CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS). Bounded polling (up to 10s)
      establishes grandchild creation (C9, C11) and process creation timestamps (C13)
      before continuous heartbeats begin. Exits nonzero (exit 3) if hazard shape
      is not established.
  --mode check-survivors:
      Executes post-kill in the same job (same VM) via if: always(). Two-phase
      observation: Phase A (immediate, record-only) and Phase B (after a bounded
      15s delay to observe whether the runner reaped the tree). Four-state tracking:
      confirmed / reused / unknown / exited. Kill-eligibility strictly limited to
      captured creation-pinned members verified on a single held handle (K1, K2).
      Observation closure preserves partial mid-scan tables observation-only (rev 8b)
      and seeds from all historical tracked PIDs plus accumulated discovered descendants.
      All late descendants classify unknown (never confirmed, never killed).
      Phase A never feeds the verdict (D2). Full-tree containment is never proven or claimed.
      Any survivor, unowned, reused, or inconclusive scan in Phase B/C fails closed.
  --mode verify-timeout:
      Polls GitHub Actions REST API with pagination and exact check_run_url suffix
      matching (C3). Empty check_run_id or name mismatch fails closed without fallback.
      Enforces heartbeat coverage reaching within one cadence of timeout duration (C4, C12);
      provides observational evidence of beats up to the boundary, not absolute proof
      of uninterrupted liveness.
  --mode leave:
      Creates local principal and atomically writes receipt (C7), intentionally
      leaving account active.
  --mode cleanup:
      Accepts --receipt; unconditionally sweeps/terminates processes for the receipt
      SID FIRST (even if account appears absent in SAM) before deleting the account (C7).
      Fails closed on missing or unreadable receipt.

Source Identity:
- Enforces on-host SHA-256 checks against EXPECTED_SHA256 before modifying system
  state or spawning unmanaged processes.

Status: SOURCE PREPARED, NOT RUN.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

EXPECTED_SHA256 = {
    "contain_core.py": "14d6a4fe28736db2b2a1694a82d8f308e0e624bbc3e68e4912207b06b30e06f0",
    "contain_core_selftest.py": "fbcdb4ea002d90568d1f391fbd65afceeb7246cf9ee064196e6c72d98beb676c",
}

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
ERROR_NO_MORE_FILES = 18
DEFAULT_HEARTBEAT_INTERVAL = 5.0


class FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


def _bind_probe_win32_apis(kernel32) -> None:
    """Declare probe-local Win32 API signatures not present in the frozen core."""
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _verify_hashes_or_abort() -> None:
    for name, want in EXPECTED_SHA256.items():
        target = SELF_DIR / name
        if not target.exists():
            print(f"ABORT: Required file '{name}' missing", flush=True)
            sys.exit(2)
        got = _sha256(target)
        if got != want:
            print(f"ABORT: SHA-256 mismatch for '{name}': want {want}, got {got}", flush=True)
            sys.exit(2)


def _persist_atomic_receipt(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _filetime_to_int(ft: FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def _query_process_identity_on_handle(
    kernel32, h_proc: wintypes.HANDLE
) -> tuple[int | None, bool | None, int]:
    """Queries (creation_time_int, is_still_active, gle) on an already-opened handle.

    Returns (None, None, gle) if GetProcessTimes or GetExitCodeProcess fails.
    Never guesses still_active on query failure.
    """
    ft_creation = FILETIME()
    ft_exit = FILETIME()
    ft_kernel = FILETIME()
    ft_user = FILETIME()
    if not kernel32.GetProcessTimes(
        h_proc,
        ctypes.byref(ft_creation),
        ctypes.byref(ft_exit),
        ctypes.byref(ft_kernel),
        ctypes.byref(ft_user),
    ):
        return None, None, ctypes.GetLastError()

    exit_code = wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(h_proc, ctypes.byref(exit_code)):
        return None, None, ctypes.GetLastError()

    still_active = exit_code.value == 259  # STILL_ACTIVE
    return _filetime_to_int(ft_creation), still_active, 0


def _query_process_identity_by_pid(
    kernel32, pid: int
) -> tuple[int | None, bool | None, int]:
    """Opens PROCESS_QUERY_LIMITED_INFORMATION and queries identity and liveness."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h_proc:
        return None, None, ctypes.GetLastError()
    try:
        return _query_process_identity_on_handle(kernel32, h_proc)
    finally:
        kernel32.CloseHandle(h_proc)


def _is_pid_in_fresh_snapshot(kernel32, pid: int) -> tuple[bool | None, int]:
    """Cross-checks ONE fresh snapshot to distinguish absent/exited from present-but-denied (G4).

    Returns (is_present: bool | None, gle: int).
    """
    import contain_core

    h_snap = kernel32.CreateToolhelp32Snapshot(contain_core.TH32CS_SNAPPROCESS, 0)
    if h_snap == contain_core.INVALID_HANDLE_VALUE or not h_snap:
        return None, ctypes.GetLastError()

    try:
        pe = contain_core.PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
        more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
        if not more:
            gle = ctypes.GetLastError()
            if gle != ERROR_NO_MORE_FILES:
                return None, gle
            return False, 0

        while more:
            if int(pe.th32ProcessID) == pid:
                return True, 0
            pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
            more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))
            if not more:
                gle = ctypes.GetLastError()
                if gle != ERROR_NO_MORE_FILES:
                    return None, gle
                break
        return False, 0
    finally:
        kernel32.CloseHandle(h_snap)


def _read_snapshot_processes(
    kernel32,
) -> tuple[bool, dict[int, dict], dict[int, list[int]], int]:
    """Enumerates all processes from a fresh Toolhelp32 snapshot.

    Returns (conclusive: bool, proc_table: dict, children_map: dict, gle: int).
    Delta 1 (rev 8b): Mid-scan Process32NextW failure preserves partial tables observation-only
    (conclusive=False, proc_table, children_map, scan_gle). Snapshot creation or first failure
    returns conclusive=False with empty tables.
    """
    import contain_core

    h_snap = kernel32.CreateToolhelp32Snapshot(contain_core.TH32CS_SNAPPROCESS, 0)
    if h_snap == contain_core.INVALID_HANDLE_VALUE or not h_snap:
        return False, {}, {}, ctypes.GetLastError()

    proc_table: dict[int, dict] = {}
    children_map: dict[int, list[int]] = {}
    scan_gle = 0

    try:
        pe = contain_core.PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
        more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
        if not more:
            gle = ctypes.GetLastError()
            return False, {}, {}, gle

        while more:
            pid = int(pe.th32ProcessID)
            ppid = int(pe.th32ParentProcessID)
            exe = pe.szExeFile
            proc_table[pid] = {"pid": pid, "ppid": ppid, "exe": exe}
            children_map.setdefault(ppid, []).append(pid)

            pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
            more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))
            if not more:
                gle = ctypes.GetLastError()
                if gle != ERROR_NO_MORE_FILES:
                    scan_gle = gle
                break
    finally:
        kernel32.CloseHandle(h_snap)

    if scan_gle != 0:
        return False, proc_table, children_map, scan_gle

    return True, proc_table, children_map, 0


def _scan_process_tree(
    kernel32,
    tracked_info: dict[int, dict],
    tracked_pids: set[int],
    observation_seed_pids: set[int] | None = None,
) -> tuple[bool, dict, str | None]:
    """Snapshot enumeration with four-state classification and read-only observation closure.

    States: confirmed / reused / unknown / exited.
    Delta 2 (rev 8b): On mid-scan failure, preserves partial tables and runs observation closure;
    every partial discovery is unknown (never kill-eligible) and scan remains conclusive=False.
    Returns (conclusive: bool, scan_result: dict, error_str: str | None).
    """
    conclusive, proc_table, children_map, gle = _read_snapshot_processes(kernel32)
    if not conclusive and not proc_table:
        return False, {}, f"Snapshot enumeration failed (gle={gle})"

    scan_error = f"Mid-scan snapshot enumeration failed (gle={gle})" if not conclusive else None
    classified: dict[int, dict] = {}

    # 1. Classify initial tracked PIDs strictly against pinned creation time (D1)
    for pid in sorted(tracked_pids):
        exp_ctime = tracked_info.get(pid, {}).get("create_time")

        # Unpinned tracked member (D1): no captured create_time -> state unknown
        if exp_ctime is None:
            if pid not in proc_table:
                classified[pid] = {
                    "pid": pid,
                    "state": "unknown" if not conclusive else "exited",
                    "exe": tracked_info.get(pid, {}).get("exe", "unknown"),
                    "create_time": None,
                    "gle": gle if not conclusive else 0,
                    "error": (
                        "Mid-scan enumeration incomplete; absence unverified"
                        if not conclusive
                        else "Unpinned tracked member: no captured creation time"
                    ),
                }
            else:
                classified[pid] = {
                    "pid": pid,
                    "state": "unknown",
                    "exe": proc_table[pid]["exe"],
                    "ppid": proc_table[pid]["ppid"],
                    "create_time": None,
                    "gle": 0,
                    "error": "Unpinned tracked PID: no captured creation time",
                }
            continue

        if pid not in proc_table:
            classified[pid] = {
                "pid": pid,
                "state": "unknown" if not conclusive else "exited",
                "exe": tracked_info.get(pid, {}).get("exe", "unknown"),
                "create_time": exp_ctime,
                "gle": gle if not conclusive else 0,
                "error": (
                    "Mid-scan enumeration incomplete; absence unverified"
                    if not conclusive
                    else None
                ),
            }
            continue

        ctime, is_act, q_gle = _query_process_identity_by_pid(kernel32, pid)
        if ctime is None or is_act is None:
            # Query failed: cross-check fresh snapshot to separate exited from present-unknown
            present, snap_gle = _is_pid_in_fresh_snapshot(kernel32, pid)
            if present is False and conclusive:
                classified[pid] = {
                    "pid": pid,
                    "state": "exited",
                    "exe": proc_table[pid]["exe"],
                    "create_time": None,
                    "gle": q_gle,
                }
            else:
                classified[pid] = {
                    "pid": pid,
                    "state": "unknown",
                    "exe": proc_table[pid]["exe"],
                    "create_time": None,
                    "gle": q_gle,
                    "error": f"Identity query failed on present process (gle={q_gle})",
                }
        elif not is_act:
            classified[pid] = {
                "pid": pid,
                "state": "exited",
                "exe": proc_table[pid]["exe"],
                "create_time": ctime,
                "gle": 0,
            }
        else:
            # Pinned + active: compare creation time
            if ctime != exp_ctime:
                classified[pid] = {
                    "pid": pid,
                    "state": "reused",
                    "exe": proc_table[pid]["exe"],
                    "ppid": proc_table[pid]["ppid"],
                    "create_time": ctime,
                    "expected_create_time": exp_ctime,
                    "gle": 0,
                }
            else:
                classified[pid] = {
                    "pid": pid,
                    "state": "confirmed",
                    "exe": proc_table[pid]["exe"],
                    "ppid": proc_table[pid]["ppid"],
                    "create_time": exp_ctime,
                    "gle": 0,
                    "match": "tracked_tree_pid",
                }

    # 2. Observation closure (read-only): seed from ALL historical observation seed PIDs (rev 8a/8b).
    # Recurse the full children_map; EVERY discovered PID classifies unknown.
    # Discovered PIDs never become kill candidates and never change any tracked member's state (D3).
    seeds = set(observation_seed_pids) if observation_seed_pids is not None else set(tracked_pids)
    queue = list(sorted(seeds))
    visited_pids = set(classified.keys())

    while queue:
        curr_pid = queue.pop(0)
        for child_pid in children_map.get(curr_pid, []):
            if child_pid in visited_pids:
                continue
            visited_pids.add(child_pid)

            if child_pid not in proc_table:
                classified[child_pid] = {
                    "pid": child_pid,
                    "state": "unknown" if not conclusive else "exited",
                    "ppid": curr_pid,
                    "create_time": None,
                    "gle": gle if not conclusive else 0,
                    "error": (
                        "Mid-scan enumeration incomplete; absence unverified"
                        if not conclusive
                        else None
                    ),
                }
                continue

            c_ctime, c_act, c_gle = _query_process_identity_by_pid(kernel32, child_pid)
            if not c_act and c_ctime is not None and conclusive:
                classified[child_pid] = {
                    "pid": child_pid,
                    "state": "exited",
                    "ppid": curr_pid,
                    "exe": proc_table[child_pid]["exe"],
                    "create_time": c_ctime,
                    "gle": 0,
                }
            else:
                # ALL discovered late descendants classify UNKNOWN (never confirmed, never killed)
                classified[child_pid] = {
                    "pid": child_pid,
                    "state": "unknown",
                    "ppid": curr_pid,
                    "exe": proc_table[child_pid]["exe"],
                    "create_time": c_ctime,
                    "gle": c_gle,
                    "error": (
                        "Late descendant: observed, unpinned, kill refused"
                        if c_ctime is not None
                        else f"Late descendant: query failed (gle={c_gle})"
                    ),
                }
            # Continue recursing children map through all discovered descendants for full observation
            queue.append(child_pid)

    confirmed_survivors = [c for c in classified.values() if c["state"] == "confirmed"]
    reused_survivors = [c for c in classified.values() if c["state"] == "reused"]
    unknown_survivors = [c for c in classified.values() if c["state"] == "unknown"]
    exited_entries = [c for c in classified.values() if c["state"] == "exited"]

    all_survivors = confirmed_survivors + reused_survivors + unknown_survivors
    clean = len(all_survivors) == 0 and conclusive

    scan_result = {
        "conclusive": conclusive,
        "clean": clean,
        "confirmed": confirmed_survivors,
        "reused": reused_survivors,
        "unknown": unknown_survivors,
        "exited": exited_entries,
        "survivors": all_survivors,
        "observation_seeds_count": len(seeds),
        "ownership_scope_limit": (
            "Containment claimed ONLY over captured pinned members with verified handle-identity. "
            "All late descendants and unpinned members classify unknown and are never killed. "
            "Observation closure seeded from all historical tracked PIDs; stranger trees "
            "reached via reused seeds classify unknown and force red, never kill."
        ),
        "fixture_evidence_note": (
            "CONTROLLED loopback fixture evidence NOT general hostile parentage spoof proof."
        ),
    }

    return conclusive, scan_result, scan_error


def _kill_confirmed_candidate(kernel32, pid: int, expected_ctime: int) -> dict:
    """K1, K2: Single-handle identity-verified kill invariant.

    Opens ONE handle with PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE.
    Verifies creation time on THAT handle before TerminateProcess on THAT SAME handle.
    Never falls through to a PID-only kill.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_TERMINATE = 0x0001
    DESIRED_ACCESS = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE

    h_proc = kernel32.OpenProcess(DESIRED_ACCESS, False, pid)
    if not h_proc:
        open_gle = ctypes.GetLastError()
        present, snap_gle = _is_pid_in_fresh_snapshot(kernel32, pid)
        if present is False:
            return {
                "pid": pid,
                "state": "exited",
                "opened": False,
                "terminated": False,
                "gle": open_gle,
                "status": "already_exited",
            }
        else:
            return {
                "pid": pid,
                "state": "unknown",
                "opened": False,
                "terminated": False,
                "gle": open_gle,
                "status": "open_failed_unknown",
            }

    try:
        ft_creation = FILETIME()
        ft_exit = FILETIME()
        ft_kernel = FILETIME()
        ft_user = FILETIME()
        if not kernel32.GetProcessTimes(
            h_proc,
            ctypes.byref(ft_creation),
            ctypes.byref(ft_exit),
            ctypes.byref(ft_kernel),
            ctypes.byref(ft_user),
        ):
            q_gle = ctypes.GetLastError()
            return {
                "pid": pid,
                "state": "unknown",
                "opened": True,
                "terminated": False,
                "gle": q_gle,
                "status": "query_failed_unknown",
            }

        live_ctime = _filetime_to_int(ft_creation)
        if live_ctime != expected_ctime:
            return {
                "pid": pid,
                "state": "reused",
                "opened": True,
                "terminated": False,
                "gle": 0,
                "status": "reused_pid_refused_kill",
            }

        term_ok = bool(kernel32.TerminateProcess(h_proc, 1))
        t_gle = 0 if term_ok else ctypes.GetLastError()
        return {
            "pid": pid,
            "state": "confirmed",
            "opened": True,
            "terminated": term_ok,
            "gle": t_gle,
            "status": "terminated" if term_ok else "terminate_failed",
        }
    finally:
        kernel32.CloseHandle(h_proc)


def mode_timeout(nonce: str, log_path: Path, interval: float) -> int:
    """Spawns detached process tree, establishes hazard tree, and emits heartbeats."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("timeout probe requires Windows host", flush=True)
        return 2

    import contain_core

    advapi32, kernel32, _ = contain_core._WIN32
    _bind_probe_win32_apis(kernel32)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_ts = _utcnow()
    my_pid = os.getpid()

    header = f"START pid={my_pid} nonce={nonce} ts={start_ts}\n"
    print(header, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header)
        f.flush()

    # R4 / F4, C11: Spawn detached child + grandchild hazard shape directly.
    # Tree identity rests entirely on logged captured PIDs and Toolhelp32 parent-PID chains (C2);
    # in-command nonce markers are omitted so cmd.exe directly executes loopback ping.exe without
    # command chaining or title/environment side effects.
    cmd_exe = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "cmd.exe")
    child_cmd = f'"{cmd_exe}" /c "ping -n 3600 127.0.0.1 > nul"'

    flags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    proc = subprocess.Popen(
        child_cmd,
        creationflags=flags,
        close_fds=True,
    )
    child_pid = proc.pid
    spawn_msg = f"SPAWNED_HAZARD_CHILD pid={child_pid} nonce={nonce} ts={_utcnow()}\n"
    print(spawn_msg, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(spawn_msg)
        f.flush()

    # C9, C11, C13: Bounded polling (up to 10s at 0.5s cadence) to establish grandchild existence
    # and record process creation timestamps
    poll_deadline = time.monotonic() + 10.0
    attempts = 0
    tree_members: list[dict] = []

    while time.monotonic() < poll_deadline:
        attempts += 1
        conclusive, proc_table, children_map, _ = _read_snapshot_processes(kernel32)
        if conclusive:
            # Recursive closure over child_pid
            descendants: list[dict] = []
            queue = list(children_map.get(child_pid, []))
            visited = set(queue)
            while queue:
                curr = queue.pop(0)
                if curr in proc_table:
                    descendants.append(proc_table[curr])
                for ch in children_map.get(curr, []):
                    if ch not in visited:
                        visited.add(ch)
                        queue.append(ch)

            # MUST require actual captured member exe.lower() == 'ping.exe' BEFORE shape accepted
            if any(d["exe"].lower() == "ping.exe" for d in descendants):
                members = [{"pid": child_pid, "exe": "cmd.exe", "ppid": my_pid}]
                for d in descendants:
                    members.append({"pid": d["pid"], "exe": d["exe"], "ppid": d["ppid"]})

                # Query creation times for all tree members (C13)
                all_times_ok = True
                for m in members:
                    ctime, is_act, _ = _query_process_identity_by_pid(kernel32, m["pid"])
                    if ctime is None or not is_act:
                        all_times_ok = False
                        break
                    m["create_time"] = ctime

                if all_times_ok:
                    tree_members = members
                    break
        time.sleep(0.5)

    if not tree_members or len(tree_members) < 2:
        fail_msg = (
            f"ABORT_HAZARD_NOT_ESTABLISHED child_pid={child_pid} nonce={nonce} "
            f"attempts={attempts} ts={_utcnow()}\n"
        )
        print(fail_msg, end="", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(fail_msg)
            f.flush()
        # Fail closed: never start heartbeats if hazard tree was not established
        proc.terminate()
        return 3

    tree_pids_str = ",".join(str(m["pid"]) for m in tree_members)
    tree_msg = (
        f"SPAWNED_HAZARD_TREE pids={tree_pids_str} child_pid={child_pid} nonce={nonce} "
        f"attempts={attempts} members={json.dumps(tree_members, separators=(',', ':'))} ts={_utcnow()} "
        f"note=point_in_time_tree_capture_before_timeout\n"
    )
    print(tree_msg, end="", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(tree_msg)
        f.flush()

    t0 = time.monotonic()
    seq = 0
    # R1: Continuous beats throughout stuck period until runner kills step
    while True:
        elapsed = time.monotonic() - t0
        beat_line = f"BEAT nonce={nonce} seq={seq} ts={_utcnow()} elapsed={elapsed:.2f}\n"
        print(beat_line, end="", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(beat_line)
            f.flush()
        seq += 1
        time.sleep(interval)


def mode_check_survivors(nonce: str, log_path: Path) -> int:
    """R2, R4, C2, C13: Two-phase observation, honest cleanup, recursive closure, and fail-closed verdict."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("check-survivors requires Windows host", flush=True)
        return 2

    import contain_core

    advapi32, kernel32, _ = contain_core._WIN32
    _bind_probe_win32_apis(kernel32)

    tracked_info: dict[int, dict] = {}
    tracked_pids: set[int] = set()

    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if f"nonce={nonce}" not in line:
                continue
            if line.startswith("SPAWNED_HAZARD_TREE "):
                m_match = re.search(r"members=(\[.*?\])(?:\s+ts=|\s*$)", line)
                if m_match:
                    try:
                        members_list = json.loads(m_match.group(1))
                        for m in members_list:
                            pid = int(m["pid"])
                            tracked_info[pid] = m
                            tracked_pids.add(pid)
                    except Exception:
                        pass
                for token in line.split():
                    if token.startswith("pids="):
                        pids_raw = token.split("=")[1]
                        for p in pids_raw.split(","):
                            try:
                                tracked_pids.add(int(p))
                            except ValueError:
                                pass
            elif line.startswith("SPAWNED_HAZARD_CHILD "):
                for token in line.split():
                    if token.startswith("pid="):
                        try:
                            tracked_pids.add(int(token.split("=")[1]))
                        except ValueError:
                            pass

    if not tracked_pids:
        err_record = {
            "nonce": nonce,
            "error": "No tracked PIDs found in log for nonce; cannot conduct survivor check",
            "clean": False,
            "ts": _utcnow(),
        }
        print(f"SURVIVOR_CHECK: {json.dumps(err_record)}", flush=True)
        return 1

    # Rev 8a/8b: Maintain accumulated historical seed set across all observation phases
    historical_pids: set[int] = set(tracked_pids)

    # Phase A: Immediate scan (record-only; races step interruption; evidence only, never verdict)
    phase_a_ok, phase_a_res, phase_a_err = _scan_process_tree(
        kernel32,
        tracked_info,
        tracked_pids,
        observation_seed_pids=historical_pids,
    )
    # Delta 3 (rev 8b): accumulate historical observations even from an inconclusive A
    if phase_a_res:
        for key in ("confirmed", "reused", "unknown", "exited"):
            for entry in phase_a_res.get(key, []):
                historical_pids.add(entry["pid"])

    phase_a_clean = bool(phase_a_ok and phase_a_res.get("clean", False))
    phase_a_record = {
        "conclusive": phase_a_ok,
        "error": phase_a_err,
        "scan": phase_a_res,
        "observed_snapshot_clean": phase_a_clean,
        "ts": _utcnow(),
        "note": "Phase A immediate scan (record-only; races runner step interruption)",
    }

    # Bounded settle: 15s observational window to observe whether the runner reaped the tree
    SETTLE_DELAY_SECONDS = 15.0
    time.sleep(SETTLE_DELAY_SECONDS)

    # Phase B: Post-settle observation (conclusive Phase B with zero observed survivors including unknown discoveries)
    phase_b_ok, phase_b_res, phase_b_err = _scan_process_tree(
        kernel32,
        tracked_info,
        tracked_pids,
        observation_seed_pids=historical_pids,
    )
    # Accumulate discovered observations from Phase B
    if phase_b_res:
        for key in ("confirmed", "reused", "unknown", "exited"):
            for entry in phase_b_res.get(key, []):
                historical_pids.add(entry["pid"])

    phase_b_clean = bool(phase_b_ok and phase_b_res.get("clean", False))
    observed_phase_b_no_survivors = bool(
        phase_b_ok and len(phase_b_res.get("survivors", [])) == 0
    )
    phase_b_record = {
        "conclusive": phase_b_ok,
        "error": phase_b_err,
        "scan": phase_b_res,
        "observed_snapshot_clean": phase_b_clean,
        "observed_phase_b_no_survivors": observed_phase_b_no_survivors,
        "ts": _utcnow(),
        "settle_delay_seconds": SETTLE_DELAY_SECONDS,
        "note": "Phase B post-settle observation (conclusive Phase B with zero observed survivors including unknown discoveries)",
    }

    # Cleanup after Phase B: target ONLY candidates whose state is confirmed (K1-K4).
    # Expected ctime taken strictly from captured record. Reused and unknown are reported, never killed.
    kill_log: list[dict] = []
    unowned_refused: list[dict] = []
    cleanup_attempted = False

    # Execute emergency cleanup if Phase B found survivors OR if scans were inconclusive
    if not observed_phase_b_no_survivors or not phase_b_ok:
        cleanup_attempted = True
        kill_candidates: list[tuple[int, int]] = []

        if phase_b_ok and phase_b_res:
            for c in phase_b_res.get("confirmed", []):
                pid = c["pid"]
                exp_ctime = tracked_info.get(pid, {}).get("create_time")
                if exp_ctime is not None:
                    kill_candidates.append((pid, exp_ctime))
                else:
                    unowned_refused.append({
                        "pid": pid,
                        "state": "unknown",
                        "reason": "Unpinned tracked member: kill refused",
                    })
            for r in phase_b_res.get("reused", []):
                unowned_refused.append({
                    "pid": r["pid"],
                    "state": "reused",
                    "reason": "PID reused; kill refused",
                })
            for u in phase_b_res.get("unknown", []):
                unowned_refused.append({
                    "pid": u["pid"],
                    "state": "unknown",
                    "reason": "Lineage/identity unknown or late descendant: kill refused",
                })
        else:
            # Inconclusive fallback: attempt kill ONLY on tracked PIDs with recorded creation time
            for pid in sorted(tracked_pids):
                exp_ctime = tracked_info.get(pid, {}).get("create_time")
                if exp_ctime is not None:
                    kill_candidates.append((pid, exp_ctime))
                else:
                    unowned_refused.append({
                        "pid": pid,
                        "state": "unknown",
                        "reason": "No captured creation time; kill refused",
                    })

        for pid, exp_ctime in kill_candidates:
            res = _kill_confirmed_candidate(kernel32, pid, exp_ctime)
            kill_log.append(res)

    # Phase C: Bounded 2.0s post-kill settle + confirmation scan (G5)
    PHASE_C_SETTLE_SECONDS = 2.0
    if cleanup_attempted:
        time.sleep(PHASE_C_SETTLE_SECONDS)

    phase_c_ok, phase_c_res, phase_c_err = _scan_process_tree(
        kernel32,
        tracked_info,
        tracked_pids,
        observation_seed_pids=historical_pids,
    )
    if phase_c_res:
        for key in ("confirmed", "reused", "unknown", "exited"):
            for entry in phase_c_res.get(key, []):
                historical_pids.add(entry["pid"])

    phase_c_clean = bool(phase_c_ok and phase_c_res.get("clean", False))
    phase_c_record = {
        "conclusive": phase_c_ok,
        "error": phase_c_err,
        "scan": phase_c_res,
        "observed_snapshot_clean": phase_c_clean,
        "ts": _utcnow(),
        "settle_delay_seconds": PHASE_C_SETTLE_SECONDS if cleanup_attempted else 0.0,
        "note": "Phase C post-cleanup confirmation scan",
    }

    # Coverage classification (Delta 4 / Rev 9)
    inconclusive_phases: list[str] = []
    if not phase_a_ok:
        inconclusive_phases.append("Phase A")
    if not phase_b_ok:
        inconclusive_phases.append("Phase B")
    if not phase_c_ok:
        inconclusive_phases.append("Phase C")

    coverage_label = "blind_window" if inconclusive_phases else "known_lineage_observed"

    # Verdict fail-closed: exit nonzero if Phase B found any survivor, OR any unowned/unknown candidate
    # was encountered, OR any kill failed/unknown, OR Phase B or C scan was inconclusive, OR Phase C finds anything.
    # Phase A is evidence-only and NEVER feeds the verdict (D2).
    kills_clean = (
        all(k.get("status") in ("terminated", "already_exited") for k in kill_log)
        and len(unowned_refused) == 0
    ) if cleanup_attempted else True

    all_clean = bool(
        phase_b_ok
        and observed_phase_b_no_survivors
        and phase_c_ok
        and phase_c_clean
        and kills_clean
    )

    record = {
        "nonce": nonce,
        "tracked_pids": sorted(tracked_pids),
        "historical_observation_seeds": sorted(historical_pids),
        "phase_a": phase_a_record,
        "settle_delay_seconds": SETTLE_DELAY_SECONDS,
        "phase_b": phase_b_record,
        "cleanup_attempted": cleanup_attempted,
        "cleanup_kills": kill_log,
        "cleanup_unowned_refused": unowned_refused,
        "phase_c": phase_c_record,
        "observed_phase_b_no_survivors": observed_phase_b_no_survivors,
        "observed_snapshot_clean": {
            "phase_a": phase_a_clean,
            "phase_b": phase_b_clean,
            "phase_c": phase_c_clean,
        },
        "coverage": coverage_label,
        "coverage_phases": inconclusive_phases,
        "full_tree_containment_proven": False,
        "unobserved_residual": (
            "Descendants whose connecting link was born and died between snapshots or "
            "inside a blind window are inherently unobserved; this record claims nothing "
            "about them. No rederivation from tracked seeds is made."
        ),
        "clean": all_clean,
        "pid_reuse_audit": "PID matching cannot rule out OS PID reuse; recorded fail-closed for audit",
        "ts": _utcnow(),
    }
    print(f"SURVIVOR_CHECK: {json.dumps(record)}", flush=True)

    return 0 if all_clean else 1


def _parse_iso(ts_str: str) -> datetime.datetime:
    ts_str = ts_str.rstrip("Z")
    return datetime.datetime.fromisoformat(ts_str).replace(tzinfo=datetime.timezone.utc)


def mode_verify_timeout(
    nonce: str,
    log_path: Path,
    timeout_seconds: int,
    run_id: str,
    check_run_id: str,
    run_attempt: str,
    job_name: str,
    owner: str,
    repo: str,
    stuck_outcome: str,
    step_name: str = "Deliberately stuck process (step timeout must kill it)",
    poll_seconds: int = 60,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
) -> int:
    """R1 / F1, C3, C4, C8, C12: Verifies timeout via exact check_run_url mapping and cadence-aware coverage."""
    verification: dict = {
        "nonce": nonce,
        "stuck_step_outcome": stuck_outcome,
        "timeout_seconds": timeout_seconds,
        "interval_seconds": interval,
        "check_run_id": check_run_id,
        "run_attempt": run_attempt,
        "checks": {},
        "pass": False,
        "ts": _utcnow(),
    }

    # C8: check_run_id is required; no fallback permitted
    if not check_run_id:
        verification["checks"]["check_run_id_provided"] = {
            "pass": False,
            "error": "check_run_id was empty; exact check_run_url mapping requires job.check_run_id",
        }
        print(f"VERIFY_TIMEOUT_FAIL: {json.dumps(verification, indent=2)}", flush=True)
        return 1

    # 1. Outcome assertion: must be failure or cancelled
    outcome_ok = stuck_outcome.lower() in ("failure", "cancelled")
    verification["checks"]["stuck_outcome_in_failure_or_cancelled"] = {
        "expected": ["failure", "cancelled"],
        "actual": stuck_outcome,
        "pass": outcome_ok,
    }

    # 2. Parse client-side log for nonce continuity and cadence-aware beat coverage (C4, C12)
    if not log_path.exists():
        verification["checks"]["log_exists"] = {"pass": False, "error": f"file not found: {log_path}"}
        print(f"VERIFY_TIMEOUT_FAIL: {json.dumps(verification, indent=2)}", flush=True)
        return 1

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    beats = []
    start_line_found = False

    for line in lines:
        if f"nonce={nonce}" not in line:
            continue
        if line.startswith("START "):
            start_line_found = True
        elif line.startswith("BEAT "):
            parts = dict(token.split("=", 1) for token in line.split() if "=" in token)
            if "seq" in parts and "ts" in parts and "elapsed" in parts:
                try:
                    beats.append({
                        "seq": int(parts["seq"]),
                        "ts": parts["ts"],
                        "elapsed": float(parts["elapsed"]),
                    })
                except ValueError:
                    pass

    verification["checks"]["nonce_start_line_present"] = {"pass": start_line_found}
    verification["checks"]["beats_recorded"] = {"count": len(beats), "pass": len(beats) >= 2}

    coverage_ok = False
    max_gap = 0.0
    max_beat_elapsed = beats[-1]["elapsed"] if beats else 0.0
    coverage_threshold = float(timeout_seconds) - float(interval)
    max_allowed_gap = float(interval) * 3.0

    if len(beats) >= 2:
        # C4, C12: Coverage requires beats to reach within one cadence of timeout boundary
        coverage_ok = max_beat_elapsed >= coverage_threshold
        for i in range(1, len(beats)):
            gap = beats[i]["elapsed"] - beats[i - 1]["elapsed"]
            if gap > max_gap:
                max_gap = gap
        gap_ok = max_gap <= max_allowed_gap
    else:
        gap_ok = False

    uncovered_window = max(0.0, float(timeout_seconds) - max_beat_elapsed)
    if coverage_ok:
        coverage_detail = (
            f"heartbeats reached {max_beat_elapsed:.2f}s (>= threshold {coverage_threshold:.2f}s = "
            f"timeout {timeout_seconds}s - interval {interval}s); uncovered window is {uncovered_window:.2f}s "
            "(only a crash inside final sub-cadence window is indistinguishable; observational evidence of "
            "beats reaching boundary within cadence, not absolute proof of uninterrupted liveness)"
        )
    else:
        coverage_detail = (
            f"heartbeats stopped at {max_beat_elapsed:.2f}s (< threshold {coverage_threshold:.2f}s = "
            f"timeout {timeout_seconds}s - interval {interval}s); uncovered window is {uncovered_window:.2f}s; "
            "early crash cannot be ruled out"
        )

    verification["checks"]["beat_coverage"] = {
        "timeout_seconds": timeout_seconds,
        "interval_seconds": interval,
        "coverage_threshold_seconds": coverage_threshold,
        "max_beat_elapsed": max_beat_elapsed,
        "uncovered_window_seconds": round(uncovered_window, 2),
        "pass": coverage_ok,
        "final_subcadence_window_unobservable": coverage_ok,
        "detail": coverage_detail,
    }
    verification["checks"]["beat_gap_bounded"] = {
        "max_observed_gap": max_gap,
        "max_allowed_gap": max_allowed_gap,
        "pass": gap_ok,
    }

    # 3. Server-side step elapsed verification via GitHub REST API (C3, C8: exact check_run_url suffix)
    server_elapsed = None
    kill_latency_overshoot = None
    server_check_pass = False
    matched_step = None
    matched_job_info = None

    t_poll_start = time.monotonic()
    expected_suffix = f"/check-runs/{check_run_id}"

    while (time.monotonic() - t_poll_start) < poll_seconds:
        page = 1
        while True:
            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
                f"?per_page=100&page={page}"
            )
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "nexus-e2e-proof-verifier", "Accept": "application/vnd.github.v3+json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    jobs = data.get("jobs", [])
                    for j in jobs:
                        check_url = j.get("check_run_url", "")
                        # C8: Matched ONLY by exact check_run_url suffix
                        if check_url.endswith(expected_suffix):
                            # Strict cross-check on job name: mismatch fails
                            if j.get("name") != job_name:
                                verification["checks"]["job_name_cross_check"] = {
                                    "pass": False,
                                    "expected": job_name,
                                    "actual": j.get("name"),
                                    "error": "check_run_url matched job but job name mismatched",
                                }
                                out_file = log_path.parent / "probe_timeout_verification.json"
                                out_file.write_text(json.dumps(verification, indent=2))
                                return 1

                            matched_job_info = {
                                "id": j.get("id"),
                                "name": j.get("name"),
                                "check_run_url": check_url,
                                "run_attempt": j.get("run_attempt"),
                            }
                            # C8: Step matched by exact name only; no substring fallback
                            for s in j.get("steps", []):
                                if s.get("name") == step_name:
                                    matched_step = s
                                    break
                            break

                    if matched_step or len(jobs) < 100:
                        break
                    page += 1
            except urllib.error.URLError as e:
                verification["server_poll_warning"] = str(e)
                break

        if matched_step and matched_step.get("completed_at"):
            break
        time.sleep(5)

    verification["matched_job"] = matched_job_info

    if matched_step and matched_step.get("started_at") and matched_step.get("completed_at"):
        t_start = _parse_iso(matched_step["started_at"])
        t_end = _parse_iso(matched_step["completed_at"])
        server_elapsed = (t_end - t_start).total_seconds()
        kill_latency_overshoot = server_elapsed - timeout_seconds
        server_check_pass = server_elapsed >= timeout_seconds
        verification["checks"]["server_duration_ge_timeout"] = {
            "started_at": matched_step["started_at"],
            "completed_at": matched_step["completed_at"],
            "server_elapsed": server_elapsed,
            "kill_latency_overshoot": kill_latency_overshoot,
            "pass": server_check_pass,
        }
    else:
        verification["checks"]["server_duration_ge_timeout"] = {
            "pass": False,
            "error": "server step completed_at timestamp remained unavailable after poll",
        }

    all_checks_passed = bool(
        outcome_ok
        and start_line_found
        and coverage_ok
        and gap_ok
        and server_check_pass
    )
    verification["pass"] = all_checks_passed

    out_file = log_path.parent / "probe_timeout_verification.json"
    out_file.write_text(json.dumps(verification, indent=2))
    print(f"VERIFY_TIMEOUT_RESULT: {json.dumps(verification, indent=2)}", flush=True)

    return 0 if all_checks_passed else 1


def mode_leave(receipt_path: Path) -> int:
    """R7, R3, C7: Simulates leftover account and writes atomic SID receipt."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("leave probe requires Windows host", flush=True)
        return 2

    import contain_core

    sid, _ = contain_core.create_principal(contain_core.DEFAULT_PRINCIPAL_NAME)
    receipt_data = {
        "sid": sid,
        "username": contain_core.DEFAULT_PRINCIPAL_NAME,
        "ts": _utcnow(),
    }
    _persist_atomic_receipt(receipt_path, receipt_data)
    print(f"LEFTOVER principal created: name={contain_core.DEFAULT_PRINCIPAL_NAME} sid={sid}", flush=True)
    print(f"persisted atomic receipt to {receipt_path.name}", flush=True)
    return 0


def mode_cleanup(receipt_path: Path, evidence_path: Path | None = None) -> int:
    """R7, R3, C7: Unconditionally terminates/sweeps processes for receipt SID, then deletes account."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("cleanup probe requires Windows host", flush=True)
        return 2

    import contain_core

    cleanup_record: dict = {
        "ts": _utcnow(),
        "receipt_path": str(receipt_path),
        "receipt_read": False,
        "target_sid": None,
        "terminate_results": [],
        "account_existed": False,
        "account_deleted": False,
        "post_sweep": {},
        "clean": False,
    }

    # C7: Missing, unreadable, or ambiguous receipt fails closed
    if not receipt_path.exists():
        cleanup_record["error"] = f"receipt file missing at {receipt_path}; cannot verify target SID"
        print(f"CLEANUP_FAIL: {json.dumps(cleanup_record, indent=2)}", flush=True)
        if evidence_path:
            evidence_path.write_text(json.dumps(cleanup_record, indent=2))
        return 1

    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        target_sid = data.get("sid")
        if not target_sid or not target_sid.startswith("S-1-5-21-"):
            raise ValueError(f"invalid or ambiguous SID in receipt: {target_sid}")
        cleanup_record["receipt_read"] = True
        cleanup_record["target_sid"] = target_sid
    except Exception as e:
        cleanup_record["error"] = f"failed to read valid SID receipt: {type(e).__name__}: {e}"
        print(f"CLEANUP_FAIL: {json.dumps(cleanup_record, indent=2)}", flush=True)
        if evidence_path:
            evidence_path.write_text(json.dumps(cleanup_record, indent=2))
        return 1

    # C7: ALWAYS run terminate_processes_by_sid + sweep_processes_by_sid FIRST
    # (tokens survive account deletion, so sweep is load-bearing even if account is absent)
    term_res = contain_core.terminate_processes_by_sid(target_sid)
    cleanup_record["terminate_results"] = [
        {"pid": p, "error": str(err)} for p, err in term_res
    ]

    # Check if account currently exists in SAM
    advapi32, kernel32, _ = contain_core._WIN32
    cb_sid = wintypes.DWORD(0)
    cch_domain = wintypes.DWORD(0)
    pe_use = wintypes.DWORD(0)

    advapi32.LookupAccountNameW(
        None, contain_core.DEFAULT_PRINCIPAL_NAME, None,
        ctypes.byref(cb_sid), None, ctypes.byref(cch_domain), ctypes.byref(pe_use)
    )

    if cb_sid.value > 0:
        cleanup_record["account_existed"] = True
        deleted = contain_core.delete_principal(captured_sid=target_sid, username=contain_core.DEFAULT_PRINCIPAL_NAME)
        cleanup_record["account_deleted"] = deleted
    else:
        cleanup_record["account_existed"] = False
        cleanup_record["account_deleted"] = False

    # Unconditional post-sweep
    post_sweep = contain_core.sweep_processes_by_sid(target_sid)
    cleanup_record["post_sweep"] = {
        "clean": post_sweep.clean,
        "child_pids": post_sweep.child_pids,
        "indeterminate_pids": post_sweep.indeterminate_pids,
    }

    # Clean only if account is not present AND post-sweep is clean
    clean_ok = (not cleanup_record["account_existed"] or cleanup_record["account_deleted"]) and post_sweep.clean
    cleanup_record["clean"] = clean_ok

    if evidence_path:
        evidence_path.write_text(json.dumps(cleanup_record, indent=2))
    print(f"CLEANUP_RESULT: {json.dumps(cleanup_record, indent=2)}", flush=True)

    return 0 if clean_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timeout", "check-survivors", "verify-timeout", "leave", "cleanup"], required=True)
    ap.add_argument("--nonce", default="probe_nonce_default")
    ap.add_argument("--log", default="probe_timeout.log")
    ap.add_argument("--receipt", default="proof_principal_receipt.json")
    ap.add_argument("--cleanup-evidence", default="")
    ap.add_argument("--interval", type=float, default=DEFAULT_HEARTBEAT_INTERVAL)
    ap.add_argument("--timeout-seconds", type=int, default=120)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--check-run-id", default="")
    ap.add_argument("--run-attempt", default="1")
    ap.add_argument("--job-name", default="probe-timeout")
    ap.add_argument("--owner", default="Yash1bajpai")
    ap.add_argument("--repo", default="nexus-agent")
    ap.add_argument("--stuck-step-outcome", default="failure")
    ap.add_argument("--step-name", default="Deliberately stuck process (step timeout must kill it)")

    args = ap.parse_args()
    log_path = Path(args.log).resolve()
    receipt_path = Path(args.receipt).resolve()
    cleanup_evidence_path = Path(args.cleanup_evidence).resolve() if args.cleanup_evidence else None

    if args.mode == "timeout":
        return mode_timeout(args.nonce, log_path, args.interval)
    elif args.mode == "check-survivors":
        return mode_check_survivors(args.nonce, log_path)
    elif args.mode == "verify-timeout":
        return mode_verify_timeout(
            nonce=args.nonce,
            log_path=log_path,
            timeout_seconds=args.timeout_seconds,
            run_id=args.run_id,
            check_run_id=args.check_run_id,
            run_attempt=args.run_attempt,
            job_name=args.job_name,
            owner=args.owner,
            repo=args.repo,
            stuck_outcome=args.stuck_step_outcome,
            step_name=args.step_name,
            interval=args.interval,
        )
    elif args.mode == "leave":
        return mode_leave(receipt_path)
    elif args.mode == "cleanup":
        return mode_cleanup(receipt_path, cleanup_evidence_path)

    return 2


if __name__ == "__main__":
    sys.exit(main())
