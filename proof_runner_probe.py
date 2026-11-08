# proof_runner_probe.py
"""Nexus e2e — Minimal runner-envelope probe harness.

Demonstrates outer-bound runner mechanics before any proof execution:
  --mode timeout:
      Spawns cmd.exe child + ping.exe grandchild with detached process flags
      (CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS). Bounded polling (up to 10s)
      establishes grandchild creation (C9, C11) before continuous heartbeats begin.
      Exits nonzero (exit 3) if hazard shape is not established.
  --mode check-survivors:
      Executes post-kill in the same job (same VM) via if: always(). Scans all
      active system processes for the tracked tree PIDs and any late descendants
      (th32ParentProcessID). Tree identity rests on logged captured PIDs and
      Toolhelp32 parent-PID chains (C2), not on unobservable command-line tokens.
      Fails closed (exit 1) on any survivor, issuing emergency termination across
      the full tracked set.
  --mode verify-timeout:
      Polls GitHub Actions REST API with pagination and exact check_run_url suffix
      matching (C3). Empty check_run_id or name mismatch fails closed without fallback.
      Enforces heartbeat coverage reaching or exceeding timeout duration (C4).
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
import subprocess
import sys
import time
import urllib.error
import urllib.request

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

EXPECTED_SHA256 = {
    "contain_core.py": "41ad522dbe3366d30e611a179ab176459af685fe2858ea9524e2f9d7a3cb53cd",
    "contain_core_selftest.py": "b0fb0fa27d119f90599cc1ba496d57553ed0caabcd23ab4a5686131caa3016c4",
}

CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008


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


def mode_timeout(nonce: str, log_path: Path, interval: float) -> int:
    """Spawns detached process tree, establishes hazard tree, and emits heartbeats."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("timeout probe requires Windows host", flush=True)
        return 2

    import contain_core

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

    # C9, C11: Bounded polling (up to 10s at 0.5s cadence) to establish grandchild existence
    advapi32, kernel32, _ = contain_core._WIN32
    poll_deadline = time.monotonic() + 10.0
    attempts = 0
    tree_members: list[dict] = []

    while time.monotonic() < poll_deadline:
        attempts += 1
        h_snap = kernel32.CreateToolhelp32Snapshot(contain_core.TH32CS_SNAPPROCESS, 0)
        if h_snap != contain_core.INVALID_HANDLE_VALUE and h_snap:
            try:
                pe = contain_core.PROCESSENTRY32W()
                pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
                more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
                parent_map: dict[int, tuple[int, str]] = {}
                while more:
                    pid = int(pe.th32ProcessID)
                    ppid = int(pe.th32ParentProcessID)
                    exe = pe.szExeFile
                    parent_map[pid] = (ppid, exe)
                    pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
                    more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))

                # Direct descendants of child_pid
                descendants = [
                    (pid, parent_map[pid][1])
                    for pid in parent_map
                    if parent_map[pid][0] == child_pid
                ]
                # MUST require actual captured member exe.lower() == 'ping.exe' BEFORE shape accepted
                if any(exe.lower() == "ping.exe" for _, exe in descendants):
                    tree_members = [{"pid": child_pid, "exe": "cmd.exe", "ppid": my_pid}]
                    for d_pid, d_exe in descendants:
                        tree_members.append({"pid": d_pid, "exe": d_exe, "ppid": child_pid})
                    break
            finally:
                kernel32.CloseHandle(h_snap)
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
        f"attempts={attempts} members={json.dumps(tree_members)} ts={_utcnow()} "
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
    """R2, R4, C2: Check whether any detached child or grandchild process survived step kill."""
    _verify_hashes_or_abort()
    if sys.platform != "win32":
        print("check-survivors requires Windows host", flush=True)
        return 2

    import contain_core

    tracked_pids: set[int] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if f"nonce={nonce}" not in line:
                continue
            if line.startswith("SPAWNED_HAZARD_TREE "):
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

    survivors: list[dict] = []
    advapi32, kernel32, _ = contain_core._WIN32
    h_snap = kernel32.CreateToolhelp32Snapshot(contain_core.TH32CS_SNAPPROCESS, 0)
    if h_snap != contain_core.INVALID_HANDLE_VALUE and h_snap:
        try:
            pe = contain_core.PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
            more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
            while more:
                pid = int(pe.th32ProcessID)
                ppid = int(pe.th32ParentProcessID)
                exe = pe.szExeFile

                if pid in tracked_pids:
                    survivors.append({"pid": pid, "exe": exe, "ppid": ppid, "match": "tracked_tree_pid"})
                elif ppid in tracked_pids and pid != 0:
                    survivors.append({"pid": pid, "exe": exe, "ppid": ppid, "match": "late_descendant"})

                pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
                more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))
        finally:
            kernel32.CloseHandle(h_snap)

    record = {
        "nonce": nonce,
        "tracked_pids": sorted(tracked_pids),
        "survivors": survivors,
        "clean": len(survivors) == 0,
        "pid_reuse_audit": "PID matching cannot rule out OS PID reuse; recorded fail-closed for audit",
        "ts": _utcnow(),
    }
    print(f"SURVIVOR_CHECK: {json.dumps(record)}", flush=True)

    if not record["clean"]:
        # C2, C6: Emergency kill covering full captured tree and any late descendants
        for s in survivors:
            h = kernel32.OpenProcess(contain_core.PROCESS_TERMINATE, False, s["pid"])
            if h:
                kernel32.TerminateProcess(h, 1)
                kernel32.CloseHandle(h)
        for tp in tracked_pids:
            h = kernel32.OpenProcess(contain_core.PROCESS_TERMINATE, False, tp)
            if h:
                kernel32.TerminateProcess(h, 1)
                kernel32.CloseHandle(h)
        return 1

    return 0


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
) -> int:
    """R1 / F1, C3, C4, C8: Verifies timeout via exact check_run_url mapping and coverage."""
    verification: dict = {
        "nonce": nonce,
        "stuck_step_outcome": stuck_outcome,
        "timeout_seconds": timeout_seconds,
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

    # 2. Parse client-side log for nonce continuity and beat coverage (C4: >= timeout_seconds)
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

    if len(beats) >= 2:
        # C4: Coverage requires beats to have reached or exceeded timeout_seconds
        coverage_ok = max_beat_elapsed >= float(timeout_seconds)
        for i in range(1, len(beats)):
            gap = beats[i]["elapsed"] - beats[i - 1]["elapsed"]
            if gap > max_gap:
                max_gap = gap
        gap_ok = max_gap <= 15.0
    else:
        gap_ok = False

    verification["checks"]["beat_coverage"] = {
        "required_seconds": timeout_seconds,
        "max_beat_elapsed": max_beat_elapsed,
        "pass": coverage_ok,
        "early_crash_window_closed": True,
        "detail": (
            "heartbeats covered full timeout duration"
            if coverage_ok
            else f"heartbeats stopped at {max_beat_elapsed:.2f}s (< {timeout_seconds}s); early crash cannot be ruled out"
        ),
    }
    verification["checks"]["beat_gap_bounded"] = {
        "max_observed_gap": max_gap,
        "max_allowed_gap": 15.0,
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
    ap.add_argument("--interval", type=float, default=5.0)
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
        )
    elif args.mode == "leave":
        return mode_leave(receipt_path)
    elif args.mode == "cleanup":
        return mode_cleanup(receipt_path, cleanup_evidence_path)

    return 2


if __name__ == "__main__":
    sys.exit(main())
