# proof_cplw_diag.py
"""Nexus e2e — Bounded CreateProcessWithLogonW cwd-context diagnostic (repaired).

Scope:
- Discriminate operative variable for CPLW failure (err=267) observed on fresh principal:
  runner profile tree location vs default ACL vs explicit SID-form ACE grant.
- Enforces strict gate precondition before ANY fixture or resource is created.
- Exercises single fresh diagnostic principal ('nexuse2ediag') with atomic receipt.
- Executes bounded variant matrix V0..V4 with cmd.exe /c exit 0 payload.
- Follows load-bearing per-variant lifecycle protocol (job kill-on-close, suspended spawn,
  direct-handle termination on unassigned failure, bounded wait, handle verification).
- Performs full teardown before proof entry (principal deletion + probe cleanup subprocess +
  removal of the three probe directories).
- Records operational conclusion with strictly observational phrasing.

Status: SOURCE PREPARED, NOT RUN.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

import contain_core

DIAG_PRINCIPAL_NAME = "nexuse2ediag"
RECEIPT_PATH = SELF_DIR / "diag_principal_receipt.json"
CLEANUP_EVIDENCE_PATH = SELF_DIR / "diag_cleanup_evidence.json"
DEFAULT_EVIDENCE_PATH = SELF_DIR / "cplw_diag_evidence.json"
DEFAULT_GATE_PATH = SELF_DIR / "proof_gate_eval.json"

STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _persist_atomic_receipt(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _get_icacls(path: Path | str) -> str:
    try:
        cp = subprocess.run(["icacls", str(path)], capture_output=True, text=True, timeout=10)
        return cp.stdout.strip()
    except Exception as e:
        return f"icacls query error: {e}"


def _close_handle_checked(
    kernel32: Any, handle: wintypes.HANDLE | None, name: str, rec: dict[str, Any]
) -> bool:
    """Close a Win32 handle and record boolean success or close failure."""
    if not handle:
        return True
    ok = bool(kernel32.CloseHandle(handle))
    rec["handles"][f"{name}_closed"] = ok
    if not ok:
        gle = ctypes.get_last_error()
        rec["close_failures"].append({"handle": name, "gle": gle})
    return ok


def _terminate_direct_process(
    kernel32: Any, h_process: wintypes.HANDLE, timeout_ms: int = 10000
) -> tuple[bool, int | None, int]:
    """Terminate process via directly held handle, bounded wait, and exit code confirmation.

    Returns (confirmed_exited, exit_code, gle).
    """
    term_ok = bool(kernel32.TerminateProcess(h_process, 1))
    t_gle = 0 if term_ok else ctypes.get_last_error()
    wait_res = kernel32.WaitForSingleObject(h_process, timeout_ms)
    ec = wintypes.DWORD(0)
    ec_ok = bool(kernel32.GetExitCodeProcess(h_process, ctypes.byref(ec)))
    ec_val = int(ec.value) if ec_ok else None
    confirmed = (wait_res == WAIT_OBJECT_0) and (ec_val is not None) and (ec_val != STILL_ACTIVE)
    return confirmed, ec_val, t_gle


def _verify_job_empty(kernel32: Any, h_job: wintypes.HANDLE, timeout_s: float = 5.0) -> tuple[bool, int, int]:
    """Poll QueryInformationJobObject for empty assigned and in-list process count."""
    deadline = time.monotonic() + timeout_s
    assigned = -1
    count = -1
    while True:
        pid_list = contain_core.JOBOBJECT_BASIC_PROCESS_ID_LIST_256()
        ret_len = wintypes.DWORD(0)
        success = kernel32.QueryInformationJobObject(
            h_job,
            contain_core.JobObjectBasicProcessIdList,
            ctypes.byref(pid_list),
            ctypes.sizeof(pid_list),
            ctypes.byref(ret_len),
        )
        if success:
            assigned = int(pid_list.NumberOfAssignedProcesses)
            count = int(pid_list.NumberOfProcessIdsInList)
            if assigned == 0 and count == 0:
                return True, 0, 0
        if time.monotonic() >= deadline:
            return False, assigned, count
        time.sleep(0.05)


def check_gate_precondition(gate_path: Path, evidence_path: Path) -> bool:
    """Validate gate precondition strictly before ANY fixture or resource is created.

    REQUIRE: parseable JSON AND key overall_pass present AND its value IS the exact
    JSON bool True (type(v) is bool and v is True).
    """
    reason = None
    if not gate_path.is_file():
        reason = f"Gate evaluation file missing: {gate_path}"
    else:
        try:
            content = gate_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if not isinstance(data, dict):
                reason = f"Gate evaluation JSON is not an object: {type(data).__name__}"
            elif "overall_pass" not in data:
                reason = "Gate evaluation JSON missing 'overall_pass' key"
            else:
                v = data["overall_pass"]
                if type(v) is not bool:
                    reason = f"Gate 'overall_pass' value is not a strict boolean: {v!r} ({type(v).__name__})"
                elif v is not True:
                    reason = "Gate 'overall_pass' is False"
        except Exception as e:
            reason = f"Gate evaluation JSON unparseable: {e}"

    if reason is not None:
        evidence = {
            "verdict": "diagnostic_not_run_gate_not_true",
            "reason": reason,
            "timestamp_utc": _utcnow(),
        }
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"ABORT: {reason}", file=sys.stderr, flush=True)
        return False

    return True


def run_variant(
    advapi32: Any,
    kernel32: Any,
    full_cmd: str,
    variant_id: str,
    name: str,
    cwd: str | None,
    username: str,
    password: str,
    diag_sid: str,
) -> dict[str, Any]:
    """Execute a single CPLW diagnostic variant following the load-bearing lifecycle protocol."""
    t0 = time.monotonic()
    rec: dict[str, Any] = {
        "variant_id": variant_id,
        "name": name,
        "cwd_arg": cwd,
        "dir_exists": Path(cwd).is_dir() if cwd is not None else True,
        "icacls_pre_call": None,
        "grant_result": None,
        "cplw_success": False,
        "cplw_gle": None,
        "assign_result": None,
        "verify_in_job_result": None,
        "resume_result": None,
        "wait_result": None,
        "exit_code": None,
        "wall_time_s": 0.0,
        "handles": {
            "job_created": False,
            "job_closed": False,
            "process_opened": False,
            "process_closed": False,
            "thread_opened": False,
            "thread_closed": False,
        },
        "close_failures": [],
        "terminate_direct_gle": None,
        "job_empty_verification": None,
        "outcome": "not_started",
        "kill_unconfirmed": False,
    }

    # R7: For V4 apply grant; on failure classify grant_failed (infrastructure) and skip CPLW
    if cwd is not None and variant_id in ("V2", "V3", "V4"):
        if variant_id == "V4":
            grant_cmd = ["icacls", cwd, "/grant", f"*{diag_sid}:(OI)(CI)F"]
            grant_ok = False
            try:
                cp_grant = subprocess.run(grant_cmd, capture_output=True, text=True, timeout=10)
                rec["grant_result"] = {
                    "cmd": grant_cmd,
                    "stdout": cp_grant.stdout.strip(),
                    "stderr": cp_grant.stderr.strip(),
                    "returncode": cp_grant.returncode,
                }
                grant_ok = (cp_grant.returncode == 0)
            except Exception as e:
                rec["grant_result"] = {"cmd": grant_cmd, "error": str(e), "returncode": -1}
                grant_ok = False

            if not grant_ok:
                rec["outcome"] = "grant_failed"
                rec["wall_time_s"] = round(time.monotonic() - t0, 3)
                return rec

        rec["icacls_pre_call"] = _get_icacls(cwd)

    h_job = None
    h_process = None
    h_thread = None
    assign_succeeded = False

    try:
        # Step 1: Create fresh Job Object; set JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        h_job = kernel32.CreateJobObjectW(None, None)
        if not h_job:
            rec["outcome"] = "job_create_failed"
            rec["cplw_gle"] = ctypes.get_last_error()
            return rec
        rec["handles"]["job_created"] = True

        job_info = contain_core.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = contain_core.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            h_job,
            contain_core.JobObjectExtendedLimitInformation,
            ctypes.byref(job_info),
            ctypes.sizeof(job_info),
        ):
            rec["outcome"] = "job_set_info_failed"
            rec["cplw_gle"] = ctypes.get_last_error()
            return rec

        # Step 2: CPLW suspended
        si = contain_core.STARTUPINFOW()
        si.cb = ctypes.sizeof(contain_core.STARTUPINFOW)
        pi = contain_core.PROCESS_INFORMATION()

        flags = contain_core.CREATE_SUSPENDED | contain_core.CREATE_UNICODE_ENVIRONMENT
        lp_cwd = cwd if cwd is not None else None
        env_workdir = cwd if (cwd is not None and Path(cwd).is_dir()) else os.getcwd()
        child_env = contain_core.build_child_env(env_workdir)
        env_block = contain_core._format_unicode_env_block(child_env)

        success = bool(
            advapi32.CreateProcessWithLogonW(
                username,
                None,
                password,
                contain_core.LOGON_WITH_PROFILE,
                None,
                full_cmd,
                flags,
                ctypes.cast(env_block, ctypes.c_void_p),
                lp_cwd,
                ctypes.byref(si),
                ctypes.byref(pi),
            )
        )

        rec["cplw_success"] = success
        if not success:
            gle = ctypes.get_last_error()
            rec["cplw_gle"] = gle
            rec["outcome"] = "spawn_error"
            _close_handle_checked(kernel32, h_job, "job", rec)
            h_job = None
            return rec

        h_process = pi.hProcess
        h_thread = pi.hThread
        rec["handles"]["process_opened"] = True
        rec["handles"]["thread_opened"] = True
        rec["pid"] = int(pi.dwProcessId)

        # Step 3: AssignProcessToJobObject
        if not kernel32.AssignProcessToJobObject(h_job, h_process):
            gle = ctypes.get_last_error()
            rec["assign_result"] = {"success": False, "gle": gle}
            rec["outcome"] = "assign_failed"
            # M1: Record TerminateProcess GLE in rec
            confirmed, ec_val, t_gle = _terminate_direct_process(kernel32, h_process, 10000)
            rec["terminate_direct_gle"] = t_gle
            if not confirmed:
                rec["kill_unconfirmed"] = True
            rec["exit_code"] = ec_val
            return rec

        assign_succeeded = True
        rec["assign_result"] = {"success": True, "gle": 0}

        # Step 4: IsProcessInJob verify
        in_job = wintypes.BOOL(False)
        if not kernel32.IsProcessInJob(h_process, h_job, ctypes.byref(in_job)) or not in_job.value:
            gle = ctypes.get_last_error()
            rec["verify_in_job_result"] = {"in_job": bool(in_job.value), "success": False, "gle": gle}
            rec["outcome"] = "verify_failed"
            # M1: Record TerminateProcess GLE in rec
            confirmed, ec_val, t_gle = _terminate_direct_process(kernel32, h_process, 10000)
            rec["terminate_direct_gle"] = t_gle
            if not confirmed:
                rec["kill_unconfirmed"] = True
            rec["exit_code"] = ec_val
            return rec
        rec["verify_in_job_result"] = {"in_job": True, "success": True, "gle": 0}

        # Step 5: ResumeThread (return -1 = failure)
        res_ret = kernel32.ResumeThread(h_thread)
        if res_ret == 0xFFFFFFFF:
            gle = ctypes.get_last_error()
            rec["resume_result"] = {"success": False, "gle": gle, "ret": -1}
            rec["outcome"] = "resume_failed"
            kernel32.TerminateJobObject(h_job, 1)
            wait_res = kernel32.WaitForSingleObject(h_process, 10000)
            ec = wintypes.DWORD(0)
            ec_ok = bool(kernel32.GetExitCodeProcess(h_process, ctypes.byref(ec)))
            ec_val = int(ec.value) if ec_ok else None
            confirmed = (wait_res == WAIT_OBJECT_0) and (ec_val is not None) and (ec_val != STILL_ACTIVE)
            if not confirmed:
                rec["kill_unconfirmed"] = True
            rec["exit_code"] = ec_val
            return rec
        rec["resume_result"] = {"success": True, "gle": 0, "ret": int(res_ret)}

        # Step 6: WaitForSingleObject(h_process, 10000) hard timeout
        wait_res = kernel32.WaitForSingleObject(h_process, 10000)
        if wait_res == WAIT_OBJECT_0:
            ec = wintypes.DWORD(0)
            gecp_ok = bool(kernel32.GetExitCodeProcess(h_process, ctypes.byref(ec)))
            # R1: Check GECP BOOL, require exit_code == 0 for outcome completed
            if not gecp_ok:
                rec["wait_result"] = {"status": "WAIT_OBJECT_0", "raw": wait_res, "gle": 0}
                rec["exit_code"] = None
                rec["outcome"] = "exit_code_unconfirmed"
            else:
                ec_val = int(ec.value)
                rec["wait_result"] = {"status": "WAIT_OBJECT_0", "raw": wait_res, "gle": 0}
                rec["exit_code"] = ec_val
                if ec_val == 0:
                    rec["outcome"] = "completed"
                else:
                    rec["outcome"] = "payload_exit_nonzero"
        elif wait_res == WAIT_TIMEOUT:
            rec["wait_result"] = {"status": "WAIT_TIMEOUT", "raw": wait_res, "gle": 0}
            rec["outcome"] = "timed_out"
            kernel32.TerminateJobObject(h_job, 1)
            empty_ok, assigned, count = _verify_job_empty(kernel32, h_job, timeout_s=5.0)
            rec["job_empty_verification"] = {"empty": empty_ok, "assigned": assigned, "count": count}
            if not empty_ok:
                rec["kill_unconfirmed"] = True
        else:
            gle = ctypes.get_last_error()
            rec["wait_result"] = {"status": "WAIT_FAILED", "raw": wait_res, "gle": gle}
            rec["outcome"] = "wait_failed"
            kernel32.TerminateJobObject(h_job, 1)
            empty_ok, assigned, count = _verify_job_empty(kernel32, h_job, timeout_s=5.0)
            rec["job_empty_verification"] = {"empty": empty_ok, "assigned": assigned, "count": count}
            if not empty_ok:
                rec["kill_unconfirmed"] = True

        # Step 7: Close thread and process handles
        if h_thread:
            _close_handle_checked(kernel32, h_thread, "thread", rec)
            h_thread = None

        if h_process:
            _close_handle_checked(kernel32, h_process, "process", rec)
            h_process = None

        # R2: Normal-path job empty verification; nonempty sets kill_unconfirmed = True
        if h_job and rec["job_empty_verification"] is None:
            empty_ok, assigned, count = _verify_job_empty(kernel32, h_job, timeout_s=5.0)
            rec["job_empty_verification"] = {"empty": empty_ok, "assigned": assigned, "count": count}
            if not empty_ok:
                rec["kill_unconfirmed"] = True

        if h_job:
            _close_handle_checked(kernel32, h_job, "job", rec)
            h_job = None

    finally:
        # R4: Close any open handles and record failure/leak
        if h_thread:
            _close_handle_checked(kernel32, h_thread, "thread", rec)
        if h_process:
            _close_handle_checked(kernel32, h_process, "process", rec)

        # R3: If assign succeeded, process was a member; verify job empty before closing h_job
        if h_job and rec["job_empty_verification"] is None and assign_succeeded:
            empty_ok, assigned, count = _verify_job_empty(kernel32, h_job, timeout_s=5.0)
            rec["job_empty_verification"] = {"empty": empty_ok, "assigned": assigned, "count": count}
            if not empty_ok:
                rec["kill_unconfirmed"] = True

        if h_job:
            _close_handle_checked(kernel32, h_job, "job", rec)

        rec["wall_time_s"] = round(time.monotonic() - t0, 3)

    return rec


def _run_diagnostic(gate_path: Path, evidence_path: Path) -> int:
    """Core diagnostic lifecycle under strict exception, teardown, and evidence contracts."""
    nonce = f"diag_{secrets.token_hex(6)}"

    v2_root: Path | None = None
    v3_dir: Path | None = None
    v4_dir: Path | None = None
    diag_sid: str | None = None
    diag_password: str | None = None
    variant_results: list[dict[str, Any]] = []
    profile_exists_after_v0 = False
    teardown_errors: list[str] = []
    cleanup_data: dict[str, Any] = {}
    cp_cleanup = None
    probe_dirs_removal: list[dict[str, Any]] = []

    # Touchup (1): Win32 setup, command payload verification, directory creation, and execution enclosed in try
    try:
        if not contain_core._WIN32:
            raise RuntimeError("contain_core._WIN32 bindings unavailable")
        advapi32, kernel32, _ = contain_core._WIN32

        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        system_drive = os.environ.get("SystemDrive", "C:")
        cmd_exe = os.path.join(system_root, "System32", "cmd.exe")
        _, full_cmd = contain_core._verify_cmdline_absolute([cmd_exe, "/c", "exit", "0"])

        runner_temp = tempfile.gettempdir()
        v2_root = Path(runner_temp) / f"nexus-cplw-{nonce}"
        v2_work = v2_root / "work"
        v3_dir = Path(f"{system_drive}\\Windows\\Temp") / f"nexus-cplw-{nonce}"
        v4_dir = Path(f"{system_drive}\\Windows\\Temp") / f"nexus-cplw-{nonce}-acl"

        # mkdirs inside try
        v2_work.mkdir(parents=True, exist_ok=True)
        v3_dir.mkdir(parents=True, exist_ok=True)
        v4_dir.mkdir(parents=True, exist_ok=True)

        # Create fresh diagnostic principal
        diag_sid, diag_password = contain_core.create_principal(DIAG_PRINCIPAL_NAME)

        # Persist atomic receipt immediately after creation, before any subsequent effect
        receipt_data = {
            "sid": diag_sid,
            "username": DIAG_PRINCIPAL_NAME,
            "ts": _utcnow(),
        }
        _persist_atomic_receipt(RECEIPT_PATH, receipt_data)

        # V0: lpCurrentDirectory = NULL (ctypes None)
        v0_res = run_variant(
            advapi32, kernel32, full_cmd, "V0", "cwd_null_inherit", None,
            DIAG_PRINCIPAL_NAME, diag_password, diag_sid
        )
        variant_results.append(v0_res)

        # Check whether user profile directory exists after V0 (profile-creation evidence)
        user_profile_dir = Path(f"{system_drive}\\Users\\{DIAG_PRINCIPAL_NAME}")
        profile_exists_after_v0 = user_profile_dir.exists()

        # V1: System32 (world-readable control)
        v1_res = run_variant(
            advapi32, kernel32, full_cmd, "V1", "cwd_system32", os.path.join(system_root, "System32"),
            DIAG_PRINCIPAL_NAME, diag_password, diag_sid
        )
        variant_results.append(v1_res)

        # V2: Runner profile temp (failing class)
        v2_res = run_variant(
            advapi32, kernel32, full_cmd, "V2", "cwd_runner_profile_temp", str(v2_work),
            DIAG_PRINCIPAL_NAME, diag_password, diag_sid
        )
        variant_results.append(v2_res)

        # V3: Neutral Windows Temp (no explicit ACE)
        v3_res = run_variant(
            advapi32, kernel32, full_cmd, "V3", "cwd_windows_temp_no_grant", str(v3_dir),
            DIAG_PRINCIPAL_NAME, diag_password, diag_sid
        )
        variant_results.append(v3_res)

        # V4: Neutral Windows Temp with explicit SID-form ACE grant
        v4_res = run_variant(
            advapi32, kernel32, full_cmd, "V4", "cwd_windows_temp_with_grant", str(v4_dir),
            DIAG_PRINCIPAL_NAME, diag_password, diag_sid
        )
        variant_results.append(v4_res)

    except Exception as e:
        teardown_errors.append(f"Diagnostic execution exception: {type(e).__name__}: {str(e)}")

    finally:
        # Full teardown before proof entry: delete principal, sweep tokens, remove probe dirs
        if diag_sid:
            try:
                deleted = contain_core.delete_principal(captured_sid=diag_sid, username=DIAG_PRINCIPAL_NAME)
                if not deleted:
                    teardown_errors.append("delete_principal reported account was already absent")
            except Exception as e:
                teardown_errors.append(f"delete_principal error: {type(e).__name__}: {str(e)}")

        # Run frozen probe cleanup subprocess for fail-closed sweep and cleanup evidence
        cleanup_cmd = [
            sys.executable,
            "proof_runner_probe.py",
            "--mode",
            "cleanup",
            "--receipt",
            str(RECEIPT_PATH),
            "--cleanup-evidence",
            str(CLEANUP_EVIDENCE_PATH),
        ]
        try:
            cp_cleanup = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=30)
            if cp_cleanup.returncode != 0:
                teardown_errors.append(f"Probe cleanup subprocess returned nonzero ({cp_cleanup.returncode})")
        except Exception as e:
            teardown_errors.append(f"Probe cleanup subprocess failed to run: {type(e).__name__}: {str(e)}")

        if CLEANUP_EVIDENCE_PATH.is_file():
            try:
                cleanup_data = json.loads(CLEANUP_EVIDENCE_PATH.read_text(encoding="utf-8"))
                if cleanup_data.get("clean") is not True:
                    teardown_errors.append(f"Probe cleanup clean is not True ({cleanup_data.get('clean')!r})")
                if cleanup_data.get("receipt_read") is not True:
                    teardown_errors.append(f"Probe cleanup receipt_read is not True ({cleanup_data.get('receipt_read')!r})")
            except Exception as e:
                teardown_errors.append(f"Failed to parse cleanup evidence: {type(e).__name__}: {str(e)}")
        else:
            if diag_sid or RECEIPT_PATH.is_file():
                teardown_errors.append(f"Cleanup evidence missing at {CLEANUP_EVIDENCE_PATH}")

        # R8 / M2: Remove ONLY the three probe directories by exact path (truthful reporting)
        for p_dir in (v2_root, v3_dir, v4_dir):
            if p_dir is None:
                continue
            removed_ok = False
            dir_str = str(p_dir)
            try:
                if p_dir.exists():
                    shutil.rmtree(p_dir, ignore_errors=False)
                removed_ok = not p_dir.exists()
            except Exception as e:
                teardown_errors.append(f"rmtree failed for {dir_str}: {type(e).__name__}: {str(e)}")
                removed_ok = not p_dir.exists()

            probe_dirs_removal.append({"path": dir_str, "removed": removed_ok})
            if not removed_ok:
                teardown_errors.append(f"Probe directory was not removed: {dir_str}")

    # R4 / R5: Fold handle close failures and check for kill_unconfirmed across variants
    infra_variant_ids: list[str] = []
    close_failures_total = 0
    any_kill_unconfirmed = False

    for r in variant_results:
        vid = r["variant_id"]
        outcome = r.get("outcome")

        # R4: Fold handle close failures into teardown_errors
        for cf in r.get("close_failures", []):
            close_failures_total += 1
            teardown_errors.append(
                f"Handle close failure in {vid} ({cf.get('handle')}, gle={cf.get('gle')})"
            )

        if r.get("kill_unconfirmed"):
            any_kill_unconfirmed = True
            teardown_errors.append(f"KILL_UNCONFIRMED in variant {vid}")

        # R5: Classify DATA vs INFRASTRUCTURE
        is_data = False
        if outcome == "completed" and r.get("exit_code") == 0 and r.get("cplw_success") is True:
            is_data = True
        elif outcome == "spawn_error" and r.get("cplw_success") is False and r.get("cplw_gle") is not None:
            is_data = True

        if not is_data:
            infra_variant_ids.append(vid)
            teardown_errors.append(
                f"Infrastructure failure in variant {vid}: outcome={outcome!r}, "
                f"kill_unconfirmed={r.get('kill_unconfirmed')}"
            )

    # R6: Check V0 and V1 baseline. Any non-completed-exit-0 outcome (even DATA rejection) blocks proof
    v_map = {r["variant_id"]: r for r in variant_results}
    v0 = v_map.get("V0", {})
    v1 = v_map.get("V1", {})

    v0_pass = (v0.get("outcome") == "completed" and v0.get("exit_code") == 0)
    v1_pass = (v1.get("outcome") == "completed" and v1.get("exit_code") == 0)

    baseline_data_failed = False
    if len(variant_results) >= 2:
        if not (v0_pass and v1_pass):
            baseline_data_failed = True
            teardown_errors.append(
                f"Baseline prerequisite failed: V0 (outcome={v0.get('outcome')}, gle={v0.get('cplw_gle')}) "
                f"or V1 (outcome={v1.get('outcome')}, gle={v1.get('cplw_gle')}) not completed with exit 0"
            )

    teardown_clean = (len(teardown_errors) == 0)

    # Touchup (2): ANY infrastructure OR teardown failure (including CloseHandle fail)
    # must VOID actionable directory conclusion, replace explicit:
    # "matrix incomplete/non-actionable due infrastructure or teardown failure; no operative-variable conclusion"
    has_infra_or_teardown_failure = (
        bool(infra_variant_ids)
        or len(variant_results) < 5
        or any_kill_unconfirmed
        or close_failures_total > 0
        or (len(teardown_errors) > (1 if baseline_data_failed else 0))
    )

    if has_infra_or_teardown_failure or not teardown_clean:
        if baseline_data_failed and not (infra_variant_ids or any_kill_unconfirmed or close_failures_total > 0 or len(teardown_errors) > 1):
            conclusion = (
                f"Baseline execution failed (V0 outcome={v0.get('outcome')}, gle={v0.get('cplw_gle')}; "
                f"V1 outcome={v1.get('outcome')}, gle={v1.get('cplw_gle')}); session initialization or "
                "credential machinery broken on this runner image."
            )
        else:
            conclusion = (
                "matrix incomplete/non-actionable due infrastructure or teardown failure; "
                "no operative-variable conclusion"
            )
    elif not (v0_pass and v1_pass):
        conclusion = (
            f"Baseline execution failed (V0 outcome={v0.get('outcome')}, gle={v0.get('cplw_gle')}; "
            f"V1 outcome={v1.get('outcome')}, gle={v1.get('cplw_gle')}); session initialization or "
            "credential machinery broken on this runner image."
        )
    else:
        v2 = v_map.get("V2", {})
        v3 = v_map.get("V3", {})
        v4 = v_map.get("V4", {})

        diff_class: list[str] = []
        for vid, vrec in [("V2", v2), ("V3", v3), ("V4", v4)]:
            if vrec.get("outcome") == "spawn_error" and vrec.get("cplw_gle") != 267:
                diff_class.append(f"{vid} rejected with GLE {vrec.get('cplw_gle')}")

        if diff_class:
            conclusion = (
                f"{'; '.join(diff_class)}; not the err-267 class; no cwd-context conclusion"
            )
        else:
            v2_pass = (v2.get("outcome") == "completed" and v2.get("exit_code") == 0)
            v3_pass = (v3.get("outcome") == "completed" and v3.get("exit_code") == 0)
            v4_pass = (v4.get("outcome") == "completed" and v4.get("exit_code") == 0)

            v2_err267 = (v2.get("outcome") == "spawn_error" and v2.get("cplw_gle") == 267)
            v3_err267 = (v3.get("outcome") == "spawn_error" and v3.get("cplw_gle") == 267)
            v4_err267 = (v4.get("outcome") == "spawn_error" and v4.get("cplw_gle") == 267)

            if v2_err267 and v3_pass:
                conclusion = (
                    "Operational conclusion: neutral default-ACL cwd (V3) accepted by CPLW while runner "
                    "profile-temp cwd (V2) was rejected with err=267 for this principal class on this runner image; "
                    "runner profile tree location is operative."
                )
            elif v2_err267 and v3_err267 and v4_pass:
                conclusion = (
                    "Operational conclusion: neutral default-ACL cwd (V3) rejected with err=267 while explicit-grant "
                    "cwd (V4) was accepted for this principal class on this runner image; explicit ACE grant is operative."
                )
            elif v2_err267 and v3_err267 and v4_err267:
                conclusion = (
                    "Operational conclusion: both neutral default-ACL cwd (V3) and explicit-grant cwd (V4) "
                    "rejected with err=267 for this principal class on this runner image; directory ACL/location alone "
                    "is not operative."
                )
            else:
                conclusion = (
                    f"Operational conclusion: observed variant outcomes recorded: V0(null)={v0_pass}, "
                    f"V1(System32)={v1_pass}, V2(profile-temp)={v2_pass}, V3(neutral-no-grant)={v3_pass}, "
                    f"V4(neutral-with-grant)={v4_pass} for this principal class on this runner image."
                )

    evidence_out: dict[str, Any] = {
        "verdict": "diagnostic_complete" if teardown_clean else "diagnostic_failed_or_dirty",
        "timestamp_utc": _utcnow(),
        "nonce": nonce,
        "diagnostic_sid": diag_sid,
        "diagnostic_username": DIAG_PRINCIPAL_NAME,
        "profile_created_after_v0": profile_exists_after_v0,
        "variants": variant_results,
        "teardown": {
            "cleanup_subprocess_exit_code": cp_cleanup.returncode if cp_cleanup else None,
            "cleanup_evidence": cleanup_data,
            "any_kill_unconfirmed": any_kill_unconfirmed,
            "close_failures_total": close_failures_total,
            "probe_dirs_removal": probe_dirs_removal,
            "teardown_clean": teardown_clean,
            "teardown_errors": teardown_errors,
        },
        "operational_conclusion": conclusion,
    }

    evidence_path.write_text(json.dumps(evidence_out, indent=2), encoding="utf-8")
    print(f"DIAGNOSTIC: written to {evidence_path} (clean={teardown_clean})", flush=True)
    print(conclusion, flush=True)

    if not teardown_clean:
        print("DIAGNOSTIC_TEARDOWN_FAIL: one or more teardown checks failed:", file=sys.stderr, flush=True)
        for err in teardown_errors:
            print(f"  - {err}", file=sys.stderr, flush=True)
        return 1

    return 0


def main() -> int:
    evidence_path = DEFAULT_EVIDENCE_PATH
    try:
        parser = argparse.ArgumentParser(description="CreateProcessWithLogonW cwd-context diagnostic")
        parser.add_argument("--gate", default=str(DEFAULT_GATE_PATH), help="Path to proof_gate_eval.json")
        parser.add_argument("--evidence", default=str(DEFAULT_EVIDENCE_PATH), help="Path to write cplw_diag_evidence.json")
        args = parser.parse_args()

        gate_path = Path(args.gate).resolve()
        evidence_path = Path(args.evidence).resolve()

        # Precondition: gate must evaluate to exact bool True BEFORE any fixture
        if not check_gate_precondition(gate_path, evidence_path):
            return 1

        if sys.platform != "win32":
            evidence = {
                "verdict": "diagnostic_not_run_non_windows",
                "reason": f"Diagnostic requires Windows host, got {sys.platform}",
                "timestamp_utc": _utcnow(),
            }
            evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            return 1

        return _run_diagnostic(gate_path, evidence_path)
    except Exception as e:
        # Touchup (1): Top-level minimal evidence handler guarantees diagnostic_failed_or_dirty JSON before exit 1
        # No secrets in errors/evidence
        err_msg = f"{type(e).__name__}: {str(e)}"
        try:
            fallback_evidence = {
                "verdict": "diagnostic_failed_or_dirty",
                "reason": f"Top-level exception in diagnostic execution: {err_msg}",
                "timestamp_utc": _utcnow(),
            }
            evidence_path.write_text(json.dumps(fallback_evidence, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"DIAGNOSTIC_FATAL: {err_msg}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
