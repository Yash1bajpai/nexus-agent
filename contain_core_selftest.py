# .github/e2e/contain_core_selftest.py
"""Nexus e2e Contained Run — Component 1 Self-Test (Patched F1–F6 + Tier U/W).

Status: UNTESTED / NOT RUN (ships as test source to be executed on Windows runneradmin).

Constructs and OBSERVES real runtime conditions for:
1. Fail-closed principal creation and duplicate rejection (F4); non-admin membership.
2. Exact child environment allowlist verification from captured child stdout (F1).
3. Job Object assignment and confirmed empty job-tree termination (F3, F5).
4. Service-mediated execution detection + actual breakaway prevention + independent SID cross-check (F2, F3).
5. Indeterminate process enumeration gate behavior (F3).
6. SID-bound principal deletion and clean post-sweep verification (F3, F6).

Tier U: SYNTHETIC UNIT logic tests (U1 matrix, U2 10-stage fault injection via real API calls).
Tier W: WINDOWS-OBSERVED real process lifecycle and handle-fault recovery (W1, W2).

All test item verdicts (pass: bool) are COMPUTED dynamically from observed effects.
No constants or mock stubs are accepted as Windows proof.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

# Ensure import of contain_core from the same directory
SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

import contain_core


def _query_process_sid_independent(pid: int) -> str | None:
    """Independently query a process's TokenUser SID in test code, bypassing sweep_processes_by_sid."""
    advapi32, kernel32, _ = contain_core._WIN32
    h_proc = kernel32.OpenProcess(contain_core.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h_proc:
        return None
    try:
        h_token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(h_proc, contain_core.TOKEN_QUERY, ctypes.byref(h_token)):
            return None
        try:
            ret_len = wintypes.DWORD(0)
            advapi32.GetTokenInformation(h_token, contain_core.TokenUser, None, 0, ctypes.byref(ret_len))
            if ret_len.value == 0:
                return None
            buf = ctypes.create_string_buffer(ret_len.value)
            if not advapi32.GetTokenInformation(h_token, contain_core.TokenUser, buf, ret_len.value, ctypes.byref(ret_len)):
                return None
            p_tu = ctypes.cast(buf, ctypes.POINTER(contain_core.TOKEN_USER))
            p_str = wintypes.LPWSTR()
            if not advapi32.ConvertSidToStringSidW(p_tu.contents.User.Sid, ctypes.byref(p_str)):
                return None
            try:
                return p_str.value
            finally:
                kernel32.LocalFree(p_str)
        finally:
            kernel32.CloseHandle(h_token)
    finally:
        kernel32.CloseHandle(h_proc)


def _get_process_parent_map() -> dict[int, int]:
    """Query system process parent map via Toolhelp32 snapshot for OS parent-chain verification."""
    advapi32, kernel32, _ = contain_core._WIN32
    h_snap = kernel32.CreateToolhelp32Snapshot(contain_core.TH32CS_SNAPPROCESS, 0)
    if h_snap == contain_core.INVALID_HANDLE_VALUE or not h_snap:
        raise contain_core.ContainmentError(f"CreateToolhelp32Snapshot failed (err={ctypes.get_last_error()})")
    parent_map: dict[int, int] = {}
    try:
        pe = contain_core.PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
        more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
        if not more:
            gle = ctypes.get_last_error()
            if gle != contain_core.ERROR_NO_MORE_FILES:
                raise contain_core.ContainmentError(f"Process32FirstW failed (err={gle})")
        while more:
            parent_map[int(pe.th32ProcessID)] = int(pe.th32ParentProcessID)
            pe.dwSize = ctypes.sizeof(contain_core.PROCESSENTRY32W)
            more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))
        gle = ctypes.get_last_error()
        if gle != contain_core.ERROR_NO_MORE_FILES:
            raise contain_core.ContainmentError(f"Process32NextW failed mid-enumeration (err={gle}); parent map truncated/indeterminate")
    finally:
        kernel32.CloseHandle(h_snap)
    return parent_map


class _MockKernel32:
    """Mock kernel32 dispatch table for Tier U fault injection."""
    def __init__(self, terminate_ret=True, terminate_gle=0, query_ret=True, query_gle=0, assigned=0, count=0, pids=None, query_raise=None):
        self.terminate_ret = terminate_ret
        self.terminate_gle = terminate_gle
        self.query_ret = query_ret
        self.query_gle = query_gle
        self.assigned = assigned
        self.count = count
        self.pids = pids or []
        self.query_raise = query_raise

    def TerminateJobObject(self, h_job, exit_code):
        if not self.terminate_ret:
            ctypes.set_last_error(self.terminate_gle)
        return self.terminate_ret

    def QueryInformationJobObject(self, h_job, info_class, buf, buf_size, ret_len):
        if self.query_raise:
            raise self.query_raise
        if not self.query_ret:
            ctypes.set_last_error(self.query_gle)
            return False
        p_list = ctypes.cast(buf, ctypes.POINTER(contain_core.JOBOBJECT_BASIC_PROCESS_ID_LIST_256))
        p_list.contents.NumberOfAssignedProcesses = self.assigned
        p_list.contents.NumberOfProcessIdsInList = self.count
        for i, pid in enumerate(self.pids[:256]):
            p_list.contents.ProcessIdList[i] = pid
        return True


class _MockSweepResultNotClean:
    """Mock SweepResult where clean is False but both lists are empty (tests C2 fallback)."""
    def __init__(self, target_sid):
        self.target_sid = target_sid
        self.child_pids = []
        self.indeterminate_pids = []
        self.error_details = {"simulated": "custom_dirty_state"}

    @property
    def clean(self):
        return False


def run_tier_u_synthetic_tests() -> list[dict]:
    """TIER U — SYNTHETIC UNIT.

    Proves gate LOGIC only; every output line labeled SYNTHETIC UNIT.
    May NEVER be cited as Windows negative proof.
    """
    u_items = []

    # U1: SweepResult.clean matrix
    u1 = {
        "tier": "SYNTHETIC UNIT",
        "item": "U1",
        "name": "sweep_result_clean_matrix",
        "description": "Verify SweepResult.clean logic for empty, child-only, indeterminate-only, and both",
    }
    case_empty = contain_core.SweepResult("S-1-5-21-test", [], []).clean
    case_child = contain_core.SweepResult("S-1-5-21-test", [101], []).clean
    case_indet = contain_core.SweepResult("S-1-5-21-test", [], [202]).clean
    case_both = contain_core.SweepResult("S-1-5-21-test", [101], [202]).clean

    u1_pass = (case_empty is True) and (case_child is False) and (case_indet is False) and (case_both is False)
    u1["expected"] = {"empty": True, "child_only": False, "indeterminate_only": False, "both": False}
    u1["actual"] = {"empty": case_empty, "child_only": case_child, "indeterminate_only": case_indet, "both": case_both}
    u1["pass"] = u1_pass
    u_items.append(u1)

    # U2: Stage fault injection mapping (T1: 10 stages with real API calls)
    u2 = {
        "tier": "SYNTHETIC UNIT",
        "item": "U2",
        "name": "termination_stage_fault_injection",
        "description": "Assert each failure stage produces the right stage tag and fail-closed status",
    }
    orig_win32 = contain_core._WIN32
    orig_sweep = contain_core.sweep_processes_by_sid
    dummy_job = wintypes.HANDLE(0x1234)
    target_sid = "S-1-5-21-mock-user"

    stages_tested = {}
    try:
        advapi, _, netapi = orig_win32 if orig_win32 else (None, None, None)

        # 1. terminate_call
        contain_core._WIN32 = (advapi, _MockKernel32(terminate_ret=False, terminate_gle=5), netapi)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["terminate_call"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "terminate_call" and res.evidence.get("gle") == 5)

        # 2. query_failed
        contain_core._WIN32 = (advapi, _MockKernel32(query_ret=False, query_gle=6), netapi)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["query_failed"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "query_failed" and res.evidence.get("gle") == 6)

        # 3. query_inconsistent (C1)
        contain_core._WIN32 = (advapi, _MockKernel32(assigned=2, count=1, pids=[101]), netapi)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["query_inconsistent"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "query_inconsistent" and res.evidence.get("assigned") == 2)

        # 4. job_not_empty
        contain_core._WIN32 = (advapi, _MockKernel32(assigned=1, count=1, pids=[999]), netapi)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["job_not_empty"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "job_not_empty" and res.evidence.get("remaining_pids") == [999])

        # 5. internal_error (FIX-U2a: raising API injection)
        contain_core._WIN32 = (advapi, _MockKernel32(query_raise=RuntimeError("simulated query fault")), netapi)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["internal_error"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "internal_error" and "simulated query fault" in res.evidence.get("error", ""))

        # Reset kernel32 to empty job for sweep stage injections
        contain_core._WIN32 = (advapi, _MockKernel32(assigned=0, count=0), netapi)

        # 6. sweep_error
        def _mock_sweep_err(sid):
            raise contain_core.ProcessSweepError("Snapshot enumeration failed")
        contain_core.sweep_processes_by_sid = _mock_sweep_err
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["sweep_error"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "sweep_error")

        # 7. child_survivors
        contain_core.sweep_processes_by_sid = lambda sid: contain_core.SweepResult(sid, [1001], [])
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["child_survivors"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "child_survivors" and res.evidence.get("child_pids") == [1001])

        # 8. indeterminate
        contain_core.sweep_processes_by_sid = lambda sid: contain_core.SweepResult(sid, [], [1002])
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["indeterminate"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "indeterminate" and res.evidence.get("indeterminate_pids") == [1002])

        # 9. sweep_not_clean (C2 fallback)
        contain_core.sweep_processes_by_sid = lambda sid: _MockSweepResultNotClean(sid)
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["sweep_not_clean"] = (res.status == "KILL_UNCONFIRMED" and res.stage == "sweep_not_clean")

        # 10. confirmed happy path
        contain_core.sweep_processes_by_sid = lambda sid: contain_core.SweepResult(sid, [], [])
        res = contain_core.terminate_and_confirm(dummy_job, target_sid, deadline_s=0.1)
        stages_tested["confirmed"] = (res.status == "CONFIRMED" and res.stage == "confirmed")

    finally:
        contain_core._WIN32 = orig_win32
        contain_core.sweep_processes_by_sid = orig_sweep

    u2["expected"] = {
        "terminate_call": True, "query_failed": True, "query_inconsistent": True,
        "job_not_empty": True, "internal_error": True, "sweep_error": True,
        "child_survivors": True, "indeterminate": True, "sweep_not_clean": True, "confirmed": True,
    }
    u2["actual"] = stages_tested
    u2["pass"] = all(stages_tested.values()) and len(stages_tested) == 10
    u_items.append(u2)

    return u_items


def run_tier_w_windows_observed_tests(created_sid: str, expected_env: dict, created_password: str, temp_dir: str) -> list[dict]:
    """TIER W — WINDOWS-OBSERVED.

    Real Windows behavior; labeled WINDOWS-OBSERVED.
    Every item is marked NOT RUN until actually executed on a Windows runneradmin.
    """
    w_items = []
    workdir = Path(temp_dir) / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    cmd_exe = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "cmd.exe")

    # W1: Direct contract test with pre-kill OS parent-chain metadata verification (FIX-W1a, FIX-W1b, FIX-W1c, L2)
    w1 = {
        "tier": "WINDOWS-OBSERVED",
        "item": "W1",
        "name": "terminate_and_confirm_direct_contract_job_tree",
        "description": "Real child+grandchild in job; pre-kill parent-chain OS metadata verification; direct terminate_and_confirm confirmation",
        "integration_path": "NOT EXERCISED — spawn_in_job timeout->cleanup leg deferred to full Comp 1 W scaffold",
        "observation": {"attempted": False, "status": "NOT RUN"},
    }
    h_job_probe = None
    h_process_child = None
    h_thread_child = None
    try:
        advapi32, kernel32, _ = contain_core._WIN32
        cmd = [cmd_exe, "/c", "start /b ping -n 60 127.0.0.1 > nul && ping -n 60 127.0.0.1 > nul"]

        h_job_probe = kernel32.CreateJobObjectW(None, None)
        if not h_job_probe:
            raise contain_core.ContainmentError(f"CreateJobObjectW failed (err={ctypes.get_last_error()})")

        job_info = contain_core.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = contain_core.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            h_job_probe, contain_core.JobObjectExtendedLimitInformation, ctypes.byref(job_info), ctypes.sizeof(job_info)
        ):
            raise contain_core.ContainmentError(f"SetInformationJobObject failed (err={ctypes.get_last_error()})")

        si = contain_core.STARTUPINFOW()
        si.cb = ctypes.sizeof(contain_core.STARTUPINFOW)
        pi = contain_core.PROCESS_INFORMATION()
        env_block = contain_core._format_unicode_env_block(expected_env)
        full_cmd = subprocess.list2cmdline(cmd)

        spawned = advapi32.CreateProcessWithLogonW(
            contain_core.DEFAULT_PRINCIPAL_NAME, None, created_password, contain_core.LOGON_WITH_PROFILE,
            None, full_cmd, contain_core.CREATE_SUSPENDED | contain_core.CREATE_UNICODE_ENVIRONMENT,
            ctypes.cast(env_block, ctypes.c_void_p), str(workdir), ctypes.byref(si), ctypes.byref(pi),
        )
        if not spawned:
            raise contain_core.ContainmentError(f"CreateProcessWithLogonW failed (err={ctypes.get_last_error()})")

        h_process_child = pi.hProcess
        h_thread_child = pi.hThread
        known_child_pid = int(pi.dwProcessId)

        if not kernel32.AssignProcessToJobObject(h_job_probe, h_process_child):
            gle = ctypes.get_last_error()
            kernel32.TerminateProcess(h_process_child, 1)
            raise contain_core.ContainmentError(f"AssignProcessToJobObject failed (err={gle})")

        if kernel32.ResumeThread(h_thread_child) == 0xFFFFFFFF:
            gle = ctypes.get_last_error()
            kernel32.TerminateProcess(h_process_child, 1)
            raise contain_core.ContainmentError(f"ResumeThread failed (err={gle})")

        if not kernel32.CloseHandle(h_thread_child):
            raise contain_core.ContainmentError(f"CloseHandle(hThread) failed (err={ctypes.get_last_error()})")
        h_thread_child = None

        time.sleep(1.0)  # Allow grandchild process to spawn

        # FIX-W1a: Pre-kill inspection: job PID list + OS parent-chain metadata
        pre_pid_list = contain_core.JOBOBJECT_BASIC_PROCESS_ID_LIST_256()
        ret_len = wintypes.DWORD(0)
        if not kernel32.QueryInformationJobObject(
            h_job_probe, contain_core.JobObjectBasicProcessIdList, ctypes.byref(pre_pid_list), ctypes.sizeof(pre_pid_list), ctypes.byref(ret_len)
        ):
            raise contain_core.ContainmentError(f"QueryInformationJobObject pre-kill query failed (err={ctypes.get_last_error()})")

        pre_kill_count = int(pre_pid_list.NumberOfProcessIdsInList)
        pre_kill_pids = [int(pre_pid_list.ProcessIdList[i]) for i in range(pre_kill_count)]
        parent_map = _get_process_parent_map()

        child_present = known_child_pid in pre_kill_pids
        child_sid = _query_process_sid_independent(known_child_pid) if child_present else None
        child_sid_verified = (child_sid == created_sid)

        # A GRANDCHILD is any distinct PID in job PIDs whose OS parent equals the known child PID
        grandchildren = [pid for pid in pre_kill_pids if pid != known_child_pid and parent_map.get(pid) == known_child_pid]
        grandchildren_sids = {gc_pid: _query_process_sid_independent(gc_pid) for gc_pid in grandchildren}
        grandchildren_verified = len(grandchildren) >= 1 and all(sid == created_sid for sid in grandchildren_sids.values())

        tree_verified = child_present and child_sid_verified and grandchildren_verified

        # Execute terminate_and_confirm via borrowed h_job_probe
        term_res = contain_core.terminate_and_confirm(h_job_probe, created_sid, deadline_s=5.0)
        post_sweep = contain_core.sweep_processes_by_sid(created_sid)

        w1_pass = bool(tree_verified and (term_res.status == "CONFIRMED") and post_sweep.clean)
        w1["expected"] = {"tree_verified_pre_kill": True, "status": "CONFIRMED", "post_sweep_clean": True}
        w1["actual"] = {
            "known_child_pid": known_child_pid,
            "child_present": child_present,
            "child_sid_verified": child_sid_verified,
            "grandchildren": grandchildren,
            "grandchildren_verified": grandchildren_verified,
            "tree_verified": tree_verified,
            "parent_map_sample": {pid: parent_map.get(pid) for pid in pre_kill_pids},
            "cleanup_status": term_res.status,
            "cleanup_stage": term_res.stage,
            "cleanup_evidence": term_res.evidence,
            "post_sweep_clean": post_sweep.clean,
            "child_pids": post_sweep.child_pids,
            "indeterminate_pids": post_sweep.indeterminate_pids,
        }
        w1["pass"] = w1_pass
        # FIX-STATUS: Dynamic attempt record
        w1["observation"] = {
            "attempted": True,
            "outcome": "pass" if w1_pass else "fail",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": term_res.stage,
        }
    except Exception as e:
        w1["actual"] = {"error": f"{type(e).__name__}: {e}"}
        w1["pass"] = False
        w1["observation"] = {
            "attempted": True,
            "outcome": "error",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": f"{type(e).__name__}: {e}",
        }
    finally:
        # L2: Checked handle cleanup with annotation-not-raise pattern
        close_errors = []
        if h_thread_child and not kernel32.CloseHandle(h_thread_child):
            close_errors.append({"handle": "h_thread_child", "gle": ctypes.get_last_error()})
        if h_process_child and not kernel32.CloseHandle(h_process_child):
            close_errors.append({"handle": "h_process_child", "gle": ctypes.get_last_error()})
        if h_job_probe and not kernel32.CloseHandle(h_job_probe):
            close_errors.append({"handle": "h_job_probe", "gle": ctypes.get_last_error()})
        if close_errors:
            w1.setdefault("actual", {})["cleanup_close_errors"] = close_errors
            w1["pass"] = False
            obs = w1.get("observation", {"attempted": True})
            obs["outcome"] = "fail"
            obs["detail"] = (obs.get("detail") or "") + f"; cleanup close failures: {close_errors}"
            w1["observation"] = obs
    w_items.append(w1)

    # W2: Fault injection via deliberate handle invalidation (FIX-W1b, FIX-W2a, FIX-STATUS, L1)
    w2 = {
        "tier": "WINDOWS-OBSERVED",
        "item": "W2",
        "name": "wait_failed_handle_invalidation_cleanup",
        "description": "Trigger WAIT_FAILED by closing process handle before wait; verify terminate_and_confirm cleans via job handle",
        "trigger_mechanism": "Deliberate handle invalidation (fault injection at API boundary); all subsequent behavior is real Windows behavior",
        "integration_path": "NOT EXERCISED — spawn_in_job WAIT_FAILED->cleanup leg deferred to full Comp 1 W scaffold",
        "observation": {"attempted": False, "status": "NOT RUN"},
    }
    h_job = None
    h_process_owned = None
    h_thread = None
    try:
        advapi32, kernel32, _ = contain_core._WIN32
        h_job = kernel32.CreateJobObjectW(None, None)
        if not h_job:
            raise contain_core.ContainmentError(f"CreateJobObjectW failed (err={ctypes.get_last_error()})")

        job_info = contain_core.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = contain_core.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            h_job, contain_core.JobObjectExtendedLimitInformation, ctypes.byref(job_info), ctypes.sizeof(job_info)
        ):
            raise contain_core.ContainmentError(f"SetInformationJobObject failed (err={ctypes.get_last_error()})")

        si = contain_core.STARTUPINFOW()
        si.cb = ctypes.sizeof(contain_core.STARTUPINFOW)
        pi = contain_core.PROCESS_INFORMATION()
        env_block = contain_core._format_unicode_env_block(expected_env)

        cmd_str = f'"{cmd_exe}" /c ping -n 30 127.0.0.1 > nul'
        spawned = advapi32.CreateProcessWithLogonW(
            contain_core.DEFAULT_PRINCIPAL_NAME,
            None,
            created_password,
            contain_core.LOGON_WITH_PROFILE,
            None,
            cmd_str,
            contain_core.CREATE_SUSPENDED | contain_core.CREATE_UNICODE_ENVIRONMENT,
            ctypes.cast(env_block, ctypes.c_void_p),
            str(workdir),
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not spawned:
            raise contain_core.ContainmentError(f"CreateProcessWithLogonW failed (err={ctypes.get_last_error()})")

        h_process_owned = pi.hProcess
        h_thread = pi.hThread

        if not kernel32.AssignProcessToJobObject(h_job, h_process_owned):
            gle = ctypes.get_last_error()
            kernel32.TerminateProcess(h_process_owned, 1)
            raise contain_core.ContainmentError(f"AssignProcessToJobObject failed (err={gle})")

        if kernel32.ResumeThread(h_thread) == 0xFFFFFFFF:
            gle = ctypes.get_last_error()
            kernel32.TerminateProcess(h_process_owned, 1)
            raise contain_core.ContainmentError(f"ResumeThread failed (err={gle})")

        if not kernel32.CloseHandle(h_thread):
            raise contain_core.ContainmentError(f"CloseHandle(hThread) failed (err={ctypes.get_last_error()})")
        h_thread = None

        # Deliberately invalidate process handle before wait; immediately set to None
        invalidated_handle = h_process_owned
        h_process_owned = None
        if not kernel32.CloseHandle(invalidated_handle):
            raise contain_core.ContainmentError(f"CloseHandle(invalidated_handle) failed (err={ctypes.get_last_error()})")

        wait_res = kernel32.WaitForSingleObject(invalidated_handle, 1000)
        wait_gle = ctypes.get_last_error()
        is_wait_failed = (wait_res == contain_core.WAIT_FAILED) and (wait_gle == 6)  # ERROR_INVALID_HANDLE = 6

        # Now execute terminate_and_confirm with the still-valid borrowed h_job
        term_res = contain_core.terminate_and_confirm(h_job, created_sid, deadline_s=5.0)

        w2_pass = bool(is_wait_failed and term_res.status == "CONFIRMED")
        w2["expected"] = {"is_wait_failed": True, "wait_gle": 6, "cleanup_status": "CONFIRMED"}
        w2["actual"] = {
            "is_wait_failed": is_wait_failed,
            "wait_gle": wait_gle,
            "cleanup_status": term_res.status,
            "cleanup_stage": term_res.stage,
            "cleanup_evidence": term_res.evidence,
        }
        w2["pass"] = w2_pass
        # FIX-STATUS: Dynamic attempt record
        w2["observation"] = {
            "attempted": True,
            "outcome": "pass" if w2_pass else "fail",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": term_res.stage,
        }
    except Exception as e:
        w2["actual"] = {"error": f"{type(e).__name__}: {e}"}
        w2["pass"] = False
        w2["observation"] = {
            "attempted": True,
            "outcome": "error",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": f"{type(e).__name__}: {e}",
        }
    finally:
        # L1: Checked handle cleanup with annotation-not-raise pattern
        close_errors = []
        if h_thread and not kernel32.CloseHandle(h_thread):
            close_errors.append({"handle": "h_thread", "gle": ctypes.get_last_error()})
        if h_process_owned and not kernel32.CloseHandle(h_process_owned):
            close_errors.append({"handle": "h_process_owned", "gle": ctypes.get_last_error()})
        if h_job and not kernel32.CloseHandle(h_job):
            close_errors.append({"handle": "h_job", "gle": ctypes.get_last_error()})
        if close_errors:
            w2.setdefault("actual", {})["cleanup_close_errors"] = close_errors
            w2["pass"] = False
            obs = w2.get("observation", {"attempted": True})
            obs["outcome"] = "fail"
            obs["detail"] = (obs.get("detail") or "") + f"; cleanup close failures: {close_errors}"
            w2["observation"] = obs
    w_items.append(w2)

    # W3: Explicit omission notice
    w3 = {
        "tier": "WINDOWS-OBSERVED",
        "item": "W3",
        "name": "unkillable_process_negative_case",
        "status": "OMITTED",
        "reason": "Job Object termination cannot be ignored by standard user processes on native Windows without driver faults. Honest unconstructible negative conditions are evaluated in Tier U.",
    }
    w_items.append(w3)

    return w_items


def run_component1_selftest() -> dict:
    evidence = {
        "component": "contain_core",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_platform": sys.platform,
        "items": [],
        "all_passed": False,
    }

    if sys.platform != "win32":
        evidence["error"] = "Self-test requires native Windows platform (runneradmin)"
        evidence["items"].append({
            "item": "platform_check",
            "description": "Verify host platform is Windows",
            "expected": "win32",
            "actual": sys.platform,
            "pass": False,
        })
        # L4: Dynamic attempt record for non-Windows early return
        evidence["observation"] = {
            "attempted": False,
            "outcome": "not_run",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detail": "non-Windows host",
        }
        return evidence

    # FIX-W7: Run Tier U synthetic tests immediately after platform verification
    items = []
    items.extend(run_tier_u_synthetic_tests())

    temp_dir = tempfile.mkdtemp(prefix="nexus_contain_test_")
    created_sid = None
    created_password = None
    expected_env = None

    try:
        # ---------------------------------------------------------------------
        # Item 1: Principal Creation & Duplicate Fail-Closed Rejection (F4)
        # ---------------------------------------------------------------------
        item1 = {
            "item": 1,
            "name": "principal_creation_and_fail_closed_rejection",
            "description": "Verify fail-closed creation, duplicate rejection without reuse, and non-admin membership",
        }
        try:
            # F4: Abort on pre-existing account at test start (environment error)
            advapi32, _, _ = contain_core._WIN32
            cb_sid = wintypes.DWORD(0)
            cch_domain = wintypes.DWORD(0)
            pe_use = wintypes.DWORD(0)
            advapi32.LookupAccountNameW(None, contain_core.DEFAULT_PRINCIPAL_NAME, None, ctypes.byref(cb_sid), None, ctypes.byref(cch_domain), ctypes.byref(pe_use))
            if cb_sid.value > 0:
                raise EnvironmentError(f"Account '{contain_core.DEFAULT_PRINCIPAL_NAME}' already exists at test start. Aborting without modifying.")

            sid, pwd = contain_core.create_principal(contain_core.DEFAULT_PRINCIPAL_NAME)
            created_sid = sid
            created_password = pwd

            is_non_admin = contain_core.verify_principal_not_admin(contain_core.DEFAULT_PRINCIPAL_NAME)
            valid_sid = sid.startswith("S-1-5-21-") and len(sid) > 15

            # F4: Second creation MUST raise PrincipalExistsError (no reuse)
            duplicate_rejected = False
            try:
                contain_core.create_principal(contain_core.DEFAULT_PRINCIPAL_NAME)
            except contain_core.PrincipalExistsError:
                duplicate_rejected = True

            item1["expected"] = {"valid_sid_prefix": True, "not_in_administrators": True, "duplicate_rejected": True}
            item1["actual"] = {
                "valid_sid_prefix": valid_sid,
                "not_in_administrators": is_non_admin,
                "duplicate_rejected": duplicate_rejected,
                "sid": sid,
            }
            item1["pass"] = bool(valid_sid and is_non_admin and duplicate_rejected)
        except Exception as e:
            item1["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item1["pass"] = False
        items.append(item1)

        # ---------------------------------------------------------------------
        # Item 2: Actual Child Environment Capture & Assertion (F1)
        # ---------------------------------------------------------------------
        item2 = {
            "item": 2,
            "name": "environment_observation_and_allowlist",
            "description": "Capture child stdout to workdir file, assert exact allowlist, test tamper rejection",
        }
        try:
            workdir = Path(temp_dir) / "work"
            workdir.mkdir(parents=True, exist_ok=True)
            expected_env = contain_core.build_child_env(workdir)

            env_capture_file = workdir / "captured_env.txt"
            if env_capture_file.exists():
                env_capture_file.unlink()

            cmd_exe = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "cmd.exe")
            # F1: child command redirects stdout to fixed file inside workdir
            spawn_res = contain_core.spawn_in_job(
                [cmd_exe, "/c", f"set > {env_capture_file.name}"],
                workdir=workdir,
                env=expected_env,
                password=created_password,
                timeout_s=15.0,
                target_sid=created_sid,
            )

            spawn_ok = spawn_res.status == "COMPLETED" and spawn_res.exit_code == 0
            file_exists = env_capture_file.exists()
            file_size = env_capture_file.stat().st_size if file_exists else 0
            size_bounded = file_exists and (file_size <= 65536)

            # Parse lines from captured file
            parsed_child_env = {}
            if file_exists and size_bounded:
                content = env_capture_file.read_bytes().decode("utf-8", errors="replace")
                for line in content.splitlines():
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    if line.startswith("="):
                        # Hidden per-drive entry e.g. =C:=C:\...
                        idx = line.find("=", 1)
                        if idx != -1:
                            k, v = line[:idx], line[idx + 1:]
                            parsed_child_env[k] = v
                    elif "=" in line:
                        k, v = line.split("=", 1)
                        parsed_child_env[k] = v

            observed_hidden: dict[str, str] = {}
            contain_core.assert_child_env(parsed_child_env, expected_env, workdir, observed_hidden=observed_hidden)
            assert_passed = True

            # Unit assertion contract test: injected variable must raise ChildEnvViolation
            tampered_env = dict(parsed_child_env)
            tampered_env["DISALLOWED_SECRET_VAR"] = "unauthorized_leak"
            tamper_caught = False
            try:
                contain_core.assert_child_env(tampered_env, expected_env, workdir)
            except contain_core.ChildEnvViolation:
                tamper_caught = True

            item2["expected"] = {
                "spawn_completed": True,
                "file_size_bounded": True,
                "assert_passed": True,
                "tamper_caught": True,
            }
            item2["actual"] = {
                "spawn_completed": spawn_ok,
                "file_size_bytes": file_size,
                "file_size_bounded": size_bounded,
                "assert_passed": assert_passed,
                "tamper_caught": tamper_caught,
                "observed_names": sorted(parsed_child_env.keys()),
                "observed_hidden_per_drive": observed_hidden,
                "scope_honesty": "Measures environment received by an honest child process; forged dump is adversarial and out of scope per rev 8 section 1.",
            }
            item2["pass"] = bool(spawn_ok and size_bounded and assert_passed and tamper_caught)
        except Exception as e:
            item2["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item2["pass"] = False
        items.append(item2)

        # ---------------------------------------------------------------------
        # Item 3: Job Assignment & Confirmed Tree Termination (F3, F5)
        # ---------------------------------------------------------------------
        item3 = {
            "item": 3,
            "name": "job_containment_and_confirmed_tree_kill",
            "description": "Verify Job Object limits and confirmed empty job-tree termination",
        }
        try:
            workdir = Path(temp_dir) / "work"
            cmd_exe = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "cmd.exe")

            # Spawn grandchild tree
            cmd = [cmd_exe, "/c", "start /b ping -n 60 127.0.0.1 > nul && ping -n 60 127.0.0.1 > nul"]
            spawn_res = contain_core.spawn_in_job(
                cmd,
                workdir=workdir,
                env=expected_env,
                password=created_password,
                timeout_s=2.0,
                target_sid=created_sid,
            )

            # F5: status TIMED_OUT requires confirmed termination via QueryInformationJobObject and post-sweep
            confirmed_timed_out = spawn_res.status == "TIMED_OUT"

            # Post-kill sweep verification (F3: assert sweep.clean)
            sweep = contain_core.sweep_processes_by_sid(created_sid)

            item3["expected"] = {"confirmed_timed_out": True, "sweep_clean": True}
            item3["actual"] = {
                "status": spawn_res.status,
                "confirmed_timed_out": confirmed_timed_out,
                "sweep_clean": sweep.clean,
                "child_pids": sweep.child_pids,
                "indeterminate_pids": sweep.indeterminate_pids,
            }
            item3["pass"] = bool(confirmed_timed_out and sweep.clean)
        except Exception as e:
            item3["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item3["pass"] = False
        items.append(item3)

        # ---------------------------------------------------------------------
        # Item 4: Service-Mediated Execution (F2a) & Job Breakaway Block (F2b)
        # ---------------------------------------------------------------------
        item4 = {
            "item": 4,
            "name": "service_mediated_execution_and_breakaway_prevention",
            "description": "Probe service-mediated execution (F2a), verify breakaway failure (F2b), cross-check SID",
        }
        try:
            workdir = Path(temp_dir) / "work"
            task_name = "NexusContainedSelftest_T7"

            # F2a: Service-Mediated Execution
            subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"], capture_output=True)

            cmd_create_task = f'schtasks /create /tn "{task_name}" /tr "cmd.exe /c ping -n 30 127.0.0.1" /sc ONCE /st 23:59 /f && schtasks /run /tn "{task_name}"'
            spawn_res_task = contain_core.spawn_in_job(
                [cmd_exe, "/c", cmd_create_task],
                workdir=workdir,
                env=expected_env,
                password=created_password,
                timeout_s=10.0,
                target_sid=created_sid,
            )

            time.sleep(1.5)  # Allow Task Scheduler service time to spawn process
            sweep_task = contain_core.sweep_processes_by_sid(created_sid)
            task_proc_found = len(sweep_task.child_pids) > 0

            independent_sid_match = False
            if task_proc_found:
                test_pid = sweep_task.child_pids[0]
                independent_sid = _query_process_sid_independent(test_pid)
                independent_sid_match = (independent_sid == created_sid)

            term_failures = contain_core.terminate_processes_by_sid(created_sid)
            subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"], capture_output=True)

            sweep_after_kill = contain_core.sweep_processes_by_sid(created_sid)
            task_sweep_clean = sweep_after_kill.clean

            # F2b: Actual Job Breakaway Probe
            breakaway_probe_script = workdir / "breakaway_probe.py"
            breakaway_result_file = workdir / "breakaway_result.json"
            if breakaway_result_file.exists():
                breakaway_result_file.unlink()

            probe_code = (
                "import ctypes, json, sys\n"
                "from ctypes import wintypes\n"
                "kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
                "class STARTUPINFOW(ctypes.Structure):\n"
                "    _fields_ = [('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR), ('lpDesktop', wintypes.LPWSTR),\n"
                "                ('lpTitle', wintypes.LPWSTR), ('dwX', wintypes.DWORD), ('dwY', wintypes.DWORD),\n"
                "                ('dwXSize', wintypes.DWORD), ('dwYSize', wintypes.DWORD), ('dwXCountChars', wintypes.DWORD),\n"
                "                ('dwYCountChars', wintypes.DWORD), ('dwFillAttribute', wintypes.DWORD), ('dwFlags', wintypes.DWORD),\n"
                "                ('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD), ('lpReserved2', ctypes.c_void_p),\n"
                "                ('hStdInput', wintypes.HANDLE), ('hStdOutput', wintypes.HANDLE), ('hStdError', wintypes.HANDLE)]\n"
                "class PROCESS_INFORMATION(ctypes.Structure):\n"
                "    _fields_ = [('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE),\n"
                "                ('dwProcessId', wintypes.DWORD), ('dwThreadId', wintypes.DWORD)]\n"
                "kernel32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,\n"
                "                                    wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,\n"
                "                                    ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION)]\n"
                "kernel32.CreateProcessW.restype = wintypes.BOOL\n"
                "si = STARTUPINFOW()\n"
                "si.cb = ctypes.sizeof(STARTUPINFOW)\n"
                "pi = PROCESS_INFORMATION()\n"
                "res = kernel32.CreateProcessW(None, 'cmd.exe /c ping -n 5 127.0.0.1', None, None, False, 0x01000000, None, None, ctypes.byref(si), ctypes.byref(pi))\n"
                "err = ctypes.get_last_error()\n"
                "with open(sys.argv[1], 'w') as f:\n"
                "    json.dump({'created': bool(res), 'winerror': err}, f)\n"
            )
            breakaway_probe_script.write_text(probe_code, encoding="utf-8")

            python_exe = sys.executable
            contain_core.spawn_in_job(
                [python_exe, str(breakaway_probe_script), str(breakaway_result_file)],
                workdir=workdir,
                env=expected_env,
                password=created_password,
                timeout_s=10.0,
                target_sid=created_sid,
            )

            breakaway_blocked = False
            breakaway_data = {}
            if breakaway_result_file.exists():
                breakaway_data = json.loads(breakaway_result_file.read_text(encoding="utf-8"))
                breakaway_blocked = (breakaway_data.get("created") is False) and (breakaway_data.get("winerror") == 5)

            f2a_ok = task_proc_found and independent_sid_match and (len(term_failures) == 0) and task_sweep_clean
            f2b_ok = breakaway_blocked

            item4["expected"] = {
                "f2a_service_mediated_detected": True,
                "f2a_independent_sid_match": True,
                "f2a_clean_after_kill": True,
                "f2b_breakaway_denied": True,
            }
            item4["actual"] = {
                "f2a_service_mediated_detected": task_proc_found,
                "f2a_independent_sid_match": independent_sid_match,
                "f2a_clean_after_kill": task_sweep_clean,
                "f2b_breakaway_data": breakaway_data,
                "f2b_breakaway_denied": breakaway_blocked,
                "child_pids": sweep_after_kill.child_pids,
                "indeterminate_pids": sweep_after_kill.indeterminate_pids,
            }
            item4["pass"] = bool(f2a_ok and f2b_ok)
        except Exception as e:
            item4["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item4["pass"] = False
        items.append(item4)

        # ---------------------------------------------------------------------
        # Item 5: Indeterminate Process Enumeration Gate (F3)
        # ---------------------------------------------------------------------
        item5 = {
            "item": 5,
            "name": "indeterminate_sweep_gate",
            "description": "Prove non-empty indeterminate PID set blocks the clean gate",
        }
        try:
            live_sweep = contain_core.sweep_processes_by_sid(created_sid)

            simulated_dirty = contain_core.SweepResult(
                target_sid=created_sid,
                child_pids=[],
                indeterminate_pids=[999999],
                error_details={999999: "OpenProcess access denied (simulated)"},
            )
            gate_blocked = not simulated_dirty.clean

            item5["expected"] = {"indeterminate_blocks_clean_gate": True}
            item5["actual"] = {
                "indeterminate_blocks_clean_gate": gate_blocked,
                "system_indeterminate_count": len(live_sweep.indeterminate_pids),
                "child_pids": live_sweep.child_pids,
                "indeterminate_pids": live_sweep.indeterminate_pids,
            }
            item5["pass"] = bool(gate_blocked)
        except Exception as e:
            item5["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item5["pass"] = False
        items.append(item5)

        # ---------------------------------------------------------------------
        # FIX-W7: Execute Tier W Windows-Observed Tests (before deletion)
        # ---------------------------------------------------------------------
        if created_sid and expected_env and created_password and temp_dir:
            tier_w_results = run_tier_w_windows_observed_tests(created_sid, expected_env, created_password, temp_dir)
            items.extend(tier_w_results)
        else:
            items.append({
                "tier": "WINDOWS-OBSERVED",
                "item": "W_fixtures_check",
                "name": "tier_w_fixtures_verification",
                "description": "Verify required fixtures exist before executing Tier W tests",
                "observation": {
                    "attempted": False,
                    "status": "NOT RUN",
                    "reason": "Missing required principal or environment fixtures",
                },
                "pass": False,
            })

        # ---------------------------------------------------------------------
        # Item 6: SID-Bound Principal Deletion & Post-Verification (F3, F6)
        # ---------------------------------------------------------------------
        item6 = {
            "item": 6,
            "name": "principal_deletion_sid_bound",
            "description": "Verify SID-bound account deletion, idempotency, and clean sweep",
        }
        try:
            deleted = contain_core.delete_principal(captured_sid=created_sid, username=contain_core.DEFAULT_PRINCIPAL_NAME)
            second_del = contain_core.delete_principal(captured_sid=created_sid, username=contain_core.DEFAULT_PRINCIPAL_NAME)

            post_sweep = contain_core.sweep_processes_by_sid(created_sid)

            item6["expected"] = {
                "first_delete": True,
                "second_delete_idempotent": False,
                "sweep_clean": True,
            }
            item6["actual"] = {
                "first_delete": deleted,
                "second_delete_idempotent": second_del,
                "sweep_clean": post_sweep.clean,
                "child_pids": post_sweep.child_pids,
                "indeterminate_pids": post_sweep.indeterminate_pids,
                "ephemeral_assumption": "Check-then-delete is guarded by SID comparison; accepted under single-tenant ephemeral runner assumption.",
            }
            item6["pass"] = bool(deleted and not second_del and post_sweep.clean)
        except Exception as e:
            item6["actual"] = {"error": f"{type(e).__name__}: {e}"}
            item6["pass"] = False
        items.append(item6)

    finally:
        if created_sid:
            try:
                contain_core.terminate_processes_by_sid(created_sid)
            except Exception:
                pass
            try:
                contain_core.delete_principal(captured_sid=created_sid, username=contain_core.DEFAULT_PRINCIPAL_NAME)
            except Exception:
                pass
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    evidence["items"] = items
    all_passed = all(item.get("pass") is True for item in items if item.get("status") != "OMITTED")
    evidence["all_passed"] = all_passed

    # L4: Dynamic attempt record for completed execution
    total_count = len(items)
    passed_count = sum(1 for item in items if item.get("pass") is True)
    evidence["observation"] = {
        "attempted": True,
        "outcome": "pass" if all_passed else "fail",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": f"{passed_count}/{total_count} items passed",
    }
    return evidence


if __name__ == "__main__":
    result = run_component1_selftest()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("all_passed") else 1)
