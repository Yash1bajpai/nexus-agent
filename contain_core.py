# .github/e2e/contain_core.py
"""Nexus e2e Contained Run — Component 1: Core Containment Primitives.

Scope (Component 1 of 4):
- In-memory principal lifecycle (standard user `nexuse2e`, netapi32 NetUserAdd/Del).
- Job Object containment (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, assignment verification,
  job-tree termination confirmation via JobObjectBasicProcessIdList).
- Strict child environment allowlist construction and exact verification
  (with documented exception for Windows per-drive hidden current-directory entries).
- Fail-closed process enumeration and termination scoped by TokenUser SID.

All Win32 APIs are explicitly typed (argtypes and restype) to prevent 64-bit handle
truncation. Errors use use_last_error=True and are captured immediately.
No credentials are written to disk, logged, or passed on a command line.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import secrets
import shutil
import string
import subprocess
import sys
import time
from typing import Any, Mapping

# -----------------------------------------------------------------------------
# Win32 Type Definitions and Constants
# -----------------------------------------------------------------------------

# Process & Thread Rights
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TokenUser = 1

# Toolhelp32
TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# Job Object Limits & Info Classes
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectBasicProcessIdList = 3
JobObjectExtendedLimitInformation = 9

# Process Creation Flags
CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
LOGON_WITH_PROFILE = 0x00000001

# Wait & Exit Status
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
INFINITE = 0xFFFFFFFF

# NetAPI32 Constants
USER_PRIV_USER = 1
UF_SCRIPT = 0x0001
UF_NORMAL_ACCOUNT = 0x0200
UF_DONT_EXPIRE_PASSWD = 0x10000
NERR_Success = 0
NERR_UserExists = 2224
NERR_UserNotFound = 2221
LG_INCLUDE_INDIRECT = 0x0001
MAX_PREFERRED_LENGTH = 0xFFFFFFFF

# Win32 Errors
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_NONE_MAPPED = 1332
ERROR_NO_MORE_FILES = 18

DEFAULT_PRINCIPAL_NAME = "nexuse2e"

# Windows hidden per-drive current directory variable pattern (e.g. =C:=C:\Windows)
_PER_DRIVE_CWD_RE = re.compile(r"^=[A-Za-z]:$")


# -----------------------------------------------------------------------------
# Win32 ctypes Structures (64-bit Aligned)
# -----------------------------------------------------------------------------

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryLimit", ctypes.c_size_t),
        ("PeakJobMemoryLimit", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_PROCESS_ID_LIST_256(ctypes.Structure):
    _fields_ = [
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 256),
    ]


class USER_INFO_1(ctypes.Structure):
    _fields_ = [
        ("usri1_name", wintypes.LPWSTR),
        ("usri1_password", wintypes.LPWSTR),
        ("usri1_password_age", wintypes.DWORD),
        ("usri1_priv", wintypes.DWORD),
        ("usri1_home_dir", wintypes.LPWSTR),
        ("usri1_comment", wintypes.LPWSTR),
        ("usri1_flags", wintypes.DWORD),
        ("usri1_script_path", wintypes.LPWSTR),
    ]


class LOCALGROUP_USERS_INFO_0(ctypes.Structure):
    _fields_ = [
        ("lgrui0_name", wintypes.LPWSTR),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", wintypes.DWORD),
    ]


class TOKEN_USER(ctypes.Structure):
    _fields_ = [
        ("User", SID_AND_ATTRIBUTES),
    ]


# -----------------------------------------------------------------------------
# Explicitly Typed Win32 API Bindings
# -----------------------------------------------------------------------------

def _bind_win32_apis():
    if sys.platform != "win32":
        return None

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    netapi32 = ctypes.WinDLL("netapi32", use_last_error=True)

    # advapi32
    advapi32.CreateProcessWithLogonW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFOW),
        ctypes.POINTER(PROCESS_INFORMATION),
    ]
    advapi32.CreateProcessWithLogonW.restype = wintypes.BOOL

    advapi32.LookupAccountNameW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.LookupAccountNameW.restype = wintypes.BOOL

    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL

    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL

    # kernel32
    kernel32.CreateJobObjectW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL

    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL

    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD

    kernel32.TerminateJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
    ]
    kernel32.TerminateJobObject.restype = wintypes.BOOL

    kernel32.TerminateProcess.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
    ]
    kernel32.TerminateProcess.restype = wintypes.BOOL

    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    kernel32.CreateToolhelp32Snapshot.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE

    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL

    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL

    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE

    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    # netapi32
    netapi32.NetUserAdd.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    netapi32.NetUserAdd.restype = wintypes.DWORD

    netapi32.NetUserDel.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
    ]
    netapi32.NetUserDel.restype = wintypes.DWORD

    netapi32.NetUserGetLocalGroups.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    netapi32.NetUserGetLocalGroups.restype = wintypes.DWORD

    netapi32.NetApiBufferFree.argtypes = [ctypes.c_void_p]
    netapi32.NetApiBufferFree.restype = wintypes.DWORD

    return advapi32, kernel32, netapi32


_WIN32 = _bind_win32_apis()


# -----------------------------------------------------------------------------
# Exceptions and Result Types
# -----------------------------------------------------------------------------

class ContainmentError(Exception):
    """Base error for containment operations."""


class PrincipalExistsError(ContainmentError):
    """Raised when the requested principal already exists (fail-closed)."""


class PrincipalCreationError(ContainmentError):
    """Raised when principal creation fails."""


class PrincipalSecurityViolation(ContainmentError):
    """Raised when account name maps to an unexpected SID during deletion guard."""


class NetApiError(ContainmentError):
    """Raised when a NetAPI32 management call fails."""


class JobAssignmentError(ContainmentError):
    """Raised when job creation, assignment, or verification fails."""


class ChildEnvViolation(ContainmentError):
    """Raised when child environment does not match the strict allowlist."""


class ProcessSweepError(ContainmentError):
    """Raised when process snapshot enumeration fails."""


@dataclass(frozen=True)
class TerminationResult:
    """Result of terminating a job and confirming empty tree + clean SID sweep."""
    status: str  # "CONFIRMED" or "KILL_UNCONFIRMED"
    stage: str   # "confirmed", "terminate_call", "query_failed" (incl. pid_buffer_exceeded),
                 # "query_inconsistent", "job_not_empty", "internal_error", "sweep_error",
                 # "child_survivors", "indeterminate", "sweep_not_clean"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepResult:
    """Fail-closed process sweep results scoped by TokenUser SID."""
    target_sid: str
    child_pids: list[int] = field(default_factory=list)
    indeterminate_pids: list[int] = field(default_factory=list)
    error_details: dict[int, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """True only if zero child processes and zero indeterminate processes exist."""
        return len(self.child_pids) == 0 and len(self.indeterminate_pids) == 0


@dataclass(frozen=True)
class SpawnResult:
    """Result of running a contained command inside a Job Object."""
    status: str  # "COMPLETED", "TIMED_OUT", "KILL_UNCONFIRMED", "WAIT_FAILED"
    exit_code: int | None
    pid: int
    stage: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# 1. Principal Lifecycle: create_principal & delete_principal (F4, F6)
# -----------------------------------------------------------------------------

def _generate_complex_password() -> str:
    """Generate a high-entropy CSPRNG password meeting Windows complexity policies."""
    chars = string.ascii_letters + string.digits + string.punctuation
    raw = "".join(secrets.choice(chars) for _ in range(32))
    return f"Nx1!{raw}#9Z"


def create_principal(username: str = "nexuse2e") -> tuple[str, str]:
    """Create local standard user via netapi32 NetUserAdd in memory.

    Password is held only in the returned tuple in caller's memory.
    If the user already exists, fails closed (does not reuse).
    Returns (sid_string, password).
    """
    if sys.platform != "win32":
        raise ContainmentError("create_principal requires Windows host platform")

    advapi32, kernel32, netapi32 = _WIN32
    password = _generate_complex_password()

    ui = USER_INFO_1()
    ui.usri1_name = username
    ui.usri1_password = password
    ui.usri1_password_age = 0
    ui.usri1_priv = USER_PRIV_USER
    ui.usri1_home_dir = None
    ui.usri1_comment = "Nexus Contained E2E Test Standard User"
    ui.usri1_flags = UF_SCRIPT | UF_NORMAL_ACCOUNT | UF_DONT_EXPIRE_PASSWD
    ui.usri1_script_path = None

    parm_err = wintypes.DWORD(0)
    status = netapi32.NetUserAdd(None, 1, ctypes.byref(ui), ctypes.byref(parm_err))

    if status == NERR_UserExists:
        raise PrincipalExistsError(f"User '{username}' already exists. Refusing reuse (fail-closed).")
    if status != NERR_Success:
        raise PrincipalCreationError(f"NetUserAdd failed with status {status} (parm_err={parm_err.value})")

    # Capture TokenUser SID via LookupAccountNameW
    cb_sid = wintypes.DWORD(0)
    cch_domain = wintypes.DWORD(0)
    pe_use = wintypes.DWORD(0)

    advapi32.LookupAccountNameW(None, username, None, ctypes.byref(cb_sid), None, ctypes.byref(cch_domain), ctypes.byref(pe_use))
    err = ctypes.get_last_error()
    if err != ERROR_INSUFFICIENT_BUFFER or cb_sid.value == 0:
        netapi32.NetUserDel(None, username)
        raise PrincipalCreationError(f"LookupAccountNameW initial query failed (err={err})")

    sid_buf = ctypes.create_string_buffer(cb_sid.value)
    domain_buf = ctypes.create_unicode_buffer(cch_domain.value)

    if not advapi32.LookupAccountNameW(
        None, username, sid_buf, ctypes.byref(cb_sid), domain_buf, ctypes.byref(cch_domain), ctypes.byref(pe_use)
    ):
        err = ctypes.get_last_error()
        netapi32.NetUserDel(None, username)
        raise PrincipalCreationError(f"LookupAccountNameW failed to retrieve SID for '{username}' (err={err})")

    p_str_sid = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid_buf, ctypes.byref(p_str_sid)):
        err = ctypes.get_last_error()
        netapi32.NetUserDel(None, username)
        raise PrincipalCreationError(f"ConvertSidToStringSidW failed (err={err})")

    try:
        sid_str = p_str_sid.value
    finally:
        kernel32.LocalFree(p_str_sid)

    return sid_str, password


def delete_principal(captured_sid: str, username: str = "nexuse2e") -> bool:
    """Delete the local account identified by captured_sid via NetUserDel.

    Immediately before deletion, resolves the username to its current SID via
    LookupAccountNameW and verifies an exact match with captured_sid.

    Assumption note:
        The check-then-delete is not atomic against a concurrent administrator deleting
        and recreating the name; accepted under the explicit ephemeral single-tenant runner
        assumption (design rev 8 R-list).

    Returns:
        True if deleted.
        False if already absent (idempotent, does not raise).
    Raises:
        PrincipalSecurityViolation: if username resolves to a SID different from captured_sid.
        NetApiError: on unexpected management failure.
    """
    if sys.platform != "win32":
        raise ContainmentError("delete_principal requires Windows host platform")

    advapi32, kernel32, netapi32 = _WIN32

    # 1. Resolve username to current SID to verify identity
    cb_sid = wintypes.DWORD(0)
    cch_domain = wintypes.DWORD(0)
    pe_use = wintypes.DWORD(0)

    advapi32.LookupAccountNameW(None, username, None, ctypes.byref(cb_sid), None, ctypes.byref(cch_domain), ctypes.byref(pe_use))
    err = ctypes.get_last_error()
    if err == ERROR_NONE_MAPPED or cb_sid.value == 0:
        return False  # Already absent

    sid_buf = ctypes.create_string_buffer(cb_sid.value)
    domain_buf = ctypes.create_unicode_buffer(cch_domain.value)

    if not advapi32.LookupAccountNameW(
        None, username, sid_buf, ctypes.byref(cb_sid), domain_buf, ctypes.byref(cch_domain), ctypes.byref(pe_use)
    ):
        err = ctypes.get_last_error()
        if err == ERROR_NONE_MAPPED:
            return False
        raise NetApiError(f"LookupAccountNameW failed during deletion guard (err={err})")

    p_str_sid = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid_buf, ctypes.byref(p_str_sid)):
        raise NetApiError(f"ConvertSidToStringSidW failed during deletion guard (err={ctypes.get_last_error()})")

    try:
        current_sid = p_str_sid.value
    finally:
        kernel32.LocalFree(p_str_sid)

    if current_sid != captured_sid:
        raise PrincipalSecurityViolation(
            f"Account '{username}' maps to SID {current_sid}, not captured SID {captured_sid}. Refusing deletion."
        )

    # 2. Issue deletion
    status = netapi32.NetUserDel(None, username)
    if status == NERR_Success:
        return True
    if status == NERR_UserNotFound:
        return False
    raise NetApiError(f"NetUserDel failed for '{username}' with status {status}")


def verify_principal_not_admin(username: str = "nexuse2e") -> bool:
    """Verify that the principal does not belong to the Administrators group."""
    if sys.platform != "win32":
        raise ContainmentError("verify_principal_not_admin requires Windows host platform")

    _, _, netapi32 = _WIN32
    p_buf = ctypes.c_void_p()
    entries_read = wintypes.DWORD(0)
    total_entries = wintypes.DWORD(0)

    status = netapi32.NetUserGetLocalGroups(
        None, username, 0, LG_INCLUDE_INDIRECT, ctypes.byref(p_buf), MAX_PREFERRED_LENGTH,
        ctypes.byref(entries_read), ctypes.byref(total_entries)
    )
    if status != NERR_Success:
        raise NetApiError(f"NetUserGetLocalGroups failed with status {status}")

    try:
        if entries_read.value == 0 or not p_buf.value:
            return True
        groups = (LOCALGROUP_USERS_INFO_0 * entries_read.value).from_address(p_buf.value)
        group_names = [groups[i].lgrui0_name.lower() for i in range(entries_read.value) if groups[i].lgrui0_name]
        return "administrators" not in group_names
    finally:
        if p_buf.value:
            netapi32.NetApiBufferFree(p_buf)


# -----------------------------------------------------------------------------
# 2. Child Environment Construction & Verification (F1)
# -----------------------------------------------------------------------------

def build_child_env(workdir: str | Path) -> dict[str, str]:
    """Construct the exact allowlist environment for the child process.

    Redirects TMP/TEMP/HOME/USERPROFILE to workdir.
    Pins PATH strictly to Python, Git, and System32 directories.
    Passes through model repo/filename if set.
    """
    workdir_abs = os.path.abspath(os.fspath(workdir))

    system_root = os.environ.get("SystemRoot") or "C:\\Windows"
    system_drive = os.environ.get("SystemDrive") or "C:"
    system32 = os.path.join(system_root, "System32")

    path_dirs: list[str] = []

    # 1. Python directory
    python_dir = os.path.abspath(os.path.dirname(sys.executable))
    path_dirs.append(python_dir)
    scripts_dir = os.path.join(python_dir, "Scripts")
    if os.path.isdir(scripts_dir):
        path_dirs.append(scripts_dir)

    # 2. Git directory
    git_path = shutil.which("git")
    if git_path:
        git_dir = os.path.abspath(os.path.dirname(git_path))
        path_dirs.append(git_dir)

    # 3. System32 directory
    if os.path.isdir(system32):
        path_dirs.append(os.path.abspath(system32))

    # De-duplicate while preserving order
    deduped_path = os.pathsep.join(dict.fromkeys(path_dirs))

    env: dict[str, str] = {
        "SystemRoot": system_root,
        "SystemDrive": system_drive,
        "PATH": deduped_path,
        "TMP": workdir_abs,
        "TEMP": workdir_abs,
        "HOME": workdir_abs,
        "USERPROFILE": workdir_abs,
    }

    for passthrough_key in ("NEXUS_AGENT_MODEL_REPO", "NEXUS_AGENT_MODEL_FILENAME"):
        if passthrough_key in os.environ:
            env[passthrough_key] = os.environ[passthrough_key]

    return env


def assert_child_env(
    reported: Mapping[str, str],
    expected: Mapping[str, str] | None = None,
    workdir: str | Path | None = None,
    observed_hidden: dict[str, str] | None = None,
) -> None:
    """Assert that the reported environment matches the allowlist exactly.

    Compares names AND values. Extra, missing, or altered keys raise ChildEnvViolation.
    Exception: Windows per-drive current-directory entries matching ^=([A-Za-z]):$
    are excluded from the extra-keys violation and recorded in observed_hidden (if provided).
    Every other extra name raises ChildEnvViolation.
    """
    if expected is None:
        if workdir is None:
            raise ValueError("Either expected env dict or workdir must be supplied to assert_child_env")
        expected = build_child_env(workdir)

    expected_map = {k.upper(): (k, v) for k, v in expected.items()}
    reported_map: dict[str, tuple[str, str]] = {}
    extra_keys: list[str] = []

    for k, v in reported.items():
        if _PER_DRIVE_CWD_RE.match(k):
            if observed_hidden is not None:
                observed_hidden[k] = v
            continue

        k_upper = k.upper()
        reported_map[k_upper] = (k, v)
        if k_upper not in expected_map:
            extra_keys.append(k)

    if extra_keys:
        raise ChildEnvViolation(f"Child reported disallowed extra environment variable(s): {sorted(extra_keys)}")

    missing_keys = set(expected_map.keys()) - set(reported_map.keys())
    if missing_keys:
        offending = [expected_map[k][0] for k in sorted(missing_keys)]
        raise ChildEnvViolation(f"Child reported missing required environment variable(s): {offending}")

    mismatches: list[str] = []
    for k_upper, (orig_k, exp_val) in expected_map.items():
        act_k, act_val = reported_map[k_upper]
        if k_upper in ("TMP", "TEMP", "HOME", "USERPROFILE"):
            if os.path.normcase(os.path.abspath(act_val)) != os.path.normcase(os.path.abspath(exp_val)):
                mismatches.append(f"{orig_k}: expected {exp_val!r}, got {act_val!r}")
        elif act_val != exp_val:
            mismatches.append(f"{orig_k}: expected {exp_val!r}, got {act_val!r}")

    if mismatches:
        raise ChildEnvViolation("Environment variable value mismatch(es):\n" + "\n".join(mismatches))


# -----------------------------------------------------------------------------
# 3. Job Containment & Spawning (F5)
# -----------------------------------------------------------------------------

def _format_unicode_env_block(env: Mapping[str, str]) -> ctypes.Array:
    """Create a double-null-terminated Unicode environment block."""
    items = sorted(env.items(), key=lambda kv: kv[0].upper())
    block_str = "\0".join(f"{k}={v}" for k, v in items) + "\0\0"
    return ctypes.create_unicode_buffer(block_str)


def _verify_cmdline_absolute(cmdline: str | list[str]) -> tuple[str, str]:
    """Verify that the target executable path is absolute and exists.

    Returns (absolute_exe_path, full_command_line_str).
    """
    if isinstance(cmdline, (list, tuple)):
        if not cmdline:
            raise ValueError("Command list cannot be empty")
        exe_path = os.path.abspath(os.fspath(cmdline[0]))
        if not os.path.isabs(exe_path) or not os.path.exists(exe_path):
            raise ValueError(f"Executable path must be absolute and existing; got {exe_path!r}")
        full_cmd = subprocess.list2cmdline([exe_path, *cmdline[1:]])
        return exe_path, full_cmd
    else:
        cmd_str = str(cmdline).strip()
        if not cmd_str:
            raise ValueError("Command line cannot be empty")
        if cmd_str.startswith('"'):
            end_quote = cmd_str.find('"', 1)
            if end_quote == -1:
                raise ValueError("Malformed command line quote")
            raw_exe = cmd_str[1:end_quote]
        else:
            raw_exe = cmd_str.split()[0]

        exe_path = os.path.abspath(raw_exe)
        if not os.path.isabs(exe_path) or not os.path.exists(exe_path):
            raise ValueError(f"Executable path must be absolute and existing; got {exe_path!r}")
        return exe_path, cmd_str


def terminate_and_confirm(
    h_job: wintypes.HANDLE,
    target_sid: str,
    deadline_s: float = 5.0,
) -> TerminationResult:
    """Terminate a Job Object and confirm both empty job-tree and clean SID sweep.

    h_job is BORROWED: caller that created it owns it and must close it.
    Caller must treat anything other than CONFIRMED as fail-closed.

    Liveness bound scope (FIX-K1):
        deadline_s bounds ONLY the Stage 2 job-empty confirmation loop. The Stage 3 SID sweep is a
        single point-in-time enumeration whose runtime depends on system process count and is NOT
        bounded by deadline_s; callers needing a hard wall-clock bound must impose it around this call.

    SID threading contract (FIX-K2):
        target_sid MUST be the TokenUser SID string of the created unprivileged principal, threaded by
        the caller from principal creation. spawn_in_job already fails closed
        (KILL_UNCONFIRMED / stage="missing_target_sid") when it is absent; the driver MUST pass the
        created principal's SID on every call so that leg can never fire in the real flow.

    Point-in-time residual (FIX-K3):
        CONFIRMED is a point-in-time result: after it returns, a same-SID process can still be started
        outside the job (services, RPC, task scheduler). As of this writing NO respawn mitigation is
        implemented or verified: the post-deletion SID-scoped firewall block probe (planned component 2)
        and account deletion (planned component 3) do not exist yet and MUST NOT be cited as proving
        anything. Even once implemented and probe-verified, those controls remain MITIGATIONS that reduce
        exposure, not guarantees; in particular, account deletion does not revoke tokens that survived
        from before the deletion, and the SID-scoped firewall probe constrains network EGRESS from the
        SID — it does not prevent a same-SID process from existing or running. NEVER cite CONFIRMED,
        alone or together with any present or future controls, as a guarantee that no same-SID process
        exists or can exist later. Until those controls exist and are probe-verified, CONFIRMED alone
        says nothing about same-SID processes existing later.
    """
    if sys.platform != "win32":
        raise ContainmentError("terminate_and_confirm requires Windows host platform")

    _, kernel32, _ = _WIN32

    # Stages 1-2: Exception normalization (C3)
    try:
        # Stage 1: TerminateJobObject
        if not kernel32.TerminateJobObject(h_job, 1):
            gle = ctypes.get_last_error()
            return TerminationResult(
                status="KILL_UNCONFIRMED",
                stage="terminate_call",
                evidence={"gle": gle},
            )

        # Stage 2: Bounded confirmation loop for empty process list in job
        deadline = time.monotonic() + deadline_s
        job_empty = False
        last_pids: list[int] = []

        while True:
            pid_list = JOBOBJECT_BASIC_PROCESS_ID_LIST_256()
            ret_len = wintypes.DWORD(0)
            success = kernel32.QueryInformationJobObject(
                h_job,
                JobObjectBasicProcessIdList,
                ctypes.byref(pid_list),
                ctypes.sizeof(pid_list),
                ctypes.byref(ret_len),
            )
            if not success:
                gle = ctypes.get_last_error()
                return TerminationResult(
                    status="KILL_UNCONFIRMED",
                    stage="query_failed",
                    evidence={"gle": gle},
                )

            assigned = int(pid_list.NumberOfAssignedProcesses)
            count = int(pid_list.NumberOfProcessIdsInList)
            if assigned > 256 or count > 256:
                return TerminationResult(
                    status="KILL_UNCONFIRMED",
                    stage="query_failed",
                    evidence={"error": "pid_buffer_exceeded", "assigned": assigned, "in_list": count},
                )

            # C1: Job-empty consistency check
            if count != assigned:
                return TerminationResult(
                    status="KILL_UNCONFIRMED",
                    stage="query_inconsistent",
                    evidence={"assigned": assigned, "in_list": count},
                )

            last_pids = [int(pid_list.ProcessIdList[i]) for i in range(count)]
            if assigned == 0 and count == 0:
                job_empty = True
                break

            if time.monotonic() >= deadline:
                break

            time.sleep(min(0.5, max(0.05, deadline - time.monotonic())))

        if not job_empty:
            return TerminationResult(
                status="KILL_UNCONFIRMED",
                stage="job_not_empty",
                evidence={"remaining_pids": last_pids},
            )
    except Exception as e:
        return TerminationResult(
            status="KILL_UNCONFIRMED",
            stage="internal_error",
            evidence={"error": f"{type(e).__name__}: {e}"},
        )

    # Stage 3: Post-termination SID sweep verification (FIX-K5 timing instrumentation)
    t0_sweep = time.monotonic()
    try:
        sweep = sweep_processes_by_sid(target_sid)
        sweep_ms = (time.monotonic() - t0_sweep) * 1000.0
    except Exception as e:
        sweep_ms = (time.monotonic() - t0_sweep) * 1000.0
        return TerminationResult(
            status="KILL_UNCONFIRMED",
            stage="sweep_error",
            evidence={"error": str(e), "sweep_ms": sweep_ms},
        )

    # C2: Clean-first classification with terminal else
    if sweep.clean:
        return TerminationResult(
            status="CONFIRMED",
            stage="confirmed",
            evidence={"target_sid": target_sid, "final_pids": last_pids, "sweep_ms": sweep_ms},
        )
    elif sweep.child_pids:
        return TerminationResult(
            status="KILL_UNCONFIRMED",
            stage="child_survivors",
            evidence={"child_pids": sweep.child_pids, "error_details": sweep.error_details, "sweep_ms": sweep_ms},
        )
    elif sweep.indeterminate_pids:
        return TerminationResult(
            status="KILL_UNCONFIRMED",
            stage="indeterminate",
            evidence={"indeterminate_pids": sweep.indeterminate_pids, "error_details": sweep.error_details, "sweep_ms": sweep_ms},
        )
    else:
        return TerminationResult(
            status="KILL_UNCONFIRMED",
            stage="sweep_not_clean",
            evidence={
                "child_pids": sweep.child_pids,
                "indeterminate_pids": sweep.indeterminate_pids,
                "error_details": sweep.error_details,
                "sweep_ms": sweep_ms,
            },
        )


def spawn_in_job(
    cmdline_abs: str | list[str],
    workdir: str | Path,
    env: Mapping[str, str],
    password: str,
    timeout_s: float | None = None,
    username: str = "nexuse2e",
    target_sid: str | None = None,
) -> SpawnResult:
    """Spawn process as principal inside an unescapable Job Object.

    1. Creates Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    2. Spawns suspended via CreateProcessWithLogonW(LOGON_WITH_PROFILE).
    3. Assigns process to job and verifies assignment (IsProcessInJob).
    4. Resumes process thread.
    5. Waits with timeout_s; terminates job on timeout.
       On timeout, verifies empty job tree via QueryInformationJobObject and
       post-termination SID sweep clean gate. Reports KILL_UNCONFIRMED on any failure.
    Fails closed on any step.
    """
    if sys.platform != "win32":
        raise ContainmentError("spawn_in_job requires Windows host platform")

    advapi32, kernel32, _ = _WIN32

    workdir_abs = os.path.abspath(os.fspath(workdir))
    if not os.path.isdir(workdir_abs):
        raise ValueError(f"Working directory does not exist: {workdir_abs!r}")

    _, full_cmd = _verify_cmdline_absolute(cmdline_abs)
    env_block = _format_unicode_env_block(env)

    # 1. Create Job Object
    h_job = kernel32.CreateJobObjectW(None, None)
    if not h_job:
        raise JobAssignmentError(f"CreateJobObjectW failed (err={ctypes.get_last_error()})")

    try:
        # Configure KILL_ON_JOB_CLOSE
        job_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            h_job, JobObjectExtendedLimitInformation, ctypes.byref(job_info), ctypes.sizeof(job_info)
        ):
            raise JobAssignmentError(f"SetInformationJobObject failed (err={ctypes.get_last_error()})")

        # 2. Spawn process suspended with profile logon
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        pi = PROCESS_INFORMATION()

        flags = CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
        success = advapi32.CreateProcessWithLogonW(
            username,
            None,
            password,
            LOGON_WITH_PROFILE,
            None,
            full_cmd,
            flags,
            ctypes.cast(env_block, ctypes.c_void_p),
            workdir_abs,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not success:
            raise ContainmentError(f"CreateProcessWithLogonW failed (err={ctypes.get_last_error()})")

        h_process = pi.hProcess
        h_thread = pi.hThread
        pid = pi.dwProcessId

        try:
            # 3. Assign process to Job Object
            if not kernel32.AssignProcessToJobObject(h_job, h_process):
                err = ctypes.get_last_error()
                kernel32.TerminateProcess(h_process, 1)
                raise JobAssignmentError(f"AssignProcessToJobObject failed (err={err})")

            # 4. Verify assignment fail-closed
            in_job = wintypes.BOOL(False)
            if not kernel32.IsProcessInJob(h_process, h_job, ctypes.byref(in_job)) or not in_job.value:
                kernel32.TerminateProcess(h_process, 1)
                raise JobAssignmentError("IsProcessInJob verification failed (process not in job)")

            # 5. Resume process thread
            if kernel32.ResumeThread(h_thread) == 0xFFFFFFFF:
                err = ctypes.get_last_error()
                kernel32.TerminateProcess(h_process, 1)
                raise ContainmentError(f"ResumeThread failed (err={err})")

            kernel32.CloseHandle(h_thread)
            h_thread = None

            # 6. Wait for process completion or timeout
            ms_timeout = int(timeout_s * 1000) if timeout_s is not None else INFINITE
            wait_res = kernel32.WaitForSingleObject(h_process, ms_timeout)

            if wait_res == WAIT_OBJECT_0:
                exit_code = wintypes.DWORD(0)
                kernel32.GetExitCodeProcess(h_process, ctypes.byref(exit_code))
                return SpawnResult(status="COMPLETED", exit_code=int(exit_code.value), pid=pid)

            elif wait_res == WAIT_TIMEOUT:
                if not target_sid:
                    kernel32.TerminateJobObject(h_job, 1)
                    return SpawnResult(status="KILL_UNCONFIRMED", exit_code=None, pid=pid, stage="missing_target_sid")
                term_res = terminate_and_confirm(h_job, target_sid, deadline_s=5.0)
                if term_res.status == "CONFIRMED":
                    return SpawnResult(status="TIMED_OUT", exit_code=None, pid=pid, stage=term_res.stage, evidence=term_res.evidence)
                return SpawnResult(status="KILL_UNCONFIRMED", exit_code=None, pid=pid, stage=term_res.stage, evidence=term_res.evidence)

            elif wait_res == WAIT_FAILED:
                wait_gle = ctypes.get_last_error()
                if not target_sid:
                    kernel32.TerminateJobObject(h_job, 1)
                    return SpawnResult(status="KILL_UNCONFIRMED", exit_code=None, pid=pid, stage="missing_target_sid", evidence={"wait_gle": wait_gle})
                term_res = terminate_and_confirm(h_job, target_sid, deadline_s=5.0)
                if term_res.status == "CONFIRMED":
                    return SpawnResult(status="WAIT_FAILED", exit_code=None, pid=pid, stage="confirmed", evidence={"wait_gle": wait_gle, "cleanup": term_res.evidence})
                return SpawnResult(status="KILL_UNCONFIRMED", exit_code=None, pid=pid, stage=term_res.stage, evidence={"wait_gle": wait_gle, "cleanup": term_res.evidence})

            else:
                kernel32.TerminateJobObject(h_job, 1)
                return SpawnResult(status="KILL_UNCONFIRMED", exit_code=None, pid=pid, stage="unknown_wait_result")

        finally:
            if h_thread:
                kernel32.CloseHandle(h_thread)
                h_thread = None
            if h_process:
                kernel32.CloseHandle(h_process)
                h_process = None

    finally:
        if h_job:
            kernel32.CloseHandle(h_job)


# -----------------------------------------------------------------------------
# 4. Fail-Closed Process Enumeration & Termination by SID (F3)
# -----------------------------------------------------------------------------

def sweep_processes_by_sid(sid: str) -> SweepResult:
    """Enumerate processes on the machine and match against target TokenUser SID.

    Fail-closed semantics:
    - If snapshot creation fails, raises ProcessSweepError.
    - If any process cannot be opened, or token cannot be queried, or SID cannot
      be converted, that PID is recorded as INDETERMINATE (never assumed 'not child').
    - SweepResult.clean is True only when both child_pids and indeterminate_pids are empty.
    """
    if sys.platform != "win32":
        raise ContainmentError("sweep_processes_by_sid requires Windows host platform")

    advapi32, kernel32, _ = _WIN32

    h_snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if h_snap == INVALID_HANDLE_VALUE or not h_snap:
        raise ProcessSweepError(f"CreateToolhelp32Snapshot failed (err={ctypes.get_last_error()})")

    child_pids: list[int] = []
    indeterminate_pids: list[int] = []
    error_details: dict[int, str] = {}

    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)

        more = kernel32.Process32FirstW(h_snap, ctypes.byref(pe))
        if not more:
            err = ctypes.get_last_error()
            if err != ERROR_NO_MORE_FILES:
                raise ProcessSweepError(f"Process32FirstW failed (err={err})")
            return SweepResult(target_sid=sid)

        while more:
            pid = int(pe.th32ProcessID)

            if pid not in (0, 4):  # Skip System Idle and System
                h_proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if not h_proc:
                    err = ctypes.get_last_error()
                    indeterminate_pids.append(pid)
                    error_details[pid] = f"OpenProcess failed (err={err})"
                else:
                    try:
                        h_token = wintypes.HANDLE()
                        if not advapi32.OpenProcessToken(h_proc, TOKEN_QUERY, ctypes.byref(h_token)):
                            err = ctypes.get_last_error()
                            indeterminate_pids.append(pid)
                            error_details[pid] = f"OpenProcessToken failed (err={err})"
                        else:
                            try:
                                ret_len = wintypes.DWORD(0)
                                advapi32.GetTokenInformation(h_token, TokenUser, None, 0, ctypes.byref(ret_len))

                                if ret_len.value == 0:
                                    err = ctypes.get_last_error()
                                    indeterminate_pids.append(pid)
                                    error_details[pid] = f"GetTokenInformation length query failed (err={err})"
                                else:
                                    token_buf = ctypes.create_string_buffer(ret_len.value)
                                    if not advapi32.GetTokenInformation(
                                        h_token, TokenUser, token_buf, ret_len.value, ctypes.byref(ret_len)
                                    ):
                                        err = ctypes.get_last_error()
                                        indeterminate_pids.append(pid)
                                        error_details[pid] = f"GetTokenInformation query failed (err={err})"
                                    else:
                                        p_token_user = ctypes.cast(token_buf, ctypes.POINTER(TOKEN_USER))
                                        proc_sid_ptr = p_token_user.contents.User.Sid

                                        p_str_sid = wintypes.LPWSTR()
                                        if not advapi32.ConvertSidToStringSidW(proc_sid_ptr, ctypes.byref(p_str_sid)):
                                            err = ctypes.get_last_error()
                                            indeterminate_pids.append(pid)
                                            error_details[pid] = f"ConvertSidToStringSidW failed (err={err})"
                                        else:
                                            try:
                                                if p_str_sid.value == sid:
                                                    child_pids.append(pid)
                                            finally:
                                                kernel32.LocalFree(p_str_sid)
                            finally:
                                kernel32.CloseHandle(h_token)
                    finally:
                        kernel32.CloseHandle(h_proc)

            pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            more = kernel32.Process32NextW(h_snap, ctypes.byref(pe))

        return SweepResult(
            target_sid=sid,
            child_pids=child_pids,
            indeterminate_pids=indeterminate_pids,
            error_details=error_details,
        )

    finally:
        kernel32.CloseHandle(h_snap)


def terminate_processes_by_sid(sid: str) -> list[tuple[int, Exception | int]]:
    """Terminate all processes matching the target SID.

    Returns a list of (pid, error) tuples for any termination failures.
    Returns empty list if all were successfully terminated.
    """
    if sys.platform != "win32":
        raise ContainmentError("terminate_processes_by_sid requires Windows host platform")

    _, kernel32, _ = _WIN32
    sweep = sweep_processes_by_sid(sid)

    failures: list[tuple[int, Exception | int]] = []
    for pid in sweep.child_pids:
        h_proc = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if not h_proc:
            failures.append((pid, ctypes.get_last_error()))
            continue
        try:
            if not kernel32.TerminateProcess(h_proc, 1):
                failures.append((pid, ctypes.get_last_error()))
        finally:
            kernel32.CloseHandle(h_proc)

    return failures
