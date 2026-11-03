"""Owned Windows process trees, not a security boundary against service brokers.

Every child starts suspended and joins our non-breakaway, kill-on-close job before
its original primary thread resumes. No PID-based discovery or termination is used.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import subprocess
import time


class WindowsJobError(RuntimeError):
    def __init__(self, cleanup):
        super().__init__("Owned Windows process startup failed")
        self.cleanup = cleanup


class _NativeAPI:
    CREATE_SUSPENDED = 0x00000004
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400

    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows Job Objects require Windows")
        import msvcrt
        self.msvcrt = msvcrt
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        size = ctypes.c_size_t
        handle = wintypes.HANDLE
        pointer = ctypes.c_void_p

        class Security(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("descriptor", pointer),
                        ("inherit", wintypes.BOOL)]

        class Startup(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("reserved", wintypes.LPWSTR),
                        ("desktop", wintypes.LPWSTR), ("title", wintypes.LPWSTR),
                        ("x", wintypes.DWORD), ("y", wintypes.DWORD),
                        ("xsize", wintypes.DWORD), ("ysize", wintypes.DWORD),
                        ("xchars", wintypes.DWORD), ("ychars", wintypes.DWORD),
                        ("fill", wintypes.DWORD), ("flags", wintypes.DWORD),
                        ("show", wintypes.WORD), ("reserved_size", wintypes.WORD),
                        ("reserved_bytes", pointer), ("stdin", handle),
                        ("stdout", handle), ("stderr", handle)]

        class StartupEx(ctypes.Structure):
            _fields_ = [("startup", Startup), ("attributes", pointer)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [("process", handle), ("thread", handle),
                        ("pid", wintypes.DWORD), ("tid", wintypes.DWORD)]

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("minimum", size), ("maximum", size),
                        ("active_limit", wintypes.DWORD), ("affinity", size),
                        ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IOCounters),
                        ("process_memory", size), ("job_memory", size),
                        ("peak_process_memory", size), ("peak_job_memory", size)]

        class Accounting(ctypes.Structure):
            _fields_ = [("user_time", ctypes.c_int64), ("kernel_time", ctypes.c_int64),
                        ("period_user", ctypes.c_int64), ("period_kernel", ctypes.c_int64),
                        ("faults", wintypes.DWORD), ("total", wintypes.DWORD),
                        ("active", wintypes.DWORD), ("terminated", wintypes.DWORD)]

        self.Security, self.StartupEx, self.ProcessInfo = Security, StartupEx, ProcessInfo
        self.ExtendedLimits, self.Accounting = ExtendedLimits, Accounting
        signatures = {
            "CreateJobObjectW": ([pointer, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": ([handle, ctypes.c_int, pointer, wintypes.DWORD], wintypes.BOOL),
            "QueryInformationJobObject": ([handle, ctypes.c_int, pointer, wintypes.DWORD, pointer], wintypes.BOOL),
            "AssignProcessToJobObject": ([handle, handle], wintypes.BOOL),
            "TerminateJobObject": ([handle, wintypes.UINT], wintypes.BOOL),
            "TerminateProcess": ([handle, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([handle], wintypes.BOOL),
            "CreatePipe": ([ctypes.POINTER(handle), ctypes.POINTER(handle), pointer, wintypes.DWORD], wintypes.BOOL),
            "SetHandleInformation": ([handle, wintypes.DWORD, wintypes.DWORD], wintypes.BOOL),
            "CreateFileW": ([wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, pointer,
                             wintypes.DWORD, wintypes.DWORD, handle], handle),
            "InitializeProcThreadAttributeList": ([pointer, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(size)], wintypes.BOOL),
            "UpdateProcThreadAttribute": ([pointer, wintypes.DWORD, size, pointer, size, pointer, pointer], wintypes.BOOL),
            "DeleteProcThreadAttributeList": ([pointer], None),
            "CreateProcessW": ([wintypes.LPCWSTR, wintypes.LPWSTR, pointer, pointer,
                                wintypes.BOOL, wintypes.DWORD, pointer, wintypes.LPCWSTR,
                                pointer, ctypes.POINTER(ProcessInfo)], wintypes.BOOL),
            "ResumeThread": ([handle], wintypes.DWORD),
            "WaitForSingleObject": ([handle, wintypes.DWORD], wintypes.DWORD),
            "GetExitCodeProcess": ([handle, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def _check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def create_job(self):
        return self._check(self.kernel.CreateJobObjectW(None, None))

    def configure_job(self, job):
        limits = self.ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE only.
        self._check(self.kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))

    def pipe(self):
        read_handle, write_handle = wintypes.HANDLE(), wintypes.HANDLE()
        security = self.Security(ctypes.sizeof(self.Security), None, True)
        self._check(self.kernel.CreatePipe(ctypes.byref(read_handle), ctypes.byref(write_handle),
                                           ctypes.byref(security), 0))
        try:
            self._check(self.kernel.SetHandleInformation(read_handle, 1, 0))
        except BaseException:
            try:
                self.close(read_handle.value)
            finally:
                self.close(write_handle.value)
            raise
        return read_handle.value, write_handle.value

    def null_input(self):
        security = self.Security(ctypes.sizeof(self.Security), None, True)
        handle = self.kernel.CreateFileW("NUL", 0x80000000, 3, ctypes.byref(security), 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return self._check(handle)

    def create_suspended(self, argv, cwd, env, handles):
        startup = self.StartupEx()
        startup.startup.cb = ctypes.sizeof(startup)
        startup.startup.flags = 0x00000100  # STARTF_USESTDHANDLES
        startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
        length = ctypes.c_size_t()
        self.kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(length))
        if not length.value:
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = ctypes.create_string_buffer(length.value)
        self._check(self.kernel.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(length)))
        try:
            startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
            inherited = (wintypes.HANDLE * len(handles))(*handles)
            self._check(self.kernel.UpdateProcThreadAttribute(attributes, 0, 0x00020002,
                         inherited, ctypes.sizeof(inherited), None, None))
            environment = ctypes.create_unicode_buffer(
                "\0".join(f"{key}={value}" for key, value in sorted(env.items(), key=lambda item: item[0].upper())) + "\0\0")
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
            info = self.ProcessInfo()
            self._check(self.kernel.CreateProcessW(
                argv[0], command, None, None, True,
                self.CREATE_SUSPENDED | self.EXTENDED_STARTUPINFO_PRESENT | self.CREATE_UNICODE_ENVIRONMENT,
                environment, os.fspath(cwd), ctypes.byref(startup), ctypes.byref(info)))
            return info.process, info.thread, info.pid
        finally:
            self.kernel.DeleteProcThreadAttributeList(attributes)

    def assign(self, job, process):
        self._check(self.kernel.AssignProcessToJobObject(job, process))

    def resume(self, thread):
        if self.kernel.ResumeThread(thread) != 1:
            raise OSError("Original suspended primary thread could not be resumed exactly once")

    def read_stream(self, handle):
        try:
            descriptor = self.msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            self.close(handle)
            raise
        try:
            return os.fdopen(descriptor, "rb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

    def wait(self, process, timeout):
        milliseconds = 0xFFFFFFFF if timeout is None else min(0xFFFFFFFE, math.ceil(max(0, timeout) * 1000))
        status = self.kernel.WaitForSingleObject(process, milliseconds)
        if status == 0x00000102:
            return None
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        self._check(self.kernel.GetExitCodeProcess(process, ctypes.byref(code)))
        return code.value

    def terminate_job(self, job):
        self._check(self.kernel.TerminateJobObject(job, 1))

    def terminate_process(self, process):
        self._check(self.kernel.TerminateProcess(process, 1))

    def active_processes(self, job):
        accounting = self.Accounting()
        self._check(self.kernel.QueryInformationJobObject(job, 1, ctypes.byref(accounting),
                                                        ctypes.sizeof(accounting), None))
        return accounting.active

    def close(self, handle):
        self._check(self.kernel.CloseHandle(handle))


class WindowsJobProcess:
    """Minimal Popen-like interface with explicit handle ownership and bounded teardown."""

    def __init__(self, argv, *, cwd, env, stdin, stdout, stderr, api=None):
        if (stdin, stdout, stderr) != (subprocess.DEVNULL, subprocess.PIPE, subprocess.PIPE):
            raise ValueError("Owned launch requires NUL stdin and separate captured output pipes")
        self.args = list(argv)
        self._api = api if api is not None else _NativeAPI()
        self._handles = {}
        self._assigned = False
        self._cleanup = None
        self.stdout = self.stderr = None
        self.returncode = None
        self.pid = None
        try:
            self._handles["job"] = self._api.create_job()
            self._api.configure_job(self._handles["job"])
            for name in ("stdout", "stderr"):
                read_handle, write_handle = self._api.pipe()
                self._handles[name + "_read"] = read_handle
                self._handles[name + "_write"] = write_handle
            self._handles["stdin"] = self._api.null_input()
            process, thread, self.pid = self._api.create_suspended(
                self.args, cwd, env, [self._handles[name] for name in ("stdin", "stdout_write", "stderr_write")])
            self._handles["process"], self._handles["thread"] = process, thread
            self._api.assign(self._handles["job"], process)
            self._assigned = True
            # The child owns the only remaining write ends before any model code runs.
            for name in ("stdin", "stdout_write", "stderr_write"):
                self._api.close(self._handles[name])
                del self._handles[name]
            for name in ("stdout", "stderr"):
                handle = self._handles.pop(name + "_read")
                stream = self._api.read_stream(handle)  # API consumes ownership, even on failure.
                setattr(self, name, stream)
            self._api.resume(thread)
        except BaseException as error:
            cleanup = self.cleanup(timeout=10)
            for stream in (self.stdout, self.stderr):
                if stream is not None:
                    stream.close()
            if not isinstance(error, Exception):
                raise
            raise WindowsJobError(cleanup) from None

    def poll(self):
        if self.returncode is None and "process" in self._handles:
            self.returncode = self._api.wait(self._handles["process"], 0)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = self._api.wait(self._handles["process"], timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode

    def cleanup(self, timeout=10):
        if self._cleanup is not None:
            return dict(self._cleanup)
        started = time.monotonic()
        deadline = started + max(0, timeout)
        errors = []
        confirmed = False
        try:
            if self._assigned:
                self._api.terminate_job(self._handles["job"])
                while True:
                    if self._api.active_processes(self._handles["job"]) == 0:
                        confirmed = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        errors.append("job_empty_timeout")
                        break
                    time.sleep(min(0.02, remaining))
                if "process" in self._handles:
                    code = self._api.wait(self._handles["process"], max(0, deadline - time.monotonic()))
                    if code is None:
                        confirmed = False
                        errors.append("owned_process_wait_timeout")
                    else:
                        self.returncode = code
            elif "process" in self._handles:
                # Assignment failed: only this original, still-suspended handle is ours.
                self._api.terminate_process(self._handles["process"])
                code = self._api.wait(self._handles["process"], max(0, deadline - time.monotonic()))
                confirmed = code is not None
                self.returncode = code
                if not confirmed:
                    errors.append("unassigned_process_wait_timeout")
            else:
                confirmed = True  # No process was ever created.
        except Exception as error:
            confirmed = False
            errors.append("termination_or_confirmation_" + type(error).__name__)
        finally:
            # Closing the last job handle is a kill fallback, NOT a confirmation.
            for name in tuple(self._handles):
                try:
                    self._api.close(self._handles[name])
                except Exception as error:
                    errors.append("close_" + name + "_" + type(error).__name__)
                finally:
                    del self._handles[name]
        self._cleanup = {"ownership": "owned_windows_job", "assigned": self._assigned,
                         "confirmed": confirmed, "errors": errors,
                         "duration_seconds": round(time.monotonic() - started, 3)}
        return dict(self._cleanup)
