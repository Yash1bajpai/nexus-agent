"""Q1 owned transport. Windows jobs are containment, not service-broker security.

The old sealed helper supplies Win32 declarations only. Its DEVNULL transport,
constructor handoff and cleanup policy are deliberately not reused. Native calls
publish directly into an already registered lease, including on BaseException.
Pending launch publication keeps every handle, descriptor and stream untouched.
Only after publication ends may the elected cleanup owner release resources,
using the original teardown deadline. Unconfirmed cleanup retains its lease;
cleanup cancellation must not replace the command's original exception.
Publication events do not prove launcher exit: only an external bounded thread
join can confirm it. Deferred self-cleanup leaves join status unconfirmed until
an independent caller joins, without renewing the original teardown deadline.
Thread starts use CPython's native start primitive plus its standard bootstrap,
without Thread.start's unbounded bootstrap wait. Attempted, target-entered and
publication-complete are distinct; an unpublished ident proves nothing after an
attempt. Unentered IO retains its streams until an external actual join.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import locale
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

STREAM_MAX_BYTES = 16 * 1024 * 1024
ROW_MAX_BYTES = 24 * 1024 * 1024
RUN_MAX_BYTES = 512 * 1024 * 1024
STATUS_MAX_BYTES = 2 * 1024 * 1024
MAX_STDIN_BYTES = 64 * 1024
TEARDOWN_SECONDS = 10.0
_RECORD_RESERVE = 16 * 1024


class Q1LifecycleError(RuntimeError):
    pass


class EvidenceIncomplete(Q1LifecycleError):
    pass


class OverallDeadlineExceeded(Q1LifecycleError):
    pass


def _start_thread_without_bootstrap_wait(thread):
    """Use the standard CPython bootstrap/join bookkeeping, never its start wait.

    Qualification pins CPython 3.12; 3.13 uses its native joinable-thread handle.
    A native-start exception may follow creation: retain limbo/lease ownership,
    rather than pretending the absent started/ident fields prove no thread.
    """
    with threading._active_limbo_lock:
        threading._limbo[thread] = thread
    if hasattr(threading, "_start_joinable_thread"):
        threading._start_joinable_thread(
            thread._bootstrap, handle=thread._handle, daemon=thread.daemon)
    else:
        threading._start_new_thread(thread._bootstrap, ())


def _native_type():
    path = Path(__file__).resolve().parents[1] / "diagnostics" / "windows_job.py"
    spec = importlib.util.spec_from_file_location("q1_sealed_win32_declarations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._NativeAPI


class _WindowsAPI:
    """Win32 adapter; every returned OS handle is published in finally first."""

    def __init__(self):
        self.raw = _native_type()()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def job_into(self, owner):
        handle = None
        try:
            handle = self.raw.create_job()
        finally:
            if handle:
                owner.handles["job"] = handle
        self.raw.configure_job(handle)

    def pipe_into(self, owner, name, *, input_pipe=False):
        read, write = wintypes.HANDLE(), wintypes.HANDLE()
        security = self.raw.Security(ctypes.sizeof(self.raw.Security), None, True)
        try:
            self.raw._check(self.raw.kernel.CreatePipe(
                ctypes.byref(read), ctypes.byref(write), ctypes.byref(security), 0))
        finally:
            if read.value:
                owner.handles[name + "_read"] = read.value
            if write.value:
                owner.handles[name + "_write"] = write.value
        parent = write if input_pipe else read
        self.raw._check(self.raw.kernel.SetHandleInformation(parent, 1, 0))

    def null_into(self, owner):
        security = self.raw.Security(ctypes.sizeof(self.raw.Security), None, True)
        handle = None
        try:
            handle = self.raw.kernel.CreateFileW(
                "NUL", 0x80000000, 3, ctypes.byref(security), 3, 0, None)
        finally:
            if handle and handle != ctypes.c_void_p(-1).value:
                owner.handles["stdin_read"] = handle
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.raw._check(handle)

    def create_into(self, owner, argv, cwd, env):
        raw = self.raw
        startup = raw.StartupEx()
        startup.startup.cb = ctypes.sizeof(startup)
        startup.startup.flags = 0x100
        handles = [owner.handles[key] for key in
                   ("stdin_read", "stdout_write", "stderr_write")]
        startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
        size = ctypes.c_size_t()
        raw.kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = ctypes.create_string_buffer(size.value)
        raw._check(raw.kernel.InitializeProcThreadAttributeList(
            attributes, 1, 0, ctypes.byref(size)))
        info = raw.ProcessInfo()
        try:
            startup.attributes = ctypes.cast(attributes, ctypes.c_void_p)
            inherited = (wintypes.HANDLE * 3)(*handles)
            raw._check(raw.kernel.UpdateProcThreadAttribute(
                attributes, 0, 0x20002, inherited, ctypes.sizeof(inherited), None, None))
            environment = ctypes.create_unicode_buffer("\0".join(
                f"{key}={value}" for key, value in sorted(
                    env.items(), key=lambda item: item[0].upper())) + "\0\0")
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
            try:
                raw._check(raw.kernel.CreateProcessW(
                    argv[0], command, None, None, True,
                    raw.CREATE_SUSPENDED | raw.EXTENDED_STARTUPINFO_PRESENT |
                    raw.CREATE_UNICODE_ENVIRONMENT, environment, cwd,
                    ctypes.byref(startup), ctypes.byref(info)))
            finally:
                # The OS fills this caller-owned structure before returning. Even
                # an exception at return/adoption leaves both original handles here.
                if info.process:
                    owner.handles["process"] = info.process
                    owner.pid = info.pid
                if info.thread:
                    owner.handles["thread"] = info.thread
        finally:
            raw.kernel.DeleteProcThreadAttributeList(attributes)

    def stream_into(self, owner, name, key, *, write=False):
        # Keep the CRT descriptor published too: conversion cannot orphan a handle.
        handle = owner.handles[key]
        descriptor = None
        try:
            descriptor = self.raw.msvcrt.open_osfhandle(
                handle, (os.O_WRONLY if write else os.O_RDONLY) | os.O_BINARY)
        finally:
            if descriptor is not None:
                owner.handles.pop(key, None)
                owner.descriptors[name] = descriptor
        stream = None
        try:
            stream = os.fdopen(descriptor, "wb" if write else "rb", buffering=0)
        finally:
            if stream is not None:
                owner.streams[name] = stream
                owner.descriptors.pop(name, None)

    def members(self, job):
        # JobObjectBasicProcessIdList, not a machine-wide process scan.
        size = 256
        while size <= 1024 * 1024:
            buffer = ctypes.create_string_buffer(size)
            if self.raw.kernel.QueryInformationJobObject(job, 3, buffer, size, None):
                assigned, count = (wintypes.DWORD * 2).from_buffer(buffer)
                if count > (size - 8) // ctypes.sizeof(ctypes.c_size_t):
                    raise EvidenceIncomplete("Invalid owned job PID list")
                pids = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                return [{"pid": int(pid)} for pid in pids]
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            size *= 2
        raise EvidenceIncomplete("Owned job PID list exceeded bound")


class _Lease:
    def __init__(self, domain, record):
        self.domain, self.record = domain, record
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.failed = threading.Event()
        self.handles, self.descriptors, self.streams = {}, {}, {}
        self.io_threads = []
        self.assigned = False
        self.proc = None
        self.pid = None
        self.returncode = None
        self.cleanup_result = None
        self.resource_cleanup_result = None
        self.finish_lock = threading.Lock()
        self.resources_finalized = False
        self.termination_errors = []
        self.cleanup_deadline = None
        self.cleaning = False
        self.errors = []
        self.buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self.counts = {"stdout": 0, "stderr": 0}
        self.hashes = {name: hashlib.sha256() for name in self.counts}
        self.row_bytes = 0
        self.evidence_lock = threading.Lock()

        # Bounded launcher thread tracking
        self.launcher = None
        self.launch_ready = threading.Event()
        self.launch_error = None
        self.launch_attempted = False
        self.launch_start_interrupted = False
        self.launch_entered = threading.Event()
        self.launch_complete = threading.Event()
        self.deferred_cleanup = False

    def failure(self, reason):
        with self.evidence_lock:
            if len(self.errors) < 16:
                self.errors.append(reason)
        self.failed.set()

    def launch(self, argv, cwd, env, payload):
        self.start_launch(argv, cwd, env, payload)
        self.wait_ready(None)

    def start_launch(self, argv, cwd, env, payload):
        with self.lock:
            if self.cleaning or self.cancel.is_set():
                raise Q1LifecycleError("Launch cancelled before construction")
            self.launcher = threading.Thread(
                target=self._launch_worker,
                args=(argv, cwd, env, payload),
                name="q1-launcher",
                daemon=True,
            )
            self.launch_attempted = True
        # No bootstrap wait or native start call under the lease lock.
        try:
            _start_thread_without_bootstrap_wait(self.launcher)
        except BaseException:
            self.launch_start_interrupted = True
            raise

    def _launch_worker(self, argv, cwd, env, payload):
        self.launch_entered.set()
        try:
            with self.lock:
                if self.cleaning or self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled before construction")
            if self.domain.platform == "nt":
                api = self.domain.api
                api.job_into(self)
                api.pipe_into(self, "stdout")
                api.pipe_into(self, "stderr")
                if payload is None:
                    api.null_into(self)
                else:
                    api.pipe_into(self, "stdin", input_pipe=True)

                if self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled before process creation")

                api.create_into(self, argv, cwd, env)

                if self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled before job assignment")

                api.assign(self.handles["job"], self.handles["process"])
                with self.lock:
                    self.assigned = True

                for key in ("stdin_read", "stdout_write", "stderr_write"):
                    if key in self.handles:
                        api.close(self.handles[key])
                        del self.handles[key]

                api.stream_into(self, "stdout", "stdout_read")
                api.stream_into(self, "stderr", "stderr_read")
                if payload is not None:
                    api.stream_into(self, "stdin", "stdin_write", write=True)

                if self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled before thread resume")

                # No child instruction runs until job adoption and parent pipe closure.
                api.resume(self.handles["thread"])
            else:
                if self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled before process creation")
                # Publish Popen before __init__: constructor exceptions retain its
                # stored pid and handles, rather than losing a returned temporary.
                self.proc = subprocess.Popen.__new__(subprocess.Popen)
                self.proc.__init__(argv, stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=cwd, env=env, start_new_session=True)
                with self.lock:
                    self.pid = self.proc.pid
                    self.streams.update(stdout=self.proc.stdout, stderr=self.proc.stderr)
                    if payload is not None:
                        self.streams["stdin"] = self.proc.stdin
            with self.lock:
                if self.cancel.is_set():
                    raise Q1LifecycleError("Launch cancelled during adoption")
        except BaseException as error:
            self.launch_error = error
        finally:
            # Publication has ended before cleanup may touch any launch resource.
            # A timed-out caller hands the one cleanup election to this thread.
            with self.lock:
                self.launch_complete.set()
                deferred = self.deferred_cleanup
            try:
                if deferred:
                    self.cleanup()
            finally:
                self.launch_ready.set()

    def wait_ready(self, timeout):
        wait_timeout = None if timeout is None else max(0.0, timeout)
        signaled = self.launch_ready.wait(timeout=wait_timeout)
        if not signaled:
            self.cancel.set()
            return False
        if self.launch_error is not None:
            raise self.launch_error
        return True

    def poll(self):
        if self.domain.platform == "nt":
            if "process" in self.handles:
                self.returncode = self.domain.api.wait(self.handles["process"], 0)
        elif self.proc is not None and getattr(self.proc, "pid", None):
            self.returncode = self.proc.poll()
        return self.returncode

    def members(self):
        if self.domain.platform == "nt":
            if "job" in self.handles:
                return self.domain.api.members(self.handles["job"])
            return []
        # Portable observer reports only the directly owned process, never native proof.
        return [{"pid": self.pid}] if self.pid and self.poll() is None else []

    def drain(self, name, capture):
        stream = self.streams[name]
        path = Path(self.record[name]["path"])
        destination = None
        try:
            try:
                destination = path.open("xb")
            except Exception as error:
                self.failure("capture_open_" + type(error).__name__)
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                with self.domain._bytes_lock:
                    self.counts[name] += len(chunk)
                    self.row_bytes += len(chunk)
                    self.domain.total_bytes += len(chunk)
                    within = (self.counts[name] <= STREAM_MAX_BYTES and
                              self.row_bytes <= ROW_MAX_BYTES and
                              self.domain.total_bytes <= RUN_MAX_BYTES - STATUS_MAX_BYTES)
                if not within:
                    self.failure("capture_cap")
                    continue  # Keep draining until the cleanup owner stops the job.
                if self.failed.is_set():
                    continue
                try:
                    if destination is None:
                        continue
                    if destination.write(chunk) != len(chunk):
                        raise OSError("short capture write")
                    self.hashes[name].update(chunk)
                    if capture:
                        self.buffers[name].extend(chunk)
                except Exception as error:
                    self.failure("capture_write_" + type(error).__name__)
        except Exception as error:
            self.failure("capture_read_" + type(error).__name__)
        finally:
            if destination is not None:
                try:
                    destination.close()
                except Exception as error:
                    self.failure("capture_close_" + type(error).__name__)
            try:
                stream.close()
            except Exception as error:
                self.failure("pipe_close_" + type(error).__name__)

    def feed(self, payload):
        stream = self.streams["stdin"]
        try:
            view = memoryview(payload)
            while view:
                count = stream.write(view)
                if not count:
                    raise OSError("short stdin write")
                view = view[count:]
        except Exception as error:
            self.failure("stdin_write_" + type(error).__name__)
        finally:
            try:
                stream.close()  # Exact finite bytes followed by EOF, even empty input.
            except Exception as error:
                self.failure("stdin_close_" + type(error).__name__)

    def _io_entry(self, info, target, args):
        with self.lock:
            admitted = not self.cancel.is_set()
            info["entered"].set()
        try:
            if admitted:
                target(*args)
        finally:
            info["complete"].set()

    def start_io(self, payload, capture, deadline):
        targets = [("stdout", self.drain, ("stdout", capture)),
                   ("stderr", self.drain, ("stderr", capture))]
        if payload is not None:
            targets.append(("stdin", self.feed, (payload,)))
        for name, target, args in targets:
            with self.lock:
                if self.cleaning or self.cancel.is_set():
                    raise Q1LifecycleError("Cleanup won before IO adoption")
                info = {"start_attempted": False, "entered": threading.Event(),
                        "complete": threading.Event(), "joined": False,
                        "start_interrupted": False}
                info["thread"] = threading.Thread(
                    target=self._io_entry, args=(info, target, args),
                    name="q1-io-" + name, daemon=True)
                self.io_threads.append(info)
                info["start_attempted"] = True
            try:
                _start_thread_without_bootstrap_wait(info["thread"])
            except BaseException:
                info["start_interrupted"] = True
                raise
            if not info["entered"].wait(max(0.0, deadline - time.monotonic())):
                return False
        return True

    def _start_facts(self):
        return {
            "launcher_start_attempted": self.launch_attempted,
            "launcher_start_interrupted": self.launch_start_interrupted,
            "launcher_entered": self.launch_entered.is_set(),
            "launcher_publication_complete": self.launch_complete.is_set(),
            "io_starts": [
                {"name": info["thread"].name,
                 "attempted": info["start_attempted"],
                 "interrupted": info["start_interrupted"],
                 "entered": info["entered"].is_set(),
                 "complete": info["complete"].is_set(),
                 "joined": info["joined"]}
                for info in self.io_threads
            ],
        }

    def cleanup(self):
        # One absolute budget and one resource owner, including deferred adoption.
        now = time.monotonic()
        with self.lock:
            if self.cleanup_deadline is None:
                self.cleanup_deadline = now + TEARDOWN_SECONDS
            deadline = self.cleanup_deadline
            self.cancel.set()
            pending = self.launch_attempted and not self.launch_complete.is_set()
        if pending and not self.launch_start_interrupted:
            self.launch_complete.wait(max(0.0, min(deadline, self.domain.deadline) - time.monotonic()))
        with self.lock:
            if not self.done.is_set():
                if self.launch_attempted and not self.launch_complete.is_set():
                    self.deferred_cleanup = True
                    self.cleanup_result = {
                        "ownership": "owned_windows_job" if self.domain.platform == "nt" else "portable_posix_group",
                        "assigned": self.assigned, "job_empty": False,
                        "process_waited": False, "io_joined": False,
                        "pipes_closed": False, "launcher_joined": False,
                        "confirmed": False,
                        "errors": ["launcher_pending" if self.launch_entered.is_set() else "launcher_start_pending"],
                        "deadline": deadline,
                    }
                    self.cleanup_result.update(self._start_facts())
                    self.record["cleanup"] = self.cleanup_result
                    return self.cleanup_result
                winner = not self.cleaning
                self.cleaning = True
            else:
                winner = False
        if winner:
            self._cleanup_owned(deadline)
        elif not self.done.is_set():
            self.done.wait(max(0.0, deadline - time.monotonic()))
        if not self.done.is_set():
            return self.cleanup_result or {"confirmed": False, "errors": ["cleanup_owner_pending"]}
        self._finish_resources(deadline)
        return self._join_launcher(deadline)

    def _join_launcher(self, deadline):
        # Never hold the lease lock while joining: the launcher needs it to exit.
        with self.lock:
            if self.cleanup_result.get("launcher_joined"):
                return self.cleanup_result
        launcher = self.launcher
        joined = not self.launch_attempted
        errors = []
        if launcher is threading.current_thread():
            # This thread may release adopted resources, but cannot attest its exit.
            errors.append("launcher_join_pending")
        elif self.launch_attempted and not self.launch_entered.is_set():
            errors.append("launcher_start_pending")
        elif not joined:
            try:
                launcher.join(max(0.0, deadline - time.monotonic()))
                joined = not launcher.is_alive()
            except BaseException as error:
                errors.append("launcher_join_" + type(error).__name__)
            if not joined and not errors:
                errors.append("launcher_join_pending")
        with self.lock:
            # Another external joiner may already have established actual exit.
            if self.cleanup_result.get("launcher_joined"):
                return self.cleanup_result
            result = dict(self.resource_cleanup_result)
            result["errors"] = (result["errors"] + errors)[:16]
            result["launcher_joined"] = joined
            result.update(self._start_facts())
            result["confirmed"] = (
                joined and result["job_empty"] and result["process_waited"]
                and result["io_joined"] and result["pipes_closed"] and not result["errors"]
            )
            self.cleanup_result = result
            self.record["cleanup"] = result
            return result

    def _cleanup_owned(self, deadline):
        errors = []
        empty = waited = False
        # Resource cleanup cannot itself attest actual launcher-thread exit.
        try:
            # 2. Terminate child processes/jobs
            if self.domain.platform == "nt":
                api = self.domain.api
                process = self.handles.get("process")
                job = self.handles.get("job")
                if self.assigned and job:
                    try:
                        api.terminate_job(job)
                        while time.monotonic() < deadline:
                            if api.active_processes(job) == 0:
                                empty = True
                                break
                            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                    except BaseException as e:
                        errors.append("terminate_job_" + type(e).__name__)
                elif process:
                    try:
                        api.terminate_process(process)  # original suspended handle only
                        empty = True
                    except BaseException as e:
                        errors.append("terminate_process_" + type(e).__name__)
                else:
                    empty = True

                if process:
                    try:
                        code = api.wait(process, max(0.0, deadline - time.monotonic()))
                        waited = code is not None
                        if self.returncode is None:
                            self.returncode = code
                    except BaseException as e:
                        errors.append("wait_process_" + type(e).__name__)
                else:
                    waited = True
            else:
                pid = self.pid or getattr(self.proc, "pid", None)
                if pid:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                    try:
                        code = self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                        if self.returncode is None:
                            self.returncode = code
                        waited = True
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    while time.monotonic() < deadline:
                        try:
                            os.killpg(pid, 0)
                        except (ProcessLookupError, OSError):
                            empty = True
                            break
                        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                else:
                    empty = waited = True
        except BaseException as error:
            # A cleanup failure is evidence, never a replacement for the caller's
            # original exception. Further cancellation cannot orphan handle closure.
            errors.append("termination_" + type(error).__name__)
        finally:
            if not empty:
                errors.append("owned_empty_unconfirmed")
            if not waited:
                errors.append("original_wait_unconfirmed")
            result = {
                "ownership": "owned_windows_job" if self.domain.platform == "nt" else "portable_posix_group",
                "assigned": self.assigned,
                "job_empty": empty, "process_waited": waited,
                "io_joined": False, "pipes_closed": False,
                "launcher_joined": False, "confirmed": False,
                "errors": errors[:16], "deadline": deadline,
            }
            with self.lock:
                self.termination_errors = errors[:16]
                self.resource_cleanup_result = result
                self.cleanup_result = result
                self.record["cleanup"] = result
                self.record["exit_code"] = self.returncode
                self.done.set()
        return self.cleanup_result

    def _finish_resources(self, deadline):
        # A pending IO bootstrap owns its streams even when ident is unpublished.
        # Termination above runs once; later external checks may join and close,
        # but neither renew the deadline nor repeat an uncertain close operation.
        if not self.finish_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return
        try:
            if self.resources_finalized:
                return
            errors = list(self.termination_errors)
            io_joined = True
            for info in self.io_threads:
                thread = info["thread"]
                if info["start_attempted"] and not info["joined"]:
                    if not info["start_interrupted"]:
                        info["entered"].wait(max(0.0, min(deadline, self.domain.deadline) - time.monotonic()))
                    if not info["entered"].is_set():
                        errors.append("io_start_pending")
                    elif thread is threading.current_thread():
                        errors.append("io_join_pending")
                    else:
                        try:
                            thread.join(max(0.0, deadline - time.monotonic()))
                            info["joined"] = info["complete"].is_set() and not thread.is_alive()
                        except BaseException as error:
                            errors.append("io_join_" + type(error).__name__)
                    if not info["joined"]:
                        io_joined = False
            pipes_closed = False
            if not io_joined:
                errors.append("io_join_unconfirmed")
            else:
                pipes_closed = True
                for name, stream in tuple(self.streams.items()):
                    try:
                        stream.close()
                    except BaseException as error:
                        pipes_closed = False
                        errors.append("stream_close_" + type(error).__name__)
                    else:
                        del self.streams[name]
                for name, descriptor in tuple(self.descriptors.items()):
                    try:
                        os.close(descriptor)
                    except BaseException as error:
                        pipes_closed = False
                        errors.append("descriptor_close_" + type(error).__name__)
                    else:
                        del self.descriptors[name]
                if self.domain.platform == "nt":
                    result = self.resource_cleanup_result
                    for name, handle in tuple(self.handles.items()):
                        if name in ("process", "job") and not (result["job_empty"] and result["process_waited"]):
                            continue
                        try:
                            self.domain.api.close(handle)
                        except BaseException as error:
                            pipes_closed = False
                            errors.append("handle_close_" + name + "_" + type(error).__name__)
                        else:
                            del self.handles[name]
                self.resources_finalized = True
            with self.lock:
                result = dict(self.resource_cleanup_result)
                result.update(io_joined=io_joined, pipes_closed=pipes_closed,
                              errors=errors[:16], launcher_joined=False, confirmed=False)
                result.update(self._start_facts())
                self.resource_cleanup_result = result
                self.cleanup_result = result
                self.record["cleanup"] = result
        finally:
            self.finish_lock.release()


class ProcessDomain:
    """A serial owned launch domain. Unknown cleanup permanently quarantines it.

    Observer receives *only* OS job-owned PID facts, not ambient processes. Its
    optional JSON return is retained as bounded evidence; exceptions propagate.
    """

    def __init__(self, output_root: Path, *, deadline: float, observer=None,
                 platform=None, api=None):
        if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError("deadline must be finite absolute monotonic time")
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.observer = observer
        self.platform = os.name if platform is None else platform
        if self.platform not in ("nt", "posix"):
            raise Q1LifecycleError("Unsupported ownership platform")
        self.api = api if api is not None else (_WindowsAPI() if self.platform == "nt" else None)
        self.records = []
        self.total_bytes = 0
        self._bytes_lock = threading.Lock()
        self._status_bytes = 0
        self._lock = threading.Lock()
        self._active = None
        self._closed = False
        self._quarantined = False

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
        return False

    def close(self):
        with self._lock:
            self._closed = True
            lease = self._active
            if lease is not None:
                lease.cancel.set()
        if lease is not None:
            result = lease.cleanup()
            if not result["confirmed"]:
                self._quarantined = True
                raise EvidenceIncomplete("Owned cleanup is unknown")
            with self._lock:
                if self._active is lease:
                    self._active = None

    def _prepare(self, argv, input, text, encoding, errors, timeout, kwargs):
        if kwargs:
            raise TypeError("Unsupported subprocess arguments: " + ", ".join(sorted(kwargs)))
        if isinstance(argv, (str, bytes)) or not isinstance(argv, (list, tuple)) or not argv:
            raise TypeError("argv must be a nonempty argument sequence")
        argv = [os.fspath(item) for item in argv]
        if any(not isinstance(item, str) or "\0" in item for item in argv):
            raise TypeError("argv must contain NUL-free text")
        if len(json.dumps(argv, ensure_ascii=True).encode()) > 4096:
            raise EvidenceIncomplete("Command metadata cap")
        if timeout is not None and (not isinstance(timeout, (int, float)) or
                                    not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be nonnegative finite or None")
        text_mode = bool(text or encoding is not None or errors is not None)
        encoding = encoding or locale.getpreferredencoding(False)
        errors = errors or "strict"
        if input is not None:
            if text_mode:
                if not isinstance(input, str):
                    raise TypeError("text input must be str")
                payload = input.encode(encoding, errors)
            else:
                if not isinstance(input, (bytes, bytearray)):
                    raise TypeError("binary input must be bytes")
                payload = bytes(input)
            if len(payload) > MAX_STDIN_BYTES:
                raise EvidenceIncomplete("Finite stdin cap")
        else:
            payload = None
        return argv, payload, text_mode, encoding, errors

    def run(self, argv, *, input=None, capture_output=False, text=False,
            encoding=None, errors=None, timeout=None, env=None, cwd=None, **kwargs):
        argv, payload, text_mode, encoding, errors = self._prepare(
            argv, input, text, encoding, errors, timeout, kwargs)
        started = time.monotonic()
        if started >= self.deadline:
            raise OverallDeadlineExceeded("Overall work deadline expired before launch")
        with self._lock:
            if self._closed or self._quarantined:
                raise EvidenceIncomplete("Process domain closed or cleanup unknown")
            if self._active is not None:
                raise Q1LifecycleError("Concurrent runs are unsupported")
            if self._status_bytes + _RECORD_RESERVE > STATUS_MAX_BYTES:
                raise EvidenceIncomplete("Status evidence cap")
            command_id = len(self.records) + 1
            record = {"command_id": command_id, "argv": argv,
                      "input_bytes": len(payload) if payload is not None else 0,
                      "input_sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
                      "requested_timeout": timeout, "exit_code": None,
                      "timed_out": False, "overall_expired": False,
                      "capture_complete": False, "observer": []}
            for name in ("stdout", "stderr"):
                record[name] = {"bytes": 0, "path": str(self.output_root / f"command-{command_id:04d}.{name}"),
                                "sha256": None, "complete": False}
            lease = _Lease(self, record)
            self.records.append(record)
            self._active = lease  # Durable owner BEFORE any constructor/native launch.
        original = None
        cleanup = {"confirmed": False, "errors": ["cleanup_not_returned"]}
        cleanup_returned = False
        try:
            lease.start_launch(argv, os.fspath(cwd) if cwd is not None else os.getcwd(),
                               dict(os.environ) if env is None else dict(env), payload)

            now = time.monotonic()
            command_deadline = started + timeout if timeout is not None else math.inf
            launch_timeout = min(command_deadline, self.deadline) - now
            if launch_timeout <= 0:
                record["overall_expired"] = True
                lease.cancel.set()
                raise OverallDeadlineExceeded("Overall work deadline expired during launch")

            ready = lease.wait_ready(launch_timeout)
            if not ready:
                now = time.monotonic()
                if now >= self.deadline:
                    record["overall_expired"] = True
                    raise OverallDeadlineExceeded("Overall work deadline expired during launch")
                else:
                    record["timed_out"] = True
                    raise subprocess.TimeoutExpired(argv, timeout)

            # All bootstrap admission waits are explicit and deadline-bounded.
            if not lease.start_io(payload, capture_output, min(command_deadline, self.deadline)):
                if time.monotonic() >= self.deadline:
                    record["overall_expired"] = True
                    raise OverallDeadlineExceeded("Overall work deadline expired during IO adoption")
                record["timed_out"] = True
                raise subprocess.TimeoutExpired(argv, timeout)

            observed = set()
            observer_bytes = 0
            while True:
                if lease.cancel.is_set():
                    raise Q1LifecycleError("Process domain cancelled")
                if lease.failed.is_set():
                    raise EvidenceIncomplete("Owned transport evidence incomplete")
                now = time.monotonic()
                if now >= self.deadline:
                    record["overall_expired"] = True
                    raise OverallDeadlineExceeded("Overall work deadline expired")
                if now >= command_deadline:
                    record["timed_out"] = True
                    raise subprocess.TimeoutExpired(argv, timeout)
                if self.observer is not None:
                    value = self.observer(lease.members())
                    if value is not None:
                        serialized = json.dumps(value, ensure_ascii=True, allow_nan=False)
                        if serialized not in observed:
                            observer_bytes += len(serialized.encode())
                            if observer_bytes > 8192:
                                raise EvidenceIncomplete("Observer evidence cap")
                            record["observer"].append(json.loads(serialized))
                            observed.add(serialized)
                if lease.poll() is not None:
                    break
                time.sleep(min(.05, max(0.0, min(command_deadline, self.deadline) - time.monotonic())))
        except BaseException as error:
            original = error
            raise
        finally:
            try:
                cleanup = lease.cleanup()
                cleanup_returned = True
                with lease.lock:
                    cleanup = lease.cleanup_result or cleanup
                    record["cleanup"] = cleanup
                record["exit_code"] = lease.returncode
                complete = cleanup["confirmed"] and not lease.failed.is_set()
                record["capture_complete"] = complete
                record["transport_errors"] = lease.errors[:16]
                for name in ("stdout", "stderr"):
                    record[name].update(bytes=lease.counts[name], sha256=lease.hashes[name].hexdigest(),
                                        complete=complete)
                if not cleanup["confirmed"]:
                    self._quarantined = True
                self._status_bytes += len(json.dumps(record, ensure_ascii=True).encode())
                if original is None and (not complete or self._status_bytes > STATUS_MAX_BYTES):
                    raise EvidenceIncomplete("Owned cleanup or evidence incomplete")
            except BaseException as cleanup_error:
                self._quarantined = True
                if not cleanup_returned:
                    cleanup = {"confirmed": False, "errors": ["cleanup_" + type(cleanup_error).__name__]}
                    record["cleanup"] = cleanup
                if original is None:
                    raise
            finally:
                with self._lock:
                    if cleanup.get("confirmed") is True and not self._quarantined:
                        self._active = None
        def decoded(name):
            data = bytes(lease.buffers[name])
            if not text_mode:
                return data
            return data.decode(encoding, errors).replace("\r\n", "\n").replace("\r", "\n")
        return subprocess.CompletedProcess(argv, lease.returncode,
                                           decoded("stdout") if capture_output else None,
                                           decoded("stderr") if capture_output else None)
