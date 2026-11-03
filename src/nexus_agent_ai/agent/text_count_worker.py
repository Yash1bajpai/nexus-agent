"""Fixed, read-only physical-text worker and bounded parent transport.

This is process containment, not an atomic filesystem snapshot or hardlink policy.
Popen itself and uninterruptible OS I/O cannot be given an absolute Python-level
wallclock guarantee. Unconfirmed owned cleanup is quarantined, never hidden.
Unsupported opened-handle validation fails closed *before* any content read.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import secrets
import stat
import struct
import subprocess
import sys
import threading
import time
import types

# -I -S deliberately excludes the project from sys.path. Load only fixed trusted
# siblings, never __init__, tools, configuration, site hooks, or a workspace path.
if __name__ == "__main__":
    _package = types.ModuleType("_nexus_text_count")
    _package.__path__ = []
    sys.modules[_package.__name__] = _package
    __package__ = _package.__name__
    for _name in ("refusals", "_workspace_paths", "text_count_limits", "text_count_scan"):
        _spec = importlib.util.spec_from_file_location(
            f"{__package__}.{_name}", Path(__file__).absolute().with_name(f"{_name}.py")
        )
        _module = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _module
        _spec.loader.exec_module(_module)

from ._workspace_paths import _is_sensitive_path, _validate_workspace_path
from .refusals import RefusalClass
from .text_count_limits import MAX_LITERAL_BYTES, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_SCAN_BYTES


class CountArgumentError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class CountExecutionUncertain(RuntimeError):
    """ACK delivery could not be established; callers must not claim not executed."""
    def __init__(self, reason, *, receipt, cleanup):
        self.reason = reason
        self.receipt = receipt
        self.cleanup = cleanup
        super().__init__(reason)


class _ProtocolError(Exception):
    pass


class _PathRefusal(Exception):
    def __init__(self, reason):
        self.reason = reason


def _encode(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def validate_query(path, literal, whole_word=True, line=None):
    """Validate public arguments without resolving, stating, or opening any path."""
    if type(path) is not str or not path or "\x00" in path:
        raise CountArgumentError("invalid_path")
    if len(path) > MAX_REQUEST_BYTES:
        raise CountArgumentError("request_limit")
    if type(literal) is not str or not literal or "\r" in literal or "\n" in literal:
        raise CountArgumentError("invalid_literal")
    if len(literal) > MAX_LITERAL_BYTES:
        raise CountArgumentError("literal_limit")
    try:
        if len(literal.encode("utf-8")) > MAX_LITERAL_BYTES:
            raise CountArgumentError("literal_limit")
        path.encode("utf-8")
    except UnicodeEncodeError:
        raise CountArgumentError("invalid_unicode") from None
    if type(whole_word) is not bool:
        raise CountArgumentError("invalid_whole_word")
    if line is not None and (type(line) is not int or line < 1):
        raise CountArgumentError("invalid_line")
    args = {"path": path, "literal": literal, "whole_word": whole_word, "line": line}
    try:
        oversized = len(_encode(args)) > MAX_REQUEST_BYTES
    except (ValueError, OverflowError):
        raise CountArgumentError("request_limit") from None
    if oversized:
        raise CountArgumentError("request_limit")
    return args


def _receipt(args, reason, *, source=None):
    return {
        "status": "incomplete", "reason": reason,
        "path": args["path"], "literal": args["literal"], "whole_word": args["whole_word"],
        "scope": {"kind": "whole_file"} if args["line"] is None else {"kind": "physical_line", "line": args["line"]},
        "count": None, "total_count": None, "physical_lines": None,
        "per_line_count": {"min": None, "max": None},
        "source": source or {"bytes": None, "sha256": None, "partial": True},
        "edit_proof": False,
    }


_REASONS = frozenset((
    "not_found", "permission", "path_refused", "sensitive_path", "unsupported_file",
    "unsupported_handle_validation", "size_limit", "decode_error", "changed", "missing_line",
    "timeout", "worker_crash", "protocol_error", "cleanup_unverified", "busy",
))


def _refusal_class(reason):
    if reason in ("sensitive_path", "path_refused"):
        return RefusalClass.SENSITIVE_PATH.value
    if reason == "not_found":
        return RefusalClass.NOT_FOUND.value
    if reason == "permission":
        return RefusalClass.PERMISSION.value
    if reason in ("size_limit", "busy"):
        return RefusalClass.CAPACITY.value
    return RefusalClass.OTHER.value if reason else None


def _result(receipt, executed):
    return {"receipt": receipt, "executed": executed, "refusal_class": _refusal_class(receipt.get("reason"))}


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _reliable_identity(info):
    if not stat.S_ISREG(info.st_mode):
        raise _PathRefusal("unsupported_file")
    if not info.st_ino:
        raise _PathRefusal("unsupported_handle_validation")
    return _snapshot(info)


def _darwin_handle_path(fd):
    import fcntl
    # Darwin sys/fcntl.h F_GETPATH, MAXPATHLEN. No Linux/lexical fallback.
    value = fcntl.fcntl(fd, 50, bytes(1024))
    return Path(os.fsdecode(value.split(b"\0", 1)[0]))


def _windows_handle_path(fd):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    final_path = kernel.GetFinalPathNameByHandleW
    final_path.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
    final_path.restype = wintypes.DWORD
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
    size = final_path(handle, None, 0, 0)
    if not size or size > 32768:
        raise _PathRefusal("unsupported_handle_validation")
    buffer = ctypes.create_unicode_buffer(size + 1)
    length = final_path(handle, buffer, len(buffer), 0)
    if not length or length >= len(buffer):
        raise _PathRefusal("unsupported_handle_validation")
    name = buffer.value
    if name.startswith("\\\\?\\UNC\\"):
        name = "\\\\" + name[8:]
    elif name.startswith("\\\\?\\"):
        name = name[4:]
    return Path(name)


def _opened_path(fd):
    try:
        if sys.platform.startswith("linux"):
            name = os.readlink(f"/proc/self/fd/{fd}")
            if not os.path.isabs(name):
                raise _PathRefusal("unsupported_handle_validation")
            return Path(name)
        if sys.platform == "darwin":
            return _darwin_handle_path(fd)
        if sys.platform == "win32":
            return _windows_handle_path(fd)
    except (OSError, ValueError, AttributeError, ImportError):
        raise _PathRefusal("unsupported_handle_validation") from None
    raise _PathRefusal("unsupported_handle_validation")


def _allowed(path):
    allowed = _validate_workspace_path(path)
    if not isinstance(allowed, Path):
        # The shared policy returns a source-authored RefusalText, not prose to parse.
        raise _PathRefusal("path_refused")
    if _is_sensitive_path(Path(path)) or _is_sensitive_path(allowed):
        raise _PathRefusal("sensitive_path")
    return allowed


def _check_identity(fd, original, target, before):
    actual = _opened_path(fd)
    actual_allowed = _allowed(actual)
    if actual != target or actual_allowed != target or _allowed(original) != target:
        raise _PathRefusal("changed")
    if (_reliable_identity(os.fstat(fd)) != before
            or _reliable_identity(target.stat()) != before
            or _reliable_identity(Path(original).stat()) != before):
        raise _PathRefusal("changed")


def _open_validated(path):
    target = _allowed(path)
    initial = _reliable_identity(target.stat())
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags)
    try:
        before = _reliable_identity(os.fstat(fd))
        if initial != before:
            raise _PathRefusal("changed")
        _check_identity(fd, path, target, before)
        if before[3] > MAX_SCAN_BYTES:
            raise _PathRefusal("size_limit")
        return fd, target, before
    except BaseException:
        os.close(fd)
        raise


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise _ProtocolError()
        parts.extend(part)
    return bytes(parts)


def _read_frame(stream, maximum):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if not 1 <= size <= maximum:
        raise _ProtocolError()
    try:
        return json.loads(_read_exact(stream, size).decode("utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise _ProtocolError() from None


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate key")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _write_frame(stream, message, maximum):
    data = _encode(message)
    if len(data) > maximum:
        raise _ProtocolError()
    data = struct.pack("!I", len(data)) + data
    offset = 0
    while offset < len(data):
        written = stream.write(data[offset:])
        if not written:
            raise _ProtocolError()
        offset += written
    stream.flush()


def _child_main():
    from .text_count_scan import scan_stream
    incoming = sys.stdin.buffer
    outgoing = sys.stdout.buffer
    request = _read_frame(incoming, MAX_REQUEST_BYTES)
    if type(request) is not dict or set(request) != {"nonce", "args"}:
        raise _ProtocolError()
    nonce, args = request["nonce"], request["args"]
    if type(nonce) is not str or len(nonce) != 32:
        raise _ProtocolError()
    if type(args) is not dict or set(args) != {"path", "literal", "whole_word", "line"}:
        raise _ProtocolError()
    if validate_query(**args) != args:
        raise _ProtocolError()
    # Inherit the parent's actual working directory. Reopening a captured name
    # can select a different workspace after rename and makes getcwd a parent
    # filesystem operation outside the deadline.
    receipt = _receipt(args, "worker_crash")
    acknowledged = False
    fd = None
    try:
        fd, target, before = _open_validated(args["path"])
        _write_frame(outgoing, {"type": "started", "nonce": nonce, "args": args}, MAX_RESPONSE_BYTES)
        if incoming.read(1) != b"\x06":
            raise _ProtocolError()
        acknowledged = True
        # ACK may have been delayed: revalidate the held target before data I/O.
        _check_identity(fd, args["path"], target, before)
        # FileIO is unbuffered: never prefetch beyond the scanner's bounded reads.
        with os.fdopen(fd, "rb", buffering=0) as stream:
            fd = None
            facts = scan_stream(stream, args["literal"], args["whole_word"], args["line"])
            receipt.update(facts)
            try:
                _check_identity(stream.fileno(), args["path"], target, before)
                if not receipt["source"]["partial"] and receipt["source"]["bytes"] != before[3]:
                    raise _PathRefusal("changed")
            except (OSError, _PathRefusal):
                receipt.update(status="incomplete", reason="changed", count=None, total_count=None,
                               physical_lines=None, per_line_count={"min": None, "max": None})
                receipt["source"]["partial"] = True
    except _PathRefusal as error:
        receipt = _receipt(args, error.reason)
    except FileNotFoundError:
        receipt = _receipt(args, "not_found")
    except PermissionError:
        receipt = _receipt(args, "permission")
    except OSError:
        receipt = _receipt(args, "unsupported_file")
    finally:
        if fd is not None:
            os.close(fd)
    _write_frame(outgoing, {"type": "result", "nonce": nonce, "args": args,
                           "executed": acknowledged, "receipt": receipt}, MAX_RESPONSE_BYTES)


def _validate_receipt(receipt, args):
    expected = _receipt(args, "worker_crash")
    if type(receipt) is not dict or set(receipt) != set(expected):
        raise _ProtocolError()
    for key in ("path", "literal", "whole_word", "scope", "edit_proof"):
        if type(receipt[key]) is not type(expected[key]) or receipt[key] != expected[key]:
            raise _ProtocolError()
    if args["line"] is not None and type(receipt["scope"]["line"]) is not int:
        raise _ProtocolError()
    if receipt["status"] not in ("complete", "incomplete"):
        raise _ProtocolError()
    complete = receipt["status"] == "complete"
    if (complete and receipt["reason"] is not None) or (
            not complete and (type(receipt["reason"]) is not str or receipt["reason"] not in _REASONS)):
        raise _ProtocolError()
    per_line = receipt["per_line_count"]
    source = receipt["source"]
    if type(per_line) is not dict or set(per_line) != {"min", "max"}:
        raise _ProtocolError()
    if type(source) is not dict or set(source) != {"bytes", "sha256", "partial"} or type(source["partial"]) is not bool:
        raise _ProtocolError()
    for value in (receipt["count"], receipt["total_count"], receipt["physical_lines"], per_line["min"], per_line["max"], source["bytes"]):
        if value is not None and (type(value) is not int or not 0 <= value <= MAX_SCAN_BYTES + 1):
            raise _ProtocolError()
    digest = source["sha256"]
    if digest is not None and (type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise _ProtocolError()
    if (digest is None) != (source["bytes"] is None):
        raise _ProtocolError()
    if complete:
        if (source["partial"] or digest is None or receipt["count"] is None
                or receipt["total_count"] is None or receipt["physical_lines"] is None):
            raise _ProtocolError()
        if receipt["physical_lines"] == 0:
            if per_line != {"min": None, "max": None} or receipt["total_count"] != 0 or receipt["count"] != 0:
                raise _ProtocolError()
        elif per_line["min"] is None or per_line["max"] is None or per_line["min"] > per_line["max"]:
            raise _ProtocolError()
        if receipt["count"] > receipt["total_count"] or (args["line"] is None and receipt["count"] != receipt["total_count"]):
            raise _ProtocolError()
    elif any(value is not None for value in (receipt["count"], receipt["total_count"], per_line["min"], per_line["max"])):
        raise _ProtocolError()
    if not complete and receipt["reason"] != "missing_line":
        if receipt["physical_lines"] is not None or not source["partial"]:
            raise _ProtocolError()
    if receipt["reason"] == "missing_line" and (
            source["partial"] or digest is None or receipt["physical_lines"] is None):
        raise _ProtocolError()
    return receipt


_CALL_LOCK = threading.Lock()
_QUARANTINE = None
_LAST_CLEANUP = None


def _minimal_env():
    # Values are paths only. No inherited API credentials, proxies or Python hooks.
    names = ("PATH", "SYSTEMROOT", "WINDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "NEXUS_AGENT_HOME")
    return {key: os.environ[key] for key in names if key in os.environ}


def _reap_owned(proc, deadline):
    terminated = killed = False
    try:
        if proc.poll() is None:
            proc.terminate()
            terminated = True
            try:
                proc.wait(timeout=max(0.0, (deadline - time.monotonic()) / 2))
            except subprocess.TimeoutExpired:
                proc.kill()
                killed = True
        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"pid": proc.pid, "terminated": terminated, "killed": killed,
            "reaped": proc.poll() is not None, "returncode": proc.poll()}


def _close_pipes(proc):
    closed = True
    for stream in (proc.stdin, proc.stdout):
        try:
            stream.close()
        except Exception:
            closed = False
    return closed


def _cleanup(proc, thread, deadline, *, normal=False, state=None):
    if normal:
        try:
            proc.wait(timeout=max(0.0, (deadline - time.monotonic()) / 2))
        except subprocess.TimeoutExpired:
            pass
    facts = _reap_owned(proc, deadline)
    thread.join(timeout=max(0.0, deadline - time.monotonic()))
    joined = not thread.is_alive()
    # Only the I/O owner closes active pipes: Windows CRT close can otherwise
    # wait on a read/write fd lock beyond this deadline.
    closed = (state.get("pipes_closed") is True if state is not None
              else _close_pipes(proc)) if joined else False
    facts.update(io_joined=joined, pipes_closed=closed,
                 confirmed=facts["reaped"] and joined and closed)
    return facts


def _cleanup_unstarted(proc, deadline):
    facts = _reap_owned(proc, deadline)
    closed = _close_pipes(proc)
    facts.update(io_joined=True, pipes_closed=closed, confirmed=facts["reaped"] and closed)
    return facts


class _LaunchLease:
    """One durable owner from before native start through verified cleanup.

    Publication, adoption and the single cleanup claim share a lock. Native
    creation, pipe I/O, joins and reaping never hold it. The registry retains
    an uncertain launch/cleanup; a later call cannot guess that it is gone.
    """

    def __init__(self, deadline, timeout):
        self.lock = threading.Lock()
        self.deadline = deadline
        self.timeout = timeout
        self.ready = None
        self.launcher = None
        self.launch_start_attempted = False
        self.launch_joined = False
        self.launch_error = None
        self.spawn_failed = False
        self.entered = False
        self.proc = None
        self.adopted = False
        self.cancelled = False
        self.cleanup_owner = None
        self.cleanup_started = False
        self.cleanup = None
        self.cleanup_error = None
        self.close_deadline = None
        self.io_thread = None
        self.io_start_attempted = False
        self.normal_completion = False
        self.state = {"request_attempted": False, "ack_attempted": False,
                      "ack_sent": False, "pipes_closed": None}

    def _claim_locked(self, owner):
        if self.cleanup_owner is not None or (owner == "launcher" and self.adopted):
            return False
        if self.close_deadline is None:
            self.close_deadline = time.monotonic() + min(1.0, self.timeout / 4)
        self.cleanup_owner = owner
        return True

    def _failed_launch(self, error):
        with self.lock:
            if self.launch_error is None:
                self.launch_error = error
            self.cancelled = True
            claimed = self.proc is not None and self._claim_locked("launcher")
            resume = self.cleanup_owner == "launcher" and not self.cleanup_started
        if claimed or resume:
            self._run_cleanup("launcher", original=error)

    def _launch(self):
        try:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-S", "-B", str(Path(__file__).absolute())],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=0, env=_minimal_env(), close_fds=True,
                )
            except OSError:
                with self.lock:
                    self.spawn_failed = True
                return
            with self.lock:
                self.proc = proc
                claimed = self.cancelled and self._claim_locked("launcher")
            if claimed:
                self._run_cleanup("launcher")
        except BaseException as error:
            self._failed_launch(error)
        finally:
            try:
                self.ready.set()
            except BaseException as error:
                self._failed_launch(error)

    def _observe_launcher(self, deadline):
        thread = self.launcher
        if thread is None:
            return not self.launch_start_attempted
        if thread is threading.current_thread():
            return False
        try:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        except RuntimeError:
            # A start that unwound need not have reached Python's started event.
            # Absence of ident/is_alive alone is not proof it cannot start late.
            return False
        finished = not thread.is_alive()
        with self.lock:
            self.launch_joined = finished
        return finished

    def __enter__(self):
        with self.lock:
            if self.entered:
                raise RuntimeError("A text-count launch lease can only be entered once.")
            self.entered = True
        try:
            self.ready = threading.Event()
            self.launcher = threading.Thread(
                target=self._launch, name="nexus-text-count-launch", daemon=True,
            )
            with self.lock:
                self.launch_start_attempted = True
            self.launcher.start()
            self.ready.wait(timeout=max(0.0, self.deadline - time.monotonic()))
            self._observe_launcher(self.deadline)
            if self.launch_error is not None:
                raise self.launch_error
            return self
        except BaseException as error:
            self.close(original=error)
            raise

    def adopt(self):
        with self.lock:
            if (self.adopted or self.cancelled or self.cleanup_owner is not None
                    or self.proc is None or not self.launch_joined
                    or self.launch_error is not None or time.monotonic() >= self.deadline):
                return False
            self.adopted = True
            return True

    def start_io(self, target):
        thread = threading.Thread(target=target, name="nexus-text-count-io", daemon=True)
        with self.lock:
            self.io_thread = thread
            if self.cancelled:
                return False
            # Assign pipe closing to the I/O owner before start can unwind.
            self.io_start_attempted = True
        thread.start()
        return True

    def begin_request(self):
        with self.lock:
            if not self.adopted or self.cancelled:
                return False
            self.state["request_attempted"] = True
            return True

    def begin_ack(self):
        with self.lock:
            if not self.adopted or self.cancelled:
                return False
            self.state["ack_attempted"] = True
            return True

    def execution_snapshot(self):
        with self.lock:
            return dict(self.state)

    def snapshot_cleanup(self):
        self._observe_launcher(time.monotonic())
        with self.lock:
            facts = dict(self.cleanup) if isinstance(self.cleanup, dict) else {}
            launcher_done = not self.launch_start_attempted or self.launch_joined
            reaped = facts.get("reaped") is True
            joined = facts.get("io_joined") is True
            closed = facts.get("pipes_closed") is True
            confirmed = (facts.get("confirmed") is True and reaped and joined and closed
                         and launcher_done and self.cleanup_error is None)
            return {
                "pid": facts.get("pid", getattr(self.proc, "pid", None)),
                "terminated": facts.get("terminated") is True,
                "killed": facts.get("killed") is True,
                "reaped": reaped, "returncode": facts.get("returncode"),
                "io_joined": joined, "pipes_closed": closed,
                "launch_joined": launcher_done, "confirmed": confirmed,
                "launch_pending": not launcher_done,
                "cleanup_error": type(self.cleanup_error).__name__ if self.cleanup_error else None,
            }

    def _run_cleanup(self, owner, *, original=None, normal=False):
        global _LAST_CLEANUP
        with self.lock:
            if self.cleanup_owner != owner or self.cleanup_started:
                return
            self.cleanup_started = True
            proc, thread, io_attempted = self.proc, self.io_thread, self.io_start_attempted
            deadline = self.close_deadline
        try:
            facts = (_cleanup(proc, thread, deadline, normal=normal, state=self.state)
                     if io_attempted else _cleanup_unstarted(proc, deadline))
            with self.lock:
                self.cleanup = facts if isinstance(facts, dict) else None
        except BaseException as error:
            with self.lock:
                if self.cleanup_error is None:
                    self.cleanup_error = error
            if original is None:
                raise
        finally:
            _LAST_CLEANUP = self.snapshot_cleanup()

    def close(self, *, original=None, normal=False):
        global _LAST_CLEANUP
        try:
            with self.lock:
                self.cancelled = True
                if self.close_deadline is None:
                    self.close_deadline = time.monotonic() + min(1.0, self.timeout / 4)
                deadline = self.close_deadline
                claimed = self.proc is not None and self._claim_locked("caller")
                resume = self.cleanup_owner == "caller" and not self.cleanup_started
                no_process = (self.proc is None and (
                    not self.launch_start_attempted or (self.launch_joined and self.spawn_failed)
                ))
                if no_process and self.cleanup_owner is None:
                    self.cleanup_owner = "caller"
                    self.cleanup_started = True
                    self.cleanup = {"pid": None, "reaped": True, "returncode": None,
                                    "io_joined": True, "pipes_closed": True, "confirmed": True}
            if claimed or resume:
                self._run_cleanup("caller", original=original, normal=normal)
            self._observe_launcher(deadline)
            _LAST_CLEANUP = self.snapshot_cleanup()
        except BaseException as error:
            with self.lock:
                if self.cleanup_error is None:
                    self.cleanup_error = error
            # The lease was registered before start and remains retained even if
            # a second cancellation prevents any cleanup facts being produced.
            if original is None:
                raise
        if original is None and self.launch_error is not None:
            raise self.launch_error

    def busy(self):
        return not self.snapshot_cleanup()["confirmed"]

    def __exit__(self, exc_type, exc, traceback):
        self.close(original=exc, normal=self.normal_completion)
        return False


def _launch_owned(deadline, timeout):
    """Create an unstarted lease; the caller registers it before __enter__."""
    return _LaunchLease(deadline, timeout)


def _quarantine_busy(lease):
    return lease.busy()


def inspect_text(path, literal, whole_word=True, line=None, *, timeout):
    """Inspect via one registered lease, preserving uncertain ACK/cleanup facts."""
    global _QUARANTINE, _LAST_CLEANUP
    args = validate_query(path, literal, whole_word, line)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid private deadline")
    if not _CALL_LOCK.acquire(blocking=False):
        return _result(_receipt(args, "busy"), False)
    try:
        if _QUARANTINE is not None:
            if _quarantine_busy(_QUARANTINE):
                return _result(_receipt(args, "busy"), False)
            _QUARANTINE = None
        nonce = secrets.token_hex(16)
        request = {"nonce": nonce, "args": args}
        if len(_encode(request)) > MAX_REQUEST_BYTES:
            raise CountArgumentError("request_limit")
        deadline = time.monotonic() + timeout
        lease = _launch_owned(deadline, timeout)
        reason, receipt, started = None, None, False
        # This outer scope deliberately covers __enter__ failure: __exit__ is
        # never called by Python if entering the context raises.
        try:
            _QUARANTINE = lease
            with lease:
                if not lease.adopt():
                    reason = ("worker_crash" if lease.spawn_failed else
                              "timeout" if time.monotonic() >= deadline else "cleanup_unverified")
                else:
                    proc = lease.proc
                    events = queue.Queue(maxsize=3)
                    authorize = threading.Event()
                    stop = threading.Event()

                    def exchange():
                        try:
                            if not lease.begin_request():
                                return
                            _write_frame(proc.stdin, request, MAX_REQUEST_BYTES)
                            first = _read_frame(proc.stdout, MAX_RESPONSE_BYTES)
                            events.put_nowait(("message", first))
                            if type(first) is dict and first.get("type") == "started":
                                while not authorize.wait(0.01):
                                    if stop.is_set():
                                        return
                                if stop.is_set() or not lease.begin_ack():
                                    return
                                if proc.stdin.write(b"\x06") != 1:
                                    raise _ProtocolError()
                                with lease.lock:
                                    lease.state["ack_sent"] = True
                                events.put_nowait(("message", _read_frame(proc.stdout, MAX_RESPONSE_BYTES)))
                            if proc.stdout.read(1):
                                raise _ProtocolError()
                            events.put_nowait(("eof", None))
                        except (OSError, ValueError, _ProtocolError, queue.Full):
                            try:
                                events.put_nowait(("error", None))
                            except queue.Full:
                                pass
                        finally:
                            closed = _close_pipes(proc)
                            with lease.lock:
                                lease.state["pipes_closed"] = closed

                    try:
                        lease.start_io(exchange)
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                reason = "timeout"
                                break
                            try:
                                kind, message = events.get(timeout=remaining)
                            except queue.Empty:
                                reason = "timeout"
                                break
                            if kind == "error":
                                reason = "protocol_error" if proc.poll() in (None, 0) else "worker_crash"
                                break
                            if kind == "eof":
                                if receipt is None:
                                    raise _ProtocolError()
                                break
                            if (type(message) is not dict or message.get("nonce") != nonce
                                    or type(message.get("args")) is not dict
                                    or set(message["args"]) != set(args)
                                    or any(type(message["args"][key]) is not type(value)
                                           or message["args"][key] != value for key, value in args.items())):
                                raise _ProtocolError()
                            if message.get("type") == "started":
                                if started or receipt is not None or set(message) != {"type", "nonce", "args"}:
                                    raise _ProtocolError()
                                started = True
                                authorize.set()
                            elif message.get("type") == "result":
                                if receipt is not None or set(message) != {"type", "nonce", "args", "executed", "receipt"}:
                                    raise _ProtocolError()
                                if type(message["executed"]) is not bool or message["executed"] != started:
                                    raise _ProtocolError()
                                receipt = _validate_receipt(message["receipt"], args)
                                if not started and (receipt["status"] == "complete" or receipt["source"]["bytes"] is not None):
                                    raise _ProtocolError()
                            else:
                                raise _ProtocolError()
                    except _ProtocolError:
                        reason = "protocol_error"
                    finally:
                        stop.set()
                    lease.normal_completion = reason is None
        finally:
            original = sys.exc_info()[1]
            try:
                lease.close(original=original, normal=lease.normal_completion)
            except BaseException:
                if original is None:
                    raise
        cleanup = lease.snapshot_cleanup()
        _LAST_CLEANUP = cleanup
        if not cleanup["confirmed"]:
            reason = "cleanup_unverified"
        elif reason is None and cleanup["returncode"] != 0:
            reason = "worker_crash"
        state = lease.execution_snapshot()
        executed = state["ack_sent"]
        if not executed and state["ack_attempted"]:
            raise CountExecutionUncertain(
                reason or "protocol_error", receipt=_receipt(args, reason or "protocol_error"),
                cleanup=cleanup,
            )
        if reason is not None:
            receipt = _receipt(args, reason)
        return _result(receipt, executed)
    finally:
        _CALL_LOCK.release()


if __name__ == "__main__":
    try:
        _child_main()
    except BaseException:
        # No exception text, file contents or environment on the private channel.
        sys.exit(2)
