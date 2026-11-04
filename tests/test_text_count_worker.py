"""Real fixed-worker gates plus private hostile executables, never runtime hooks."""
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import threading
import time
import types

import pytest

from nexus_agent_ai.agent import _workspace_paths as paths
from nexus_agent_ai.agent import text_count_worker as worker
from nexus_agent_ai.agent import tools
from nexus_agent_ai.agent.text_count_limits import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, MAX_SCAN_BYTES
from tests._count_identity_diagnostics import count_identity_diagnostics


@pytest.fixture(autouse=True)
def clear_worker_private_state(monkeypatch):
    # Tests must not silently abandon a real owned process/thread.
    old = worker._QUARANTINE
    assert old is None or not worker._quarantine_busy(old)
    monkeypatch.setattr(worker, "_QUARANTINE", None)
    monkeypatch.setattr(worker, "_LAST_CLEANUP", None)
    yield
    current = worker._QUARANTINE
    assert current is None or not worker._quarantine_busy(current)


_TEST_PREAMBLE = '''import json, os, struct, sys

def exact(n):
    result = b""
    while len(result) < n:
        part = os.read(0, n - len(result))
        if not part:
            raise SystemExit(3)
        result += part
    return result

def receive():
    n = struct.unpack("!I", exact(4))[0]
    return json.loads(exact(n))

def send(value):
    data = json.dumps(value, separators=(",", ":")).encode()
    frame = struct.pack("!I", len(data)) + data
    while frame:
        frame = frame[os.write(1, frame):]

def started(request):
    send({"type":"started", "nonce":request["nonce"], "args":request["args"]})
    assert exact(1) == b"\\x06"
'''

_BLOCKING_PIPE_BODY = '''request = receive()
started(request)
r, w = os.pipe()
os.read(r, 1)  # w remains owned/open: an actual blocking kernel pipe read.
'''


def install_private_worker(monkeypatch, tmp_path, body):
    """Test-only executable factory; production always supplies its fixed argv.

    Main's configured-deadline smoke can reuse this fixture, which finishes the
    real STARTED/ACK exchange and blocks in os.read, not a cooperative sleep.
    """
    script = tmp_path / "private_worker.py"
    script.write_text(_TEST_PREAMBLE + body, encoding="utf-8")
    real_popen = subprocess.Popen
    launches = []

    def launch(argv, **kwargs):
        assert argv == [sys.executable, "-I", "-S", "-B", str(Path(worker.__file__).absolute())]
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["bufsize"] == 0
        assert kwargs["close_fds"] is True
        child = real_popen([sys.executable, "-I", "-S", "-B", str(script)], **kwargs)
        launches.append((child, kwargs))
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    return launches


@pytest.fixture
def private_worker(monkeypatch, tmp_path):
    return lambda body: install_private_worker(monkeypatch, tmp_path, body)


def assert_no_counts(receipt):
    assert receipt["status"] == "incomplete"
    assert receipt["count"] is None
    assert receipt["total_count"] is None
    assert receipt["per_line_count"] == {"min": None, "max": None}
    assert receipt["edit_proof"] is False


def assert_cleaned():
    assert worker._LAST_CLEANUP["confirmed"] is True
    assert worker._LAST_CLEANUP["reaped"] is True
    assert worker._LAST_CLEANUP["io_joined"] is True


def test_validation_is_pure(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("parent validation accessed filesystem")
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    assert worker.validate_query("./exact.txt", "é.*", False, 3) == {
        "path": "./exact.txt", "literal": "é.*", "whole_word": False, "line": 3,
    }


@pytest.mark.parametrize("args,reason", [
    (("", "a"), "invalid_path"), ((Path("a"), "a"), "invalid_path"),
    (("a\x00", "a"), "invalid_path"), (("a", ""), "invalid_literal"),
    (("a", "a\n"), "invalid_literal"), (("a", "a\r"), "invalid_literal"),
    (("a", "a", 1), "invalid_whole_word"), (("a", "a", True, True), "invalid_line"),
    (("a", "a", True, 0), "invalid_line"), (("a", "a", True, -1), "invalid_line"),
    (("a", "é" * 513), "literal_limit"), (("a", "\ud800"), "invalid_unicode"),
    (("a" * (MAX_REQUEST_BYTES + 1), "a"), "request_limit"),
])
def test_invalid_arguments_do_not_launch(monkeypatch, args, reason):
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(worker.CountArgumentError) as caught:
        worker.inspect_text(*args, timeout=1)
    assert caught.value.reason == reason


def test_real_spawn_complete_exact_receipt_and_no_edit_authority(tmp_path):
    data = b"\xef\xbb\xbfa a\r\na\n\na"
    Path("sample.txt").write_bytes(data)
    prior_state = dict(tools._FILE_STATE)
    prior_digests = dict(tools._FILE_DIGESTS)
    result = worker.inspect_text("./sample.txt", "a", line=2, timeout=5)
    assert result["executed"] is True
    assert result["refusal_class"] is None
    receipt = result["receipt"]
    assert receipt == {
        "status": "complete", "reason": None, "path": "./sample.txt", "literal": "a",
        "whole_word": True, "scope": {"kind": "physical_line", "line": 2},
        "count": 1, "total_count": 4, "physical_lines": 4,
        "per_line_count": {"min": 0, "max": 2},
        "source": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "partial": False},
        "edit_proof": False,
    }
    assert tools._FILE_STATE == prior_state
    assert tools._FILE_DIGESTS == prior_digests
    assert_cleaned()


def test_existing_prior_read_proof_unchanged():
    Path("sample.txt").write_text("a", encoding="utf-8")
    tools.execute_read_file("sample.txt")
    prior_state, prior_digests = dict(tools._FILE_STATE), dict(tools._FILE_DIGESTS)
    assert worker.inspect_text("sample.txt", "a", timeout=5)["executed"] is True
    assert tools._FILE_STATE == prior_state
    assert tools._FILE_DIGESTS == prior_digests


@pytest.mark.parametrize("data,line,reason,partial", [
    (b"a\xff", None, "decode_error", True),
    (b"a\n", 3, "missing_line", False),
    (b"a" * (MAX_SCAN_BYTES + 1), None, "size_limit", True),
])
def test_real_incomplete_never_exposes_counts(data, line, reason, partial):
    Path("sample.txt").write_bytes(data)
    result = worker.inspect_text("sample.txt", "a", line=line, timeout=5)
    assert_no_counts(result["receipt"])
    assert result["receipt"]["reason"] == reason
    assert result["receipt"]["source"]["partial"] is partial
    if reason == "missing_line":
        assert result["receipt"]["physical_lines"] == 1
        assert result["receipt"]["source"]["sha256"] == hashlib.sha256(data).hexdigest()
    if reason == "size_limit":
        assert result["executed"] is False
        assert result["receipt"]["source"]["bytes"] is None
    assert_cleaned()


@pytest.mark.parametrize("name,reason,refusal_class", [
    ("absent.txt", "not_found", "not_found"),
    (".env", "sensitive_path", "sensitive_path"),
    ("credentials.txt", "sensitive_path", "sensitive_path"),
    ("folder", "unsupported_file", "other"),
])
def test_real_prescan_refusal(name, reason, refusal_class):
    Path(".env").write_text("never read", encoding="utf-8")
    Path("credentials.txt").write_text("never read", encoding="utf-8")
    Path("folder").mkdir()
    result = worker.inspect_text(name, "never", timeout=5)
    assert result["executed"] is False
    assert result["refusal_class"] == refusal_class
    assert result["receipt"]["reason"] == reason
    assert result["receipt"]["source"] == {"bytes": None, "sha256": None, "partial": True}
    assert_no_counts(result["receipt"])
    assert_cleaned()


def test_real_outside_and_state_paths_are_never_read(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("a", encoding="utf-8")
    state = Path(os.environ["NEXUS_AGENT_HOME"])
    state.mkdir(parents=True, exist_ok=True)
    (state / "public.txt").write_text("a", encoding="utf-8")
    for name in (str(outside), str(state / "public.txt")):
        result = worker.inspect_text(name, "a", timeout=5)
        assert result["executed"] is False
        assert result["refusal_class"] == "sensitive_path"
        assert result["receipt"]["source"]["bytes"] is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_fifo_refuses_without_opening_for_content():
    os.mkfifo("pipe")
    result = worker.inspect_text("pipe", "a", timeout=2)
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "unsupported_file"
    assert_cleaned()


def test_private_blocking_pipe_deadline_owned_cleanup(private_worker):
    launches = private_worker(_BLOCKING_PIPE_BODY)
    start = time.monotonic()
    result = worker.inspect_text("target.txt", "a", timeout=1)
    elapsed = time.monotonic() - start
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "timeout"
    assert result["receipt"]["source"] == {"bytes": None, "sha256": None, "partial": True}
    assert_no_counts(result["receipt"])
    assert elapsed < 3
    assert launches[0][0].poll() is not None
    assert_cleaned()


def test_configured_deadline_stops_blocked_pipe(private_worker):
    """Operational gate: actual unchanged configured deadline, real blocked I/O."""
    launches = private_worker(_BLOCKING_PIPE_BODY)
    start = time.monotonic()
    result = worker.inspect_text("target.txt", "a", timeout=tools.CODE_EXECUTION_TIMEOUT)
    elapsed = time.monotonic() - start
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "timeout"
    assert elapsed >= tools.CODE_EXECUTION_TIMEOUT
    assert elapsed < tools.CODE_EXECUTION_TIMEOUT + 4
    assert launches[0][0].poll() is not None
    assert_cleaned()


@pytest.mark.parametrize("body", [
    "raise SystemExit(7)\n",
    "os.write(1, b'not a frame')\n",
    "os.write(1, struct.pack('!I', 0xffffffff))\nr,w=os.pipe()\nos.read(r,1)\n",
    "os.write(1, struct.pack('!I', 1) + b'{')\n",
    "request=receive()\nrequest['nonce']='0'*32\nstarted(request)\n",
    "request=receive()\nrequest['args']['whole_word']=1\nstarted(request)\n",
])
def test_crash_and_invalid_protocol_never_acknowledge(private_worker, body):
    private_worker(body)
    result = worker.inspect_text("target.txt", "a", timeout=2)
    assert result["executed"] is False
    assert result["receipt"]["reason"] in ("worker_crash", "protocol_error")
    assert_no_counts(result["receipt"])
    assert_cleaned()


def test_started_then_crash_preserves_execution(private_worker):
    private_worker("request=receive()\nstarted(request)\nraise SystemExit(7)\n")
    result = worker.inspect_text("target.txt", "a", timeout=2)
    assert result["executed"] is True
    assert result["receipt"]["reason"] in ("worker_crash", "protocol_error")
    assert_no_counts(result["receipt"])
    assert_cleaned()


def test_stuck_before_request_read_is_bounded(private_worker):
    private_worker("r,w=os.pipe()\nos.read(r,1)\n")
    result = worker.inspect_text("p" * 8000, "a", timeout=1)
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "timeout"
    assert_cleaned()


def test_environment_drops_credentials_and_python_hooks(private_worker, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-child")
    monkeypatch.setenv("HTTPS_PROXY", "not-for-child")
    monkeypatch.setenv("PYTHONPATH", "not-for-child")
    launches = private_worker("raise SystemExit(7)\n")
    worker.inspect_text("target.txt", "a", timeout=2)
    env = launches[0][1]["env"]
    assert not {"OPENAI_API_KEY", "HTTPS_PROXY", "PYTHONPATH"} & set(env)
    assert env["NEXUS_AGENT_HOME"] == os.environ["NEXUS_AGENT_HOME"]


def test_workspace_stdlib_shadows_are_not_imported():
    for name in ("json.py", "sitecustomize.py", "hashlib.py", "subprocess.py"):
        Path(name).write_text("raise RuntimeError('workspace code executed')\n", encoding="utf-8")
    Path("sample.txt").write_text("a", encoding="utf-8")
    result = worker.inspect_text("sample.txt", "a", timeout=5)
    assert result["receipt"]["status"] == "complete"


def test_child_waits_for_ack_before_content_read(monkeypatch):
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    data = worker._encode(request)
    incoming = io.BytesIO(struct.pack("!I", len(data)) + data)  # Deliberately no ACK.
    outgoing = io.BytesIO()
    Path("sample.txt").write_text("a", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=incoming))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=outgoing))
    monkeypatch.setattr(worker.os, "fdopen", lambda *a, **k: pytest.fail("content opened before ACK"))
    with pytest.raises(worker._ProtocolError):
        worker._child_main()
    assert worker._read_frame(io.BytesIO(outgoing.getvalue()), MAX_RESPONSE_BYTES)["type"] == "started"


def test_unsupported_handle_validation_is_no_read(monkeypatch):
    Path("sample.txt").write_text("a", encoding="utf-8")
    monkeypatch.setattr(worker.sys, "platform", "unsupported-os")
    with pytest.raises(worker._PathRefusal) as caught:
        worker._open_validated("sample.txt")
    assert caught.value.reason == "unsupported_handle_validation"


def test_handle_actual_outside_rejected(monkeypatch, tmp_path):
    Path("sample.txt").write_text("a", encoding="utf-8")
    monkeypatch.setattr(worker, "_opened_path", lambda fd: tmp_path.parent / "outside.txt")
    with pytest.raises(worker._PathRefusal) as caught:
        worker._open_validated("sample.txt")
    assert caught.value.reason == "path_refused"


@pytest.mark.skipif(os.name != "posix", reason="POSIX rename of an open file")
def test_path_identity_swap_rejected_before_read(monkeypatch):
    Path("sample.txt").write_text("a", encoding="utf-8")
    Path("replacement.txt").write_text("b", encoding="utf-8")
    real_opened = worker._opened_path

    def replace_after_open(fd):
        actual = real_opened(fd)
        os.replace("replacement.txt", "sample.txt")
        return actual

    monkeypatch.setattr(worker, "_opened_path", replace_after_open)
    with pytest.raises(worker._PathRefusal) as caught:
        worker._open_validated("sample.txt")
    assert caught.value.reason == "changed"


def test_same_handle_mutation_rejected_after_scan():
    Path("sample.txt").write_text("a", encoding="utf-8")
    fd, target, before = worker._open_validated("sample.txt")
    try:
        Path("sample.txt").write_text("changed", encoding="utf-8")
        with pytest.raises(worker._PathRefusal) as caught:
            worker._check_identity(fd, "sample.txt", target, before)
        assert caught.value.reason == "changed"
    finally:
        os.close(fd)


def test_sensitive_lexical_alias_never_opens(monkeypatch):
    Path("sample.txt").write_text("a", encoding="utf-8")
    try:
        Path(".env").symlink_to("sample.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink privilege unavailable")
    monkeypatch.setattr(worker.os, "open", lambda *a, **k: pytest.fail("sensitive alias opened"))
    with pytest.raises(worker._PathRefusal) as caught:
        worker._open_validated(".env")
    assert caught.value.reason == "sensitive_path"


def test_stable_inside_symlink_and_outside_symlink(tmp_path):
    Path("sample.txt").write_text("a", encoding="utf-8")
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("a", encoding="utf-8")
    try:
        Path("inside-link").symlink_to("sample.txt")
        Path("outside-link").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink privilege unavailable")
    assert worker.inspect_text("inside-link", "a", timeout=5)["receipt"]["status"] == "complete"
    result = worker.inspect_text("outside-link", "a", timeout=5)
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "path_refused"



def test_frame_length_rejected_before_body_read():
    class HeaderOnly:
        calls = 0
        def read(self, size):
            self.calls += 1
            assert self.calls == 1
            assert size == 4
            return struct.pack("!I", MAX_REQUEST_BYTES + 1)
    with pytest.raises(worker._ProtocolError):
        worker._read_frame(HeaderOnly(), MAX_REQUEST_BYTES)


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{', b'\xff'])
def test_invalid_frame_json_rejected(payload):
    with pytest.raises(worker._ProtocolError):
        worker._read_frame(io.BytesIO(struct.pack("!I", len(payload)) + payload), MAX_RESPONSE_BYTES)


def test_darwin_opened_path_portable_double(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(fcntl=lambda fd, cmd, arg: calls.append((fd, cmd, len(arg))) or b"/project/a\0" + bytes(1013))
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    assert worker._darwin_handle_path(17) == Path("/project/a")
    assert calls == [(17, 50, 1024)]


@pytest.mark.parametrize("name", ["\\\\?\\C:\\project\\a.txt", "\\\\?\\UNC\\server\\share\\a.txt"])
def test_windows_opened_path_portable_double(monkeypatch, name):
    calls = []
    class FinalPath:
        def __call__(self, handle, buffer, size, flags):
            calls.append((handle.value, size, flags))
            if buffer is None:
                return len(name) + 1
            buffer.value = name
            return len(name)
    final = FinalPath()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: types.SimpleNamespace(GetFinalPathNameByHandleW=final), raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: 2**40 + fd))
    result = worker._windows_handle_path(17)
    assert str(result) == ("\\\\" + name[8:] if "UNC" in name else name[4:])
    assert calls[0][0] == 2**40 + 17
    assert len(final.argtypes) == 4
    assert final.restype is ctypes.wintypes.DWORD


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows handle validation only")
def test_native_windows_final_handle_and_identity():
    Path("sample.txt").write_text("a", encoding="utf-8")
    fd, target, before = worker._open_validated("sample.txt")
    try:
        assert worker._windows_handle_path(fd) == target
        assert os.fstat(fd).st_ino != 0
        worker._check_identity(fd, "sample.txt", target, before)
    finally:
        os.close(fd)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows junction/reparse test only")
def test_native_windows_reparse_outside_refused(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "sample.txt").write_text("a", encoding="utf-8")
    try:
        Path("reparse").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Windows reparse creation privilege unavailable")
    result = worker.inspect_text("reparse/sample.txt", "a", timeout=5)
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "path_refused"


def test_stalled_launch_is_bounded_and_quarantined(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_popen = subprocess.Popen
    calls = []
    def stalled(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(worker.subprocess, "Popen", stalled)
    try:
        start = time.monotonic()
        result = worker.inspect_text("must-not-be-read.txt", "a", timeout=0.1)
        assert time.monotonic() - start < 1
        assert entered.is_set()
        assert result["executed"] is False
        assert result["receipt"]["reason"] == "cleanup_unverified"
        assert worker._LAST_CLEANUP["confirmed"] is False
        assert worker.inspect_text("other.txt", "a", timeout=0.1)["receipt"]["reason"] == "busy"
        assert calls == [1]
    finally:
        slot = worker._QUARANTINE
        release.set()
        slot.launcher.join(timeout=3)
        # The runtime's shared deadline may already be exhausted when Popen
        # finally returns. Reap only this test-owned child outside its lease.
        slot.proc.wait(timeout=3)
    facts = slot.snapshot_cleanup()
    assert slot.proc.poll() is not None
    if facts["confirmed"]:
        assert not worker._quarantine_busy(slot)
    else:
        assert worker._quarantine_busy(slot)
        assert worker.inspect_text("other.txt", "a", timeout=0.1)["receipt"]["reason"] == "busy"
        # Test-only external reaping cannot manufacture missing runtime facts.
        monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_unconfirmed_cleanup_quarantines_only_one_owned_process(monkeypatch, private_worker):
    launches = private_worker(_BLOCKING_PIPE_BODY)
    real_cleanup = worker._cleanup
    def unconfirmed(proc, thread, timeout, **kwargs):
        facts = real_cleanup(proc, thread, timeout, **kwargs)
        facts["confirmed"] = False
        return facts
    monkeypatch.setattr(worker, "_cleanup", unconfirmed)
    result = worker.inspect_text("target.txt", "a", timeout=1)
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "cleanup_unverified"
    assert worker._LAST_CLEANUP["confirmed"] is False
    assert worker._QUARANTINE.proc is launches[0][0]
    assert launches[0][0].poll() is not None
    assert_no_counts(result["receipt"])
    assert worker._quarantine_busy(worker._QUARANTINE)
    # The injected audit uncertainty must not authorize a new runtime launch.
    # This fixture alone can prove the real process and both pipes are gone.
    assert launches[0][0].stdin.closed and launches[0][0].stdout.closed
    monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_ack_delivery_error_is_typed_uncertainty(monkeypatch, private_worker):
    private_worker(_BLOCKING_PIPE_BODY)
    executable_factory = worker.subprocess.Popen

    class AckFailure:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            if data == b"\x06":
                raise BrokenPipeError()
            return self.stream.write(data)
        def flush(self):
            return self.stream.flush()
        def close(self):
            return self.stream.close()

    def launch(*args, **kwargs):
        child = executable_factory(*args, **kwargs)
        child.stdin = AckFailure(child.stdin)
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    with pytest.raises(worker.CountExecutionUncertain) as caught:
        worker.inspect_text("target.txt", "a", timeout=2)
    assert caught.value.reason in ("protocol_error", "worker_crash")
    assert_no_counts(caught.value.receipt)
    assert caught.value.cleanup["confirmed"] is True


def test_output_flood_is_stopped_at_declared_frame_bound(private_worker):
    private_worker("request=receive()\nstarted(request)\nos.write(1, struct.pack('!I', 0xffffffff))\n"
                   "block=b'x'*4096\nwhile True: os.write(1,block)\n")
    result = worker.inspect_text("target.txt", "a", timeout=2)
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "protocol_error"
    assert_no_counts(result["receipt"])
    assert_cleaned()


def test_trailing_output_after_result_is_protocol_error(private_worker):
    receipt = worker._receipt(worker.validate_query("target.txt", "a"), "not_found")
    private_worker("request=receive()\nreceipt=json.loads(" + repr(json.dumps(receipt)) + ")\n"
                   "send({'type':'result','nonce':request['nonce'],'args':request['args'],"
                   "'executed':False,'receipt':receipt})\nos.write(1,b'extra')\n")
    result = worker.inspect_text("target.txt", "a", timeout=2)
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "protocol_error"
    assert_cleaned()


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM escalation")
def test_owned_worker_ignoring_terminate_is_killed(private_worker):
    private_worker("import signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n" + _BLOCKING_PIPE_BODY)
    result = worker.inspect_text("target.txt", "a", timeout=1)
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "timeout"
    assert worker._LAST_CLEANUP["terminated"] is True
    assert worker._LAST_CLEANUP["killed"] is True
    assert_cleaned()


def test_unreliable_inode_fails_closed():
    info = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_ino=0)
    with pytest.raises(worker._PathRefusal) as caught:
        worker._reliable_identity(info)
    assert caught.value.reason == "unsupported_handle_validation"


def test_linux_proc_handle_unavailable_fails_closed(monkeypatch):
    monkeypatch.setattr(worker.sys, "platform", "linux")
    def unavailable(*args):
        raise FileNotFoundError()
    monkeypatch.setattr(worker.os, "readlink", unavailable)
    with pytest.raises(worker._PathRefusal) as caught:
        worker._opened_path(17)
    assert caught.value.reason == "unsupported_handle_validation"


@pytest.mark.parametrize("returned", [0, 40000])
def test_windows_api_unverifiable_path_fails_closed(monkeypatch, returned):
    class FinalPath:
        def __call__(self, *args):
            return returned
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: types.SimpleNamespace(GetFinalPathNameByHandleW=FinalPath()), raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(get_osfhandle=lambda fd: 17))
    with pytest.raises(worker._PathRefusal) as caught:
        worker._windows_handle_path(17)
    assert caught.value.reason == "unsupported_handle_validation"


@pytest.mark.parametrize("mutation", [
    {"reason": []}, {"reason": {}}, {"status": []},
    {"count": 0}, {"physical_lines": 0},
    {"source": {"bytes": None, "sha256": None, "partial": False}},
    {"source": {"bytes": 0, "sha256": "not-a-digest", "partial": True}},
    {"edit_proof": True},
])
def test_malformed_result_receipts_are_typed_protocol_errors(mutation):
    args = worker.validate_query("target.txt", "a")
    receipt = worker._receipt(args, "not_found")
    receipt.update(mutation)
    with pytest.raises(worker._ProtocolError):
        worker._validate_receipt(receipt, args)


def test_nonfinite_json_rejected():
    data = b'{"value":NaN}'
    with pytest.raises(worker._ProtocolError):
        worker._read_frame(io.BytesIO(struct.pack("!I", len(data)) + data), MAX_RESPONSE_BYTES)


def test_child_prescan_permission_failure_retains_no_read_proof(monkeypatch):
    Path("sample.txt").write_text("a", encoding="utf-8")
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    data = worker._encode(request)
    outgoing = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(struct.pack("!I", len(data)) + data)))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=outgoing))
    def denied(*args, **kwargs):
        raise PermissionError()
    monkeypatch.setattr(worker.os, "open", denied)
    worker._child_main()
    result = worker._read_frame(io.BytesIO(outgoing.getvalue()), MAX_RESPONSE_BYTES)
    assert result["type"] == "result"
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "permission"
    assert result["receipt"]["source"]["bytes"] is None


def test_child_rechecks_identity_after_ack_before_content(monkeypatch):
    Path("sample.txt").write_text("a", encoding="utf-8")
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    data = worker._encode(request)
    class DelayedAck(io.BytesIO):
        def read(self, size=-1):
            if self.tell() == len(data) + 4:
                Path("sample.txt").write_text("changed during ACK", encoding="utf-8")
            return super().read(size)
    outgoing = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=DelayedAck(struct.pack("!I", len(data)) + data + b"\x06")))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=outgoing))
    monkeypatch.setattr(worker.os, "fdopen", lambda *a, **k: pytest.fail("read after detected change"))
    worker._child_main()
    frames = io.BytesIO(outgoing.getvalue())
    assert worker._read_frame(frames, MAX_RESPONSE_BYTES)["type"] == "started"
    result = worker._read_frame(frames, MAX_RESPONSE_BYTES)
    assert result["executed"] is True
    assert result["receipt"]["reason"] == "changed"
    assert result["receipt"]["source"]["bytes"] is None


def test_quarantined_live_owner_prevents_another_spawn(monkeypatch):
    status = {"returncode": None}
    owner = types.SimpleNamespace(poll=lambda: status["returncode"])
    lease = worker._LaunchLease(time.monotonic() + 1, 1)
    lease.proc = owner
    monkeypatch.setattr(worker, "_QUARANTINE", lease)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("second worker spawned"))
    try:
        for _ in range(3):
            result = worker.inspect_text("target.txt", "a", timeout=1)
            assert result["executed"] is False
            assert result["receipt"]["reason"] == "busy"
            assert worker._QUARANTINE.proc is owner
    finally:
        status["returncode"] = 0
        # Fake owner has no native resources; missing facts still stayed busy.
        monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_child_postscan_change_invalidates_counts_but_preserves_prefix(monkeypatch):
    from nexus_agent_ai.agent import text_count_scan as scanner
    data = b"a a\n"
    Path("sample.txt").write_bytes(data)
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    encoded = worker._encode(request)
    outgoing = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(
        struct.pack("!I", len(encoded)) + encoded + b"\x06")))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=outgoing))
    real_scan = scanner.scan_stream

    def mutate_after_scan(*args, **kwargs):
        facts = real_scan(*args, **kwargs)
        Path("sample.txt").write_bytes(b"changed")
        return facts

    monkeypatch.setattr(scanner, "scan_stream", mutate_after_scan)
    worker._child_main()
    frames = io.BytesIO(outgoing.getvalue())
    worker._read_frame(frames, MAX_RESPONSE_BYTES)
    result = worker._read_frame(frames, MAX_RESPONSE_BYTES)
    assert result["executed"] is True
    receipt = result["receipt"]
    assert_no_counts(receipt)
    assert receipt["reason"] == "changed"
    assert receipt["physical_lines"] is None
    assert receipt["source"] == {
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "partial": True,
    }


def test_cleanup_does_not_close_pipes_owned_by_active_io(monkeypatch):
    def forbidden_close():
        pytest.fail("closing a pipe with active I/O can block in the Windows CRT")
    proc = types.SimpleNamespace(
        stdin=types.SimpleNamespace(close=forbidden_close),
        stdout=types.SimpleNamespace(close=forbidden_close),
    )
    active_io = types.SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    monkeypatch.setattr(worker, "_reap_owned", lambda *a: {
        "pid": 123, "terminated": True, "killed": True, "reaped": False,
        "returncode": None,
    })
    facts = worker._cleanup(proc, active_io, time.monotonic() + 1)
    assert facts["confirmed"] is False
    assert facts["io_joined"] is False


def test_positive_physical_line_is_not_an_arbitrary_machine_integer_limit():
    Path("sample.txt").write_text("a\n", encoding="utf-8")
    result = worker.inspect_text("sample.txt", "a", line=2**64, timeout=5)
    assert result["executed"] is True
    assert_no_counts(result["receipt"])
    assert result["receipt"]["reason"] == "missing_line"
    assert result["receipt"]["physical_lines"] == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux proc path spelling")
def test_literal_deleted_suffix_is_a_valid_regular_filename():
    Path("data (deleted)").write_text("a a\n", encoding="utf-8")
    result = worker.inspect_text("data (deleted)", "a", timeout=5)
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["count"] == 2


def test_premature_eof_cannot_claim_a_complete_source(monkeypatch):
    from nexus_agent_ai.agent import text_count_scan as scanner
    Path("sample.txt").write_bytes(b"a a\n")
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    encoded = worker._encode(request)
    outgoing = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(
        struct.pack("!I", len(encoded)) + encoded + b"\x06")))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=outgoing))
    real_scan = scanner.scan_stream
    monkeypatch.setattr(scanner, "scan_stream",
                        lambda *a, **k: real_scan(io.BytesIO(b"a"), "a"))
    worker._child_main()
    frames = io.BytesIO(outgoing.getvalue())
    worker._read_frame(frames, MAX_RESPONSE_BYTES)
    result = worker._read_frame(frames, MAX_RESPONSE_BYTES)
    assert result["executed"] is True
    assert_no_counts(result["receipt"])
    assert result["receipt"]["reason"] == "changed"
    assert result["receipt"]["source"] == {
        "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest(), "partial": True,
    }


def test_pipe_close_failure_is_explicit_and_quarantines_owner(monkeypatch, private_worker):
    private_worker(_BLOCKING_PIPE_BODY)
    executable_factory = worker.subprocess.Popen
    owned = []

    class CloseFailure:
        def __init__(self, stream):
            self.stream = stream
        def read(self, size):
            return self.stream.read(size)
        def close(self):
            self.stream.close()
            raise OSError("synthetic cleanup failure")

    def launch(*args, **kwargs):
        child = executable_factory(*args, **kwargs)
        child.stdout = CloseFailure(child.stdout)
        owned.append(child)
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    try:
        result = worker.inspect_text("target.txt", "a", timeout=1)
        assert result["executed"] is True
        assert_no_counts(result["receipt"])
        assert result["receipt"]["reason"] == "cleanup_unverified"
        assert worker._LAST_CLEANUP["pipes_closed"] is False
        assert worker.inspect_text("another.txt", "a", timeout=1)["receipt"]["reason"] == "busy"
        assert len(owned) == 1
    finally:
        # This test double raises after successful native closure. Release only
        # after proving that its real owned process and both pipes are gone.
        assert owned[0].poll() is not None
        assert owned[0].stdin.closed and owned[0].stdout.stream.closed
        monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_parent_does_not_make_filesystem_cwd_probe_before_deadline(monkeypatch):
    Path("sample.txt").write_text("a a", encoding="utf-8")
    real_getcwd = os.getcwd
    parent_thread = threading.current_thread()

    def blocked_parent_getcwd():
        if threading.current_thread() is parent_thread:
            pytest.fail("getcwd in the caller is not deadline-contained")
        return real_getcwd()

    with monkeypatch.context() as patch:
        patch.setattr(worker.os, "getcwd", blocked_parent_getcwd)
        result = worker.inspect_text("sample.txt", "a", timeout=5)
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["count"] == 2
    assert_cleaned()


@pytest.mark.skipif(os.name != "posix", reason="POSIX inherited working-directory rename")
def test_spawn_keeps_actual_workspace_when_its_old_name_is_replaced(tmp_path, monkeypatch):
    workspace, moved = tmp_path / "workspace", tmp_path / "moved"
    workspace.mkdir()
    (workspace / "sample.txt").write_bytes(b"a\n")
    monkeypatch.chdir(workspace)
    real_popen = subprocess.Popen

    def rename_before_spawn(*args, **kwargs):
        workspace.rename(moved)
        workspace.mkdir()
        (workspace / "sample.txt").write_bytes(b"a a a a\n")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(worker.subprocess, "Popen", rename_before_spawn)
    result = worker.inspect_text("sample.txt", "a", timeout=5)
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["count"] == 1
    assert result["receipt"]["source"]["sha256"] == hashlib.sha256(b"a\n").hexdigest()
    assert_cleaned()


def test_bounded_response_reports_a_long_request_failure_without_protocol_loss():
    # This fits the request contract, not the host's filename limit. Both
    # argument echoes must fit so the OS refusal is not misreported as IPC loss.
    target = "x" * (MAX_REQUEST_BYTES * 3 // 4)
    result = worker.inspect_text(target, "a", timeout=5)
    assert result["executed"] is False
    assert result["receipt"]["reason"] in ("unsupported_file", "not_found")
    assert result["receipt"]["path"] == target
    assert_no_counts(result["receipt"])
    assert_cleaned()


def test_launch_thread_creation_failure_has_no_fabricated_execution(monkeypatch):
    error = RuntimeError("synthetic native thread creation failure")
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(worker.threading.Thread, "start", unavailable)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("no launch thread existed"))
    try:
        with pytest.raises(RuntimeError) as caught:
            worker.inspect_text("sample.txt", "a", timeout=1)
        assert caught.value is error
        assert worker._quarantine_busy(worker._QUARANTINE)
        assert worker.inspect_text("sample.txt", "a", timeout=1)["receipt"]["reason"] == "busy"
    finally:
        # The test replacement never called native start; release only its fake.
        monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_fresh_fixed_worker_does_not_write_package_bytecode(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted-install"
    trusted.mkdir()
    source = Path(worker.__file__).parent
    for name in ("text_count_worker", "refusals", "_workspace_paths",
                 "text_count_limits", "text_count_scan"):
        (trusted / (name + ".py")).write_bytes((source / (name + ".py")).read_bytes())
    monkeypatch.setattr(worker, "__file__", str(trusted / "text_count_worker.py"))
    Path("sample.txt").write_text("a a\n", encoding="utf-8")
    result = worker.inspect_text("sample.txt", "a", timeout=5)
    assert result["receipt"]["status"] == "complete"
    assert result["receipt"]["count"] == 2
    assert not (trusted / "__pycache__").exists()
    assert_cleaned()


def test_cancelled_pending_launch_reaps_late_child_without_request(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_event = threading.Event
    original_popen = subprocess.Popen
    created = []
    owned = []
    writes = []

    class InterruptReady(original_event):
        def __init__(self):
            super().__init__()
            created.append(self)
        def wait(self, timeout=None):
            if self is created[0]:
                assert entered.wait(3)
                raise KeyboardInterrupt()
            return super().wait(timeout)

    class NoRequest:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            writes.append(data)
            return self.stream.write(data)
        def close(self):
            self.stream.close()

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        child = original_popen(*args, **kwargs)
        child.stdin = NoRequest(child.stdin)
        owned.append(child)
        return child

    monkeypatch.setattr(worker.threading, "Event", InterruptReady)
    monkeypatch.setattr(worker.subprocess, "Popen", delayed)
    try:
        with pytest.raises(KeyboardInterrupt):
            worker.inspect_text("must-not-read.txt", "a", timeout=1)
        slot = worker._QUARANTINE
        assert worker._quarantine_busy(slot)
        assert worker.inspect_text("another.txt", "a", timeout=1)["receipt"]["reason"] == "busy"
    finally:
        release.set()
        if worker._QUARANTINE is not None:
            worker._QUARANTINE.launcher.join(timeout=5)
            worker._QUARANTINE.proc.wait(timeout=3)
    assert len(owned) == 1 and owned[0].poll() is not None
    assert writes == []
    if slot.snapshot_cleanup()["confirmed"]:
        assert not worker._quarantine_busy(slot)
    else:
        assert worker._quarantine_busy(slot)
        assert worker.inspect_text("another.txt", "a", timeout=1)["receipt"]["reason"] == "busy"
        monkeypatch.setattr(worker, "_QUARANTINE", None)


def test_cancelled_acknowledged_io_reaps_owned_worker(monkeypatch, private_worker):
    private_worker(_BLOCKING_PIPE_BODY)
    factory = worker.subprocess.Popen
    ack = threading.Event()
    owned = []
    original_queue = worker.queue.Queue

    class AckStream:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            size = self.stream.write(data)
            if data == b"\x06" and size == 1:
                ack.set()
            return size
        def flush(self):
            self.stream.flush()
        def close(self):
            self.stream.close()

    class InterruptAfterStarted(original_queue):
        def get(self, *args, **kwargs):
            if getattr(self, "received_started", False):
                assert ack.wait(3)
                raise KeyboardInterrupt()
            item = super().get(*args, **kwargs)
            self.received_started = True
            return item

    def launch(*args, **kwargs):
        child = factory(*args, **kwargs)
        child.stdin = AckStream(child.stdin)
        owned.append(child)
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    monkeypatch.setattr(worker.queue, "Queue", InterruptAfterStarted)
    with pytest.raises(KeyboardInterrupt):
        worker.inspect_text("target.txt", "a", timeout=5)
    assert ack.is_set()
    assert len(owned) == 1 and owned[0].poll() is not None
    assert_cleaned()


@pytest.mark.parametrize("data", [b"a a\n", b"\xff", None])
def test_diagnostic_child_preserves_actual_receipt_and_consumed_checks(tmp_path, data):
    """Execute original worker bytes, not an in-process substitute for the child."""
    from tests import _count_identity_diagnostics as diagnostic
    if data is not None:
        Path("sample.txt").write_bytes(data)
    args = worker.validate_query("sample.txt", "a")
    request = {"nonce": "a" * 32, "args": args}
    wire = io.BytesIO()
    worker._write_frame(wire, request, MAX_REQUEST_BYTES)
    payload = wire.getvalue() + b"\x06"
    actual_worker = os.path.abspath(worker.__file__)
    original = [sys.executable, "-I", "-S", "-B", actual_worker]
    artifact = tmp_path / "observed.json"
    traced = [*original[:4], os.path.abspath(diagnostic.__file__), actual_worker, str(artifact)]
    baseline = subprocess.run(original, input=payload, capture_output=True, timeout=5)
    observed = subprocess.run(traced, input=payload, capture_output=True, timeout=5)
    assert (observed.returncode, observed.stdout, observed.stderr) == (
        baseline.returncode, baseline.stdout, baseline.stderr)
    raw = artifact.read_bytes()
    assert len(raw) <= diagnostic.MAX_BYTES
    facts = json.loads(raw)
    assert facts["worker_sha256"] == hashlib.sha256(Path(actual_worker).read_bytes()).hexdigest()
    phases = facts["phases"]
    if data is None:
        assert phases[0]["status"] == "ERROR"
        assert phases[0]["error"] == "FileNotFoundError"
        assert phases[1]["status"] == phases[2]["status"] == "NOT_REACHED"
        assert phases[3]["reason"] == "not_found"
    else:
        assert phases[0]["stats"][0]["consumed"]["st_size"] == len(data)
        assert phases[0]["checks"][0]["actual_equals_target"] is True
        assert phases[1]["actual_allowed_equals_target"] is True
        assert phases[3]["receipt_status"] == ("complete" if data == b"a a\n" else "incomplete")
        assert phases[3]["reason"] == (None if data == b"a a\n" else "decode_error")
        assert len(phases[1]["stats"]) == 3


def test_identity_trace_preserves_short_circuit_stat_order_and_exception(tmp_path, monkeypatch):
    from tests._count_identity_diagnostics import WorkerTrace
    Path("sample.txt").write_bytes(b"a")
    fd, target, before = worker._open_validated("sample.txt")
    Path("sample.txt").write_bytes(b"changed")
    calls = []
    real_stat, real_fstat = os.stat, os.fstat

    def logged_stat(*args, **kwargs):
        calls.append("stat")
        return real_stat(*args, **kwargs)

    def logged_fstat(*args, **kwargs):
        calls.append("fstat")
        return real_fstat(*args, **kwargs)

    monkeypatch.setattr(os, "stat", logged_stat)
    monkeypatch.setattr(os, "fstat", logged_fstat)
    try:
        with pytest.raises(worker._PathRefusal) as baseline:
            worker._check_identity(fd, "sample.txt", target, before)
        baseline_calls = list(calls)
        calls.clear()
        capture = WorkerTrace(worker.__file__)
        with capture, pytest.raises(worker._PathRefusal) as observed:
            worker._check_identity(fd, "sample.txt", target, before)
        assert observed.value.reason == baseline.value.reason == "changed"
        assert calls == baseline_calls
        check = capture.payload["phases"][1]
        assert check["status"] == "ERROR"
        assert len(check["stats"]) == 1  # Later stat predicates were short-circuited.
        assert check["stats"][0]["consumed"]["st_size"] == 7
        error = PermissionError("sentinel")
        monkeypatch.setattr(worker, "_opened_path", lambda fd: (_ for _ in ()).throw(error))
        with WorkerTrace(worker.__file__) as failed, pytest.raises(PermissionError) as raised:
            worker._check_identity(fd, "sample.txt", target, before)
        assert raised.value is error
        assert failed.payload["phases"][1]["actual_equals_target"] == "NOT_REACHED"
    finally:
        os.close(fd)


def test_identity_trace_caps_final_envelope_and_retains_not_reached(tmp_path):
    from tests._count_identity_diagnostics import MAX_BYTES, WorkerTrace
    capture = WorkerTrace(worker.__file__)
    capture._save(0, {"phase": "open", "status": "ERROR", "values": [1] * MAX_BYTES})
    artifact = tmp_path / "bounded.json"
    capture.write(artifact)
    raw = artifact.read_bytes()
    assert len(raw) <= MAX_BYTES
    assert json.loads(raw)["cap_exceeded"] is True
    phases = json.loads(raw)["phases"]
    assert phases[0]["status"] == "OMITTED_CAP"
    assert all(phase["status"] == "NOT_REACHED" for phase in phases[1:])


def test_diagnostic_launcher_matches_windows_lexical_case_and_separator_aliases(tmp_path, monkeypatch):
    import ntpath
    from tests._count_identity_diagnostics import _prepare_worker_launch
    script = r"d:/SITE/agent/temporary/../TEXT_COUNT_WORKER.PY"
    installed = r"D:\site\agent\text_count_worker.py"
    argv = ["python.exe", "-I", "-S", "-B", script]
    artifact = tmp_path / "matched.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("launcher matching probed the filesystem")

    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", forbidden)
        patch.setattr(os, "lstat", forbidden)
        patch.setattr(Path, "resolve", forbidden)
        actual = _prepare_worker_launch(argv, installed, "diagnostic-helper.py", artifact, paths=ntpath)
    assert actual == [*argv[:4], "diagnostic-helper.py", script, str(artifact)]
    assert not artifact.with_suffix(".unmatched.json").exists()


@pytest.mark.parametrize("flags,script,reason", [
    (["-I", "-S", "-B"], r"E:\site\agent\text_count_worker.py", "path_mismatch"),
    (["-I", "-S", "-X"], r"D:\site\agent\text_count_worker.py", "argv_shape"),
])
def test_diagnostic_launcher_records_unmatched_worker_without_changing_launch(tmp_path, flags, script, reason):
    import ntpath
    from tests._count_identity_diagnostics import MAX_BYTES, _prepare_worker_launch
    argv = ["python.exe", *flags, script]
    artifact = tmp_path / "invocation.json"
    actual = _prepare_worker_launch(
        argv, r"D:\site\agent\text_count_worker.py", "diagnostic-helper.py", artifact, paths=ntpath)
    assert actual is argv
    marker = artifact.with_suffix(".unmatched.json").read_bytes()
    assert json.loads(marker) == {"status": "UNMATCHED_WORKER_LAUNCH", "reason": reason}
    assert len(marker) <= MAX_BYTES
    assert not artifact.exists()


def test_diagnostic_launcher_preserves_posix_case_distinction(tmp_path):
    import posixpath
    from tests._count_identity_diagnostics import _prepare_worker_launch
    argv = ["python", "-I", "-S", "-B", "/Site/agent/text_count_worker.py"]
    artifact = tmp_path / "posix.json"
    actual = _prepare_worker_launch(
        argv, "/site/agent/text_count_worker.py", "diagnostic-helper.py", artifact, paths=posixpath)
    assert actual is argv
    assert json.loads(artifact.with_suffix(".unmatched.json").read_bytes())["reason"] == "path_mismatch"


def test_windows_snapshot_omits_ctime_ns_to_prevent_false_changed_refusal(monkeypatch):
    """On Windows, identity snapshot verifies dev, ino, mode, size, and mtime; ctime is excluded."""
    import types
    stat_base = types.SimpleNamespace(st_dev=1, st_ino=100, st_mode=stat.S_IFREG | 0o600,
                                      st_size=42, st_mtime_ns=1000, st_ctime_ns=5000)
    stat_ctime_drift = types.SimpleNamespace(st_dev=1, st_ino=100, st_mode=stat.S_IFREG | 0o600,
                                             st_size=42, st_mtime_ns=1000, st_ctime_ns=999999)
    stat_mtime_drift = types.SimpleNamespace(st_dev=1, st_ino=100, st_mode=stat.S_IFREG | 0o600,
                                             st_size=42, st_mtime_ns=2000, st_ctime_ns=5000)

    # Under simulated win32:
    monkeypatch.setattr(worker.sys, "platform", "win32")
    snap1 = worker._snapshot(stat_base)
    snap2 = worker._snapshot(stat_ctime_drift)
    snap3 = worker._snapshot(stat_mtime_drift)
    assert len(snap1) == 5
    assert snap1 == snap2, "Windows snapshot must not reject on ctime_ns drift alone"
    assert snap1 != snap3, "Windows snapshot must still reject on mtime_ns drift"

    # Under simulated linux:
    monkeypatch.setattr(worker.sys, "platform", "linux")
    snap_posix1 = worker._snapshot(stat_base)
    snap_posix2 = worker._snapshot(stat_ctime_drift)
    assert len(snap_posix1) == 6
    assert snap_posix1 != snap_posix2, "POSIX snapshot must continue checking ctime_ns"
