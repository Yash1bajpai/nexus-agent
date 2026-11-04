"""Local transport/fault tests and explicit, non-substitutable Windows cases."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

_PATH = Path(__file__).resolve().parents[1] / ".github" / "qualification" / "windows_process.py"
_SPEC = importlib.util.spec_from_file_location("q1_windows_process_tests", _PATH)
q1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(q1)


def domain(root, **kwargs):
    return q1.ProcessDomain(root, deadline=time.monotonic() + 15, **kwargs)


def python(source, *args):
    return [sys.executable, "-I", "-c", source, *map(str, args)]


class Cancellation(BaseException):
    pass


def native_start_name():
    # Exercise the actual start primitive used by this CPython, not a fake ident.
    return "_start_joinable_thread" if hasattr(threading, "_start_joinable_thread") else "_start_new_thread"


@pytest.fixture
def gated_native_start(monkeypatch):
    releases, owners = [], []

    def install(role, interrupt):
        release, entered, finished = (threading.Event() for _ in range(3))
        releases.append(release)
        primitive = native_start_name()
        actual = getattr(threading, primitive)
        original = Cancellation("interrupted native return")
        observed = {}

        def native_start(target, *args, **kwargs):
            owner = getattr(target, "__self__", None)
            if getattr(owner, "name", None) != role:
                return actual(target, *args, **kwargs)
            observed["owner"] = owner
            owners.append(owner)

            def bootstrap(*target_args):
                entered.set()
                assert release.wait(5), "test did not release bootstrap"
                try:
                    target(*target_args)
                finally:
                    finished.set()

            result = actual(bootstrap, *args, **kwargs)
            if interrupt:
                raise original
            return result

        monkeypatch.setattr(threading, primitive, native_start)
        return release, entered, finished, original, observed

    yield install
    for release in releases:
        release.set()
    for owner in owners:
        # Fixture cleanup is not production proof; production must do its own join.
        if owner.ident is not None:
            owner.join(2)


class FakeAPI:
    """Faults publish ownership before raising, as required of the native adapter."""

    def __init__(self, fault=None, error=None, *, empty=True):
        self.fault = fault
        self.error = error or Cancellation("injected")
        self.empty = empty
        self.events = []
        self.live = set()
        self.next_handle = 100
        self.running = False

    def step(self, name):
        self.events.append(name)
        if name == self.fault:
            raise self.error

    def handle(self, owner, key):
        self.next_handle += 1
        self.live.add(self.next_handle)
        owner.handles[key] = self.next_handle

    def job_into(self, owner):
        self.handle(owner, "job")
        self.step("job")

    def pipe_into(self, owner, name, *, input_pipe=False):
        self.handle(owner, name + "_read")
        self.handle(owner, name + "_write")
        self.step(name + "_pipe")

    def null_into(self, owner):
        self.handle(owner, "stdin_read")
        self.step("null")

    def create_into(self, owner, argv, cwd, env):
        self.handle(owner, "process")
        self.handle(owner, "thread")
        owner.pid = 321
        self.step("create")

    def assign(self, job, process):
        self.step("assign")

    def resume(self, thread):
        self.running = True
        self.step("resume")

    def stream_into(self, owner, name, key, *, write=False):
        self.close(owner.handles.pop(key))
        owner.streams[name] = io.BytesIO() if write else io.BytesIO(name.encode())
        self.step(name + "_stream")

    def wait(self, process, timeout):
        self.step("wait")
        return 0

    def terminate_job(self, job):
        self.step("terminate_job")
        self.running = False

    def terminate_process(self, process):
        self.step("terminate_process")

    def active_processes(self, job):
        self.step("active_processes")
        return 0 if self.empty else 1

    def members(self, job):
        return [{"pid": 321}]

    def close(self, handle):
        assert handle in self.live, "double or unowned close"
        self.live.remove(handle)
        self.events.append("close")


@pytest.mark.parametrize("fault", ["job", "stdout_pipe", "stderr_pipe", "null", "create",
                                  "assign", "stdout_stream", "stderr_stream", "resume"])
def test_durable_owner_constructor_and_adoption_cancellation(tmp_path, fault):
    error = Cancellation(fault)
    api = FakeAPI(fault, error)
    d = domain(tmp_path, platform="nt", api=api)
    with pytest.raises(Cancellation) as raised:
        d.run(["synthetic"], capture_output=True)
    assert raised.value is error
    assert not api.live
    assert d.records[0]["cleanup"]["confirmed"]
    if fault == "create" or fault == "assign":
        assert "terminate_process" in api.events
    if fault in ("stdout_stream", "stderr_stream", "resume"):
        assert "terminate_job" in api.events
    assert d._active is None


def test_assign_precedes_resume_and_owned_observer_only(tmp_path):
    api = FakeAPI()
    seen = []
    d = domain(tmp_path, platform="nt", api=api,
               observer=lambda members: seen.extend(members) or {"members": members})
    result = d.run(["synthetic"], capture_output=True)
    assert result.stdout == b"stdout"
    assert result.stderr == b"stderr"
    assert api.events.index("assign") < api.events.index("resume")
    assert seen == [{"pid": 321}]
    assert d.records[0]["observer"] == [{"members": [{"pid": 321}]}]
    assert d.records[0]["requested_timeout"] is None
    assert not api.live


def test_unknown_cleanup_quarantines_without_relaunch(tmp_path, monkeypatch):
    monkeypatch.setattr(q1, "TEARDOWN_SECONDS", .03)
    api = FakeAPI(empty=False)
    d = domain(tmp_path, platform="nt", api=api)
    with pytest.raises(q1.EvidenceIncomplete):
        d.run(["synthetic"])
    count = api.events.count("create")
    with pytest.raises(q1.EvidenceIncomplete):
        d.run(["not launched"])
    assert api.events.count("create") == count
    assert not d.records[0]["cleanup"]["confirmed"]


def test_original_exception_survives_cleanup_failure(tmp_path):
    error = Cancellation("original identity")
    api = FakeAPI("terminate_job", RuntimeError("cleanup failure"))
    d = domain(tmp_path, platform="nt", api=api,
               observer=lambda _: (_ for _ in ()).throw(error))
    with pytest.raises(Cancellation) as raised:
        d.run(["synthetic"])
    assert raised.value is error
    assert not d.records[0]["cleanup"]["confirmed"]
    assert d._active is not None
    assert set(d._active.handles) == {"job", "process"}


@pytest.mark.parametrize("role", ["q1-launcher", "q1-io-stdout", "q1-io-stderr", "q1-io-stdin"])
def test_genuine_pre_native_failure_retains_attempt_uncertainty(tmp_path, monkeypatch, role):
    api = FakeAPI()
    error = Cancellation("genuine pre-native failure")
    primitive = native_start_name()
    actual = getattr(threading, primitive)
    failed = []

    def fail_before_native(target, *args, **kwargs):
        owner = getattr(target, "__self__", None)
        if getattr(owner, "name", None) == role:
            failed.append(owner)
            raise error  # The fixture knows no native call occurred; production cannot.
        return actual(target, *args, **kwargs)

    monkeypatch.setattr(threading, primitive, fail_before_native)
    d = domain(tmp_path, platform="nt", api=api)
    try:
        with pytest.raises(Cancellation) as raised:
            d.run(["synthetic"], input=b"finite input")
        assert raised.value is error
        assert d._active is not None and d._quarantined
        result = d.records[0]["cleanup"]
        assert not result["confirmed"]
        if role == "q1-launcher":
            assert result["launcher_start_attempted"] and not result["launcher_entered"]
            assert not result["launcher_publication_complete"] and not result["launcher_joined"]
            assert api.events == [] and not api.live
        else:
            fact = next(f for f in result["io_starts"] if f["name"] == role)
            assert fact["attempted"] and fact["interrupted"] and not fact["entered"] and not fact["joined"]
            assert not d._active.streams[role.removeprefix("q1-io-")].closed
        # Repeated checks cannot transform absent ident into completion proof.
        assert not d._active.cleanup()["confirmed"]
    finally:
        # Only this injection knows that native creation never happened. Remove
        # its test-only limbo entries without granting production a false proof.
        with threading._active_limbo_lock:
            for owner in failed:
                threading._limbo.pop(owner, None)


def test_failure_before_any_start_attempt_can_confirm_no_launcher(tmp_path, monkeypatch):
    error = Cancellation("thread construction failed")
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: (_ for _ in ()).throw(error))
    api = FakeAPI()
    d = domain(tmp_path, platform="nt", api=api)
    with pytest.raises(Cancellation) as raised:
        d.run(["synthetic"])
    assert raised.value is error
    assert d.records[0]["cleanup"]["confirmed"]
    assert not d.records[0]["cleanup"]["launcher_start_attempted"]
    assert not d.records[0]["cleanup"]["launcher_entered"]
    assert not api.events and d._active is None


@pytest.mark.parametrize("role", ["q1-launcher", "q1-io-stdout", "q1-io-stderr", "q1-io-stdin"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_native_bootstrap_pending_is_bounded_and_owned(tmp_path, monkeypatch, gated_native_start, role, interrupt):
    monkeypatch.setattr(q1, "TEARDOWN_SECONDS", .04)
    release, native_entered, finished, original, observed = gated_native_start(role, interrupt)
    api = FakeAPI()
    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + .05, platform="nt", api=api)
    output = {}

    def run():
        try:
            d.run(["synthetic"], input=b"finite input")
        except BaseException as error:
            output["error"] = error

    caller = threading.Thread(target=run)
    caller.start()
    try:
        assert native_entered.wait(1)
        caller.join(.3)
        assert not caller.is_alive(), "bootstrap wait exceeded the caller's deadline"
        assert not finished.is_set()
        assert observed["owner"].ident is None
        lease = d._active
        assert lease is not None and d._quarantined
        result = d.records[0]["cleanup"]
        deadline = result["deadline"]
        assert not result["confirmed"]
        if interrupt:
            assert output["error"] is original
        else:
            assert isinstance(output["error"], (q1.OverallDeadlineExceeded, q1.EvidenceIncomplete))
        if role == "q1-launcher":
            assert result["launcher_start_attempted"] and not result["launcher_entered"]
            assert not result["launcher_publication_complete"] and not result["launcher_joined"]
            assert not api.events and not api.live
        else:
            fact = next(f for f in result["io_starts"] if f["name"] == role)
            assert fact["attempted"] and not fact["entered"] and not fact["complete"] and not fact["joined"]
            assert not result["io_joined"] and not result["pipes_closed"]
            assert not lease.streams[role.removeprefix("q1-io-")].closed
        assert not lease.cleanup()["confirmed"]
        assert lease.cleanup_deadline == deadline
    finally:
        release.set()
        caller.join(2)
    assert finished.wait(2)
    observed["owner"].join(2)
    d.close()
    final = d.records[0]["cleanup"]
    assert final["confirmed"] and final["launcher_joined"]
    assert final["deadline"] == deadline
    assert d._active is None
    assert api.events.count("terminate_job") + api.events.count("terminate_process") <= 1


def test_single_cleanup_winner_and_repeated_deadline(tmp_path):
    api = FakeAPI()
    d = domain(tmp_path, platform="nt", api=api)
    lease = q1._Lease(d, {})
    lease.launch(["synthetic"], str(tmp_path), {}, None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(lease.cleanup())) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert len(results) == 3
    assert all(result == results[0] for result in results)
    assert api.events.count("terminate_job") == 1
    assert lease.cleanup()["deadline"] == results[0]["deadline"]


def test_all_cleanup_stages_share_absolute_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(q1, "TEARDOWN_SECONDS", .06)
    api = FakeAPI()
    timeouts = []
    original_wait = api.wait
    def wait(process, timeout):
        timeouts.append(timeout)
        return original_wait(process, timeout)
    api.wait = wait
    def active(job):
        time.sleep(.04)
        return 0
    api.active_processes = active
    d = domain(tmp_path, platform="nt", api=api)
    lease = q1._Lease(d, {})
    lease.launch(["synthetic"], str(tmp_path), {}, None)
    result = lease.cleanup()
    assert result["confirmed"]
    assert 0 <= timeouts[0] < .04


@pytest.mark.parametrize("kwargs", [{"shell": True}, {"check": True}, {"stdin": subprocess.PIPE},
                                    {"stdout": subprocess.PIPE}, {"creationflags": 0}])
def test_unsupported_arguments_rejected_before_launch(tmp_path, kwargs):
    api = FakeAPI()
    with pytest.raises(TypeError):
        domain(tmp_path, platform="nt", api=api).run(["synthetic"], **kwargs)
    assert not api.events


def test_input_cap_before_launch(tmp_path):
    api = FakeAPI()
    with pytest.raises(q1.EvidenceIncomplete):
        domain(tmp_path, platform="nt", api=api).run(["synthetic"], input=b"x" * (q1.MAX_STDIN_BYTES + 1))
    assert not api.events

def test_blocked_native_create_does_not_block_caller_past_deadline(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockAPI(FakeAPI):
        def create_into(self, owner, argv, cwd, env):
            entered.set()
            assert release.wait(2)
            return super().create_into(owner, argv, cwd, env)

    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 0.05, platform="nt", api=BlockAPI())
    out = {}

    def run():
        try:
            d.run(["synthetic"], capture_output=True)
        except BaseException as e:
            out["exception"] = type(e).__name__

    t = threading.Thread(target=run)
    t.start()
    assert entered.wait(1)
    t.join(0.15)
    still_blocked = t.is_alive()
    release.set()
    t.join(2)
    assert not still_blocked, "caller must not be blocked past overall deadline"
    assert out.get("exception") == "OverallDeadlineExceeded"


def test_late_publish_failed_terminate_retains_quarantined_ownership_and_errors(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class LateAPI(FakeAPI):
        def create_into(self, owner, *args):
            self.owner = owner
            entered.set()
            assert release.wait(2)
            super().create_into(owner, *args)

        def terminate_process(self, process):
            self.events.append("terminate_failed")
            raise OSError("injected terminate failure")

    api = LateAPI()
    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 0.05, platform="nt", api=api)
    with pytest.raises(q1.OverallDeadlineExceeded):
        d.run(["synthetic"])

    assert d._quarantined is True
    assert d._active is not None
    assert d.records[0]["cleanup"]["launcher_joined"] is False
    assert "launcher_pending" in d.records[0]["cleanup"]["errors"]

    release.set()
    api.owner.launch_ready.wait(2)

    assert "terminate_failed" in api.events
    assert "process" in api.owner.handles
    assert d._active is not None
    assert "terminate_process_OSError" in d.records[0]["cleanup"]["errors"]
    assert d.records[0]["cleanup"]["launcher_joined"] is False
    assert d.records[0]["cleanup"]["process_waited"] is True
    api.owner.launcher.join(2)
    assert api.owner.cleanup()["launcher_joined"] is True


@pytest.mark.parametrize("boundary", ["create", "assign", "stdout_stream"])
@pytest.mark.parametrize("failure", [None, "terminate", "close"])
def test_pending_launch_resources_untouched_until_single_cleanup(tmp_path, boundary, failure):
    entered, release = threading.Event(), threading.Event()

    class BlockAPI(FakeAPI):
        def block(self, owner):
            self.owner = owner
            owner.streams["probe"] = io.BytesIO(b"still owned")
            owner.descriptors["probe"] = os.open(os.devnull, os.O_RDONLY)
            self.before_handles = dict(owner.handles)
            self.before_streams = dict(owner.streams)
            self.before_descriptors = dict(owner.descriptors)
            entered.set()
            assert release.wait(5), "test failed to release launcher"
            self.untouched = (
                owner.handles == self.before_handles
                and all(h in self.live for h in self.before_handles.values())
                and owner.streams == self.before_streams
                and all(not s.closed for s in self.before_streams.values())
                and owner.descriptors == self.before_descriptors
            )
            for fd in self.before_descriptors.values():
                os.fstat(fd)

        def create_into(self, owner, *args):
            if boundary == "create":
                self.block(owner)
            super().create_into(owner, *args)

        def assign(self, job, process):
            if boundary == "assign":
                self.block(lease)
            super().assign(job, process)

        def stream_into(self, owner, name, key, **kwargs):
            if boundary == "stdout_stream" and name == "stdout":
                self.block(owner)
            super().stream_into(owner, name, key, **kwargs)

        def terminate_process(self, process):
            super().terminate_process(process)
            if failure == "terminate":
                raise OSError("termination interrupted")

        def terminate_job(self, job):
            super().terminate_job(job)
            if failure == "terminate":
                raise OSError("termination interrupted")

        def close(self, handle):
            if failure == "close" and handle == getattr(self, "before_handles", {}).get("job"):
                raise OSError("close interrupted")
            super().close(handle)

    api = BlockAPI()
    d = domain(tmp_path, platform="nt", api=api)
    lease = q1._Lease(d, {})
    d._active = lease
    lease.start_launch(["synthetic"], str(tmp_path), {}, None)
    try:
        assert entered.wait(2)
        d.deadline = time.monotonic()
        pending = lease.cleanup()
        deadline = pending["deadline"]
        assert pending["errors"] == ["launcher_pending"]
        assert not lease.done.is_set()
        assert lease.handles == api.before_handles
        assert all(h in api.live for h in api.before_handles.values())
        assert lease.streams == api.before_streams
        assert all(not stream.closed for stream in api.before_streams.values())
        for fd in api.before_descriptors.values():
            os.fstat(fd)
        assert lease.cleanup()["deadline"] == deadline
        assert "terminate_process" not in api.events and "terminate_job" not in api.events
    finally:
        release.set()
        lease.launcher.join(3)
    assert not lease.launcher.is_alive()
    assert api.untouched
    assert not lease.record["cleanup"]["launcher_joined"]
    final = lease.cleanup()  # The owned external path must attest the actual join.
    assert final is lease.cleanup_result
    assert final["launcher_joined"]
    assert final["deadline"] == deadline
    assert "launcher_pending" not in final["errors"]
    assert api.events.count("terminate_process") + api.events.count("terminate_job") == 1
    assert lease.cleanup() is final
    if failure:
        assert not final["confirmed"]
        assert lease.handles  # Unknown ownership is never discarded.
        assert any(failure in error for error in final["errors"])
    else:
        assert final["confirmed"]
        assert not api.live and not lease.handles and not lease.streams and not lease.descriptors
    for stream in api.before_streams.values():
        assert stream.closed
    for fd in api.before_descriptors.values():
        with pytest.raises(OSError):
            os.fstat(fd)

@pytest.mark.parametrize("deferred", [False, True])
def test_actual_launcher_exit_required_after_publication(tmp_path, monkeypatch, deferred):
    """Target completion is not native thread exit, including self-cleanup."""
    monkeypatch.setattr(q1, "TEARDOWN_SECONDS", .05)
    target_returned, release_exit = threading.Event(), threading.Event()
    create_entered, release_create = threading.Event(), threading.Event()
    original_worker = q1._Lease._launch_worker
    actual_join = threading.Thread.join
    join_attempts = []

    def delayed_exit(self, *args):
        original_worker(self, *args)
        target_returned.set()
        assert release_exit.wait(5), "test did not release actual launcher exit"

    def observe_join(thread, timeout=None):
        if thread.name == "q1-launcher":
            assert thread is not threading.current_thread(), "deferred cleanup attempted self-join"
            join_attempts.append(timeout)
        return actual_join(thread, timeout)

    class BlockCreate(FakeAPI):
        def create_into(self, owner, *args):
            create_entered.set()
            assert release_create.wait(5)
            super().create_into(owner, *args)

    monkeypatch.setattr(q1._Lease, "_launch_worker", delayed_exit)
    monkeypatch.setattr(threading.Thread, "join", observe_join)
    api = BlockCreate() if deferred else FakeAPI()
    d = domain(tmp_path, platform="nt", api=api)
    started = time.monotonic()
    try:
        expected = subprocess.TimeoutExpired if deferred else q1.EvidenceIncomplete
        with pytest.raises(expected):
            d.run(["synthetic"], timeout=.05 if deferred else None)
        assert time.monotonic() - started < 1
        lease = d._active
        assert lease is not None and d._quarantined
        if deferred:
            assert create_entered.is_set()
            assert d.records[0]["cleanup"]["errors"] == ["launcher_pending"]
            release_create.set()
        assert target_returned.wait(2)
        assert lease.launch_complete.is_set() and lease.launch_ready.is_set()
        assert lease.launcher.is_alive()
        result = d.records[0]["cleanup"]
        deadline = result["deadline"]
        assert not result["launcher_joined"] and not result["confirmed"]
        assert "launcher_join_pending" in result["errors"]
        assert "launcher_join_RuntimeError" not in result["errors"]
        assert d._active is lease
        second = lease.cleanup()
        assert not second["confirmed"] and not second["launcher_joined"]
        assert second["deadline"] == deadline
        assert lease.launcher.is_alive()
        assert all(timeout is not None and 0 <= timeout <= .05 for timeout in join_attempts)
    finally:
        release_create.set()
        release_exit.set()
        if d._active is not None:
            actual_join(d._active.launcher, 2)
    assert not lease.launcher.is_alive()
    d.close()  # Independent owned join confirmation, without a new cleanup budget.
    final = d.records[0]["cleanup"]
    assert final["deadline"] == deadline
    assert final["launcher_joined"] and final["confirmed"]
    assert final["errors"] == []
    assert d._active is None
    assert api.events.count("terminate_process") + api.events.count("terminate_job") == 1


def test_caller_waits_for_actual_launcher_exit_within_existing_budget(tmp_path, monkeypatch):
    target_returned, release_exit, join_entered = (threading.Event() for _ in range(3))
    original_worker = q1._Lease._launch_worker
    actual_join = threading.Thread.join

    def delayed_exit(self, *args):
        original_worker(self, *args)
        target_returned.set()
        assert release_exit.wait(5)

    def observe_join(thread, timeout=None):
        if thread.name == "q1-launcher":
            assert thread is not threading.current_thread()
            assert timeout is not None and 0 <= timeout <= q1.TEARDOWN_SECONDS
            join_entered.set()
        return actual_join(thread, timeout)

    monkeypatch.setattr(q1._Lease, "_launch_worker", delayed_exit)
    monkeypatch.setattr(threading.Thread, "join", observe_join)
    d = domain(tmp_path, platform="nt", api=FakeAPI())
    observed = {}

    def run():
        try:
            observed["result"] = d.run(["synthetic"])
        except BaseException as error:
            observed["error"] = error

    caller = threading.Thread(target=run)
    caller.start()
    try:
        assert target_returned.wait(2) and join_entered.wait(2)
        assert caller.is_alive() and not observed
        lease = d._active
        assert lease is not None and lease.launcher.is_alive()
        assert not d.records[0]["cleanup"]["confirmed"]
    finally:
        release_exit.set()
        actual_join(caller, 2)
    assert not caller.is_alive()
    assert "error" not in observed
    assert observed["result"].returncode == 0
    assert not lease.launcher.is_alive()
    assert d.records[0]["cleanup"]["launcher_joined"]
    assert d.records[0]["cleanup"]["confirmed"]
    assert d._active is None
def test_early_event_wait_false_does_not_cancel_or_fail_before_deadline(tmp_path, monkeypatch):
    """Early or spurious False from Event.wait must not cancel or abort launch before deadline."""
    api = FakeAPI()
    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 1.0, platform="nt", api=api)

    original_wait = threading.Event.wait
    wait_calls = []

    def spurious_first_wait(self, timeout=None):
        wait_calls.append(timeout)
        if len(wait_calls) == 1:
            return False  # Spurious wake before deadline
        return original_wait(self, timeout)

    monkeypatch.setattr(threading.Event, "wait", spurious_first_wait)
    res = d.run(["synthetic"], capture_output=True)
    assert res.stdout == b"stdout"
    assert len(wait_calls) >= 2
    assert d._active is None


def test_early_wait_ready_false_without_timeout_raises_overall_deadline_not_timeout_expired_none(tmp_path, monkeypatch):
    """When timeout=None, deadline expiration must raise OverallDeadlineExceeded, never TimeoutExpired(argv, None)."""
    api = FakeAPI()
    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 0.05, platform="nt", api=api)

    def hung_launch(owner, argv, cwd, env):
        api.handle(owner, "process")
        api.handle(owner, "thread")
        owner.pid = 321
        time.sleep(0.1)  # Sleep past domain deadline without signaling ready

    monkeypatch.setattr(api, "create_into", hung_launch)
    with pytest.raises(q1.OverallDeadlineExceeded):
        d.run(["synthetic"], timeout=None)
    record = d.records[0]
    assert record["overall_expired"] is True
    assert record["timed_out"] is False


def test_individual_timeout_shorter_than_overall_deadline_raises_timeout_expired_with_timeout_value(tmp_path, monkeypatch):
    api = FakeAPI()
    d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 10.0, platform="nt", api=api)

    def slow_launch(owner, argv, cwd, env):
        api.handle(owner, "process")
        api.handle(owner, "thread")
        owner.pid = 321
        time.sleep(0.08)  # Exceeds individual timeout (0.03), well within overall deadline (10.0)

    monkeypatch.setattr(api, "create_into", slow_launch)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        d.run(["synthetic"], timeout=0.03)
    assert raised.value.timeout == 0.03
    record = d.records[0]
    assert record["timed_out"] is True
    assert record["overall_expired"] is False



@pytest.mark.parametrize("original_error", [RuntimeError("observer error"), Cancellation("first cancellation")])
def test_cleanup_second_cancellation_preserves_original_and_unknown_owner(tmp_path, monkeypatch, original_error):
    d = domain(tmp_path, platform="nt", api=FakeAPI(),
               observer=lambda _: (_ for _ in ()).throw(original_error))
    actual_cleanup = q1._Lease.cleanup

    def interrupted_cleanup(self):
        raise KeyboardInterrupt("second cancellation")

    with monkeypatch.context() as patch:
        patch.setattr(q1._Lease, "cleanup", interrupted_cleanup)
        with pytest.raises(type(original_error)) as raised:
            d.run(["synthetic"])
    assert raised.value is original_error
    assert d._quarantined
    assert d._active is not None
    assert d._active.handles
    assert d.records[0]["cleanup"] == {"confirmed": False, "errors": ["cleanup_KeyboardInterrupt"]}
    assert actual_cleanup(d._active)["confirmed"]
def test_descendant_diagnostic_adapter_records_bounded_creation_facts(tmp_path):
    """Portable check of the bounded diagnostic recording companion with append-time bounding."""
    api = FakeAPI()
    observed_facts = []
    MAX_BYTES = 8192
    bytes_used = [2]
    cap_exceeded = [False]

    # Zero dependency on psutil on Linux: use injected portable lookup function
    def inspect_process(pid):
        if pid == 321:
            return {"create_time": 1000.0, "status": "owned_active"}
        return None  # non-existent / inaccessible

    def diagnostic_observer(members):
        for m in members:
            pid = m.get("pid")
            info = inspect_process(pid)
            if info is not None:
                create_time = info["create_time"]
                status = info["status"]
            else:
                create_time = None
                status = "missing_UNKNOWN"

            # Do NOT fabricate job membership on portable test:
            is_in_job = True if ("assign" in api.events and pid == 321) else "UNKNOWN"

            entry = {
                "pid": pid,
                "create_time": create_time,
                "is_in_job": is_in_job,
                "status": status,
                "observed_at": time.monotonic(),
            }
            candidate_payload = {
                "facts": observed_facts + [entry],
                "cap_exceeded": cap_exceeded[0],
            }
            raw_size = len(json.dumps(candidate_payload, ensure_ascii=True).encode("utf-8"))
            if raw_size <= MAX_BYTES:
                observed_facts.append(entry)
                bytes_used[0] = raw_size
            else:
                cap_exceeded[0] = True
        return {"members": members}

    out_dir = Path(os.environ.get("NEXUS_Q1_OUTPUT_DIR", str(tmp_path)))
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "descendant_diagnostic_facts.json"
    try:
        d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + 5.0, platform="nt", api=api, observer=diagnostic_observer)
        result = d.run(["synthetic"], capture_output=True)
        assert result.stdout == b"stdout"
        assert len(observed_facts) >= 1
        assert observed_facts[0]["pid"] == 321
        assert observed_facts[0]["status"] in ("owned_active", "missing_UNKNOWN")

        # Exercise the REAL adapter logic with an actual non-existent / inaccessible PID
        diagnostic_observer([{"pid": 99999999}])
        assert len(observed_facts) >= 2
        missing_entry = observed_facts[-1]
        assert missing_entry["pid"] == 99999999
        assert missing_entry["status"] == "missing_UNKNOWN"
        assert missing_entry["create_time"] is None
        assert missing_entry["is_in_job"] == "UNKNOWN"

        # Exercise append-time bounding: flood with members and verify bound is enforced AT APPEND TIME
        flood_members = [{"pid": 10000000 + i} for i in range(500)]
        diagnostic_observer(flood_members)
        assert cap_exceeded[0] is True
        assert bytes_used[0] <= MAX_BYTES
        assert len(json.dumps({"facts": observed_facts, "cap_exceeded": True}).encode("utf-8")) <= MAX_BYTES
    finally:
        payload = json.dumps({
            "facts": observed_facts,
            "cap_exceeded": cap_exceeded[0],
            "bytes_used": bytes_used[0],
        }, ensure_ascii=True)
        assert len(payload.encode("utf-8")) <= MAX_BYTES
        artifact_path.write_text(payload, encoding="utf-8")
        assert artifact_path.is_file()

@pytest.mark.skipif(os.name != "posix", reason="portable POSIX transport test, not native Windows proof")
class TestPortableTransport:
    def test_exact_binary_input_eof_and_separate_capture(self, tmp_path):
        payload = b"\x00a\r\n\xff" * 300
        d = domain(tmp_path)
        result = d.run(python("import sys; b=sys.stdin.buffer.read();sys.stdout.buffer.write(b);sys.stderr.buffer.write(b[::-1])"),
                       input=payload, capture_output=True)
        assert result.stdout == payload
        assert result.stderr == payload[::-1]
        assert d.records[0]["input_sha256"] == hashlib.sha256(payload).hexdigest()
        assert d.records[0]["capture_complete"]
        for name, data in (("stdout", payload), ("stderr", payload[::-1])):
            record = d.records[0][name]
            assert Path(record["path"]).read_bytes() == data
            assert record["sha256"] == hashlib.sha256(data).hexdigest()

    def test_text_unicode_and_universal_newlines(self, tmp_path):
        result = domain(tmp_path).run(python("import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"),
                                     input="雪\r\nline\r", text=True, encoding="utf-8", capture_output=True)
        assert result.stdout == "雪\nline\n"

    def test_empty_input_and_devnull_eof(self, tmp_path):
        d = domain(tmp_path)
        for payload in (None, b""):
            result = d.run(python("import sys;print(len(sys.stdin.buffer.read()))"),
                           input=payload, capture_output=True)
            assert result.stdout == b"0\n"

    def test_concurrent_drains_beyond_pipe_capacity(self, tmp_path):
        source = "import sys,threading; t=threading.Thread(target=lambda:sys.stderr.buffer.write(b'e'*300000));t.start();sys.stdout.buffer.write(b'o'*300000);t.join()"
        result = domain(tmp_path).run(python(source), capture_output=True)
        assert result.stdout == b"o" * 300000
        assert result.stderr == b"e" * 300000

    def test_finite_timeout_is_real_timeout_and_can_continue(self, tmp_path):
        d = domain(tmp_path)
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            d.run(python("import time;time.sleep(30)"), timeout=.1)
        assert raised.value.timeout == .1
        assert d.records[0]["timed_out"]
        assert not d.records[0]["overall_expired"]
        assert d.records[0]["cleanup"]["confirmed"]
        assert d.run(python("pass")).returncode == 0

    def test_none_timeout_preserved_but_overall_deadline_applies(self, tmp_path):
        d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + .1)
        with pytest.raises(q1.OverallDeadlineExceeded):
            d.run(python("import time;time.sleep(30)"), timeout=None)
        assert d.records[0]["requested_timeout"] is None
        assert d.records[0]["overall_expired"]
        assert not d.records[0]["timed_out"]
        with pytest.raises(q1.OverallDeadlineExceeded):
            d.run(python("pass"))
        assert len(d.records) == 1

    def test_stream_cap_aborts_and_drains_without_truncated_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(q1, "STREAM_MAX_BYTES", 1024)
        d = domain(tmp_path)
        with pytest.raises(q1.EvidenceIncomplete):
            d.run(python("import os;\nwhile True: os.write(1,b'x'*4096)"), capture_output=True)
        assert not d.records[0]["capture_complete"]
        assert d.records[0]["cleanup"]["confirmed"]
        assert Path(d.records[0]["stdout"]["path"]).stat().st_size <= 1024

    def test_row_cap_combines_streams(self, tmp_path, monkeypatch):
        monkeypatch.setattr(q1, "ROW_MAX_BYTES", 1000)
        d = domain(tmp_path)
        with pytest.raises(q1.EvidenceIncomplete):
            d.run(python("import os;os.write(1,b'x'*600);os.write(2,b'y'*600)"))
        assert not d.records[0]["capture_complete"]

    def test_aggregate_cap_spans_commands_with_reserved_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(q1, "RUN_MAX_BYTES", q1.STATUS_MAX_BYTES + 1000)
        d = domain(tmp_path)
        d.run(python("import os;os.write(1,b'x'*600)"))
        with pytest.raises(q1.EvidenceIncomplete):
            d.run(python("import os;os.write(2,b'y'*600)"))
        assert d.total_bytes == 1200
        assert d.records[0]["capture_complete"]
        assert not d.records[1]["capture_complete"]

    def test_capture_open_failure_still_cleans_owned_child(self, tmp_path, monkeypatch):
        original = Path.open
        def broken(path, mode="r", *args, **kwargs):
            if mode == "xb":
                raise OSError("capture disk unavailable")
            return original(path, mode, *args, **kwargs)
        monkeypatch.setattr(Path, "open", broken)
        d = domain(tmp_path)
        with pytest.raises(q1.EvidenceIncomplete):
            d.run(python("import os;\nwhile True:os.write(1,b'x'*65536)"))
        assert d.records[0]["cleanup"]["confirmed"]
        assert any("capture_open" in value for value in d.records[0]["transport_errors"])

    def test_cancel_preserves_exception_identity(self, tmp_path):
        error = Cancellation("observer cancellation")
        def observer(members):
            assert members
            raise error
        d = domain(tmp_path, observer=observer)
        with pytest.raises(Cancellation) as raised:
            d.run(python("import time;time.sleep(30)"))
        assert raised.value is error
        assert d.records[0]["cleanup"]["confirmed"]

    def test_no_capture_returns_none_but_keeps_bounded_artifacts(self, tmp_path):
        d = domain(tmp_path)
        result = d.run(python("print('raw')"))
        assert result.stdout is result.stderr is None
        assert Path(d.records[0]["stdout"]["path"]).read_bytes() == b"raw\n"


@pytest.mark.skipif(os.name != "nt", reason="requires actual Windows job objects; POSIX is not native proof")
class TestNativeWindows:
    @staticmethod
    def tree_source(exit_code=0, delay=.3):
        # The grandchild inherits stdout/stderr and outlives its original CLI.
        grandchild = "import time;time.sleep(60)"
        child = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-I','-c',"
                 + repr(grandchild) + "]);print('GRANDCHILD',p.pid,flush=True);time.sleep(60)")
        return ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-I','-c',"
                + repr(child) + "]);print('CHILD',p.pid,flush=True);time.sleep("
                + repr(delay) + ");sys.exit(" + repr(exit_code) + ")")

    @staticmethod
    def assert_members_dead(record):
        import psutil
        pids = []
        for line in Path(record["stdout"]["path"]).read_text().splitlines():
            if line.startswith(("CHILD ", "GRANDCHILD ")):
                pids.append(int(line.split()[1]))
        assert len(pids) == 2, "both actual descendants must have launched"
        assert all(not psutil.pid_exists(pid) for pid in pids)
        assert record["cleanup"]["confirmed"]
        assert record["cleanup"]["job_empty"]
        assert record["cleanup"]["io_joined"]

    def test_normal_descendants(self, tmp_path):
        d = domain(tmp_path)
        assert d.run(python(self.tree_source()), capture_output=True).returncode == 0
        self.assert_members_dead(d.records[0])

    def test_abnormal_descendants(self, tmp_path):
        d = domain(tmp_path)
        assert d.run(python(self.tree_source(7)), capture_output=True).returncode == 7
        self.assert_members_dead(d.records[0])

    def test_timeout_descendants(self, tmp_path):
        d = domain(tmp_path)
        with pytest.raises(subprocess.TimeoutExpired):
            d.run(python(self.tree_source(delay=60)), timeout=.8)
        self.assert_members_dead(d.records[0])

    def test_cancel_descendants(self, tmp_path):
        error = Cancellation("native cancellation")
        started = time.monotonic()
        def observer(members):
            if time.monotonic() - started > .8:
                assert len(members) >= 3
                raise error
        d = domain(tmp_path, observer=observer)
        with pytest.raises(Cancellation) as raised:
            d.run(python(self.tree_source(delay=60)))
        assert raised.value is error
        self.assert_members_dead(d.records[0])

    def test_piped_input_and_eof(self, tmp_path):
        payload = "雪\n\"quotes\"\x00end\n"
        d = domain(tmp_path)
        result = d.run(python("import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"),
                       input=payload, capture_output=True, text=True, encoding="utf-8")
        assert result.stdout == payload
        assert d.records[0]["input_bytes"] == len(payload.encode())
        assert d.records[0]["capture_complete"]

    def test_inherited_pipes_and_sentinel(self, tmp_path):
        sentinel = subprocess.Popen(python("import time;time.sleep(60)"),
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        try:
            d = domain(tmp_path)
            assert d.run(python(self.tree_source()), capture_output=True).returncode == 0
            self.assert_members_dead(d.records[0])
            assert sentinel.poll() is None, "unrelated process must survive owned job cleanup"
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)

    def test_blocked_stdin_deadline(self, tmp_path):
        d = q1.ProcessDomain(tmp_path, deadline=time.monotonic() + .8)
        started = time.monotonic()
        with pytest.raises(q1.OverallDeadlineExceeded):
            d.run(python("import time;time.sleep(60)"), input=b"x" * q1.MAX_STDIN_BYTES)
        assert time.monotonic() - started < 11
        assert d.records[0]["overall_expired"]
        assert d.records[0]["cleanup"]["confirmed"]
        assert d.records[0]["cleanup"]["io_joined"]

    def test_owned_member_observer(self, tmp_path):
        observed = set()
        def observer(members):
            assert all(set(fact) == {"pid"} for fact in members)
            observed.update(fact["pid"] for fact in members)
        d = domain(tmp_path, observer=observer)
        d.run(python(self.tree_source()), capture_output=True)
        self.assert_members_dead(d.records[0])
        descendant_pids = {int(line.split()[1]) for line in
                           Path(d.records[0]["stdout"]["path"]).read_text().splitlines()}
        assert descendant_pids <= observed
        assert len(observed) >= 3

    def test_owned_member_observer_diagnostic_companion(self, tmp_path):
        """Diagnostic companion recording creation timestamps, typed handle states, and exit facts alongside mandatory observer."""
        import psutil
        observed_diagnostics = []
        MAX_BYTES = 8192
        bytes_used = [2]
        cap_exceeded = [False]

        def observer(members):
            for fact in members:
                pid = fact.get("pid")
                create_time = None
                is_in_job = None
                handle_opened = False
                exit_code = None
                close_time = None
                status = "missing_UNKNOWN"
                try:
                    p = psutil.Process(pid)
                    create_time = p.create_time()
                    status = "owned_active" if p.is_running() else "terminated"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    status = "missing_UNKNOWN"

                # Query typed handle states on native Windows directly via PID without private domain APIs
                if hasattr(psutil, "WINDOWS") and psutil.WINDOWS:
                    import ctypes
                    from ctypes import wintypes
                    kernel32 = ctypes.windll.kernel32
                    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                    kernel32.OpenProcess.restype = wintypes.HANDLE
                    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                    kernel32.CloseHandle.restype = wintypes.BOOL
                    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                    kernel32.WaitForSingleObject.restype = wintypes.DWORD
                    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                    kernel32.GetExitCodeProcess.restype = wintypes.BOOL

                    # Process membership in job is grounded by QueryInformationJobObject via members parameter
                    is_in_job = True
                    h = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
                    if h:
                        handle_opened = True
                        try:
                            if kernel32.WaitForSingleObject(h, 0) == 0:  # WAIT_OBJECT_0
                                ec = wintypes.DWORD()
                                if kernel32.GetExitCodeProcess(h, ctypes.byref(ec)):
                                    exit_code = int(ec.value)
                        finally:
                            kernel32.CloseHandle(h)
                            close_time = time.monotonic()
                    elif status == "owned_active":
                        status = "inaccessible_UNKNOWN"

                entry = {
                    "pid": pid,
                    "create_time": create_time,
                    "is_in_job": is_in_job if is_in_job is not None else "UNKNOWN",
                    "handle_opened": handle_opened,
                    "exit_code": exit_code,
                    "close_time": close_time,
                    "status": status,
                    "observed_at": time.monotonic(),
                }
                candidate_payload = {
                    "facts": observed_diagnostics + [entry],
                    "cap_exceeded": cap_exceeded[0],
                }
                raw_size = len(json.dumps(candidate_payload, ensure_ascii=True).encode("utf-8"))
                if raw_size <= MAX_BYTES:
                    observed_diagnostics.append(entry)
                    bytes_used[0] = raw_size
                else:
                    cap_exceeded[0] = True

        out_dir = Path(os.environ.get("NEXUS_Q1_OUTPUT_DIR", str(tmp_path)))
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / "descendant_diagnostic_facts.json"
        try:
            d = domain(tmp_path, observer=observer)
            d.run(python(self.tree_source()), capture_output=True)
            self.assert_members_dead(d.records[0])
            assert len(observed_diagnostics) >= 3
            assert all(item["status"] in ("owned_active", "terminated", "missing_UNKNOWN", "inaccessible_UNKNOWN") for item in observed_diagnostics)
            assert bytes_used[0] <= MAX_BYTES
            assert len(json.dumps({"facts": observed_diagnostics, "cap_exceeded": True}).encode("utf-8")) <= MAX_BYTES
        finally:
            payload = json.dumps({
                "facts": observed_diagnostics,
                "cap_exceeded": cap_exceeded[0],
                "bytes_used": bytes_used[0],
            }, ensure_ascii=True)
            assert len(payload.encode("utf-8")) <= MAX_BYTES
            artifact_path.write_text(payload, encoding="utf-8")
            assert artifact_path.is_file()
