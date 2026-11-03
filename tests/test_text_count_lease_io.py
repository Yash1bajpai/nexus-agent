"""Lease ownership at the real inspect_text / bounded fake-pipe boundary.

No subprocess is created here: Popen alone is replaced, while the launcher,
lease, inspect scope, protocol exchange, and cleanup primitives remain real.
"""
import json
import queue
import struct
import threading
import types

import pytest

from nexus_agent_ai.agent import text_count_worker as worker


_WAIT = 5


class _ReadPipe:
    def __init__(self):
        self.buffer = bytearray()
        self.closed = False
        self.close_calls = 0

    def feed(self, message):
        payload = json.dumps(message, separators=(",", ":")).encode("ascii")
        assert len(payload) <= worker.MAX_RESPONSE_BYTES
        assert not self.buffer
        self.buffer.extend(struct.pack("!I", len(payload)) + payload)

    def read(self, size):
        assert not self.closed
        assert size > 0
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def close(self):
        self.close_calls += 1
        self.closed = True


class _WritePipe:
    def __init__(self, proc):
        self.proc = proc
        self.closed = False
        self.close_calls = 0
        self.writes = []
        self.request = bytearray()
        self.requests = []
        self.acks = []

    def write(self, data):
        assert not self.closed
        data = bytes(data)
        self.writes.append(data)
        if data == b"\x06":
            self.proc.ack_entered.set()
            assert self.proc.ack_release.wait(_WAIT), "ACK barrier was not released"
            if self.proc.ack_outcome == "error":
                raise OSError("bounded fake ACK delivery failure")
            self.acks.append(data)
            return 1
        assert not self.requests, "a second request reached the fake child"
        self.request.extend(data)
        assert len(self.request) <= worker.MAX_REQUEST_BYTES + 4
        if len(self.request) >= 4:
            size = struct.unpack("!I", self.request[:4])[0]
            if len(self.request) == size + 4:
                message = json.loads(bytes(self.request[4:]))
                self.requests.append(message)
                self.proc.stdout.feed({"type": "started", **message})
        return len(data)

    def flush(self):
        assert not self.closed

    def close(self):
        self.close_calls += 1
        self.closed = True


class _FakeProcess:
    """Bounded in-memory streams; no pid, descriptor, or native child is owned."""
    pid = 424242

    def __init__(self):
        self.stdout = _ReadPipe()
        self.stdin = _WritePipe(self)
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.ack_entered = threading.Event()
        self.ack_release = threading.Event()
        self.io_release = threading.Event()
        self.ack_outcome = "sent"
        self.release_ack_on_terminate = True

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15
        self.io_release.set()
        if self.release_ack_on_terminate:
            self.ack_release.set()

    def kill(self):
        self.kill_calls += 1
        self.terminate()

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            raise worker.subprocess.TimeoutExpired("fake-child", timeout)
        return self.returncode


@pytest.fixture
def fake_launch(monkeypatch):
    previous = worker._QUARANTINE
    assert previous is None or not worker._quarantine_busy(previous)
    monkeypatch.setattr(worker, "_QUARANTINE", None)
    monkeypatch.setattr(worker, "_LAST_CLEANUP", None)
    proc = _FakeProcess()
    captured = types.SimpleNamespace(proc=proc, leases=[], cleanups=[], threads=[])

    def popen(*args, **kwargs):
        lease = worker._QUARANTINE
        assert isinstance(lease, worker._LaunchLease)
        assert lease.launch_start_attempted
        captured.leases.append(lease)
        assert len(captured.leases) == 1, "a rejected path launched another child"
        return proc

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    for name in ("_cleanup", "_cleanup_unstarted"):
        primitive = getattr(worker, name)

        def cleanup(*args, _name=name, _primitive=primitive, **kwargs):
            lease = captured.leases[0]
            assert worker._QUARANTINE is lease
            assert lease.proc is proc
            assert lease.cancelled
            captured.cleanups.append(_name)
            return _primitive(*args, **kwargs)

        monkeypatch.setattr(worker, name, cleanup)
    yield captured
    # Unknown cleanup is intentional in fault tests. Only fake resources may be
    # discarded, and every actual launcher/IO/observer thread must first finish.
    proc.io_release.set()
    proc.ack_release.set()
    for lease in captured.leases:
        for thread in (lease.launcher, lease.io_thread):
            if thread is not None and thread.ident is not None:
                thread.join(_WAIT)
                assert not thread.is_alive()
    for thread in captured.threads:
        thread.join(_WAIT)
        assert not thread.is_alive()
    assert type(proc) is _FakeProcess
    for stream in (proc.stdin, proc.stdout):
        if not stream.closed:
            stream.close()
    current = worker._QUARANTINE
    assert current is None or any(current is lease for lease in captured.leases)
    monkeypatch.setattr(worker, "_QUARANTINE", None)


def _assert_no_request_or_ack(captured):
    proc = captured.proc
    lease, = captured.leases
    assert proc.stdin.writes == []
    assert proc.stdin.requests == []
    assert proc.stdin.acks == []
    assert lease.state["request_attempted"] is False
    assert lease.state["ack_attempted"] is False
    assert lease.state["ack_sent"] is False
    assert lease.launch_joined
    assert not lease.launcher.is_alive()
    assert not worker._CALL_LOCK.locked()


def _assert_confirmed(captured):
    lease, = captured.leases
    facts = lease.snapshot_cleanup()
    assert facts["confirmed"] is True
    assert facts["reaped"] is True
    assert facts["io_joined"] is True
    assert facts["pipes_closed"] is True
    assert captured.proc.returncode is not None
    assert captured.proc.stdin.close_calls == 1
    assert captured.proc.stdout.close_calls == 1
    assert not worker._quarantine_busy(lease)


def _threading_proxy(**overrides):
    return types.SimpleNamespace(
        Lock=threading.Lock,
        Event=overrides.get("Event", threading.Event),
        Thread=overrides.get("Thread", threading.Thread),
        current_thread=threading.current_thread,
    )


@pytest.mark.parametrize("failure", [
    "queue", "authorize_event", "stop_event", "io_construct",
    "io_start_before_native", "io_start_after_native",
])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
def test_inspect_retains_adopted_owner_across_io_setup_failure(
    monkeypatch, fake_launch, failure, error_type,
):
    """Adapt the supplied Queue cancellation repro to the actual lease scope."""
    original = error_type("original post-adoption/pre-IO failure")
    proc = fake_launch.proc
    event_calls = 0
    target_entered = threading.Event()

    def event_factory(*args, **kwargs):
        nonlocal event_calls
        lease = worker._QUARANTINE
        if lease is not None and lease.adopted:
            event_calls += 1
            wanted = 1 if failure == "authorize_event" else 2
            if failure in ("authorize_event", "stop_event") and event_calls == wanted:
                raise original
        return threading.Event(*args, **kwargs)

    def queue_factory(*args, **kwargs):
        assert worker._QUARANTINE.adopted
        if failure == "queue":
            raise original
        return queue.Queue(*args, **kwargs)

    class FaultThread(threading.Thread):
        def start(self):
            if self.name != "nexus-text-count-io":
                return super().start()
            if failure == "io_start_before_native":
                raise original
            if failure == "io_start_after_native":
                super().start()
                assert target_entered.wait(_WAIT)
                raise original
            return super().start()

    def thread_factory(*args, **kwargs):
        if kwargs.get("name") == "nexus-text-count-io":
            assert worker._QUARANTINE.adopted
            if failure == "io_construct":
                raise original
            if failure == "io_start_after_native":
                target = kwargs["target"]

                def gated_target():
                    target_entered.set()
                    assert proc.io_release.wait(_WAIT)
                    target()

                kwargs["target"] = gated_target
        return FaultThread(*args, **kwargs)

    monkeypatch.setattr(worker, "queue", types.SimpleNamespace(
        Queue=queue_factory, Empty=queue.Empty, Full=queue.Full,
    ))
    monkeypatch.setattr(worker, "threading", _threading_proxy(
        Event=event_factory, Thread=thread_factory,
    ))
    with pytest.raises(error_type) as caught:
        worker.inspect_text("physical.txt", "a", timeout=2)
    assert caught.value is original
    lease, = fake_launch.leases
    assert lease.adopted
    assert lease.cancelled
    assert len(fake_launch.cleanups) == 1
    expected = "_cleanup" if failure.startswith("io_start_") else "_cleanup_unstarted"
    assert fake_launch.cleanups == [expected]
    _assert_no_request_or_ack(fake_launch)
    if failure == "io_start_before_native":
        assert lease.io_start_attempted
        assert lease.snapshot_cleanup()["confirmed"] is False
        assert worker._QUARANTINE is lease
        assert worker._quarantine_busy(lease)
        assert proc.stdin.close_calls == proc.stdout.close_calls == 0
        rejected = worker.inspect_text("must-not-launch.txt", "b", timeout=2)
        assert rejected["executed"] is False
        assert rejected["receipt"]["reason"] == "busy"
        assert len(fake_launch.leases) == 1
    else:
        _assert_confirmed(fake_launch)
    # Both with-exit and inspect's explicit finally have already run. A third
    # cancellation cannot acquire another cleanup claim or close pipes twice.
    lease.close(original=original)
    assert len(fake_launch.cleanups) == 1


def test_request_and_ack_permissions_reject_before_adoption(monkeypatch, fake_launch):
    adopt = worker._LaunchLease.adopt
    observations = []
    original = KeyboardInterrupt("stop after permission boundary was observed")

    def checked_adopt(lease):
        assert lease.proc is fake_launch.proc
        assert lease.launch_joined
        assert not lease.adopted
        observations.append((lease.begin_request(), lease.begin_ack()))
        assert lease.state["request_attempted"] is False
        assert lease.state["ack_attempted"] is False
        return adopt(lease)

    def cancel_queue(*args, **kwargs):
        raise original

    monkeypatch.setattr(worker._LaunchLease, "adopt", checked_adopt)
    monkeypatch.setattr(worker, "queue", types.SimpleNamespace(
        Queue=cancel_queue, Empty=queue.Empty, Full=queue.Full,
    ))
    with pytest.raises(KeyboardInterrupt) as caught:
        worker.inspect_text("physical.txt", "a", timeout=2)
    assert caught.value is original
    assert observations == [(False, False)]
    _assert_no_request_or_ack(fake_launch)
    _assert_confirmed(fake_launch)


def test_rejected_adoption_never_constructs_io_or_sends_request(monkeypatch, fake_launch):
    observations = []

    def reject(lease):
        assert lease.proc is fake_launch.proc
        assert lease.launch_joined
        observations.append((lease.begin_request(), lease.begin_ack()))
        return False

    def forbidden_queue(*args, **kwargs):
        pytest.fail("rejected adoption reached IO construction")

    monkeypatch.setattr(worker._LaunchLease, "adopt", reject)
    monkeypatch.setattr(worker, "queue", types.SimpleNamespace(
        Queue=forbidden_queue, Empty=queue.Empty, Full=queue.Full,
    ))
    result = worker.inspect_text("must-not-scan.txt", "a", timeout=2)
    assert observations == [(False, False)]
    assert result["executed"] is False
    assert result["receipt"]["status"] == "incomplete"
    assert result["receipt"]["count"] is None
    assert result["receipt"]["edit_proof"] is False
    lease, = fake_launch.leases
    assert not lease.adopted
    assert lease.io_thread is None
    assert not lease.io_start_attempted
    assert fake_launch.cleanups == ["_cleanup_unstarted"]
    _assert_no_request_or_ack(fake_launch)
    _assert_confirmed(fake_launch)


@pytest.mark.parametrize("cleanup_error_type", [KeyboardInterrupt, RuntimeError])
def test_post_adoption_cleanup_second_cancellation_preserves_original(
    monkeypatch, fake_launch, cleanup_error_type,
):
    original = KeyboardInterrupt("original Queue construction cancellation")
    secondary = cleanup_error_type("second cancellation during cleanup")
    cleanup = worker._cleanup_unstarted

    def cancel_queue(*args, **kwargs):
        assert worker._QUARANTINE.adopted
        raise original

    def interrupted_cleanup(*args, **kwargs):
        # Really reap/close the fake resources before interrupting delivery of
        # the cleanup facts. A lost return is not proof of confirmed cleanup.
        cleanup(*args, **kwargs)
        raise secondary

    monkeypatch.setattr(worker, "queue", types.SimpleNamespace(
        Queue=cancel_queue, Empty=queue.Empty, Full=queue.Full,
    ))
    monkeypatch.setattr(worker, "_cleanup_unstarted", interrupted_cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        worker.inspect_text("physical.txt", "a", timeout=2)
    assert caught.value is original
    lease, = fake_launch.leases
    assert lease.cleanup_error is secondary
    assert worker._QUARANTINE is lease
    assert lease.snapshot_cleanup()["confirmed"] is False
    assert worker._quarantine_busy(lease)
    lease.close(original=original)
    assert fake_launch.cleanups == ["_cleanup_unstarted"]
    assert fake_launch.proc.stdin.close_calls == 1
    assert fake_launch.proc.stdout.close_calls == 1
    _assert_no_request_or_ack(fake_launch)
    rejected = worker.inspect_text("must-not-launch.txt", "b", timeout=2)
    assert rejected["executed"] is False
    assert rejected["receipt"]["reason"] == "busy"
    assert len(fake_launch.leases) == 1


@pytest.mark.parametrize("ack_outcome", ["sent", "error", "blocked"])
@pytest.mark.parametrize("cancel_original", [False, True])
def test_ack_race_retains_attempt_sent_or_unknown_without_holding_owner_lock(
    monkeypatch, fake_launch, ack_outcome, cancel_original,
):
    proc = fake_launch.proc
    proc.ack_outcome = ack_outcome
    proc.release_ack_on_terminate = ack_outcome != "blocked"
    original = KeyboardInterrupt("original parent cancellation after ACK began")
    observations = []
    queue_gets = 0

    class BoundaryQueue(queue.Queue):
        def get(self, *args, **kwargs):
            nonlocal queue_gets
            queue_gets += 1
            if queue_gets == 1:
                return super().get(*args, **kwargs)
            assert proc.ack_entered.wait(_WAIT)
            lease = fake_launch.leases[0]
            assert lease.state["request_attempted"] is True
            assert lease.state["ack_attempted"] is True
            assert lease.state["ack_sent"] is False
            lock_available = threading.Event()

            def observe_owner():
                # Competing adoption must reject promptly even while the real
                # exchange target is inside a blocking stdin.write call.
                observations.append(lease.adopt())
                lock_available.set()

            observer = threading.Thread(target=observe_owner, daemon=True)
            fake_launch.threads.append(observer)
            observer.start()
            assert lock_available.wait(_WAIT), "ownership lock held across pipe write"
            observer.join(_WAIT)
            if cancel_original:
                raise original
            raise queue.Empty

    monkeypatch.setattr(worker, "queue", types.SimpleNamespace(
        Queue=BoundaryQueue, Empty=queue.Empty, Full=queue.Full,
    ))
    if cancel_original:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker.inspect_text("physical.txt", "a", timeout=2)
        assert caught.value is original
    elif ack_outcome == "sent":
        result = worker.inspect_text("physical.txt", "a", timeout=2)
        assert result["executed"] is True
        assert result["receipt"]["status"] == "incomplete"
        assert result["receipt"]["count"] is None
    else:
        with pytest.raises(worker.CountExecutionUncertain) as caught:
            worker.inspect_text("physical.txt", "a", timeout=2)
        assert caught.value.receipt["status"] == "incomplete"
        assert caught.value.receipt["count"] is None
    lease, = fake_launch.leases
    assert observations == [False]
    assert lease.adopted and lease.cancelled
    assert lease.state["request_attempted"] is True
    assert lease.state["ack_attempted"] is True
    assert lease.state["ack_sent"] is (ack_outcome == "sent")
    assert len(proc.stdin.requests) == 1
    assert proc.stdin.acks == ([b"\x06"] if ack_outcome == "sent" else [])
    assert fake_launch.cleanups == ["_cleanup"]
    assert lease.launch_joined and not lease.launcher.is_alive()
    assert not worker._CALL_LOCK.locked()
    if ack_outcome == "blocked":
        assert lease.io_thread.is_alive()
        assert lease.state["pipes_closed"] is not True
        assert lease.snapshot_cleanup()["confirmed"] is False
        assert worker._QUARANTINE is lease
        assert worker._quarantine_busy(lease)
        assert proc.stdin.close_calls == proc.stdout.close_calls == 0
        rejected = worker.inspect_text("must-not-launch.txt", "b", timeout=2)
        assert rejected["executed"] is False
        assert rejected["receipt"]["reason"] == "busy"
        assert len(fake_launch.leases) == 1
        # Release only after proving the caller did not close an active pipe.
        proc.ack_release.set()
        lease.io_thread.join(_WAIT)
        assert not lease.io_thread.is_alive()
        assert proc.stdin.close_calls == proc.stdout.close_calls == 1
    else:
        _assert_confirmed(fake_launch)
    lease.close(original=original if cancel_original else None)
    assert fake_launch.cleanups == ["_cleanup"]
