"""Deterministic lease ownership boundaries, using fake children and real threads."""
import threading
import time
import types

import pytest

from nexus_agent_ai.agent import text_count_worker as worker


_WAIT = 5
_TIMEOUT = 30


class _Pipe:
    def __init__(self):
        self.closed = False
        self.close_calls = 0
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        raise AssertionError("an unadopted lease sent a request or ACK")

    def read(self, size=-1):
        raise AssertionError("an unadopted lease began pipe I/O")

    def close(self):
        self.close_calls += 1
        self.closed = True


class _Proc:
    """No native process or descriptor exists behind these explicit facts."""

    def __init__(self, pid):
        self.pid = pid
        self.stdin = _Pipe()
        self.stdout = _Pipe()
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        assert self.returncode is not None, "cleanup did not terminate the fake child"
        return self.returncode


class _Harness:
    def __init__(self):
        self.procs = []
        self.threads = []
        self.releases = []
        self.launches = []
        self.cleanups = []
        self.real_cleanup = worker._cleanup_unstarted

    def event(self):
        event = threading.Event()
        self.releases.append(event)
        return event

    def thread(self, *args, **kwargs):
        thread = threading.Thread(*args, **kwargs)
        self.threads.append(thread)
        return thread

    def start(self, target):
        outcome = {}

        def invoke():
            try:
                outcome["value"] = target()
            except BaseException as error:
                outcome["error"] = error

        thread = self.thread(target=invoke, daemon=True)
        thread.start()
        return thread, outcome

    def proc(self):
        proc = _Proc(1000 + len(self.procs))
        self.procs.append(proc)
        return proc

    def popen(self, *args, **kwargs):
        proc = self.proc()
        self.launches.append(proc)
        return proc

    def cleanup(self, proc, deadline):
        lease = worker._QUARANTINE
        self.cleanups.append((proc, threading.current_thread(), lease.cleanup_owner))
        assert lease.proc is proc
        assert lease.cancelled is True
        assert lease.cleanup_owner is not None
        facts = self.real_cleanup(proc, deadline)
        assert set(facts) == {
            "pid", "terminated", "killed", "reaped", "returncode",
            "io_joined", "pipes_closed", "confirmed",
        }
        assert facts["confirmed"] is True
        return facts

    @staticmethod
    def join(thread):
        thread.join(timeout=_WAIT)
        assert not thread.is_alive(), "a controlled native test thread did not finish"


@pytest.fixture
def lease_harness(monkeypatch):
    previous = worker._QUARANTINE
    assert previous is None or not worker._quarantine_busy(previous)
    harness = _Harness()
    # Patch this module's threading namespace, not Python's shared module: real
    # Thread internals and fixture barriers must retain their normal Events.
    namespace = types.SimpleNamespace(**vars(threading))
    namespace.Thread = harness.thread
    monkeypatch.setattr(worker, "threading", namespace)
    monkeypatch.setattr(worker, "_QUARANTINE", None)
    monkeypatch.setattr(worker, "_LAST_CLEANUP", None)
    monkeypatch.setattr(worker.subprocess, "Popen", harness.popen)
    monkeypatch.setattr(worker, "_cleanup_unstarted", harness.cleanup)
    yield harness
    for release in harness.releases:
        release.set()
    for thread in harness.threads:
        if thread.ident is not None:
            harness.join(thread)
        else:
            assert not thread.is_alive()
    # Unknown cleanup is intentionally quarantined in several tests. Only after
    # every real thread has joined may this fake-only fixture dispose its model
    # resources and clear that state. This does not manufacture lease facts.
    for proc in harness.procs:
        if proc.poll() is None:
            proc.terminate()
        if not proc.stdin.closed:
            proc.stdin.close()
        if not proc.stdout.closed:
            proc.stdout.close()
        assert proc.poll() is not None and proc.stdin.closed and proc.stdout.closed
    monkeypatch.setattr(worker, "_QUARANTINE", None)


def _new_lease():
    lease = worker._launch_owned(time.monotonic() + _TIMEOUT, _TIMEOUT)
    worker._QUARANTINE = lease
    return lease


def _inspect():
    return worker.inspect_text("must-not-read.txt", "a", timeout=_TIMEOUT)


def _assert_no_exchange(lease):
    assert lease.adopted is False
    assert lease.io_thread is None
    assert lease.io_start_attempted is False
    assert lease.state["request_attempted"] is False
    assert lease.state["ack_attempted"] is False
    assert lease.state["ack_sent"] is False
    if lease.proc is not None:
        assert lease.proc.stdin.writes == []


def _assert_one_cleanup(harness, lease):
    assert len(harness.cleanups) == 1
    assert harness.cleanups[0][0] is lease.proc
    assert lease.proc.terminate_calls == 1
    assert lease.proc.kill_calls == 0
    assert lease.proc.stdin.close_calls == 1
    assert lease.proc.stdout.close_calls == 1


def _assert_busy_without_launch(harness, lease):
    launches = len(harness.launches)
    assert lease.busy() is True
    assert worker._quarantine_busy(lease) is True
    for _ in range(2):
        result = _inspect()
        assert result["executed"] is False
        assert result["receipt"]["reason"] == "busy"
        assert result["receipt"]["status"] == "incomplete"
        assert result["receipt"]["count"] is None
        assert worker._QUARANTINE is lease
    assert len(harness.launches) == launches
    assert not worker._CALL_LOCK.locked()


def test_factory_creates_no_event_thread_or_process(lease_harness, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure lease construction started native launch machinery")

    monkeypatch.setattr(worker.threading, "Event", forbidden)
    monkeypatch.setattr(worker.threading, "Thread", forbidden)
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    lease = worker._launch_owned(time.monotonic() + _TIMEOUT, _TIMEOUT)
    assert isinstance(lease, worker._LaunchLease)
    assert lease.proc is None
    assert lease.launcher is None and lease.ready is None
    assert lease.launch_start_attempted is False
    assert lease.launch_joined is False
    assert lease_harness.threads == []
    assert lease_harness.launches == []


def test_registry_precedes_native_start_and_start_error_escapes_exactly(
        lease_harness, monkeypatch):
    original = KeyboardInterrupt("native start cancellation")
    observed = []

    class InterruptedStart(threading.Thread):
        def start(self):
            lease = worker._QUARANTINE
            assert isinstance(lease, worker._LaunchLease)
            assert lease.launcher is self
            assert lease.launch_start_attempted is True
            observed.append(lease)
            raise original

    def thread(*args, **kwargs):
        result = InterruptedStart(*args, **kwargs)
        lease_harness.threads.append(result)
        return result

    monkeypatch.setattr(worker.threading, "Thread", thread)
    with pytest.raises(KeyboardInterrupt) as caught:
        _inspect()
    assert caught.value is original
    lease, = observed
    assert worker._QUARANTINE is lease
    assert lease.cancelled is True
    assert lease.launch_joined is False
    assert lease_harness.launches == []
    assert lease_harness.cleanups == []
    _assert_no_exchange(lease)
    _assert_busy_without_launch(lease_harness, lease)


def test_enter_failure_uses_explicit_outer_finally_not_exit(lease_harness, monkeypatch):
    original = SystemExit("after launch, before context entry completes")
    real_enter = worker._LaunchLease.__enter__
    exits = []

    def fail_enter(lease):
        real_enter(lease)
        assert lease.proc is not None and lease.launch_joined is True
        raise original

    def forbidden_exit(*args):
        exits.append(args)
        pytest.fail("Python must not call __exit__ after __enter__ failed")

    monkeypatch.setattr(worker._LaunchLease, "__enter__", fail_enter)
    monkeypatch.setattr(worker._LaunchLease, "__exit__", forbidden_exit)
    with pytest.raises(SystemExit) as caught:
        _inspect()
    assert caught.value is original
    lease = worker._QUARANTINE
    assert exits == []
    assert lease.cancelled is True
    _assert_no_exchange(lease)
    _assert_one_cleanup(lease_harness, lease)
    assert lease.snapshot_cleanup()["confirmed"] is True
    assert not worker._CALL_LOCK.locked()


def test_cancelled_ready_wait_owns_late_child_without_double_cleanup(
        lease_harness, monkeypatch):
    entered = threading.Event()
    release = lease_harness.event()
    cleanup_entered = threading.Event()
    release_cleanup = lease_harness.event()
    original = KeyboardInterrupt("cancel while Popen has not returned")

    class InterruptReady(threading.Event):
        def wait(self, timeout=None):
            assert entered.wait(_WAIT)
            raise original

    def blocked_popen(*args, **kwargs):
        entered.set()
        assert release.wait(_WAIT)
        return lease_harness.popen(*args, **kwargs)

    def blocked_cleanup(proc, deadline):
        cleanup_entered.set()
        assert release_cleanup.wait(_WAIT)
        return lease_harness.cleanup(proc, deadline)

    monkeypatch.setattr(worker.threading, "Event", InterruptReady)
    monkeypatch.setattr(worker.subprocess, "Popen", blocked_popen)
    monkeypatch.setattr(worker, "_cleanup_unstarted", blocked_cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        _inspect()
    assert caught.value is original
    lease = worker._QUARANTINE
    assert lease.proc is None and lease.cancelled is True
    assert lease.launcher.is_alive() and not lease.launch_joined
    assert lease_harness.cleanups == []
    _assert_no_exchange(lease)
    _assert_busy_without_launch(lease_harness, lease)
    release.set()
    assert cleanup_entered.wait(_WAIT)
    owner = lease.cleanup_owner
    assert owner is not None and lease.cleanup is None
    lease.close(original=original)
    assert lease.cleanup_owner == owner
    assert lease_harness.cleanups == []
    assert worker._QUARANTINE is lease
    release_cleanup.set()
    lease_harness.join(lease.launcher)
    assert worker._quarantine_busy(lease) is False
    assert lease.launch_joined is True
    assert lease_harness.cleanups[0][1] is lease.launcher
    lease.close(original=original)
    lease.close(original=original)
    _assert_no_exchange(lease)
    _assert_one_cleanup(lease_harness, lease)
    assert lease.snapshot_cleanup()["confirmed"] is True


def test_publication_before_adoption_cancellation_has_one_caller_reaper(
        lease_harness, monkeypatch):
    original = KeyboardInterrupt("published but not adopted")
    caller = threading.current_thread()

    class InterruptPublished(threading.Event):
        def wait(self, timeout=None):
            assert super().wait(timeout)
            lease = worker._QUARANTINE
            assert lease.proc is not None
            assert lease.adopted is False
            raise original

    monkeypatch.setattr(worker.threading, "Event", InterruptPublished)
    with pytest.raises(KeyboardInterrupt) as caught:
        _inspect()
    assert caught.value is original
    lease = worker._QUARANTINE
    lease_harness.join(lease.launcher)
    lease.close(original=original)
    _assert_one_cleanup(lease_harness, lease)
    assert lease_harness.cleanups[0][1] is caller
    _assert_no_exchange(lease)
    assert lease.snapshot_cleanup()["confirmed"] is True
    assert not worker._CALL_LOCK.locked()


def test_adoption_is_single_winner_and_permissions_require_adoption(lease_harness):
    lease = _new_lease()
    barrier = threading.Barrier(3)
    with lease:
        assert lease.launch_joined is True
        assert lease.begin_request() is False
        assert lease.begin_ack() is False
        _assert_no_exchange(lease)

        def compete():
            barrier.wait(timeout=_WAIT)
            return lease.adopt()

        first, first_result = lease_harness.start(compete)
        second, second_result = lease_harness.start(compete)
        barrier.wait(timeout=_WAIT)
        lease_harness.join(first)
        lease_harness.join(second)
        assert "error" not in first_result and "error" not in second_result
        assert sorted([first_result["value"], second_result["value"]]) == [False, True]
        assert lease.adopted is True
        assert lease.adopt() is False
        assert lease.begin_request() is True
        assert lease.state["request_attempted"] is True
        assert lease.proc.stdin.writes == []
    assert lease.adopt() is False
    assert lease.begin_request() is False
    assert lease.begin_ack() is False
    _assert_one_cleanup(lease_harness, lease)
    assert lease.snapshot_cleanup()["confirmed"] is True


def test_cleanup_claim_is_durable_without_holding_lock_over_reaping(
        lease_harness, monkeypatch):
    lease = _new_lease()
    lease.__enter__()
    cleanup_entered = threading.Event()
    release_cleanup = lease_harness.event()
    probe_done = threading.Event()
    original = KeyboardInterrupt("original cancellation")
    second = SystemExit("second cancellation")

    def blocked_cleanup(proc, deadline):
        cleanup_entered.set()
        assert release_cleanup.wait(_WAIT)
        return lease_harness.cleanup(proc, deadline)

    monkeypatch.setattr(worker, "_cleanup_unstarted", blocked_cleanup)
    first, first_result = lease_harness.start(lambda: lease.close(original=original))
    assert cleanup_entered.wait(_WAIT)
    assert worker._QUARANTINE is lease and lease.cancelled is True
    owner = lease.cleanup_owner
    assert owner is not None

    def competing_close():
        try:
            assert lease.adopt() is False
            assert lease.begin_request() is False
            assert lease.begin_ack() is False
            lease.close(original=second)
            assert lease.cleanup_owner == owner
        finally:
            probe_done.set()

    contender, contender_result = lease_harness.start(competing_close)
    try:
        assert probe_done.wait(_WAIT), "cleanup held the lease lock over process I/O"
        assert lease_harness.cleanups == []
        assert lease.snapshot_cleanup()["confirmed"] is False
    finally:
        release_cleanup.set()
    lease_harness.join(first)
    lease_harness.join(contender)
    assert "error" not in first_result and "error" not in contender_result
    lease.close(original=original)
    _assert_no_exchange(lease)
    _assert_one_cleanup(lease_harness, lease)
    assert lease.snapshot_cleanup()["confirmed"] is True


@pytest.mark.parametrize("secondary_type", [KeyboardInterrupt, RuntimeError])
def test_second_cancellation_or_cleanup_error_preserves_original_and_quarantine(
        lease_harness, monkeypatch, secondary_type):
    original = SystemExit("first cancellation must escape unchanged")
    secondary = secondary_type("cleanup failed before returning any facts")
    real_enter = worker._LaunchLease.__enter__
    attempts = []

    def fail_enter(lease):
        real_enter(lease)
        raise original

    def fail_cleanup(proc, deadline):
        attempts.append(proc)
        lease = worker._QUARANTINE
        assert lease.cancelled and lease.cleanup_owner is not None
        raise secondary

    monkeypatch.setattr(worker._LaunchLease, "__enter__", fail_enter)
    monkeypatch.setattr(worker, "_cleanup_unstarted", fail_cleanup)
    with pytest.raises(SystemExit) as caught:
        _inspect()
    assert caught.value is original
    lease = worker._QUARANTINE
    assert attempts == [lease.proc]
    assert lease.cleanup_error is secondary
    assert lease.cleanup is None
    assert lease.snapshot_cleanup()["confirmed"] is False
    assert lease.proc.poll() is None
    lease.close(original=original)
    _assert_busy_without_launch(lease_harness, lease)
    assert attempts == [lease.proc]
    _assert_no_exchange(lease)


def test_cleanup_error_without_original_escapes_exactly_even_after_reaping(
        lease_harness, monkeypatch):
    lease = _new_lease()
    lease.__enter__()
    original = RuntimeError("cleanup facts never reached the owner")

    def fail_after_resources(proc, timeout):
        lease_harness.cleanup(proc, timeout)
        raise original

    monkeypatch.setattr(worker, "_cleanup_unstarted", fail_after_resources)
    with pytest.raises(RuntimeError) as caught:
        lease.close()
    assert caught.value is original
    assert lease.cleanup_error is original
    assert lease.cleanup is None
    assert lease.proc.poll() is not None
    assert lease.proc.stdin.closed and lease.proc.stdout.closed
    lease.close(original=original)
    _assert_busy_without_launch(lease_harness, lease)
    _assert_no_exchange(lease)
    _assert_one_cleanup(lease_harness, lease)


@pytest.mark.parametrize("missing", [None, "confirmed", "reaped", "io_joined", "pipes_closed"])
def test_missing_or_partial_cleanup_facts_remain_unknown_after_physical_cleanup(
        lease_harness, monkeypatch, missing):
    lease = _new_lease()
    lease.__enter__()

    def lose_facts(proc, timeout):
        facts = lease_harness.cleanup(proc, timeout)
        if missing is None:
            return None
        del facts[missing]
        return facts

    monkeypatch.setattr(worker, "_cleanup_unstarted", lose_facts)
    lease.close()
    assert lease.proc.poll() is not None
    assert lease.proc.stdin.closed and lease.proc.stdout.closed
    assert lease.launch_joined is True
    assert lease.snapshot_cleanup()["confirmed"] is False
    _assert_busy_without_launch(lease_harness, lease)
    lease.close()
    _assert_one_cleanup(lease_harness, lease)
    _assert_no_exchange(lease)


def test_launcher_target_return_is_not_native_thread_completion(lease_harness, monkeypatch):
    target_returned = threading.Event()
    release_thread = lease_harness.event()
    original = KeyboardInterrupt("cancel after launcher target finally")

    def thread(*, target, **kwargs):
        def still_running():
            try:
                target()
            finally:
                target_returned.set()
                assert release_thread.wait(_WAIT)

        return lease_harness.thread(target=still_running, **kwargs)

    class InterruptAfterTarget(threading.Event):
        def wait(self, timeout=None):
            assert super().wait(timeout)
            assert target_returned.wait(_WAIT)
            raise original

    monkeypatch.setattr(worker.threading, "Thread", thread)
    monkeypatch.setattr(worker.threading, "Event", InterruptAfterTarget)
    with pytest.raises(KeyboardInterrupt) as caught:
        _inspect()
    assert caught.value is original
    lease = worker._QUARANTINE
    assert target_returned.is_set() and lease.ready.is_set()
    assert lease.launcher.is_alive()
    assert lease.cleanup["confirmed"] is True
    assert lease.proc.poll() is not None
    assert lease.proc.stdin.closed and lease.proc.stdout.closed
    facts = lease.snapshot_cleanup()
    assert lease.launch_joined is False
    assert facts["launch_joined"] is False
    assert facts["confirmed"] is False
    _assert_busy_without_launch(lease_harness, lease)
    _assert_no_exchange(lease)
    release_thread.set()
    lease_harness.join(lease.launcher)
    assert worker._quarantine_busy(lease) is False
    assert lease.launch_joined is True
    assert lease.snapshot_cleanup()["confirmed"] is True
    _assert_one_cleanup(lease_harness, lease)


def test_blocked_popen_stays_busy_without_blind_second_launch(lease_harness, monkeypatch):
    entered = threading.Event()
    release = lease_harness.event()
    clock = [time.monotonic()]
    starts = []
    time_namespace = types.SimpleNamespace(**vars(time))
    time_namespace.monotonic = lambda: clock[0]

    class DeadlineReady(threading.Event):
        def wait(self, timeout=None):
            assert entered.wait(_WAIT)
            # Expire the actual launch deadline without a sleep or scheduling
            # assumption. Popen remains blocked for every refusal assertion.
            clock[0] += timeout + 1
            return False

    def blocked_popen(*args, **kwargs):
        starts.append(threading.current_thread())
        entered.set()
        assert release.wait(_WAIT)
        return lease_harness.popen(*args, **kwargs)

    monkeypatch.setattr(worker, "time", time_namespace)
    monkeypatch.setattr(worker.threading, "Event", DeadlineReady)
    monkeypatch.setattr(worker.subprocess, "Popen", blocked_popen)
    result = _inspect()
    lease = worker._QUARANTINE
    assert result["executed"] is False
    assert result["receipt"]["reason"] == "cleanup_unverified"
    assert len(starts) == 1
    assert lease.proc is None and lease.launcher.is_alive()
    assert lease.launch_joined is False
    assert lease.snapshot_cleanup()["confirmed"] is False
    assert lease_harness.cleanups == []
    _assert_busy_without_launch(lease_harness, lease)
    assert len(starts) == 1
    _assert_no_exchange(lease)
    # The fake is blocked indefinitely until explicit test teardown; releasing
    # it here lets us join the native thread instead of abandoning a daemon.
    release.set()
    lease_harness.join(lease.launcher)
    assert worker._quarantine_busy(lease) is False
    assert len(starts) == 1 and len(lease_harness.launches) == 1
    _assert_one_cleanup(lease_harness, lease)
    _assert_no_exchange(lease)


def test_unexpected_popen_error_is_relayed_by_identity_without_continuation(
        lease_harness, monkeypatch):
    original = KeyboardInterrupt("Popen failed unexpectedly")
    observed = []

    def fail_popen(*args, **kwargs):
        lease = worker._QUARANTINE
        assert lease.launch_start_attempted is True
        observed.append(lease)
        raise original

    monkeypatch.setattr(worker.subprocess, "Popen", fail_popen)
    with pytest.raises(KeyboardInterrupt) as caught:
        _inspect()
    assert caught.value is original
    lease, = observed
    assert lease.launch_error is original
    assert lease.launch_joined is True
    assert worker._QUARANTINE is lease
    assert lease.cancelled is True
    assert lease_harness.cleanups == []
    _assert_no_exchange(lease)
    _assert_busy_without_launch(lease_harness, lease)


def test_one_lease_cannot_launch_a_second_process_on_reentry(lease_harness):
    lease = _new_lease()
    with lease:
        assert lease.adopt()
        with pytest.raises(RuntimeError):
            lease.__enter__()
        assert len(lease_harness.launches) == 1
        assert lease.state["request_attempted"] is False
    _assert_one_cleanup(lease_harness, lease)
    assert lease.snapshot_cleanup()["confirmed"] is True
