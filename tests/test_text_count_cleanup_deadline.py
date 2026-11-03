"""One absolute cleanup allowance, including helper waits and launcher joins.

Fake native handles and an explicit clock measure requested timed waits, not an
absolute wallclock guarantee for uninterruptible OS calls. No child is created.
"""
import threading
import types

import pytest

from nexus_agent_ai.agent import text_count_worker as worker


class Clock:
    def __init__(self):
        self.now = 100.0
    def monotonic(self):
        return self.now
    def advance(self, amount):
        self.now += amount


class Pipe:
    def __init__(self, closed=False):
        self.closed = closed
    def close(self):
        self.closed = True


class Process:
    pid = 424242
    def __init__(self, clock, *, confirms=True):
        self.clock = clock
        self.confirms = confirms
        self.stdin, self.stdout = Pipe(), Pipe()
        self.returncode = None
        self.terminated = 0
        self.killed = 0
        self.waits = []
        self.expected_deadline = None
    def poll(self):
        return self.returncode
    def terminate(self):
        self.terminated += 1
    def kill(self):
        self.killed += 1
    def wait(self, timeout=None):
        assert timeout is not None and timeout >= 0
        if self.expected_deadline is not None:
            assert timeout <= max(0.0, self.expected_deadline - self.clock.now) + 1e-9
        self.waits.append(timeout)
        self.clock.advance(timeout)
        if self.killed and self.confirms:
            self.returncode = -9
            return self.returncode
        raise worker.subprocess.TimeoutExpired("fake-child", timeout)


class ObservedThread:
    def __init__(self, clock, *, alive=False):
        self.clock = clock
        self.alive = alive
        self.waits = []
    def join(self, timeout=None):
        assert timeout is not None and timeout >= 0
        self.waits.append(timeout)
        if self.alive:
            self.clock.advance(timeout)
    def is_alive(self):
        return self.alive


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    namespace = types.SimpleNamespace(**vars(worker.time))
    namespace.monotonic = value.monotonic
    monkeypatch.setattr(worker, "time", namespace)
    monkeypatch.setattr(worker, "_QUARANTINE", None)
    monkeypatch.setattr(worker, "_LAST_CLEANUP", None)
    return value


@pytest.mark.parametrize("io_started", [False, True])
def test_one_cleanup_deadline_covers_delayed_helper_all_waits_and_launcher_join(clock, monkeypatch, io_started):
    lease = worker._LaunchLease(clock.now + 10, 10)
    worker._QUARANTINE = lease
    lease.proc = proc = Process(clock)
    proc.expected_deadline = clock.now + 1
    lease.launcher = launcher = ObservedThread(clock)
    lease.launch_start_attempted = True
    lease.launch_joined = True
    if io_started:
        lease.io_thread = ObservedThread(clock)
        lease.io_start_attempted = True
        lease.state["pipes_closed"] = True
        proc.stdin.closed = proc.stdout.closed = True
    name = "_cleanup" if io_started else "_cleanup_unstarted"
    primitive = getattr(worker, name)
    calls = []

    def delayed_cleanup(*args, **kwargs):
        calls.append(1)
        # Simulate time between claim and entering the helper. It must consume
        # the existing allowance, never create another one-second allowance.
        clock.advance(0.6)
        return primitive(*args, **kwargs)

    monkeypatch.setattr(worker, name, delayed_cleanup)
    lease.close(normal=io_started)
    assert lease.close_deadline == pytest.approx(101.0)
    assert clock.now <= lease.close_deadline + 1e-9
    assert proc.terminated == proc.killed == 1
    assert proc.returncode == -9
    assert sum(proc.waits) <= 0.4 + 1e-9
    assert launcher.waits and all(wait == 0 for wait in launcher.waits)
    assert lease.snapshot_cleanup()["confirmed"] is True
    first_deadline, first_waits = lease.close_deadline, list(proc.waits)
    clock.advance(100)
    lease.close()
    lease.close()
    assert calls == [1]
    assert lease.close_deadline == first_deadline
    assert proc.waits == first_waits


def test_late_launch_cleanup_uses_exhausted_original_deadline_and_stays_unknown(clock, monkeypatch):
    lease = worker._LaunchLease(clock.now + 10, 10)
    worker._QUARANTINE = lease
    lease.launch_start_attempted = True
    lease.launcher = launcher = ObservedThread(clock, alive=True)
    lease.ready = threading.Event()
    lease.close()
    original_deadline = lease.close_deadline
    assert clock.now == original_deadline
    assert worker._quarantine_busy(lease)
    clock.advance(5)
    proc = Process(clock, confirms=False)
    proc.expected_deadline = original_deadline
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: proc)
    lease._launch()  # Fixed target, fake Popen only; no native thread or child.
    launcher.alive = False
    assert lease.proc is proc
    assert lease.cleanup_owner == "launcher"
    assert lease.close_deadline == original_deadline
    assert proc.terminated == proc.killed == 1
    assert proc.waits == [0, 0]
    assert lease.snapshot_cleanup()["confirmed"] is False
    assert worker._quarantine_busy(lease)
    clock.advance(100)
    lease.close()
    assert proc.waits == [0, 0]
    assert lease.close_deadline == original_deadline
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("blind relaunch"))
    result = worker.inspect_text("not-inspected.txt", "a", timeout=10)
    assert result["executed"] is False and result["receipt"]["reason"] == "busy"
