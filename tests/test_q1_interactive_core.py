"""Unit and edge tests for Q1 Windows interactive candidate core and assessor.

Zero model launches, zero downloads, zero network calls.
Tests deterministic parsing, context arithmetic, answer attribution, Backspace probe,
outage handling, environment isolation, evidence caps, AST preservation, and cleanup filtering.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


assess = _load_module("interactive_assess_under_test", ".github/qualification/interactive_assess.py")
core = _load_module("interactive_core_under_test", ".github/qualification/interactive_core.py")


def mock_successful_ownership(monkeypatch, root: Path, pid: int = 4242) -> None:
    """Record fake facts and mock process recording without launching real processes."""
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    facts = [{"pid": pid, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "mock.exe", "cmdline": ["mock"]}]
    (out / "owned_processes.json").write_text(json.dumps(facts), encoding="utf-8")
    monkeypatch.setattr(core, "record_owned_process", lambda r, p, **kw: None)
    monkeypatch.setattr(core, "scan_and_record_descendants", lambda r, p: None)
    monkeypatch.setattr(core, "_capture_registration_identity",
                        lambda p: {"create_ticks": 1000000, "exe": "mock.exe", "creator_pid": 4242})


class FakePtyProcess:
    """Mock PtyProcess for testing waiter logic, echo rejection, and timeouts."""

    def __init__(self, chunks, pid=4242):
        self.chunks = list(chunks)
        self.pid = pid
        self._alive = True

    def read(self, size=1024):
        if self.chunks:
            return self.chunks.pop(0)
        return ""

    def isalive(self):
        return self._alive

    def write(self, text):
        pass

    def close(self):
        self._alive = False




def test_strip_ansi_removes_escape_sequences():
    raw = "\x1b[32m>> You: \x1b[0m\x1b[2KHello world\x1b[6n"
    assert assess.strip_ansi(raw) == ">> You: Hello world"


def test_parse_tty_probe():
    line = "__NEXUS_TTY_PROBE__: isatty=True win_console=True interactive=True\n"
    res = assess.parse_tty_probe(line)
    assert res["observed"] is True
    assert res["isatty"] is True
    assert res["win_console"] is True
    assert res["interactive"] is True

    bad = "Standard python output without probe"
    assert assess.parse_tty_probe(bad)["observed"] is False


def test_ready_prompt_detection_vs_echo(tmp_path, monkeypatch):
    """Verify PtySession.wait_for_ready_prompt rejects echoed input and detects clean ready prompt."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    fake = FakePtyProcess([
        ">> You: Read readme.txt\r\n",  # Echo of input
        "Processing...\r\n",
        ">> You: ",                     # Clean ready prompt
    ])
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=2.0, deadline=time.monotonic() + 5.0)
        assert ready is True
        assert consent_wait is False
        assert "Processing..." in buf
    finally:
        session.close()


def test_wait_for_ready_prompt_timeout(tmp_path, monkeypatch):
    """Verify PtySession.wait_for_ready_prompt returns False when ready prompt is not emitted."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    fake = FakePtyProcess(["Still working...\r\n"])
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=0.1, deadline=time.monotonic() + 0.1)
        assert ready is False
        assert consent_wait is False
    finally:
        session.close()


def test_canonical_diff_content_prefix_matching():
    """Verify canonical_diff handles ID changes, rewrites, and appends via content prefix."""
    before = [
        {"session_id": "s1", "role": "user", "message_json": json.dumps({"content": "hi"}), "id": 1},
        {"session_id": "s1", "role": "assistant", "message_json": json.dumps({"content": "hello"}), "id": 2},
    ]
    after = [
        {"session_id": "s1", "role": "user", "message_json": json.dumps({"content": "hi"}), "id": 10},
        {"session_id": "s1", "role": "assistant", "message_json": json.dumps({"content": "hello"}), "id": 11},
        {"session_id": "s1", "role": "user", "message_json": json.dumps({"content": "what is 2+2?"}), "id": 12},
        {"session_id": "s1", "role": "assistant", "message_json": json.dumps({"content": "4"}), "id": 13},
    ]
    delta = core.canonical_diff(before, after)
    assert len(delta) == 2
    assert json.loads(delta[0]["message_json"])["content"] == "what is 2+2?"
    assert json.loads(delta[1]["message_json"])["content"] == "4"


def test_parse_context_output_and_validation():
    sample = (
        "Configured context budget: 4096 tokens; configured output ceiling: 512 tokens.\n"
        "Stored demand: ~1100/4096 tokens (overhead + full stored history + reserve).\n"
        "Breakdown: system ~150, tools ~250, system+tools overhead ~400, "
        "full stored history ~188, output reserve ~512, window 4096 tokens.\n"
        "Next request projection: ~1050/4096 tokens (fitted history ~138, including omission note+ack ~0; kept 2 turn(s), dropped 0).\n"
        "Free for new input: ~3046 tokens (window - projection).\n"
    )
    metrics = assess.parse_context_output(sample)
    assert metrics["window"] == 4096
    assert metrics["config_ceiling"] == 512
    assert metrics["system"] == 150
    assert metrics["tools"] == 250
    assert metrics["overhead"] == 400
    assert metrics["history"] == 188
    assert metrics["reserve"] == 512
    assert metrics["stored_demand_total"] == 1100
    assert metrics["projection"] == 1050
    assert metrics["fitted"] == 138
    assert metrics["omission"] == 0
    assert metrics["free"] == 3046

    val = assess.validate_context_metrics(metrics)
    assert val["status"] == assess.PASS
    assert val["errors"] == []


def test_validate_context_metrics_detects_arithmetic_errors():
    bad_metrics = {
        "window": 4096,
        "config_ceiling": 512,
        "system": 100,
        "tools": 200,
        "overhead": 350,  # 100 + 200 != 350
        "history": 100,
        "reserve": 512,
        "stored_demand_total": 962,
        "projection": 900,
        "fitted": 100,
        "omission": 0,
        "free": 3196,
    }
    val = assess.validate_context_metrics(bad_metrics)
    assert val["status"] == assess.FAIL
    assert any("overhead" in e for e in val["errors"])


def test_validate_context_metrics_omission_bounds():
    bad_omission = {
        "window": 4096,
        "config_ceiling": 512,
        "system": 100,
        "tools": 200,
        "overhead": 300,
        "history": 100,
        "reserve": 512,
        "stored_demand_total": 912,
        "projection": 912,
        "fitted": 100,
        "omission": 150,  # omission 150 > fitted 100
        "free": 3184,
    }
    val = assess.validate_context_metrics(bad_omission)
    assert val["status"] == assess.FAIL
    assert any("omission" in e for e in val["errors"])


def test_parse_compact_output():
    text = "Compacted 3 earlier turn(s): history ~1200 -> ~450 tokens."
    parsed = assess.parse_compact_output(text)
    assert parsed["compacted"] is True
    assert parsed["turns_summarized"] == 3
    assert parsed["before_tokens"] == 1200
    assert parsed["after_tokens"] == 450

    uncompacted = "Nothing to compact yet (only the last 2 turns are kept as they are)."
    parsed_bad = assess.parse_compact_output(uncompacted)
    assert parsed_bad["compacted"] is False


def test_extract_assistant_turn_answer_isolates_final_answer():
    turn_messages = [
        {"role": "user", "content": "Read readme.txt and tell me the owner"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Project Harbor Desk\nOwner Mira Chen"},
        {"role": "assistant", "content": "The project is Harbor Desk and the owner is Mira Chen.", "tool_calls": None},
    ]
    ans = assess.extract_assistant_turn_answer(turn_messages)
    assert ans == "The project is Harbor Desk and the owner is Mira Chen."


def test_assess_task_readme_bound_tool_and_backspace():
    user_msg_good = {"role": "user", "content": "Read readme.txt and tell me the project name and owner."}
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "readme.txt"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "Project Harbor Desk\nOwner Mira Chen\n"},
        {"role": "assistant", "content": "Project Harbor Desk, owned by Mira Chen."},
    ]
    ans_good = "Project Harbor Desk, owned by Mira Chen."

    res_good = assess.assess_task_readme(ans_good, messages, user_msg_good,
                                         "Read readme.txt and tell me the project name and owner.")
    assert res_good["task"]["status"] == assess.PASS
    assert res_good["interactive"]["status"] == assess.PASS

    # Unbound tool call (mismatched ID) => FAIL
    unbound_messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "readme.txt"})},
            }],
        },
        {"role": "tool", "tool_call_id": "different_id", "name": "read_file", "content": "Project Harbor Desk\nOwner Mira Chen\n"},
    ]
    res_unbound = assess.assess_task_readme(ans_good, unbound_messages, user_msg_good,
                                           "Read readme.txt and tell me the project name and owner.")
    assert res_unbound["task"]["status"] == assess.FAIL


def test_assess_task_greeting_requires_bound_mutation_and_exact_bytes():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_write",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "greeting.txt", "content": "Hello Harbor Desk\n"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_write", "name": "write_file", "content": "OK"},
    ]
    res_ok = assess.assess_task_greeting(b"Hello Harbor Desk\n", messages)
    assert res_ok["task"]["status"] == assess.PASS

    # Missing bound mutation tool call
    res_no_tool = assess.assess_task_greeting(b"Hello Harbor Desk\n", [])
    assert res_no_tool["task"]["status"] == assess.FAIL

    # Missing newline
    res_wrong_bytes = assess.assess_task_greeting(b"Hello Harbor Desk", messages)
    assert res_wrong_bytes["task"]["status"] == assess.FAIL


def test_assess_task_calculator_review_requires_explicit_relation():
    src = "def add(a, b): return a + b\ndef multiply(a, b): return a + b\n"
    read_tools = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_read_calc",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "calculator.py"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_read_calc", "name": "read_file", "content": src},
    ]

    # Explicit relation: multiply returns addition instead of multiplying
    ans_explicit = "In calculator.py, the multiply function incorrectly returns a + b (addition) instead of multiplying."
    res = assess.assess_task_calculator_review(ans_explicit, src, src, read_tools, is_slash=False)
    assert res["task"]["status"] == assess.PASS

    # Ambiguous sentence without contrast => REVIEW, not PASS
    ans_ambiguous = "The file defines add and multiply functions which both return a + b."
    res_amb = assess.assess_task_calculator_review(ans_ambiguous, src, src, read_tools, is_slash=False)
    assert res["task"]["status"] != assess.FAIL
    assert res_amb["task"]["status"] == assess.REVIEW


def test_verify_calculator_static_never_executes_model_code(tmp_path):
    """Static verification: syntax/add-preservation/multiply-changed without execution.

    A hostile top-level statement in the model-written file must never run:
    no import, no exec, no compile-and-call from the harness."""
    workspace = tmp_path / "calc_ws"
    workspace.mkdir()
    before_src = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a + b\n"
    sentinel = workspace / "SENTINEL_EXECUTED"
    hostile = (
        "from pathlib import Path as _P\n"
        f"_P({str(sentinel)!r}).write_text('pwned')\n"
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
    )
    (workspace / "calculator.py").write_text(hostile, encoding="utf-8")
    facts = core.verify_calculator_static(workspace, before_src)
    assert facts == {"syntax_ok": True, "add_preserved": True, "multiply_changed": True}
    assert not sentinel.exists(), "model-written code must never execute during verification"

    # Broken add AST -> not preserved
    (workspace / "calculator.py").write_text(
        "def add(a, b):\n    return a + b + 0\n\ndef multiply(a, b):\n    return a * b\n",
        encoding="utf-8"
    )
    assert core.verify_calculator_static(workspace, before_src)["add_preserved"] is False

    # Unchanged multiply (bug not fixed) -> not changed
    (workspace / "calculator.py").write_text(before_src, encoding="utf-8")
    assert core.verify_calculator_static(workspace, before_src)["multiply_changed"] is False

    # Syntax error -> static fail, still no execution
    (workspace / "calculator.py").write_text("def broken(:\n", encoding="utf-8")
    assert core.verify_calculator_static(workspace, before_src)["syntax_ok"] is False

    # Missing file
    (workspace / "calculator.py").unlink()
    assert core.verify_calculator_static(workspace, before_src) == {
        "syntax_ok": False, "add_preserved": False, "multiply_changed": False}


def test_assess_task_calculator_fix_review_never_pass_and_behavior_not_tested():
    """Static-complete fix yields REVIEW (behavior NOT_TESTED), never PASS; gaps yield FAIL."""
    mutation_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
          "function": {"name": "patch_file", "arguments": json.dumps({"path": "calculator.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "patch_file", "content": "patched",
         "metadata": {"executed": True}},
    ]
    before_src = "def add(a, b): return a + b\ndef multiply(a, b): return a + b\n"
    after_src = "def add(a, b): return a + b\ndef multiply(a, b): return a * b\n"
    good = {"syntax_ok": True, "add_preserved": True, "multiply_changed": True}
    res = assess.assess_task_calculator_fix(good, before_src, after_src, mutation_msgs)
    assert res["task"]["status"] == assess.REVIEW
    assert res["task"]["status"] != assess.PASS
    assert "NOT_TESTED" in res["task"]["evidence"][0]["behavior"]
    assert res["task"]["evidence"][0]["static_only"] is True

    # No bound mutation -> FAIL
    res_no_mut = assess.assess_task_calculator_fix(good, before_src, after_src, [])
    assert res_no_mut["task"]["status"] == assess.FAIL

    # Static gap (multiply unchanged) -> FAIL
    res_static_gap = assess.assess_task_calculator_fix(
        {"syntax_ok": True, "add_preserved": True, "multiply_changed": False},
        before_src, before_src, mutation_msgs)
    assert res_static_gap["task"]["status"] == assess.FAIL


def _strict_psutil(monkeypatch, processes):
    """Fake psutil over a {pid: [create_time, exe, running]} table (mutable running)."""
    class FakeProc:
        def __init__(self, pid):
            if pid not in processes:
                raise FakePsutil.NoSuchProcess(pid)
            self.pid = pid
        def create_time(self):
            return processes[self.pid][0]
        def exe(self):
            return processes[self.pid][1]
        def is_running(self):
            return processes[self.pid][2]

    class FakePsutil:
        class NoSuchProcess(Exception):
            pass
        @staticmethod
        def Process(pid):
            return FakeProc(pid)

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    return processes


def _owned(tmp_path, entries):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "owned_processes.json").write_text(json.dumps(entries), encoding="utf-8")
    return out


def test_terminate_kills_verified_identity_persists_receipt_then_allows_handoff(tmp_path, monkeypatch):
    """Success handoff: verified termination, exit proof, receipt on disk before return."""
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    out = _owned(tmp_path, [entry])
    processes = _strict_psutil(monkeypatch, {4321: [1000.0, "ours.exe", True]})
    kills = []

    def fake_terminate(e, result):
        kills.append(e["pid"])
        processes[e["pid"]][2] = False
        return True

    monkeypatch.setattr(core, "_terminate_verified", fake_terminate)
    closed = type("S", (), {"pty": type("P", (), {"isalive": lambda self: False})()})()
    tracker = [0]
    res = core.terminate_owned_processes(tmp_path, tracker, "after_session_001", pty=closed, settle_seconds=1.0)
    assert res["terminated"] is True
    assert kills == [4321]
    receipt = json.loads((out / "termination_after_session_001.json").read_text(encoding="utf-8"))
    assert receipt["terminated"] is True and receipt["killed_pids"] == [4321] and receipt["pty_closed"] is True
    assert tracker[0] > 0, "receipt must be charged against the evidence budget"


def test_terminate_fails_closed_on_survivor(tmp_path, monkeypatch):
    entry = {"pid": 5555, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    out = _owned(tmp_path, [entry])
    _strict_psutil(monkeypatch, {5555: [1000.0, "ours.exe", True]})  # never stops running
    monkeypatch.setattr(core, "_terminate_verified", lambda e, result: True)
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.3)
    assert res["terminated"] is False
    assert 5555 in res["remaining"]
    receipt = json.loads((out / "termination_after_session_001.json").read_text(encoding="utf-8"))
    assert receipt["terminated"] is False and 5555 in receipt["remaining"]


def test_terminate_pid_reuse_never_kills_occupant(tmp_path, monkeypatch):
    """Same PID, different create_time/exe: occupant untouched, mismatch reported, no absolute
    PID-reuse claim - the recorded identity is simply gone."""
    entry = {"pid": 7777, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    _owned(tmp_path, [entry])
    _strict_psutil(monkeypatch, {7777: [2000.0, "occupant.exe", True]})
    kills = []
    monkeypatch.setattr(core, "_terminate_verified", lambda e, result: kills.append(e["pid"]))
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.2)
    assert kills == [], "a PID-reuse occupant must never be terminated"
    assert res["skipped_identity_mismatch"] == [7777]
    assert res["terminated"] is True


def test_terminate_never_adopts_ambient_server(tmp_path, monkeypatch):
    """An unrecorded llama-server pointing at this root is never terminated (no adoption)."""
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    _owned(tmp_path, [entry])
    _strict_psutil(monkeypatch, {9999: [500.0, "llama-server.exe", True]})  # ambient only
    kills = []
    monkeypatch.setattr(core, "_terminate_verified", lambda e, result: kills.append(e["pid"]))
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.2)
    assert kills == []
    assert res["terminated"] is True


def test_terminate_zero_kill_on_strict_metadata_violations(tmp_path, monkeypatch):
    """Every malformed identity shape fails closed with zero kills."""
    kills = []
    monkeypatch.setattr(core, "_terminate_verified",
                        lambda e, result: (_ for _ in ()).throw(AssertionError("kill attempted")))
    _strict_psutil(monkeypatch, {4321: [1000.0, "ours.exe", True]})
    bad_shapes = [
        "not json",
        "[]",
        json.dumps([{"pid": True, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}]),          # bool pid
        json.dumps([{"pid": "4321", "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}]),        # string pid
        json.dumps([{"pid": -1, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}]),            # negative pid
        json.dumps([{"pid": 4321, "create_time": True, "exe": "ours.exe"}]),            # bool time
        json.dumps([{"pid": 4321, "create_time": 0, "exe": "ours.exe"}]),               # zero time
        json.dumps([{"pid": 4321, "create_time": float("inf"), "exe": "ours.exe"}]),    # inf time
        json.dumps([{"pid": 4321, "create_time": float("nan"), "exe": "ours.exe"}]),    # nan time
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000}]),                             # missing exe
        json.dumps([{"pid": 4321, "create_time": 1000.0, "exe": ""}]),                  # empty exe
        json.dumps([{"pid": 4321, "create_time": 1000.0, "exe": "   "}]),               # blank exe
        json.dumps([{"pid": 4321, "create_time": 1000.0, "exe": "ours.exe"}]),          # missing ticks (legacy)
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": True, "exe": "ours.exe"}]),       # bool ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 0, "exe": "ours.exe"}]),          # zero ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": -5, "exe": "ours.exe"}]),         # negative ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000.5, "exe": "ours.exe"}]),     # float ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": "1000000", "exe": "ours.exe"}]),  # string ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "exe": "ours.exe"}]),  # missing creator (legacy)
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": True, "exe": "ours.exe"}]),   # bool creator
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 0, "exe": "ours.exe"}]),     # zero creator
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": "4242", "exe": "ours.exe"}]), # string creator
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "creator_ticks": True, "exe": "ours.exe"}]),   # bool creator_ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "creator_ticks": 0, "exe": "ours.exe"}]),     # zero creator_ticks
        json.dumps([{"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "creator_ticks": 1.5, "exe": "ours.exe"}]),   # float creator_ticks
    ]
    for bad in bad_shapes:
        res = core.terminate_owned_processes(_owned(tmp_path, []) .parent.parent if False else
                                             _write_bad(tmp_path, bad), [0], "r", settle_seconds=0.05)
        assert res["terminated"] is False, bad
        assert res["errors"], bad
    assert kills == []


def _write_bad(tmp_path, bad):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "owned_processes.json").write_text(bad, encoding="utf-8")
    return tmp_path


def test_terminate_fails_closed_on_observation_errors(tmp_path, monkeypatch):
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    out = _owned(tmp_path, [entry])
    (out / "observation_errors.json").write_text(json.dumps(["descendant scan failed"]), encoding="utf-8")
    _strict_psutil(monkeypatch, {4321: [1000.0, "ours.exe", True]})
    monkeypatch.setattr(core, "_terminate_verified",
                        lambda e, result: (_ for _ in ()).throw(AssertionError("kill attempted")))
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.1)
    assert res["terminated"] is False


def test_terminate_fails_closed_on_failed_inspection(tmp_path, monkeypatch):
    """psutil inspection error (not NoSuchProcess) -> fail closed, no termination."""
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    _owned(tmp_path, [entry])

    class FakePsutil:
        class NoSuchProcess(Exception):
            pass
        @staticmethod
        def Process(pid):
            raise RuntimeError("inspection failed")

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "_terminate_verified",
                        lambda e, result: (_ for _ in ()).throw(AssertionError("kill attempted")))
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.1)
    assert res["terminated"] is False
    assert any("pre-kill inspection" in e for e in res["errors"])


def test_terminate_fails_closed_on_live_pty(tmp_path, monkeypatch):
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    _owned(tmp_path, [entry])
    _strict_psutil(monkeypatch, {4321: [1000.0, "ours.exe", True]})
    monkeypatch.setattr(core, "_terminate_verified",
                        lambda e, result: (_ for _ in ()).throw(AssertionError("kill attempted")))
    live = type("S", (), {"pty": type("P", (), {"isalive": lambda self: True})()})()
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", pty=live, settle_seconds=0.1)
    assert res["terminated"] is False
    assert res["pty_closed"] is False


def test_terminate_fails_closed_on_receipt_write_failure(tmp_path, monkeypatch):
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe", "cmdline": ["ours"]}
    _owned(tmp_path, [entry])
    processes = _strict_psutil(monkeypatch, {4321: [1000.0, "ours.exe", True]})

    def fake_terminate(e, result):
        processes[e["pid"]][2] = False
        return True

    monkeypatch.setattr(core, "_terminate_verified", fake_terminate)
    monkeypatch.setattr(core, "charge_and_write",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("evidence cap")))
    res = core.terminate_owned_processes(tmp_path, [0], "after_session_001", settle_seconds=0.5)
    assert res["terminated"] is False, "no receipt, no handoff"
    assert any("receipt persist failed" in e for e in res["errors"])


class _RegistrationPsutil:
    """Fake psutil for registration tests: {pid: [create_time, exe, cmdline]}."""

    class NoSuchProcess(Exception):
        pass

    table = {}

    @classmethod
    def Process(cls, pid):
        if pid not in cls.table:
            raise cls.NoSuchProcess(pid)
        create_time, exe, cmdline = cls.table[pid]
        return type("P", (), {"create_time": staticmethod(lambda: create_time),
                              "exe": staticmethod(lambda: exe),
                              "cmdline": staticmethod(lambda: cmdline)})()


def _registration_rig(monkeypatch, tmp_path, identities, table=None):
    """Fake psutil + fake single-channel capture over {pid: identity|None}."""
    monkeypatch.setitem(sys.modules, "psutil", _RegistrationPsutil)
    _RegistrationPsutil.table = table or {pid: [1000.0, "ours.exe", ["ours", "--flag"]]
                                          for pid in identities}
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "_capture_registration_identity", lambda pid: identities.get(pid))
    return tmp_path


def _read_owned(root):
    f = root / "output" / "owned_processes.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _obs_errors(root):
    f = root / "output" / "observation_errors.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def test_registration_single_channel_identity_recorded(tmp_path, monkeypatch):
    """Root spawn: identity (ticks + exe + creator) from ONE handle, creator is
    the harness, psutil fields retained as display-only observation anchors."""
    identity = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": os.getpid()}
    root = _registration_rig(monkeypatch, tmp_path, {4321: identity})
    core.record_owned_process(root, 4321, expected=identity, expected_creator=os.getpid())
    stored = _read_owned(root)
    assert stored == [{"pid": 4321, "create_time": 1000.0, "create_ticks": 5000000,
                       "exe": "ours.exe", "creator_pid": os.getpid(), "creator_ticks": None,
                       "observed_exe": "ours.exe", "cmdline": ["ours", "--flag"]}]
    assert core._valid_owned_entry(stored[0]) is True


def test_registration_anchor_mismatch_fails_closed(tmp_path, monkeypatch):
    """Process replaced between earliest observation and registration: exact
    anchor mismatch -> raise, nothing registered, occupant never adopted."""
    anchor = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": os.getpid()}
    occupant = {"create_ticks": 9000000, "exe": "ours.exe", "creator_pid": os.getpid()}
    root = _registration_rig(monkeypatch, tmp_path, {4321: occupant})
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 4321, expected=anchor, expected_creator=os.getpid())
    assert _read_owned(root) == []
    assert any("anchor mismatch" in str(e) for e in _obs_errors(root))


def test_registration_foreign_creator_fails_closed(tmp_path, monkeypatch):
    """Creator is neither the harness nor a registered parent: unowned
    occupant refused even with a plausible identity."""
    identity = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": 99999}
    root = _registration_rig(monkeypatch, tmp_path, {4321: identity})
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 4321, expected=identity)
    assert _read_owned(root) == []
    assert any("creator" in str(e).lower() for e in _obs_errors(root))


def test_registration_expected_creator_mismatch_fails_closed(tmp_path, monkeypatch):
    identity = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": 99999}
    root = _registration_rig(monkeypatch, tmp_path, {4321: identity})
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 4321, expected=identity, expected_creator=os.getpid())
    assert _read_owned(root) == []


def test_registration_missing_required_anchor_fails_closed(tmp_path, monkeypatch):
    """First capture failed (expected=None) but the registration-time capture
    succeeds: refuse - the earliest-observation comparison and the first
    capture's failure (e.g. CloseHandle) must never be skipped."""
    identity = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": os.getpid()}
    root = _registration_rig(monkeypatch, tmp_path, {4321: identity})
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 4321, expected=None, expected_creator=os.getpid())
    assert _read_owned(root) == []
    assert any("anchor missing" in str(e) for e in _obs_errors(root))


def test_ptysession_first_capture_failure_skips_registration(tmp_path, monkeypatch):
    """Root anchor capture returns None (first attempt failed): registration
    is never reached, no second capture can bypass it, no protocol input."""
    writes = []
    calls = {"record": 0, "capture": 0}

    class RecPty:
        pid = 4242
        def read(self, size=1024): return ""
        def isalive(self): return True
        def write(self, text): writes.append(text)
        def close(self): pass

    def fake_capture(pid):
        calls["capture"] += 1
        return None  # first (anchor) capture failed, e.g. CloseHandle failure

    monkeypatch.setattr(core, "_capture_registration_identity", fake_capture)
    monkeypatch.setattr(core, "record_owned_process",
                        lambda *a, **kw: calls.__setitem__("record", calls["record"] + 1))
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=RecPty())
    assert session.error_flag is not None and session.error_flag.startswith("root_ownership_failed")
    assert calls["record"] == 0, "registration must not run without a valid anchor"
    assert calls["capture"] == 1, "no bypass second capture"
    assert writes == []
    assert not hasattr(session, "_reader_thread")


def test_registration_identity_unavailable_raises(tmp_path, monkeypatch):
    """No single-channel capture (incl. CloseHandle failure): registration
    raises and records an observation error - never a ticks-None record that
    lets the protocol start without a cleanup-valid root."""
    identity = {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": os.getpid()}
    root = _registration_rig(monkeypatch, tmp_path, {4321: None})
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 4321, expected=identity, expected_creator=os.getpid())
    assert _read_owned(root) == []
    assert any("identity unavailable" in str(e) for e in _obs_errors(root))


def test_registration_descendant_of_verified_parent(tmp_path, monkeypatch):
    """Descendant: creator is a registered parent whose identity re-verifies
    NOW; child records the parent's verified ticks as creator_ticks."""
    parent_identity = {"create_ticks": 4000000, "exe": "parent.exe", "creator_pid": os.getpid()}
    child_identity = {"create_ticks": 5000000, "exe": "child.exe", "creator_pid": 4321}
    table = {4321: [1000.0, "parent.exe", ["parent"]], 5555: [1001.0, "child.exe", ["child"]]}
    root = _registration_rig(monkeypatch, tmp_path, {4321: parent_identity, 5555: child_identity},
                             table=table)
    core.record_owned_process(root, 4321, expected=parent_identity, expected_creator=os.getpid())
    core.record_owned_process(root, 5555, expected=child_identity)
    stored = _read_owned(root)
    assert stored[1]["creator_pid"] == 4321
    assert stored[1]["creator_ticks"] == 4000000
    assert core._valid_owned_entry(stored[1]) is True


def test_registration_descendant_parent_drift_fails_closed(tmp_path, monkeypatch):
    """Registered parent was recycled after registration (re-verification
    returns different ticks): the child is refused, not adopted."""
    parent_identity = {"create_ticks": 4000000, "exe": "parent.exe", "creator_pid": os.getpid()}
    drifted_parent = {"create_ticks": 4100000, "exe": "parent.exe", "creator_pid": os.getpid()}
    child_identity = {"create_ticks": 5000000, "exe": "child.exe", "creator_pid": 4321}
    table = {4321: [1000.0, "parent.exe", ["parent"]], 5555: [1001.0, "child.exe", ["child"]]}
    captures = {"count": 0}

    def fake_capture(pid):
        if pid == 4321:
            captures["count"] += 1
            return parent_identity if captures["count"] == 1 else drifted_parent
        return child_identity

    monkeypatch.setitem(sys.modules, "psutil", _RegistrationPsutil)
    _RegistrationPsutil.table = table
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "_capture_registration_identity", fake_capture)
    core.record_owned_process(tmp_path, 4321, expected=parent_identity, expected_creator=os.getpid())
    with pytest.raises(RuntimeError):
        core.record_owned_process(tmp_path, 5555, expected=child_identity)
    stored = _read_owned(tmp_path)
    assert [e["pid"] for e in stored] == [4321]


def test_registration_child_predates_parent_fails_closed(tmp_path, monkeypatch):
    parent_identity = {"create_ticks": 5000000, "exe": "parent.exe", "creator_pid": os.getpid()}
    child_identity = {"create_ticks": 4000000, "exe": "child.exe", "creator_pid": 4321}
    table = {4321: [1000.0, "parent.exe", ["parent"]], 5555: [999.0, "child.exe", ["child"]]}
    root = _registration_rig(monkeypatch, tmp_path, {4321: parent_identity, 5555: child_identity},
                             table=table)
    core.record_owned_process(root, 4321, expected=parent_identity, expected_creator=os.getpid())
    with pytest.raises(RuntimeError):
        core.record_owned_process(root, 5555, expected=child_identity)
    assert [e["pid"] for e in _read_owned(root)] == [4321]


def test_ptysession_no_protocol_input_when_root_ownership_fails(tmp_path, monkeypatch):
    """Regression: identity capture/registration failure raises BEFORE any
    protocol input - the session records root_ownership_failed, closes, the
    reader never starts, and zero bytes are written to the PTY."""
    writes = []

    class RecPty:
        pid = 4242
        def read(self, size=1024): return ""
        def isalive(self): return True
        def write(self, text): writes.append(text)
        def close(self): pass

    monkeypatch.setattr(core, "record_owned_process",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("capture failed")))
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=RecPty())
    assert session.error_flag is not None and session.error_flag.startswith("root_ownership_failed")
    assert writes == [], "no protocol input may be sent after ownership failure"
    assert not hasattr(session, "_reader_thread"), "reader starts only after ownership is valid"
    assert any("root_ownership_failed" in str(e) for e in _obs_errors(tmp_path))


@pytest.mark.skipif(os.name != "nt", reason="Windows handle semantics only verifiable on Windows")
def test_terminate_verified_real_os_handle_identity(tmp_path):
    """Real OS proof (no mocks) through the PRODUCTION registration path.

    A safe disposable child spawned by THIS test (not a phone/user process)
    is anchored immediately, registered via production record_owned_process,
    read back through strict metadata validation, then: ticks off by one
    refused with the child alive, wrong image refused alive, and production
    terminate_owned_processes kills exactly the recorded identity
    (child.wait confirms exit). The creator binding must report this test
    process as the parent."""
    import subprocess as sp
    child = sp.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        import psutil
        anchor = core._capture_registration_identity(child.pid)
        assert anchor is not None, "single-channel capture must succeed on the Windows gate"
        assert type(anchor["create_ticks"]) is int and anchor["create_ticks"] > 0
        assert anchor["creator_pid"] == os.getpid(), "subprocess child's creator is this test process"

        core.record_owned_process(tmp_path, child.pid, expected=anchor,
                                  expected_creator=os.getpid())
        read_back = {"errors": []}
        entries = core._read_strict_owned_metadata(tmp_path, read_back)
        assert entries and not read_back["errors"], read_back
        entry = entries[0]
        assert entry["creator_pid"] == os.getpid() and entry["create_ticks"] == anchor["create_ticks"]

        res_bad_ticks = {"errors": []}
        assert core._terminate_verified({**entry, "create_ticks": entry["create_ticks"] + 1},
                                        res_bad_ticks) is False
        assert psutil.pid_exists(child.pid), "wrong identity must never terminate"

        res_bad_exe = {"errors": []}
        assert core._terminate_verified({**entry, "exe": entry["exe"] + ".wrong"},
                                        res_bad_exe) is False
        assert psutil.pid_exists(child.pid)

        res = core.terminate_owned_processes(tmp_path, [0], "native_test", settle_seconds=0.5)
        assert res["terminated"] is True, res
        child.wait(timeout=10)
        assert child.poll() is not None, "exact recorded identity must terminate"
        receipt_dir = os.environ.get("NEXUS_Q1_NATIVE_RECEIPT_DIR")
        if receipt_dir:
            # Retain the bounded child receipt in the harness upload output,
            # only after the real termination outcome above has been asserted.
            (Path(receipt_dir) / "termination_native_test.json").write_bytes(
                (tmp_path / "output" / "termination_native_test.json").read_bytes())
    finally:
        if child.poll() is None:
            child.kill()


def test_terminate_verified_fails_closed_without_stable_handle(monkeypatch):
    """Non-Windows: no stable handle, no termination, explicit error."""
    monkeypatch.setattr(core, "_IS_WINDOWS", False)
    result = {"errors": []}
    assert core._terminate_verified({"pid": 1, "create_time": 1.0, "exe": "x"}, result) is False
    assert any("stable process handle unavailable" in e for e in result["errors"])


class _FakeKernel32:
    """Logic-level fake for the handle lifecycle (NOT OS proof): drives the
    real _terminate_verified control flow with scripted Win32 outcomes."""

    def __init__(self, close_ok=True, terminate_ok=True, ticks=1000000, image="ours.exe"):
        self.close_ok = close_ok
        self.terminate_ok = terminate_ok
        self.ticks = ticks
        self.image = image
        self.terminated = False
        self.closed = False

    def OpenProcess(self, access, inherit, pid):
        return 1234

    def GetProcessTimes(self, handle, creation, exit_time, kernel, user):
        creation._obj.dwHighDateTime = self.ticks >> 32
        creation._obj.dwLowDateTime = self.ticks & 0xFFFFFFFF
        return True

    def QueryFullProcessImageNameW(self, handle, flags, buf, size):
        buf.value = self.image
        return True

    def TerminateProcess(self, handle, code):
        self.terminated = True
        return self.terminate_ok

    def CloseHandle(self, handle):
        self.closed = True
        return self.close_ok


def _fake_windows(monkeypatch, fake):
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "_load_kernel32", lambda: fake)


def test_terminate_verified_logic_exact_identity_success(monkeypatch):
    """Mock-control-flow only: exact ticks + image -> terminate issued, True."""
    fake = _FakeKernel32()
    _fake_windows(monkeypatch, fake)
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}
    result = {"errors": []}
    assert core._terminate_verified(entry, result) is True
    assert fake.terminated and fake.closed
    assert result["errors"] == []


def test_terminate_verified_logic_ticks_mismatch_no_terminate(monkeypatch):
    """Ticks off by one: refused, no terminate issued, handle still closed."""
    fake = _FakeKernel32(ticks=1000001)
    _fake_windows(monkeypatch, fake)
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}
    result = {"errors": []}
    assert core._terminate_verified(entry, result) is False
    assert fake.terminated is False and fake.closed is True
    assert any("exact creation ticks differ" in e for e in result["errors"])


def test_terminate_verified_fails_closed_on_close_handle_failure(monkeypatch):
    """CloseHandle failure: diagnostics recorded, no success claim even though
    TerminateProcess was issued on the verified handle."""
    fake = _FakeKernel32(close_ok=False)
    _fake_windows(monkeypatch, fake)
    entry = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}
    result = {"errors": []}
    assert core._terminate_verified(entry, result) is False
    assert fake.terminated is True
    assert any("CloseHandle failed" in e for e in result["errors"])


class _FakeNtdll:
    """Scripted NtQueryInformationProcess (logic-level, NOT OS proof)."""

    def __init__(self, status=0, unique=4321, creator=4242, returned=None):
        self.status = status
        self.unique = unique
        self.creator = creator
        self.returned = returned

    def NtQueryInformationProcess(self, handle, info_class, buf, length, ret_len):
        import ctypes as _c
        info = buf._obj
        info.UniqueProcessId = self.unique
        info.InheritedFromUniqueProcessId = self.creator
        ret_len._obj.value = self.returned if self.returned is not None else _c.sizeof(info)
        return self.status


def test_read_creator_pid_success():
    fake = _FakeNtdll(status=0, unique=4321, creator=4242)
    assert core._read_creator_pid(fake, 1234, 4321) == 4242


def test_read_creator_pid_fails_closed_on_status():
    assert core._read_creator_pid(_FakeNtdll(status=-1073741823), 1234, 4321) is None


def test_read_creator_pid_fails_closed_on_short_return():
    """A genuinely short ReturnLength (one byte under the native full size;
    24 would be a VALID full x86 read, not short) fails closed."""
    import ctypes as _c
    full = _c.sizeof(core._process_basic_information_type())
    assert core._read_creator_pid(_FakeNtdll(returned=full - 1), 1234, 4321) is None


def test_read_creator_pid_fails_closed_on_layout_drift(monkeypatch):
    monkeypatch.setattr(core, "_pbi_layout_supported", lambda pbi: False)
    assert core._read_creator_pid(_FakeNtdll(), 1234, 4321) is None


def test_read_creator_pid_fails_closed_on_unique_mismatch():
    """UniqueProcessId not equal to the opened pid: wrong process or shifted
    struct - never trust the creator field."""
    assert core._read_creator_pid(_FakeNtdll(unique=9999), 1234, 4321) is None


def test_read_creator_pid_fails_closed_on_nonpositive_creator():
    assert core._read_creator_pid(_FakeNtdll(creator=0), 1234, 4321) is None


def test_capture_registration_identity_single_channel(monkeypatch):
    """All three identity parts come through ONE handle; CloseHandle failure
    voids the whole capture."""
    fake_k = _FakeKernel32(ticks=5000000, image="ours.exe")
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "_load_kernel32", lambda: fake_k)
    monkeypatch.setattr(core, "_load_ntdll", lambda: _FakeNtdll(unique=4321, creator=4242))
    identity = core._capture_registration_identity(4321)
    assert identity == {"create_ticks": 5000000, "exe": "ours.exe", "creator_pid": 4242}

    monkeypatch.setattr(core, "_load_kernel32", lambda: _FakeKernel32(close_ok=False, ticks=5000000))
    assert core._capture_registration_identity(4321) is None, "unaccounted handle lifecycle voids capture"


def test_valid_owned_entry_strict_shapes():
    good = {"pid": 4321, "create_time": 1000.0, "create_ticks": 1000000, "creator_pid": 4242, "exe": "ours.exe"}
    assert core._valid_owned_entry(good) is True
    assert core._valid_owned_entry({**good, "pid": True}) is False
    assert core._valid_owned_entry({**good, "pid": "4321"}) is False
    assert core._valid_owned_entry({**good, "pid": 0}) is False
    assert core._valid_owned_entry({**good, "create_time": True}) is False
    assert core._valid_owned_entry({**good, "create_time": float("nan")}) is False
    assert core._valid_owned_entry({**good, "create_time": float("inf")}) is False
    assert core._valid_owned_entry({**good, "create_time": -1}) is False
    assert core._valid_owned_entry({**good, "exe": ""}) is False
    assert core._valid_owned_entry({**good, "create_ticks": True}) is False
    assert core._valid_owned_entry({**good, "create_ticks": 0}) is False
    assert core._valid_owned_entry({**good, "create_ticks": -1}) is False
    assert core._valid_owned_entry({**good, "create_ticks": 1000.5}) is False
    assert core._valid_owned_entry({**good, "create_ticks": "1000000"}) is False
    assert core._valid_owned_entry({k: v for k, v in good.items() if k != "exe"}) is False
    assert core._valid_owned_entry({k: v for k, v in good.items() if k != "create_ticks"}) is False
    assert core._valid_owned_entry({k: v for k, v in good.items() if k != "creator_pid"}) is False
    assert core._valid_owned_entry({**good, "creator_pid": True}) is False
    assert core._valid_owned_entry({**good, "creator_pid": 0}) is False
    assert core._valid_owned_entry({**good, "creator_pid": "4242"}) is False
    assert core._valid_owned_entry({**good, "creator_ticks": None}) is True
    assert core._valid_owned_entry({**good, "creator_ticks": 4000000}) is True
    assert core._valid_owned_entry({**good, "creator_ticks": True}) is False
    assert core._valid_owned_entry({**good, "creator_ticks": 0}) is False
    assert core._valid_owned_entry({**good, "creator_ticks": 1.5}) is False


def _protocol(n):
    return [{"name": f"row-{i:02d}", "kind": "model", "prompt": "p", "timeout": 1.0} for i in range(n)]


def _rows(protocol):
    return [{"name": r["name"], "task": {"status": assess.NOT_RUN}} for r in protocol]


def test_orchestration_success_handoff_and_row_preservation():
    """Blocked row preserved; verified termination precedes relaunch; remaining rows run in order."""
    protocol = _protocol(4)
    rows = _rows(protocol)
    calls = []

    def run_session(num, start):
        calls.append(("run", num, start))
        if num == 1:
            rows[0]["task"] = {"status": assess.PASS}
            rows[1]["task"] = {"status": assess.INCOMPLETE, "evidence": ["consent_wait_coverage_gap"]}
            return False, 2
        for r in range(start, len(rows)):
            rows[r]["task"] = {"status": assess.PASS}
        return True, None

    def terminate(num):
        calls.append(("terminate", num))
        return {"terminated": True}

    assert core.orchestrate_task_sessions(run_session, terminate, protocol, rows) is True
    assert calls == [("run", 1, 0), ("terminate", 2), ("run", 2, 1)] or calls == [("run", 1, 0), ("terminate", 2), ("run", 2, 2)]
    assert calls[1] == ("terminate", 2) and calls[2][0] == "run", "termination must precede relaunch"
    assert rows[0]["task"]["status"] == assess.PASS
    assert rows[1]["task"]["status"] == assess.INCOMPLETE  # coverage-gap row preserved
    assert all(r["task"]["status"] == assess.PASS for r in rows[2:])


def test_orchestration_termination_failure_blocks_relaunch():
    protocol = _protocol(4)
    rows = _rows(protocol)
    calls = []

    def run_session(num, start):
        calls.append(("run", num, start))
        rows[0]["task"] = {"status": assess.INCOMPLETE, "evidence": ["consent_wait_coverage_gap"]}
        return False, 1

    def terminate(num):
        calls.append(("terminate", num))
        return {"terminated": False, "errors": ["survivor"]}

    assert core.orchestrate_task_sessions(run_session, terminate, protocol, rows) is False
    assert calls == [("run", 1, 0), ("terminate", 2)], "no relaunch without verified termination"
    assert rows[0]["task"]["status"] == assess.INCOMPLETE
    for r in rows[1:]:
        assert r["task"]["status"] == assess.INCOMPLETE
        assert any("could not be stopped" in str(e) for e in r["task"]["evidence"])


def test_orchestration_cap_exhaustion_marks_remaining_in_order():
    protocol = _protocol(6)
    rows = _rows(protocol)
    calls = []

    def run_session(num, start):
        calls.append(("run", num, start))
        rows[start]["task"] = {"status": assess.INCOMPLETE, "evidence": ["consent_wait_coverage_gap"]}
        return False, start + 1

    def terminate(num):
        calls.append(("terminate", num))
        return {"terminated": True}

    assert core.orchestrate_task_sessions(run_session, terminate, protocol, rows, max_sessions=3) is False
    assert calls == [("run", 1, 0), ("terminate", 2), ("run", 2, 1), ("terminate", 3), ("run", 3, 2)]
    for r in rows[:3]:
        assert r["task"]["status"] == assess.INCOMPLETE
        assert any("consent_wait_coverage_gap" in str(e) for e in r["task"]["evidence"])
    for r in rows[3:]:
        assert any("restart limit" in str(e) for e in r["task"]["evidence"])
    assert [r["name"] for r in rows] == [p["name"] for p in protocol], "row order preserved"


def test_orchestration_should_stop_prevents_relaunch():
    protocol = _protocol(3)
    rows = _rows(protocol)
    calls = []

    def run_session(num, start):
        calls.append(("run", num, start))
        return False, 1

    assert core.orchestrate_task_sessions(run_session, lambda n: calls.append(("t", n)),
                                          protocol, rows, should_stop=lambda: True) is False
    assert calls == [], "stop condition must prevent even the first launch"


def test_static_verification_has_no_execution_primitives():
    """verify_calculator_static must contain no execution primitive (source guard)."""
    src = (Path(__file__).resolve().parents[1] / ".github/qualification/interactive_core.py").read_text(encoding="utf-8")
    func_src = src[src.index("def verify_calculator_static"):src.index("class PtySession")]
    for forbidden in ("subprocess", "exec(", "eval(", "compile(", "__import__", "runpy", "importlib", "Popen"):
        assert forbidden not in func_src, f"static verification must not contain {forbidden}"


def test_verify_calculator_static_rejects_non_file(tmp_path):
    workspace = tmp_path / "calc_ws"
    workspace.mkdir()
    (workspace / "calculator.py").mkdir()  # directory, not a file
    facts = core.verify_calculator_static(workspace, "def add(a, b):\n    return a + b\n")
    assert facts == {"syntax_ok": False, "add_preserved": False, "multiply_changed": False}


def test_assess_task_web_search_evidence_matching_and_outage():
    url = "https://github.com/ggerganov/llama.cpp/releases/tag/b4100"
    evidence = f"Latest release tag b4100 available at {url}"
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "call_search_1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_search_1", "name": "search_web", "content": evidence},
    ]
    ans_grounded = f"The latest release is b4100 at {url}"
    res_ok = assess.assess_task_web_search(ans_grounded, tool_msgs)
    assert res_ok["task"]["status"] == assess.REVIEW
    assert res_ok["reply"]["status"] == assess.REVIEW
    assert res_ok["network"]["status"] == assess.PASS

    # Outage with error context and honest report
    outage_evidence = "ERROR: connection failed (HTTP 503 Service Unavailable)"
    tool_outage = [
        {"role": "assistant", "tool_calls": [{"id": "call_search_2", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_search_2", "name": "search_web", "content": outage_evidence},
    ]
    ans_honest = "I cannot search the web due to an HTTP 503 connection error."
    res_outage = assess.assess_task_web_search(ans_honest, tool_outage)
    assert res_outage["network"]["status"] == assess.ENVIRONMENT_BLOCKED
    assert res_outage["task"]["status"] == assess.ENVIRONMENT_BLOCKED


def test_assess_compact_recall_honest_unknown():
    res_ok = assess.assess_compact_recall("The codename is Amber-Kite-731.", "Amber-Kite-731")
    assert res_ok["task"]["status"] == assess.REVIEW
    assert res_ok["reply"]["status"] == assess.REVIEW

    res_honest = assess.assess_compact_recall("I do not recall what codename was mentioned.", "Amber-Kite-731")
    assert res_honest["task"]["status"] == assess.FAIL
    assert res_honest["reply"]["status"] == assess.PASS


def test_assess_task_brief_original_failure_task_fail():
    """Verify original actual failure ('I'll read the file...') with no tool call yields task FAIL."""
    assistant_ans = "I'll read the file to get a summary."
    delta_msgs = [
        {"role": "user", "content": "Read brief.txt and summarize it in four short bullet points."},
        {"role": "assistant", "content": assistant_ans},
    ]
    res = assess.assess_task_brief(assistant_ans, delta_msgs)
    assert res["name"] == "compact-turn-02-brief"
    assert res["task"]["status"] == assess.FAIL
    assert res["task"]["status"] != assess.PASS
    assert res["reply"]["status"] == assess.REVIEW
    evidence = res["task"]["evidence"][0]
    assert evidence["read_bound"] is False
    assert evidence["four_substantive_items"] is False
    assert evidence["assistant_answer"] == assistant_ans
    assert evidence["read_evidence"] == []


def test_assess_task_brief_four_bullets_without_read_task_fail():
    """Verify four substantive bullet points without bound successful read_file brief.txt yields task FAIL."""
    bullets = (
        "- Infrastructure migration completed across all primary clusters.\n"
        "- Database latency decreased by 18% with connection pooling.\n"
        "- Soak test detected three edge cases in background sync worker.\n"
        "- Anomalies mitigated by adjusting timeout thresholds and pool limits."
    )
    res = assess.assess_task_brief(bullets, [])
    assert res["name"] == "compact-turn-02-brief"
    assert res["task"]["status"] == assess.FAIL
    assert res["task"]["status"] != assess.PASS
    assert res["reply"]["status"] == assess.REVIEW
    evidence = res["task"]["evidence"][0]
    assert evidence["read_bound"] is False
    assert evidence["four_substantive_items"] is True
    assert len(evidence["bullet_items"]) == 4


@pytest.mark.parametrize("executed,refusal,status", [
    (True, None, assess.REVIEW),
    (False, None, assess.FAIL),
    (True, "capacity", assess.FAIL),
])
def test_assess_task_brief_read_and_substantive_four_items_review_not_pass(executed, refusal, status):
    """Verify successful bound read_file brief.txt and four substantive items yields REVIEW, never PASS."""
    brief_content = (
        "Harbor Desk Deployment Summary - Release Cycle 2026-Q1\n"
        "The infrastructure migration to containerized worker nodes has completed across all four primary clusters.\n"
        "Database latency decreased by 18% following the implementation of connection pooling and read-replica routing.\n"
        "During the staging soak test, three edge cases were detected in the background sync worker:\n"
        "1. Token expiration during long-running batch jobs resulted in transient HTTP 401 retries.\n"
        "2. File descriptors exceeded the soft limit under concurrent payload downloads exceeding 50 MB.\n"
        "3. Cache invalidation events occasionally lagged behind replica write confirmations by 150 milliseconds.\n"
        "All three anomalies were mitigated by adjusting timeout thresholds and pool limits.\n"
    )
    tool_msgs = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_brief_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "brief.txt"})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_brief_1", "name": "read_file", "content": brief_content,
         "metadata": {"executed": executed, "refusal_class": refusal}},
    ]
    bullets = (
        "Here is the deployment brief summary in four bullet points:\n"
        "1. Infrastructure migration to containerized worker nodes has completed across four primary clusters.\n"
        "2. Database latency dropped 18% following implementation of connection pooling and read-replica routing.\n"
        "3. Three edge cases detected in background sync worker during staging soak test.\n"
        "4. All anomalies mitigated by adjusting timeout thresholds and connection pool limits."
    )
    res = assess.assess_task_brief(bullets, tool_msgs)
    assert res["name"] == "compact-turn-02-brief"
    assert res["task"]["status"] == status
    assert res["task"]["status"] != assess.PASS
    assert res["reply"]["status"] == assess.REVIEW
    assert res["reply"]["status"] != assess.PASS
    evidence = res["task"]["evidence"][0]
    assert evidence["read_bound"] is (status == assess.REVIEW)
    assert evidence["four_substantive_items"] is True
    assert len(evidence["bullet_items"]) == 4
    assert evidence["read_evidence"] == [brief_content]
    assert evidence["assistant_answer"] == bullets




def test_assess_task_brief_placeholders_and_echoes_fail():
    """Verify empty bullets, placeholder bullets, prompt echoes, and non-4 item counts fail closed even with read_file."""
    brief_content = "Some brief content"
    tool_msgs = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "brief.txt"})},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": brief_content,
         "metadata": {"executed": True}},
    ]
    # Placeholders like Point 1, Point 2... (only 1 alphabetic word per item)
    placeholders = "- Point 1\n- Point 2\n- Point 3\n- Point 4"
    res_ph = assess.assess_task_brief(placeholders, tool_msgs)
    assert res_ph["task"]["status"] == assess.FAIL
    assert res_ph["task"]["evidence"][0]["four_substantive_items"] is False

    # Prompt echo
    echo = "1. Read brief.txt\n2. and summarize it\n3. in four\n4. short bullet points"
    res_echo = assess.assess_task_brief(echo, tool_msgs)
    assert res_echo["task"]["status"] == assess.FAIL
    assert res_echo["task"]["evidence"][0]["four_substantive_items"] is False

    # Empty bullets
    empty = "- \n- \n- \n- "
    res_empty = assess.assess_task_brief(empty, tool_msgs)
    assert res_empty["task"]["status"] == assess.FAIL
    assert res_empty["task"]["evidence"][0]["four_substantive_items"] is False

    # Exactly four required: 5 items (even if substantive) must fail
    five_items = (
        "- Infrastructure migration completed across primary clusters.\n"
        "- Database latency decreased by 18 percent.\n"
        "- Soak test detected three background sync edge cases.\n"
        "- Timeout thresholds and pool limits adjusted.\n"
        "- Extra fifth item summary note."
    )
    res_five = assess.assess_task_brief(five_items, tool_msgs)
    assert res_five["task"]["status"] == assess.FAIL
    assert res_five["task"]["evidence"][0]["four_substantive_items"] is False
    four_with_empty_fifth = five_items.rsplit("\n", 1)[0] + "\n- "
    assert assess.assess_task_brief(four_with_empty_fifth, tool_msgs)["task"]["status"] == assess.FAIL


def test_summarize_axes_failed_or_review_brief_blocks_automated_core_pass():
    """Verify failed or review brief row blocks automated core pass even when all other required rows pass."""
    rows = [{"name": name, "task": {"status": assess.PASS}, "reply": {"status": assess.PASS}} for name in assess.REQUIRED_TASKS]
    cleanup_axis = {"status": assess.PASS}
    tty_axis = {"status": assess.PASS}

    # Case 1: compact-turn-02-brief failed -> task axis FAIL, automated core pass False, qualified False
    for r in rows:
        if r["name"] == "compact-turn-02-brief":
            r["task"]["status"] = assess.FAIL
            r["reply"]["status"] = assess.REVIEW
    res_fail = assess.summarize_axes(rows, cleanup_axis, tty_axis, True, True, False)
    assert res_fail["axes"]["task"] == assess.FAIL
    assert res_fail["automated_core_pass"] is False
    assert res_fail["qualified"] is False

    # Case 2: compact-turn-02-brief in REVIEW -> task axis REVIEW, automated core pass False, independent review required
    for r in rows:
        if r["name"] == "compact-turn-02-brief":
            r["task"]["status"] = assess.REVIEW
            r["reply"]["status"] = assess.REVIEW
    res_review = assess.summarize_axes(rows, cleanup_axis, tty_axis, True, True, False)
    assert res_review["axes"]["task"] == assess.REVIEW
    assert res_review["automated_core_pass"] is False
    assert res_review["independent_review_required"] is True
    assert res_review["qualified"] is False

def test_cleanup_injection_via_sys_modules(tmp_path, monkeypatch):
    """Verify cleanup_root respects exact create_time and unconfirmed on missing psutil."""
    out = tmp_path / "output"
    out.mkdir()
    owned = [{"pid": 999999, "create_time": 0.0, "exe": "dummy", "cmdline": []}]
    (out / "owned_processes.json").write_text(json.dumps(owned), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "psutil", None)
    res = core.cleanup_root(tmp_path)
    assert res["confirmed"] is False


def test_protocol_timeout_and_remaining_not_run(tmp_path, monkeypatch):
    """Verify _execute_protocol_session marks timed out step INCOMPLETE and remaining NOT_RUN."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    fake_pty = FakePtyProcess([])  # Empty chunks => will time out
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake_pty)
    try:
        tasks_rows = [{"name": r["name"], "task": {"status": assess.NOT_RUN}} for r in core.PROTOCOL_TASKS]
        completed, restart_idx = core._execute_protocol_session(
            core.PROTOCOL_TASKS, "fake_session", session, tmp_path, tmp_path, tmp_path / "history.db",
            deadline=time.monotonic() + 0.05, is_tasks=True, output_dir=tmp_path / "output",
            target_rows=tasks_rows, byte_tracker=[0]
        )
        assert completed is False
        assert restart_idx is None
        assert tasks_rows[0]["task"]["status"] == assess.INCOMPLETE
        assert all(r["task"]["status"] == assess.NOT_RUN for r in tasks_rows[1:])
    finally:
        session.close()




CONSENT_BLOCK = (
    "\nNexus wants to run pytest in:\n  /tmp/ws\n"
    "This runs the project's own code on your computer, with your permissions, not sandboxed.\n"
    "  [o] once  - this folder, until Nexus exits (nothing saved)\n"
    "  [a] always - this exact folder only, saved; undo with `nexus-agent trust revoke`\n"
    "  [N] no\nAllow? [o/a/N]: "
)


def test_consent_block_is_coverage_gap_and_never_answered(tmp_path, monkeypatch):
    """The full trust-prompt block yields consent_wait promptly; the harness writes nothing."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    writes = []
    fake = FakePtyProcess(["working...\r\n", CONSENT_BLOCK])
    monkeypatch.setattr(fake, "write", lambda text: writes.append(text))
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        start = time.monotonic()
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=30.0, deadline=time.monotonic() + 30.0)
        assert consent_wait is True
        assert ready is False
        assert time.monotonic() - start < 5.0, "must not wait out the row timeout"
        assert "Allow? [o/a/N]:" in buf, "full prompt must stay in row evidence"
        assert writes == [], "harness must never answer a consent prompt"
    finally:
        session.close()


def test_spoofed_or_partial_consent_text_is_not_detected(tmp_path, monkeypatch):
    """A bare suffix or partial block is not a consent wait; even a fully spoofed
    block could only cause a fail-safe coverage gap, never execution."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    writes = []
    fake = FakePtyProcess(["model prose ending in Allow? [o/a/N]: "])
    monkeypatch.setattr(fake, "write", lambda text: writes.append(text))
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=0.2, deadline=time.monotonic() + 0.2)
        assert consent_wait is False
        assert ready is False
        assert writes == []
    finally:
        session.close()

    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    fake2 = FakePtyProcess(["Nexus wants to run pytest in:\n  /tmp/ws\n  [o] once\n>> You: "])
    session2 = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake2)
    try:
        buf2, ready2, consent_wait2 = session2.wait_for_ready_prompt(timeout=2.0, deadline=time.monotonic() + 5.0)
        assert consent_wait2 is False
        assert ready2 is True
    finally:
        session2.close()


def test_consent_wait_row_recorded_and_restart_idx_returned(tmp_path, monkeypatch):
    """A consent-blocked row is INCOMPLETE with an explicit coverage-gap marker,
    and the caller gets the next row index for a fresh session."""
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    out_dir = tmp_path / "output"
    (out_dir / "rows").mkdir(parents=True)
    fake = FakePtyProcess([CONSENT_BLOCK])
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        tasks_rows = [{"name": r["name"], "task": {"status": assess.NOT_RUN}} for r in core.PROTOCOL_TASKS]
        completed, restart_idx = core._execute_protocol_session(
            core.PROTOCOL_TASKS, "fake_session", session, tmp_path, tmp_path, tmp_path / "history.db",
            deadline=time.monotonic() + 10.0, is_tasks=True, output_dir=out_dir,
            target_rows=tasks_rows, byte_tracker=[0]
        )
        assert completed is False
        assert restart_idx == 1, "rows after the consent-blocked one continue in a fresh session"
        row0 = tasks_rows[0]["task"]
        assert row0["status"] == assess.INCOMPLETE
        assert row0["status"] != assess.PASS
        assert any("consent_wait_coverage_gap" in str(e) for e in row0["evidence"])
        assert all(r["task"]["status"] == assess.NOT_RUN for r in tasks_rows[1:])
        assert (out_dir / "rows" / "fake_session_00_row-01-readme.json").exists()
    finally:
        session.close()

def test_child_environment_disables_run_code_for_probe(tmp_path):
    """Probe REPL children inherit the run_code refusal flag (Windows has no rlimits/Job Objects)."""
    env = core.child_environment(tmp_path / "home", tmp_path / "tmp", tmp_path / "ws")
    assert env["NEXUS_AGENT_DISABLE_RUN_CODE"] == "1"


def test_deadline_from_file(tmp_path):
    valid_data = {
        "work_deadline_utc": time.time() + 3600,
        "job_limit_minutes": 120,
        "reserve_minutes": 10,
    }
    df = tmp_path / "deadline.json"
    df.write_text(json.dumps(valid_data), encoding="utf-8")
    mono = core.deadline_from_file(df)
    assert mono > time.monotonic()




def test_check_evidence_cap(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "file.txt").write_bytes(b"a" * 1024)
    total = core.check_evidence_cap(out, cap=core.EVIDENCE_CAP, reserve=core.MANIFEST_RESERVE)
    assert total == 1024

    with pytest.raises(RuntimeError, match="Evidence output cap exceeded"):
        core.check_evidence_cap(out, cap=1500, reserve=1000)


def test_reader_cap_is_incomplete_not_silent_success(tmp_path, monkeypatch):
    mock_successful_ownership(monkeypatch, tmp_path, 4242)
    fake = FakePtyProcess(["extra\n", ">> You: "])
    tracker = [core.EVIDENCE_CAP - core.MANIFEST_RESERVE]
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake, byte_tracker=tracker)
    try:
        _, ready, consent_wait = session.wait_for_ready_prompt(0.2, time.monotonic() + 0.2)
        assert not ready
        assert consent_wait is False
        assert session.error_flag == "output_cap_exceeded"
    finally:
        session.close()


def test_unrun_and_environment_blocked_are_not_product_failure():
    rows = [{"name": name, "task": {"status": assess.NOT_RUN}} for name in assess.REQUIRED_TASKS]
    result = assess.summarize_axes(rows, {"status": assess.PASS}, {"status": assess.PASS}, False, False, False)
    assert result["axes"]["task"] == assess.INCOMPLETE
    assert result["automated_core_pass"] is False
    for row in rows:
        row["task"]["status"] = assess.PASS
    rows[-1]["task"]["status"] = assess.ENVIRONMENT_BLOCKED
    result = assess.summarize_axes(rows, {"status": assess.PASS}, {"status": assess.PASS}, True, True, False)
    assert result["axes"]["task"] == assess.ENVIRONMENT_BLOCKED
    assert result["qualified"] is False


def test_regression_ownership_fail_closed_on_unrecorded_root_and_observation_failure(tmp_path, monkeypatch):
    """Verify unrecorded root PID fails closed, halts ready prompt, and rejects send_input."""
    fake = FakePtyProcess([">> You: "], pid=0)
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        assert session.error_flag is not None
        assert "root_ownership_failed" in session.error_flag
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=0.1, deadline=time.monotonic() + 0.1)
        assert ready is False
        assert consent_wait is False
        with pytest.raises(RuntimeError, match="Cannot send input"):
            session.send_input("test\n")
    finally:
        session.close()

    cleanup_res = core.cleanup_root(tmp_path)
    assert cleanup_res["confirmed"] is False
    assert any("root_ownership_failed" in err for err in cleanup_res["errors"])
    axis = assess.assess_cleanup([], cleanup_res["killed_pids"], cleanup_res["remaining_llama_servers"],
                                 cleanup_res["confirmed"], cleanup_res["errors"])
    assert axis["status"] == assess.INCOMPLETE


def test_regression_descendant_observation_failure_halts_session(tmp_path, monkeypatch):
    """Verify descendant observation scanning failure propagates error_flag, stops waiter, and rejects input."""
    mock_successful_ownership(monkeypatch, tmp_path, pid=4242)
    fake = FakePtyProcess([">> You: "], pid=4242)
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        def _failing_scan(root, pid):
            core.record_observation_error(root, "descendant scan failed")
            raise RuntimeError("descendant scan failed")

        monkeypatch.setattr(core, "scan_and_record_descendants", _failing_scan)
        buf, ready, consent_wait = session.wait_for_ready_prompt(timeout=0.2, deadline=time.monotonic() + 0.2)
        assert ready is False
        assert consent_wait is False
        assert "descendant_observation_error" in (session.error_flag or "")
        with pytest.raises(RuntimeError, match="Cannot send input"):
            session.send_input("test\n")
    finally:
        session.close()

    cleanup_res = core.cleanup_root(tmp_path)
    assert cleanup_res["confirmed"] is False
    assert any("descendant" in err for err in cleanup_res["errors"])


def test_regression_missing_owned_metadata_not_confirmed(tmp_path):
    """Verify missing owned_processes.json fails closed and never confirms empty success."""
    res = core.cleanup_root(tmp_path)
    assert res["confirmed"] is False
    assert "missing owned metadata" in res["errors"]
    axis = assess.assess_cleanup([], res["killed_pids"], res["remaining_llama_servers"], res["confirmed"], res["errors"])
    assert axis["status"] == assess.INCOMPLETE


def test_regression_search_mismatched_tag_url_cannot_pass():
    """Verify search does not PASS from independent tag/url intersections; requires REVIEW."""
    evidence = (
        "Release b1234 notes: see https://github.com/ggerganov/llama.cpp/releases/tag/b1234\n"
        "Previous release was b9999 at https://github.com/ggerganov/llama.cpp/releases/tag/b9999"
    )
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search_web", "content": evidence},
    ]
    ans_mismatched = "The latest release is b1234 at https://github.com/ggerganov/llama.cpp/releases/tag/b9999"
    res = assess.assess_task_web_search(ans_mismatched, tool_msgs)
    assert res["task"]["status"] != assess.PASS
    assert res["task"]["status"] == assess.REVIEW
    assert res["reply"]["status"] == assess.REVIEW


def test_regression_uncertain_seed_mention_cannot_pass():
    """Verify assistant mentioning seed with uncertainty does not PASS; requires REVIEW."""
    ans_uncertain = "I cannot remember whether it was Amber-Kite-731 or something else."
    res = assess.assess_compact_recall(ans_uncertain, "Amber-Kite-731")
    assert res["task"]["status"] != assess.PASS
    assert res["task"]["status"] == assess.REVIEW
    assert res["reply"]["status"] == assess.REVIEW


@pytest.mark.parametrize("observation", ["inspection_failure", "zero_create_time"])
def test_regression_positive_root_without_observed_identity_cannot_feed(tmp_path, monkeypatch, observation):
    from types import SimpleNamespace

    def create_time():
        if observation == "inspection_failure":
            raise OSError("root inspection failed")
        return 0.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        Process=lambda pid: SimpleNamespace(create_time=create_time)))
    fake = FakePtyProcess([">> You: "], pid=4242)
    monkeypatch.setattr(fake, "write", lambda text: pytest.fail("Prompt fed without root identity"))
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        assert "root_ownership_failed" in session.error_flag
        with pytest.raises(RuntimeError, match="Cannot send input"):
            session.send_input("must not run\r")
        assert not (tmp_path / "output/owned_processes.json").exists()
        errors = json.loads((tmp_path / "output/observation_errors.json").read_text())
        assert any("root_ownership_failed" in message for message in errors)
    finally:
        session.close()


def _conpty_deadline(tmp_path):
    f = tmp_path / "deadline.json"
    f.write_text(json.dumps({"work_deadline_utc": time.time() + 600,
                             "job_limit_minutes": 120, "reserve_minutes": 10}))
    return f


@pytest.mark.skipif(os.name == "nt", reason="off-Windows refusal is only exercisable off Windows")
def test_conpty_check_refuses_off_windows(tmp_path):
    """Non-Windows: explicit refusal, evidence retained, no model/qualification flags."""
    code = core.main(["conpty-check", "--root", str(tmp_path / "root"),
                      "--deadline-file", str(_conpty_deadline(tmp_path))])
    assert code == 1
    evidence = json.loads((tmp_path / "root" / "output" / "conpty-compat.json").read_text())
    assert evidence["conpty_compatibility_only"] is True
    assert evidence["interactive_qualification"] is False
    assert evidence["final_qualification"] is False
    assert evidence["model_execution_permitted"] is False
    assert evidence["network_used"] is False
    assert any("Windows" in e for e in evidence["errors"])


def test_conpty_check_refuses_without_pywinpty(tmp_path, monkeypatch):
    """Windows without pywinpty: fail closed before any spawn."""
    monkeypatch.setattr(core, "_IS_WINDOWS", True)
    monkeypatch.setattr(core, "PtyProcess", None)
    code = core.main(["conpty-check", "--root", str(tmp_path / "root"),
                      "--deadline-file", str(_conpty_deadline(tmp_path))])
    assert code == 1
    evidence = json.loads((tmp_path / "root" / "output" / "conpty-compat.json").read_text())
    assert any("pywinpty" in e for e in evidence["errors"])
    assert evidence["checks"] == {}
    assert evidence["termination_receipt_sha256"] is None
