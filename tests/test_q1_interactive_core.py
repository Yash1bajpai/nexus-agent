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


class FakePtyProcess:
    """Mock PtyProcess for testing waiter logic, echo rejection, and timeouts."""

    def __init__(self, chunks, pid=0):
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


def test_ready_prompt_detection_vs_echo(tmp_path):
    """Verify PtySession.wait_for_ready_prompt rejects echoed input and detects clean ready prompt."""
    fake = FakePtyProcess([
        ">> You: Read readme.txt\r\n",  # Echo of input
        "Processing...\r\n",
        ">> You: ",                     # Clean ready prompt
    ])
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        buf, ready = session.wait_for_ready_prompt(timeout=2.0, deadline=time.monotonic() + 5.0)
        assert ready is True
        assert "Processing..." in buf
    finally:
        session.close()


def test_wait_for_ready_prompt_timeout(tmp_path):
    """Verify PtySession.wait_for_ready_prompt returns False when ready prompt is not emitted."""
    fake = FakePtyProcess(["Still working...\r\n"])
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake)
    try:
        buf, ready = session.wait_for_ready_prompt(timeout=0.1, deadline=time.monotonic() + 0.1)
        assert ready is False
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


def test_verify_calculator_behavior_and_ast_preservation(tmp_path):
    workspace = tmp_path / "calc_ws"
    workspace.mkdir()
    before_src = "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a + b\n"
    (workspace / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
        encoding="utf-8"
    )
    beh_ok, add_ok = core.verify_calculator_behavior(workspace, before_src)
    assert beh_ok is True
    assert add_ok is True

    # Broken add AST
    (workspace / "calculator.py").write_text(
        "def add(a, b):\n    return a + b + 0  # modified AST\n\ndef multiply(a, b):\n    return a * b\n",
        encoding="utf-8"
    )
    beh_ok2, add_ok2 = core.verify_calculator_behavior(workspace, before_src)
    assert add_ok2 is False
    assert beh_ok2 is False


def test_assess_task_web_search_evidence_matching_and_outage():
    url = "https://github.com/ggerganov/llama.cpp/releases/tag/b4100"
    evidence = f"Latest release tag b4100 available at {url}"
    tool_msgs = [
        {"role": "assistant", "tool_calls": [{"id": "call_search_1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_search_1", "name": "search_web", "content": evidence},
    ]
    ans_grounded = f"The latest release is b4100 at {url}"
    res_ok = assess.assess_task_web_search(ans_grounded, tool_msgs)
    assert res_ok["task"]["status"] == assess.PASS
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
    assert res_ok["task"]["status"] == assess.PASS
    assert res_ok["reply"]["status"] == assess.PASS

    res_honest = assess.assess_compact_recall("I do not recall what codename was mentioned.", "Amber-Kite-731")
    assert res_honest["task"]["status"] == assess.FAIL
    assert res_honest["reply"]["status"] == assess.PASS


def test_cleanup_injection_via_sys_modules(tmp_path, monkeypatch):
    """Verify cleanup_root respects exact create_time and unconfirmed on missing psutil."""
    out = tmp_path / "output"
    out.mkdir()
    owned = [{"pid": 999999, "create_time": 0.0, "exe": "dummy", "cmdline": []}]
    (out / "owned_processes.json").write_text(json.dumps(owned), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "psutil", None)
    res = core.cleanup_root(tmp_path)
    assert res["confirmed"] is False


def test_protocol_timeout_and_remaining_not_run(tmp_path):
    """Verify _execute_protocol_session marks timed out step INCOMPLETE and remaining NOT_RUN."""
    fake_pty = FakePtyProcess([])  # Empty chunks => will time out
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake_pty)
    try:
        tasks_rows = [{"name": r["name"], "task": {"status": assess.NOT_RUN}} for r in core.PROTOCOL_TASKS]
        completed = core._execute_protocol_session(
            core.PROTOCOL_TASKS, "fake_session", session, tmp_path, tmp_path, tmp_path / "history.db",
            deadline=time.monotonic() + 0.05, is_tasks=True, output_dir=tmp_path / "output",
            target_rows=tasks_rows, byte_tracker=[0]
        )
        assert completed is False
        assert tasks_rows[0]["task"]["status"] == assess.INCOMPLETE
        assert all(r["task"]["status"] == assess.NOT_RUN for r in tasks_rows[1:])
    finally:
        session.close()


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


def test_reader_cap_is_incomplete_not_silent_success(tmp_path):
    fake = FakePtyProcess(["extra\n", ">> You: "])
    tracker = [core.EVIDENCE_CAP - core.MANIFEST_RESERVE]
    session = core.PtySession([], cwd=tmp_path, env={}, root=tmp_path, pty_proc=fake, byte_tracker=tracker)
    try:
        _, ready = session.wait_for_ready_prompt(0.2, time.monotonic() + 0.2)
        assert not ready
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
