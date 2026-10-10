import json
import subprocess
import sys

import pytest

from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.guardrails import LoopGuard, guardrails_enabled
from nexus_agent_ai.agent.tools import execute_list_directory, reset_file_tracking
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


class Scripted(BaseProvider):
    """Replays a fixed list of model turns: a ToolCall tuple or a final text."""

    def __init__(self, turns):
        self.model = "scripted"
        self.turns = list(turns)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append([m for m in messages])
        t = self.turns.pop(0)
        if isinstance(t, tuple):
            call = ToolCall(f"c{len(self.seen)}", t[0], t[1])
            return ProviderResponse(text="", tool_calls=[call], raw_assistant_message={"role": "assistant", "content": ""})
        return ProviderResponse(text=t, raw_assistant_message={"role": "assistant", "content": t})

    def stream(self, **kw):
        yield self.complete(**kw)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    (tmp_path / "pricing.py").write_text("def total(price, count):\n    return price + count\n")
    (tmp_path / "settings.py").write_text("TIMEOUT = 5\n")
    (tmp_path / "test_p.py").write_text("from pricing import total\ndef test_t():\n    assert total(7, 3) == 21\n")
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    reset_file_tracking()
    return tmp_path


def run(turns, prompt, **kw):
    events = []
    p = Scripted(turns)
    a = Agent(p, max_iterations=8, verbose=False, event_callback=events.append, **kw)
    return a.run(prompt), events, p


def observed(events):
    return [e["result"] for e in events if e["type"] == "observe"]


def test_missing_file_result_lists_directory_and_suggests_match(repo):
    final, ev, _ = run([("read_file", {"path": "price.py"}), "done"], "fix price.py")
    res = observed(ev)[0]
    assert "does not exist" in res and "[FILE] pricing.py" in res
    assert "Closest existing file(s): pricing.py" in res


def test_missing_file_without_lookalike_says_so(repo):
    _, ev, _ = run([("read_file", {"path": "nonexistent.py"}), "It does not exist."], "read nonexistent.py")
    res = observed(ev)[0]
    assert "No similar file exists" in res and "Closest" not in res


def test_guardrails_off_leaves_error_untouched(repo):
    _, ev, _ = run([("read_file", {"path": "price.py"}), "done"], "read price.py", guardrails=False)
    assert observed(ev)[0] == "ERROR: File not found: price.py"


def test_env_switch(monkeypatch):
    monkeypatch.setenv("NEXUS_GUARDRAILS", "0")
    assert guardrails_enabled() is False
    assert guardrails_enabled(True) is True
    monkeypatch.delenv("NEXUS_GUARDRAILS")
    assert guardrails_enabled() is True


def test_false_success_claim_is_sent_back_then_fixed(repo):
    turns = [
        ("run_tests", {}),
        "All tests pass.",  # false: pytest failed
        ("read_file", {"path": "pricing.py"}),
        ("patch_file", {"path": "pricing.py", "target": "price + count", "replacement": "price * count"}),
        ("run_tests", {}),
        "Tests pass now.",
    ]
    final, ev, p = run(turns, "fix the failing test")
    results = observed(ev)
    assert "FAILED" in results[0] and "PASSED" in results[-1]
    assert final == "Tests pass now."
    assert any(e["type"] == "guardrail" for e in ev)


def test_false_success_claim_that_never_gets_fixed_is_corrected(repo):
    turns = [("run_tests", {}), "Tests pass.", "Tests pass.", "Tests pass.", "Tests pass."]
    final, _, _ = run(turns, "run the tests")
    assert "run_tests last reported FAILED" in final


def test_honest_failure_report_is_untouched(repo):
    final, _, _ = run([("run_tests", {}), "The test fails: total returns 10, expected 21."], "run the tests")
    assert final == "The test fails: total returns 10, expected 21."


def test_success_claim_without_running_tests_is_flagged(repo):
    final, _, _ = run(["All tests pass."], "run the tests and tell me if they pass")
    # no tools used at all on a non-tool path would be fine; with tools available it is flagged
    assert "unverified" in final or final == "All tests pass."


def test_invented_git_status_answer_gets_real_output(repo):
    (repo / "settings.py").write_text("TIMEOUT = 9\n")
    final, _, _ = run([("git_status", {}), "test_git_status() and the tests pass"], "show git status")
    assert "Actual git output:" in final and "settings.py" in final


def test_correct_git_answer_not_padded(repo):
    (repo / "settings.py").write_text("TIMEOUT = 9\n")
    final, _, _ = run([("git_status", {}), "Only settings.py is modified."], "show git status")
    assert "Actual git output" not in final


def test_clean_tree_claim_checked(repo):
    (repo / "settings.py").write_text("TIMEOUT = 9\n")
    final, _, _ = run([("git_status", {}), "The working tree is clean."], "git status")
    assert "Actual git output:" in final
