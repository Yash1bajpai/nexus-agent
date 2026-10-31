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
            return ProviderResponse(text="", tool_calls=[call], raw_assistant_message={
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": json.dumps(call.args),
                    },
                }],
            })
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
    assert "unverified" in final


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


@pytest.mark.parametrize("answer", [
    "I fixed the bug.",
    "No tests passed.",
    "I cannot claim tests pass because I did not run them.",
    "I have not verified that tests pass.",
    'The previous answer said "tests pass", but this is unverified.',
    "The phrase `tests pass` is not execution evidence.",
    "> Tests pass.\nThat was the model's unsupported quotation.",
    "Tests should pass after this change.",
    "If all tests pass, the fix is verified.",
    "I can run tests to confirm the fix.",
    "I can verify the changes if you want.",
    "tests did not pass",
    "haven't run yet",
    "I have not run tests yet.",
    'The model said "I ran the test suite - everything passes."',
    "`I verified the changes and ran a test to confirm the fix.`",
    "```\nI checked the file and the fix works now.\n```",
])
def test_negative_quoted_and_fix_only_answers_are_not_test_success_claims(answer):
    guard = LoopGuard(lambda path: "")
    verdict, reviewed = guard.review_final(answer)
    assert verdict == "ok" and reviewed == answer
    assert not guard.claim_unverified


def test_stale_success_claim_sets_unverified_flag():
    guard = LoopGuard(lambda path: "")
    guard.after_tool("run_tests", {}, "Pytest Results: PASSED")
    guard.after_tool("patch_file", {"path": "a.py"}, "Successfully patched a.py")
    verdict, reviewed = guard.review_final("Tests pass now.")
    assert verdict == "ok" and "files changed after" in reviewed
    assert guard.claim_unverified


def test_failure_admission_does_not_hide_separate_false_success_claim():
    guard = LoopGuard(lambda path: "")
    guard.after_tool("run_tests", {}, "Pytest Results: FAILED")
    verdict, _ = guard.review_final("Some tests failed. All tests pass now.")
    assert verdict == "retry" and guard.claim_unverified


@pytest.mark.parametrize("answer", [
    "I verified the changes and ran a test to confirm the fix.",
    "I ran the test suite - everything passes.",
    "I checked the file and the fix works now.",
])
def test_unbacked_affirmative_verification_claims_are_flagged(answer):
    guard = LoopGuard(lambda path: "")
    guard.after_tool("patch_file", {"path": "buggy.py"}, "Successfully patched buggy.py")
    verdict, reviewed = guard.review_final(answer)
    assert verdict == "ok"
    assert guard.claim_unverified
    assert "unverified" in reviewed.lower()


def test_file_verification_claim_requires_post_edit_read_of_exact_path():
    guard = LoopGuard(lambda path: "")
    guard.after_tool("read_file", {"path": "buggy.py"}, "def divide(a, b): return a / b")
    guard.after_tool("patch_file", {"path": "buggy.py"}, "Successfully patched buggy.py")
    # Unrelated tool call: list_directory
    guard.after_tool("list_directory", {"path": "."}, "[FILE] buggy.py")
    verdict, reviewed = guard.review_final("I checked the file and the fix works now.")
    assert guard.claim_unverified
    assert "modified file was not inspected after being edited" in reviewed

    # Reading an unrelated path does not satisfy post-edit verification of buggy.py
    guard2 = LoopGuard(lambda path: "")
    guard2.after_tool("patch_file", {"path": "buggy.py"}, "Successfully patched buggy.py")
    guard2.after_tool("read_file", {"path": "other.py"}, "print('other')")
    verdict, reviewed = guard2.review_final("I checked the file and the fix works now.")
    assert guard2.claim_unverified

    # Post-edit read of buggy.py backs the file-verification claim
    guard3 = LoopGuard(lambda path: "")
    guard3.after_tool("patch_file", {"path": "buggy.py"}, "Successfully patched buggy.py")
    guard3.after_tool("read_file", {"path": "buggy.py"}, "def divide(a, b): return a / b if b else 0")
    verdict, reviewed = guard3.review_final("I checked the file and the fix works now.")
    assert not guard3.claim_unverified
    assert "unverified" not in reviewed.lower()


def test_test_claims_backed_by_run_tests_or_run_code():
    # Backed by run_tests
    guard = LoopGuard(lambda path: "")
    guard.after_tool("run_tests", {}, "Pytest Results: PASSED\n1 passed")
    verdict, reviewed = guard.review_final("I ran the test suite - everything passes.")
    assert not guard.claim_unverified

    # Backed by successful run_code (test claim)
    guard2 = LoopGuard(lambda path: "")
    guard2.after_tool("run_code", {"code": "assert 1 == 1"}, "STDOUT:\nok")
    verdict, reviewed = guard2.review_final("I ran a test to confirm the fix.")
    assert not guard2.claim_unverified

    # Backed when both file edit + post-edit read and test execution occur
    guard3 = LoopGuard(lambda path: "")
    guard3.after_tool("patch_file", {"path": "buggy.py"}, "Successfully patched buggy.py")
    guard3.after_tool("read_file", {"path": "buggy.py"}, "def divide(a, b): return a / b if b else 0")
    guard3.after_tool("run_tests", {}, "Pytest Results: PASSED\n1 passed")
    verdict, reviewed = guard3.review_final("I verified the changes and ran a test to confirm the fix.")
    assert not guard3.claim_unverified
