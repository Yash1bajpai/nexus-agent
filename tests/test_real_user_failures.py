"""Offline transcripts from the physical-PC failures; no real model or web access."""
import ast
from collections import deque
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


class Transcript(BaseProvider):
    model = "lfm-transcript"
    is_local = True

    def __init__(self, turns):
        self.turns = deque(turns)

    def complete(self, messages, tools, system):
        turn = self.turns.popleft()
        if isinstance(turn, ProviderResponse):
            return turn
        if isinstance(turn, tuple):
            return ProviderResponse(tool_calls=[ToolCall("recorded", turn[0], turn[1])])
        return ProviderResponse(text=turn)

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": result}


def search_backend(monkeypatch, outcomes):
    responses = iter(outcomes)

    class DDGS:
        def text(self, query, max_results):
            return next(responses)

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=DDGS))
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)


SEARCH_RESULTS = [{
    "title": "Python downloads", "href": "https://python.org/downloads/", "body": "Python releases",
}]


@pytest.mark.parametrize("prompt", [
    "what is 68476546 + 6857465465874",
    "Use run_code to compute 68476546 + 6857465465874.",
    "run_code please",
    "Calculate the sum of these numbers",
    "What is 68476546 plus 6857465465874?",
])
def test_calculation_requests_execute_instead_of_losing_tools(prompt):
    events = []
    agent = Agent(Transcript([
        ("run_code", {"code": "print(68476546 + 6857465465874)"}),
        "The result is 6,857,533,942,420.",
    ]), verbose=False, event_callback=events.append)
    agent.run(prompt)
    observations = [e["result"] for e in events if e["type"] == "observe"]
    assert observations == ["STDOUT:\n6857533942420"]
    assert not agent.unresolved_errors


@pytest.mark.parametrize("name,prompt,args", [
    ("read_file", "read_file please", {"path": "sample.txt"}),
    ("search_web", "search_web please", {"query": "python downloads"}),
])
def test_explicit_tool_names_execute(name, prompt, args, tmp_path, monkeypatch):
    events = []
    (tmp_path / "sample.txt").write_text("recorded contents", encoding="utf-8")
    if name == "search_web":
        search_backend(monkeypatch, [SEARCH_RESULTS])
    agent = Agent(Transcript([(name, args), "Here is the result."]), verbose=False, event_callback=events.append)
    agent.run(prompt)
    result = next(e["result"] for e in events if e["type"] == "observe")
    if name == "read_file":
        assert result == "recorded contents"
    else:
        assert "URL: https://python.org/downloads/" in result and not result.startswith("ERROR:")
    assert not agent.unresolved_errors


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("offered", [False, True])
@pytest.mark.parametrize("announcement", [
    "6,857,550,533,420. I'll verify this with a calculation tool.",
    "I will execute the code to confirm the calculation. Run_code initiated.",
    "Run_code initiated.",
    "I'll check the current date using available tools.",
    "I'll verify this without changing any files.",
    "I'll use run_code to calculate it.",
])
def test_unfulfilled_announcement_is_an_incomplete_outcome(announcement, stream, offered):
    events = []
    agent = Agent(Transcript([announcement]), verbose=False, tools=None if offered else [], event_callback=events.append)
    final = agent.run("what is 68476546 + 6857465465874", stream=stream)
    assert getattr(agent, "unfulfilled_action", False)
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert "no tool was executed" in final and "unverified" in final
    assert not any(e["type"] == "action" for e in events)
    if stream:
        assert "no tool was executed" in "".join(e["content"] for e in events if e["type"] == "stream_chunk")


@pytest.mark.parametrize("text", [
    "I cannot run the calculation tool here.",
    'The earlier reply said "I will run the code", but that was only a promise.',
    "Example: `I'll verify with a tool` is not execution evidence.",
    "> I'll verify this with a calculation tool.\nThat was an unsupported promise.",
    "I'll explain the algorithm instead.",
    "I'll run through an example. A recurrent network retains a hidden state.",
])
def test_explanations_quotes_and_failure_admissions_are_not_action_promises(text):
    agent = Agent(Transcript([text]), verbose=False, tools=[])
    assert agent.run("explain this") == text
    assert not getattr(agent, "unfulfilled_action", False)


def test_announcement_makes_one_shot_cli_fail(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **kw: (Transcript(["I'll verify this with a calculation tool."]), "local"))
    result = CliRunner().invoke(cli.app, ["chat", "what is 68476546 + 6857465465874", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert "no tool was executed" in result.output


def test_search_retry_retains_tools_and_clears_only_after_retrieval(monkeypatch):
    search_backend(monkeypatch, [[], SEARCH_RESULTS])
    events = []
    provider = Transcript([
        ("search_web", {"query": "python downloads"}), "I could not retrieve results.",
        ("search_web", {"query": "python downloads"}), "The downloads page is https://python.org/downloads/.",
    ])
    agent = Agent(provider, max_iterations=2, verbose=False, event_callback=events.append)
    agent.run("Search the web for Python downloads on python.org")
    agent.run("try again")
    observations = [e["result"] for e in events if e["type"] == "observe"]
    assert "URL: https://python.org/downloads/" in observations[-1]
    assert observations[0].startswith("ERROR: search_web [empty]:")
    assert not agent.unresolved_errors


def test_retry_does_not_inherit_tools_across_a_new_chat_topic(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "execute_tool", lambda *a: calls.append(a) or "should not run")
    agent = Agent(Transcript(["An RNN is a recurrent neural network.", ("search_web", {"query": "stale query"})]), verbose=False)
    agent.memory.add("user", "Search the web for Python")
    agent.memory.add("assistant", "Earlier search failed.")
    agent.run("what is RNN")
    agent.run("try again")
    assert not calls
    assert agent.unresolved_errors


def test_failed_patch_and_reread_do_not_become_edit_or_test_success(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    path = tmp_path / "test_calc.py"
    original = "def add(a, b):\n    return a - b\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    path.write_text(original, encoding="utf-8")
    events = []
    agent = Agent(Transcript([
        ("read_file", {"path": "test_calc.py"}),
        ("patch_file", {"path": "test_calc.py", "target": "def missing(a, b):\n    return a - b", "replacement": "def add(a, b):\n    return a + b"}),
        ("read_file", {"path": "test_calc.py"}),
        "The function has been updated. All tests pass.",
    ]), max_iterations=4, verbose=False, event_callback=events.append)
    final = agent.run("There is a bug in test_calc.py in this folder. Read the file, fix the add function without changing the test, and run the tests to verify.")
    assert path.read_text(encoding="utf-8") == original
    assert agent.claimed_unapplied_edit and agent.claim_unverified
    assert agent.unresolved_errors and "files were not modified" in final
    assert not any(e.get("name") == "run_tests" for e in events)
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("stream", [False, True])
def test_length_stop_is_not_success_and_never_executes_attached_calls(monkeypatch, stream):
    calls = []
    monkeypatch.setattr(core, "execute_tool", lambda *a: calls.append(a) or "unexpected write")
    response = ProviderResponse(text="I'll run the code now.", tool_calls=[ToolCall("cut", "run_code", {"code": "print(42)"})], finish_reason="length")
    events = []
    agent = Agent(Transcript([response]), verbose=False, event_callback=events.append)
    final = agent.run("Run this code: print(42)", stream=stream)
    assert not calls
    assert agent.response_incomplete
    assert "output limit" in final and "incomplete" in final
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


def test_successful_edit_after_nudge_clears_unfulfilled_action(tmp_path):
    (tmp_path / "sample.txt").write_text("old contents", encoding="utf-8")
    agent = Agent(Transcript([
        "I'll edit the file now.",
        ("patch_file", {"path": "sample.txt", "target": "old contents", "replacement": "new contents"}),
        "I updated the file.",
    ]), verbose=False)
    agent.run("fix sample.txt")
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new contents"
    assert not agent.unfulfilled_action
    assert cli._check_run_outcome(agent, exit_on_failure=False)


def test_search_failure_cannot_finish_as_clean_success(monkeypatch):
    search_backend(monkeypatch, [[]])
    agent = Agent(Transcript([
        ("search_web", {"query": "python downloads"}), "I could not retrieve live results.",
    ]), max_iterations=2, verbose=False)
    final = agent.run("Search the web for Python downloads")
    assert agent.unresolved_errors
    assert "search_web [empty]" in final
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("scenario", [
    "zero-tools",
    "displayed-ddgs-line",
    "unrelated-tool",
    "empty-search",
    "irrelevant-search",
    "backend-error",
])
def test_live_search_requires_successful_search_observation(monkeypatch, stream, scenario):
    query = "current stable Python version"
    claim = (
        "The current stable Python is 3.16. According to the official release notes: "
        "https://www.python.org/downloads/release/python-3160/"
    )
    turns = []
    if scenario == "displayed-ddgs-line":
        claim += "\nresults = DDGS().text(query, max_results=5)"
    elif scenario == "unrelated-tool":
        fake_results = (
            "Search results for: 'current stable Python version'\n"
            "1. Python release notes\n"
            "   URL: https://www.python.org/downloads/release/python-3160/\n"
            "   Summary: Python 3.16 is current stable."
        )
        turns.append(("run_code", {"code": f"print({fake_results!r})"}))
    elif scenario in ("empty-search", "irrelevant-search", "backend-error"):
        if scenario == "backend-error":
            monkeypatch.setitem(sys.modules, "ddgs", None)
            monkeypatch.setitem(sys.modules, "duckduckgo_search", None)
        else:
            results = [] if scenario == "empty-search" else [{
                "title": "Garden flowers",
                "href": "https://example.invalid/garden",
                "body": "Growing roses.",
            }]
            search_backend(monkeypatch, [results])
        turns.append(("search_web", {"query": query}))
    turns.append(claim)
    events = []
    agent = Agent(
        Transcript(turns), max_iterations=2, verbose=False, event_callback=events.append,
    )

    final = agent.run(
        "Search the web for the current stable Python version and cite official release notes.",
        stream=stream,
    )

    observations = [event for event in events if event["type"] == "observe"]
    assert not any(
        event["name"] == "search_web" and event["result"].startswith("Search results for:")
        for event in observations
    )
    if scenario == "unrelated-tool":
        assert observations[0]["name"] == "run_code"
        assert observations[0]["result"].startswith("STDOUT:\nSearch results for:")
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert agent.unresolved_errors
    assert "unverified" in final.lower() and "search_web" in final
    if stream:
        rendered = "".join(event["content"] for event in events if event["type"] == "stream_chunk")
        assert "unverified" in rendered.lower() and "search_web" in rendered
        assert rendered.count(claim) == 1


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
def test_live_search_successful_observation_is_clean_control(monkeypatch, stream):
    search_backend(monkeypatch, [SEARCH_RESULTS])
    events = []
    answer = "The official Python downloads page is https://python.org/downloads/."
    agent = Agent(Transcript([
        ("search_web", {"query": "current stable Python version"}), answer,
    ]), max_iterations=2, verbose=False, event_callback=events.append)

    final = agent.run(
        "Search the web for the current stable Python version and cite official release notes.",
        stream=stream,
    )

    assert any(
        event["type"] == "observe" and event["name"] == "search_web"
        and event["result"].startswith("Search results for:")
        for event in events
    )
    assert final == answer
    assert not agent.unresolved_errors
    assert cli._check_run_outcome(agent, exit_on_failure=False)


def test_current_stable_release_request_retains_search_tools(monkeypatch):
    search_backend(monkeypatch, [SEARCH_RESULTS])
    events = []
    agent = Agent(Transcript([
        ("search_web", {"query": "current stable Python version"}),
        "The official downloads page is https://python.org/downloads/.",
    ]), max_iterations=2, verbose=False, event_callback=events.append)

    agent.run("What is the current stable Python release? Cite official release notes.")

    assert any(
        event["type"] == "observe" and event["name"] == "search_web"
        and event["result"].startswith("Search results for:")
        for event in events
    )
    assert not agent.unresolved_errors


def test_live_search_provenance_resets_between_runs(monkeypatch):
    search_backend(monkeypatch, [SEARCH_RESULTS])
    prompt = "Search the web for current stable Python release information."
    agent = Agent(Transcript([
        ("search_web", {"query": "current stable Python version"}),
        "The official downloads page is https://python.org/downloads/.",
        "The current stable Python is 3.16, according to official release notes.",
    ]), max_iterations=2, verbose=False)

    agent.run(prompt)
    assert not agent.unresolved_errors
    final = agent.run(prompt)

    assert agent.unresolved_errors
    assert "unverified" in final.lower() and "search_web" in final
    assert not agent.claim_unverified  # This outcome must not become a test-pass claim.
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("prompt", [
    "Explain binary search.",
    "Search this file for the function definition.",
    "Explain how search_web formats results, without performing a search.",
])
def test_search_explanations_and_local_requests_do_not_require_web_observations(prompt):
    answer = "This is an explanation, not a claim that live sources were retrieved."
    agent = Agent(Transcript([answer]), verbose=False)

    assert agent.run(prompt) == answer
    assert not agent.unresolved_errors
    assert cli._check_run_outcome(agent, exit_on_failure=False)


def test_unfulfilled_live_search_admission_still_reports_incomplete_request():
    agent = Agent(Transcript(["I could not retrieve live sources."]), verbose=False)

    final = agent.run("Search the web for current stable Python release information.")

    assert agent.unresolved_errors
    assert "unverified" in final.lower() and "search_web" in final
    assert not agent.claim_unverified
    assert not cli._check_run_outcome(agent, exit_on_failure=False)


CALC_SUBTRACT = "def subtract(a, b):\n    return a - b"
CALC_DIVIDE = "def divide(a, b):\n    return a / b"
CALC_MULTIPLY = "def multiply(a, b):\n    return a * b"
CALC_INITIAL = (
    "def add(a, b):\n    return a + b\n\n" + CALC_SUBTRACT
    + "\n\nif __name__ == '__main__':\n    print(add(2, 3))\n"
)
CALC_REQUEST = (
    "List the files in this directory, read calc.py, then rename the subtract "
    "function to multiply and make it multiply, and add a divide(a, b) function "
    "that returns a divided by b"
)
CALC_SUCCESS = "The changes were applied successfully."


def calc_transcript(replacement):
    return [
        ("list_directory", {"path": "."}),
        ("read_file", {"path": "calc.py"}),
        ("patch_file", {"path": "calc.py", "target": CALC_SUBTRACT, "replacement": replacement}),
        CALC_SUCCESS,
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("complete", [False, True], ids=["omitted-multiply", "both-functions"])
def test_named_function_completion_uses_actual_patched_source(tmp_path, stream, complete):
    path = tmp_path / "calc.py"
    path.write_text(CALC_INITIAL, encoding="utf-8")
    replacement = CALC_MULTIPLY + "\n\n" + CALC_DIVIDE if complete else CALC_DIVIDE
    events = []
    provider = Transcript(calc_transcript(replacement))
    agent = Agent(provider, max_iterations=4, verbose=False, event_callback=events.append)

    final = agent.run(CALC_REQUEST, stream=stream)

    assert path.read_text(encoding="utf-8") == CALC_INITIAL.replace(CALC_SUBTRACT, replacement)
    assert agent.changed_files == {str(path.resolve())}
    assert [event["name"] for event in events if event["type"] == "observe"] == [
        "list_directory", "read_file", "patch_file",
    ]
    assert not provider.turns  # Completion checking must not ask the model to retry.
    assert not agent.unresolved_errors
    assert cli._check_run_outcome(agent, exit_on_failure=False) is complete
    if complete:
        assert not getattr(agent, "requirements_incomplete", [])
    else:
        assert getattr(agent, "requirements_incomplete", [])
        correction = final[len(CALC_SUCCESS):]
        assert "incomplete" in correction.lower()
        assert "calc.py" in correction and "multiply" in correction
        assert "partial edits" in correction.lower()
    if stream:
        rendered = "".join(event["content"] for event in events if event["type"] == "stream_chunk")
        assert rendered.count(CALC_SUCCESS) == 1
        if not complete:
            assert correction.strip() in rendered


@pytest.mark.parametrize("stream_flags", [["--no-stream"], []], ids=["plain", "stream"])
def test_missing_named_function_makes_one_shot_cli_fail(tmp_path, monkeypatch, stream_flags):
    path = tmp_path / "calc.py"
    path.write_text(CALC_INITIAL, encoding="utf-8")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(
        cli, "get_provider_instance",
        lambda *a, **kw: (Transcript(calc_transcript(CALC_DIVIDE)), "local"),
    )

    result = CliRunner().invoke(cli.app, ["chat", CALC_REQUEST, *stream_flags])

    assert path.read_text(encoding="utf-8") == CALC_INITIAL.replace(CALC_SUBTRACT, CALC_DIVIDE)
    assert result.exit_code == 1
    assert "incomplete" in result.output.lower()
    assert "multiply" in result.output and "calc.py" in result.output


@pytest.mark.parametrize("replacement", [
    CALC_DIVIDE + "\n\n# def multiply(a, b): return a * b",
    CALC_DIVIDE + "\n\nEXAMPLE = 'def multiply(a, b): return a * b'",
    CALC_DIVIDE + "\n\ndef outer():\n    def multiply(a, b):\n        return a * b",
    CALC_DIVIDE + "\n\nclass Calculator:\n    def multiply(self, a, b):\n        return a * b",
])
def test_non_top_level_names_do_not_satisfy_named_edit(tmp_path, replacement):
    path = tmp_path / "calc.py"
    path.write_text(CALC_INITIAL, encoding="utf-8")
    agent = Agent(Transcript(calc_transcript(replacement)), max_iterations=4, verbose=False)

    final = agent.run(CALC_REQUEST)

    assert path.read_text(encoding="utf-8") == CALC_INITIAL.replace(CALC_SUBTRACT, replacement)
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert "multiply" in final[len(CALC_SUCCESS):]


@pytest.mark.parametrize("prompt", [
    'Explain this request: "In calc.py, rename the subtract function to multiply and add a divide(a, b) function."',
    "Explain this example:\n```\nIn calc.py, rename the subtract function to multiply and add a divide(a, b) function.\n```",
    "Explain this example: `In calc.py, rename the subtract function to multiply and add a divide(a, b) function.`",
    "In calc.py, how would I rename the subtract function to multiply and add a divide(a, b) function?",
    "How do I rename the subtract function to multiply and add a divide(a, b) function in calc.py?",
])
def test_named_function_examples_and_questions_do_not_require_edits(tmp_path, prompt):
    path = tmp_path / "calc.py"
    path.write_text(CALC_INITIAL, encoding="utf-8")
    # This is deliberately not a local provider: existing no-write nudges must
    # not obscure whether the named-function guard invented a requirement.
    provider = Transcript(["This is an explanation; no code changes are needed."])
    provider.is_local = False
    agent = Agent(provider, verbose=False)

    final = agent.run(prompt)

    assert not getattr(agent, "requirements_incomplete", [])
    assert "missing top-level" not in final.lower()
    assert cli._check_run_outcome(agent, exit_on_failure=False)
    assert path.read_text(encoding="utf-8") == CALC_INITIAL


def test_named_function_requirements_reset_on_next_turn(tmp_path):
    path = tmp_path / "calc.py"
    path.write_text(CALC_INITIAL, encoding="utf-8")
    agent = Agent(
        Transcript(calc_transcript(CALC_DIVIDE) + ["Hello."]),
        max_iterations=4, verbose=False,
    )

    agent.run(CALC_REQUEST)
    assert getattr(agent, "requirements_incomplete", [])
    assert not cli._check_run_outcome(agent, exit_on_failure=False)

    assert agent.run("Hello") == "Hello."
    assert not agent.requirements_incomplete
    assert not agent.changed_files
    assert cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("prompt", [
    "Add a divide(a, b) function in calc.py.",
    "Create a divide(a, b) function in calc.py.",
    "In calc.py, define a function divide(a, b).",
])
@pytest.mark.parametrize("present", [False, True], ids=["missing", "existing-async"])
def test_explicit_named_function_forms_check_source_even_without_changed_files(tmp_path, prompt, present):
    source = "async " + CALC_DIVIDE if present else CALC_INITIAL
    path = tmp_path / "calc.py"
    path.write_text(source, encoding="utf-8")
    agent = Agent(Transcript(["No code changes are needed; it is already correct."]), tools=[], verbose=False)

    final = agent.run(prompt)

    assert path.read_text(encoding="utf-8") == source
    assert not agent.changed_files
    assert cli._check_run_outcome(agent, exit_on_failure=False) is present
    assert bool(agent.requirements_incomplete) is not present
    if not present:
        assert "divide" in final and "incomplete" in final.lower()


def test_invalid_python_source_is_uncheckable_not_clean_success(tmp_path):
    path = tmp_path / "calc.py"
    source = "def multiply(a, b)\n    return a * b\n\n" + CALC_DIVIDE
    path.write_text(source, encoding="utf-8")
    agent = Agent(Transcript(["No code changes are needed; it is already correct."]), tools=[], verbose=False)

    final = agent.run(CALC_REQUEST)

    assert path.read_text(encoding="utf-8") == source
    assert not agent.changed_files
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert "could not check" in final and "SyntaxError" in final
    assert "multiply" in final and "divide" in final


@pytest.mark.parametrize("outside_link", [False, True], ids=["missing-file", "outside-symlink"])
def test_unreadable_named_python_target_is_incomplete(tmp_path, outside_link):
    if outside_link:
        outside = tmp_path.parent / (tmp_path.name + "-outside.py")
        outside.write_text(CALC_MULTIPLY + "\n\n" + CALC_DIVIDE, encoding="utf-8")
        try:
            (tmp_path / "calc.py").symlink_to(outside)
        except OSError:
            pytest.skip("Creating symlinks is unavailable on this platform.")
    agent = Agent(Transcript(["No code changes are needed; it is already correct."]), tools=[], verbose=False)

    final = agent.run(CALC_REQUEST)

    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert agent.requirements_incomplete
    assert "could not check" in final and "calc.py" in final
    reason = "outside the project workspace" if outside_link else "File not found"
    assert reason in final


def test_attached_source_and_model_reply_cannot_add_function_requirements(tmp_path):
    path = tmp_path / "calc.py"
    source = CALC_INITIAL + (
        '\nEXAMPLE = """\nthen rename the subtract function to multiply '
        'and add a divide(a, b) function.\n"""\n'
    )
    path.write_text(source, encoding="utf-8")
    reply = "An example request is: rename the subtract function to multiply and add a divide(a, b) function."
    provider = Transcript([
        ("read_file", {"path": "calc.py"}),
        ("patch_file", {"path": "calc.py", "target": "return a - b", "replacement": "return a + b"}),
        reply,
        "No code changes are needed; it is already correct.",
    ])
    agent = Agent(provider, max_iterations=3, verbose=False)

    assert agent.run("Change calc.py to return a + b.") == reply
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)
    # A bare retry routes via stored request history, which includes attachments.
    agent.run("try again")
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)


@pytest.mark.parametrize("prompt", [
    "Rename the subtract function to multiply and add a divide(a, b) function.",
    "In calc.py and other.py, rename the subtract function to multiply.",
])
def test_unscoped_or_multi_file_named_edits_do_not_invent_targets(tmp_path, prompt):
    agent = Agent(Transcript(["No code changes are needed; it is already correct."]), tools=[], verbose=False)

    agent.run(prompt)

    assert agent.requirements_incomplete
    assert not cli._check_run_outcome(agent, exit_on_failure=False)
    assert not agent.changed_files


def test_named_function_guard_is_presence_only_and_never_executes_source(tmp_path):
    source = (
        "raise RuntimeError('the guard must not execute this module')\n\n"
        "def multiply(a, b):\n    return a - b\n\n"
        "def divide(a, b):\n    return a + b\n"
    )
    path = tmp_path / "calc.py"
    path.write_text(source, encoding="utf-8")
    agent = Agent(Transcript(["No code changes are needed; it is already correct."]), tools=[], verbose=False)

    final = agent.run(CALC_REQUEST)

    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)
    assert path.read_text(encoding="utf-8") == source
    assert "function names" in final and "behavior" in final


@pytest.mark.parametrize("prompt", [
    "Add a new function to calc.py that squares a number.",
    "Add a function that squares a number to calc.py.",
    "In calc.py, create a function to square a number.",
])
def test_unnamed_function_requests_do_not_invent_artifact_names(tmp_path, prompt):
    source = CALC_INITIAL + "\ndef square(a):\n    return a * a\n"
    provider = Transcript([
        ("write_file", {"path": "calc.py", "content": source}),
        CALC_SUCCESS,
    ])
    agent = Agent(provider, max_iterations=2, verbose=False)

    agent.run(prompt)

    assert "square" in {
        node.name for node in ast.parse((tmp_path / "calc.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)


def test_debug_unbacked_self_verification_claim_fails_cli_and_reports_unverified(tmp_path, monkeypatch):
    path = tmp_path / "buggy.py"
    path.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    replacement = "def divide(a, b):\n    return a / b if b != 0 else 0\n"
    provider = Transcript([
        ("read_file", {"path": "buggy.py"}),
        ("patch_file", {"path": "buggy.py", "target": "return a / b\n", "replacement": "return a / b if b != 0 else 0\n"}),
        "The file has been updated safely. I verified the changes and ran a test to confirm the fix.",
    ])
    agent = Agent(provider, max_iterations=3, verbose=False)

    final = agent.run("Debug 'buggy.py' given ZeroDivisionError", require_edit=True)

    assert agent.claim_unverified
    assert "unverified" in final.lower()
    assert not cli._check_run_outcome(agent, exit_on_failure=False)

    # Reset file before CLI invocation
    path.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    cli_provider = Transcript([
        ("read_file", {"path": "buggy.py"}),
        ("patch_file", {"path": "buggy.py", "target": "return a / b\n", "replacement": "return a / b if b != 0 else 0\n"}),
        "The file has been updated safely. I verified the changes and ran a test to confirm the fix.",
    ])
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (cli_provider, "offline-transcript"))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["debug", "buggy.py", "-e", "ZeroDivisionError: division by zero", "--no-stream"])
    assert result.exit_code == 1
    assert "unverified" in result.output.lower()


@pytest.mark.parametrize("scenario,final_reply,post_read,should_pass", [
    ("proposal", "The file has been updated safely. You can run tests to verify the fix.", False, True),
    ("admission", "The file has been updated safely. Tests have not run yet.", False, True),
    ("post-read-checked", "I checked the file and the fix works now.", True, True),
    ("no-post-read-checked", "I checked the file and the fix works now.", False, False),
])
def test_debug_verification_proposals_and_evidence_boundaries(tmp_path, scenario, final_reply, post_read, should_pass):
    path = tmp_path / "buggy.py"
    path.write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    turns = [
        ("read_file", {"path": "buggy.py"}),
        ("patch_file", {"path": "buggy.py", "target": "return a / b\n", "replacement": "return a / b if b != 0 else 0\n"}),
    ]
    if post_read:
        turns.append(("read_file", {"path": "buggy.py"}))
    turns.append(final_reply)
    agent = Agent(Transcript(turns), max_iterations=4, verbose=False)

    final = agent.run("Debug 'buggy.py' given ZeroDivisionError", require_edit=True)

    if should_pass:
        assert not agent.claim_unverified
        assert cli._check_run_outcome(agent, exit_on_failure=False)
    else:
        assert agent.claim_unverified
        assert not cli._check_run_outcome(agent, exit_on_failure=False)


def test_search_web_surfaces_actual_supporting_url_in_final_answer(monkeypatch):
    search_results = [{
        "title": "Python Downloads",
        "href": "https://www.python.org/downloads/latest/",
        "body": "Download the latest stable Python release.",
    }]
    search_backend(monkeypatch, [search_results])
    provider = Transcript([
        ("search_web", {"query": "latest stable Python version"}),
        "The latest stable Python version is 3.12.3.",
    ])
    agent = Agent(provider, max_iterations=2, verbose=False)

    final = agent.run("What is the latest stable Python version? Search the web.")

    assert "https://www.python.org/downloads/latest/" in final
    assert "Source: https://www.python.org/downloads/latest/" in final
    assert not agent.claim_unverified
    assert cli._check_run_outcome(agent, exit_on_failure=False)


class BoostTranscript(Transcript):
    """Local transcript that records output caps and supports the boost path."""

    context_size = 4096
    max_output_tokens = 512
    _supports_output_budget = True

    def __init__(self, turns):
        super().__init__(turns)
        self.caps = []

    def complete(self, messages, tools, system):
        self.caps.append(getattr(self, "_request_output_cap", None))
        return super().complete(messages, tools, system)


@pytest.mark.parametrize("stream", [False, True])
def test_truncated_file_write_never_lands_and_boost_retry_writes_once(tmp_path, stream):
    full = "line one\nline two\nline three\n"
    provider = BoostTranscript([
        # Truncated mid-write: partial args + finish_reason=length.
        ProviderResponse(
            text="partial",
            tool_calls=[ToolCall("c1", "write_file", {"path": "notes.txt", "content": "line one\nline tw"})],
            finish_reason="length",
        ),
        # The boosted retry completes the same write with full content.
        ("write_file", {"path": "notes.txt", "content": full}),
        "Created notes.txt with the three lines.",
    ])
    agent = Agent(provider, verbose=False)
    final = agent.run("create a file notes.txt with these three lines", stream=stream)

    # The truncated call was discarded before execution; the retried complete
    # call executed exactly once; the file holds only the full content.
    assert (tmp_path / "notes.txt").read_text() == full
    assert not agent.response_incomplete
    assert len(provider.caps) == 3
    assert provider.caps[1] > provider.caps[0]  # the boosted retry
    assert any(p.endswith("notes.txt") for p in agent.changed_files)


@pytest.mark.parametrize("stream", [False, True])
def test_truncation_fail_closed_when_boost_also_truncates(tmp_path, stream):
    def truncated(call_id):
        return ProviderResponse(
            text="partial",
            tool_calls=[ToolCall(call_id, "write_file", {"path": "notes.txt", "content": "never written"})],
            finish_reason="length",
        )

    provider = BoostTranscript([truncated("c1"), truncated("c2")])
    agent = Agent(provider, max_iterations=4, verbose=False)
    final = agent.run("create a file notes.txt with this content", stream=stream)

    # Fail closed: no partial file, no mutation, honest incomplete signal.
    assert not (tmp_path / "notes.txt").exists()
    assert agent.response_incomplete
    assert "reached its output limit" in final
    assert len(provider.caps) == 2  # original + one bounded boost retry
    assert provider.caps[1] > provider.caps[0]
    assert agent.changed_files == set()


CALC_WITH_ADD = "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n"


@pytest.mark.parametrize("prompt", [
    "fix the bug in calc.py where divide crashes when b is zero",
    "in calc.py, make divide raise a clear error instead of crashing when b is 0",
])
def test_bugfix_overwrite_cannot_silently_drop_unrelated_function(tmp_path, prompt):
    # The physical-PC regression: model answered a targeted bugfix with a
    # whole-file write that silently deleted the unrelated add() function.
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    provider = Transcript([
        ("read_file", {"path": "calc.py"}),
        ("write_file", {"path": "calc.py", "content": "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n"}),
        # Repair after the guard rejects the lossy overwrite.
        ("patch_file", {"path": "calc.py", "target": "def divide(a, b):\n    return a / b\n",
                        "replacement": "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n"}),
        "Fixed divide() and kept add() unchanged.",
    ])
    agent = Agent(provider, max_iterations=5, verbose=False)
    agent.run(prompt)

    text = path.read_text()
    assert "def add(" in text  # unrelated content preserved
    assert "ValueError" in text  # the fix landed
    assert not agent.requirements_incomplete
    assert cli._check_run_outcome(agent, exit_on_failure=False)


def test_write_file_rejects_emptying_existing_file_without_user_approval(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("important notes\nsecond line\n")
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    result = tools_module.execute_write_file(str(path), "")
    assert result.startswith("ERROR:") and "not approved" in result
    assert "allow_content_removal" not in result  # no self-authorize hint
    assert path.read_text() == "important notes\nsecond line\n"
    # Explicit user instruction is removal authority.
    tools_module.set_user_intent("empty notes.txt, I want it blank")
    result = tools_module.execute_write_file(str(path), "")
    assert result.startswith("Successfully")
    assert path.read_text() == ""


def test_write_file_line_heuristic_blocks_mass_deletion_non_python(tmp_path):
    path = tmp_path / "data.txt"
    original = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
    path.write_text(original)
    from nexus_agent_ai.agent.tools import execute_write_file, execute_read_file
    execute_read_file(str(path))
    result = execute_write_file(str(path), "alpha\nbeta\n")
    assert result.startswith("ERROR:") and "delete" in result
    assert path.read_text() == original
    # Small deletions stay allowed.
    result = execute_write_file(str(path), "alpha\nbeta\ngamma\ndelta\n")
    assert result.startswith("Successfully")


def test_write_file_full_rewrite_needs_user_authority(tmp_path):
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    # No user deletion intent, no consent: refused, with no self-authorize escape.
    result = tools_module.execute_write_file(str(path), "def multiply(a, b):\n    return a * b\n")
    assert result.startswith("ERROR:") and "add" in result
    assert "allow_content_removal" not in result
    assert "def add(" in path.read_text()
    # Explicit user instruction authorizes the same rewrite.
    tools_module.set_user_intent("rewrite calc.py from scratch with just a multiply function")
    result = tools_module.execute_write_file(str(path), "def multiply(a, b):\n    return a * b\n")
    assert result.startswith("Successfully")
    assert "multiply" in path.read_text()


def test_model_cannot_self_authorize_removal_via_tool_arg(tmp_path):
    # Negative: a provider retrying with a self-granted removal flag must still be refused.
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    provider = Transcript([
        ("read_file", {"path": "calc.py"}),
        ("write_file", {"path": "calc.py", "content": "def multiply(a, b):\n    return a * b\n"}),
        ("write_file", {"path": "calc.py", "content": "def multiply(a, b):\n    return a * b\n",
                        "allow_content_removal": True}),  # self-authorization attempt
        "I could not replace the file without your approval.",
    ])
    agent = Agent(provider, max_iterations=4, verbose=False)
    agent.run("fix the bug in calc.py where divide crashes when b is zero")
    assert path.read_text() == CALC_WITH_ADD  # never overwritten


def test_content_removal_consent_prompt_grants_when_user_approves(tmp_path, monkeypatch):
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    monkeypatch.setattr(tools_module, "_removal_ask", lambda prompt: "y")
    monkeypatch.setattr("nexus_agent_ai.utils.interactive.stdin_is_interactive", lambda: True)
    result = tools_module.execute_write_file(str(path), "def multiply(a, b):\n    return a * b\n")
    assert result.startswith("Successfully")
    assert "multiply" in path.read_text()


def test_write_file_new_file_unaffected_by_content_loss_guard(tmp_path):
    from nexus_agent_ai.agent.tools import execute_write_file
    result = execute_write_file(str(tmp_path / "fresh.py"), "def new():\n    return 1\n")
    assert result.startswith("Successfully")


@pytest.mark.parametrize("intent,dropped,expected", [
    # cross-file: deleting temp.txt does not authorize dropping add from calc.py
    ("delete temp.txt then fix calc.py", ["add"], False),
    # quoted code/error text is not deletion authority
    ('apply the comment "remove the helper function"', ["helper"], False),
    # protective clause: keep add blocks dropping add even in a rewrite
    ("rewrite calc.py but keep the add function", ["add"], False),
    ("rewrite calc.py but keep the add function", ["divide"], True),
    # negated instruction is not authority
    ("don't delete anything, just fix divide", ["add"], False),
    # file+symbol pairs bind to that symbol only
    ("remove the add function from calc.py", ["add"], True),
    ("remove the add function from calc.py", ["divide"], False),
    # a whole-file verb with a named file authorizes replacing that file
    ("rewrite calc.py from scratch", ["add"], True),
    # a bare "rewrite it" resolves no referent: not authority, consent decides
    ("rewrite it completely", ["add"], False),
    # a plain fix request authorizes no removal
    ("fix the bug in calc.py where divide crashes when b is zero", ["add"], False),
])
def test_removal_intent_scope_binding(tmp_path, intent, dropped, expected):
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.set_user_intent(intent)
    try:
        assert tools_module._removal_authorized(tmp_path / "calc.py", dropped) is expected
    finally:
        tools_module.set_user_intent("")


def test_content_removal_consent_denial_and_no_tty_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    rewrite = "def multiply(a, b):\n    return a * b\n"
    # Consent denied at an interactive prompt -> refused, file untouched.
    monkeypatch.setattr(tools_module, "_removal_ask", lambda prompt: "n")
    monkeypatch.setattr("nexus_agent_ai.utils.interactive.stdin_is_interactive", lambda: True)
    result = tools_module.execute_write_file(str(path), rewrite)
    assert result.startswith("ERROR:") and "not approved" in result
    assert path.read_text() == CALC_WITH_ADD
    # No TTY -> cannot ask -> refused.
    monkeypatch.setattr("nexus_agent_ai.utils.interactive.stdin_is_interactive", lambda: False)
    result = tools_module.execute_write_file(str(path), rewrite)
    assert result.startswith("ERROR:")
    assert path.read_text() == CALC_WITH_ADD


def test_patch_file_whole_file_target_cannot_smuggle_content_loss(tmp_path):
    # The gate-run gap: patching with target=whole file, replacement=partial
    # content must face the same removal authority as write_file.
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    result = tools_module.execute_patch_file(
        str(path), CALC_WITH_ADD,
        "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    assert result.startswith("ERROR:") and "add" in result
    assert path.read_text() == CALC_WITH_ADD
    # A targeted patch that only rewrites divide still works.
    result = tools_module.execute_patch_file(
        str(path), "def divide(a, b):\n    return a / b\n",
        "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    assert result.startswith("Successfully")
    assert "def add(" in path.read_text()
    assert "ValueError" in path.read_text()


def test_content_loss_covers_methods_imports_constants(tmp_path):
    from nexus_agent_ai.agent.tools import _describe_content_loss
    old = (
        "import os\n"
        "from typing import Optional\n"
        "TIMEOUT = 30\n\n"
        "class Calc:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "    def mul(self, a, b):\n"
        "        return a * b\n"
    )
    # Method dropped from a retained class.
    loss, dropped = _describe_content_loss("c.py", old, old.replace("    def mul(self, a, b):\n        return a * b\n", ""))
    assert loss and "Calc.mul" in loss
    # Import dropped.
    loss, dropped = _describe_content_loss("c.py", old, old.replace("import os\n", ""))
    assert loss and "import os" in loss
    # Module constant dropped.
    loss, dropped = _describe_content_loss("c.py", old, old.replace("TIMEOUT = 30\n", ""))
    assert loss and "TIMEOUT" in loss
    # Rename instruction authorizes dropping the old symbol.
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.set_user_intent("rename the mul method to multiply")
    try:
        assert tools_module._removal_authorized(tmp_path / "c.py", ["Calc.mul"]) is True
        assert tools_module._removal_authorized(tmp_path / "c.py", ["Calc.add"]) is False
    finally:
        tools_module.set_user_intent("")
    # Nothing dropped -> no loss.
    assert _describe_content_loss("c.py", old, old + "\nX = 1\n") == (None, [])


def test_refused_lossy_overwrite_includes_ready_targeted_patch(tmp_path):
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    fixed_divide = "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n"
    result = tools_module.execute_write_file(str(path), fixed_divide)
    assert result.startswith("ERROR:")
    # The refusal hands the model a ready-to-use targeted patch for the intended change.
    assert "patch_file with these exact arguments" in result
    assert "def divide(a, b):" in result and "ValueError" in result
    # Deletions are never suggested.
    result2 = tools_module.execute_write_file(str(path), "def add(a, b):\n    return a + b\n")
    assert "exact arguments" not in result2


def test_golden_add_function_preserving_existing_faces_no_guard(tmp_path):
    # Golden path: model appends a new function while keeping everything ->
    # no loss detected, write proceeds with no refusal and no hint noise.
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    new_content = CALC_WITH_ADD + "\ndef multiply(a, b):\n    return a * b\n"
    result = tools_module.execute_write_file(str(path), new_content)
    assert result.startswith("Successfully")
    assert "not approved" not in result and "exact arguments" not in result
    assert path.read_text() == new_content


def test_golden_targeted_patch_modification_faces_no_guard(tmp_path):
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    result = tools_module.execute_patch_file(
        str(path), "def divide(a, b):\n    return a / b\n",
        "def divide(a, b):\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n")
    assert result.startswith("Successfully")
    assert "def add(" in path.read_text() and "ValueError" in path.read_text()


def test_recovery_hint_describes_candidate_only_no_deletion_no_authority(tmp_path):
    # The hint is derived from the model's proposed content (never user intent),
    # suggests no deletions, executes nothing, and grants no authority.
    path = tmp_path / "calc.py"
    path.write_text(CALC_WITH_ADD)
    from nexus_agent_ai.agent import tools as tools_module
    tools_module.execute_read_file(str(path))
    result = tools_module.execute_write_file(str(path), "def divide(a, b):\n    return a / b if b else 0\n")
    assert "patch_file with these exact arguments" in result
    # The suggested replacement contains only the intended modification...
    assert "return a / b if b else 0" in result
    # ...and never suggests deleting the dropped function.
    import json as _json
    hint_json = result[result.index('{"path":'):result.rindex("}") + 1]
    args = _json.loads(hint_json)
    assert "def add(" not in args["replacement"] or "def add(" in args["target"]
    assert path.read_text() == CALC_WITH_ADD  # nothing executed
    # A second attempt at the same lossy write is still refused (no authority leaked).
    result2 = tools_module.execute_write_file(str(path), "def divide(a, b):\n    return a / b if b else 0\n")
    assert result2.startswith("ERROR:")
