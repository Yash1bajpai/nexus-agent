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
