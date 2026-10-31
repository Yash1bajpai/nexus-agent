"""Offline transcripts from the physical-PC failures; no real model or web access."""
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
