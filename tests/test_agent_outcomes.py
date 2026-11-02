import json

import pytest

from nexus_agent_ai.agent import core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory, is_genuine_user_message
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, RateLimitError, Tool, ToolCall
from nexus_agent_ai.providers.messages import assistant_message


class Scripted(BaseProvider):
    model = "scripted"

    def __init__(self, turns, local=False):
        self.turns = list(turns)
        self.seen = []
        self.is_local = local

    def complete(self, messages, tools, system):
        self.seen.append(list(messages))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, tuple):
            call = ToolCall(f"call_{len(self.seen)}", turn[0], turn[1])
            return ProviderResponse(tool_calls=[call], raw_assistant_message=assistant_message("", [call]))
        return ProviderResponse(text=turn, raw_assistant_message=assistant_message(turn, []))

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": result}


def tool(name):
    return Tool(name, "", {}, lambda **kwargs: "unused")


def test_require_edit_sets_outcome_without_edit_keywords_or_available_tools():
    agent = Agent(Scripted(["I fixed the anomaly."]), tools=[], verbose=False)
    final = agent.run("Investigate the anomaly", require_edit=True)
    assert agent.edit_not_applied and agent.claimed_unapplied_edit
    assert not agent.changed_files
    assert "files were not modified" in final


def test_honest_no_change_is_not_a_false_edit_claim():
    agent = Agent(Scripted(["No code changes are needed; it is already correct."]), tools=[], verbose=False)
    final = agent.run("fix this", require_edit=True)
    assert not agent.edit_not_applied and not agent.claimed_unapplied_edit
    assert "Warning:" not in final


def test_noop_write_does_not_count_as_edit_or_invalidate_tests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "result.py"
    path.write_text("original")

    def execute(name, args):
        if name == "run_tests":
            return "Pytest Results: PASSED"
        path.write_text("original")
        return "No changes: result.py already contains the requested content."

    monkeypatch.setattr(core, "execute_tool", execute)
    agent = Agent(Scripted([
        ("run_tests", {}), ("write_file", {"path": str(path), "content": "original"}),
        "Tests pass. No changes are needed.",
    ]), tools=[tool("run_tests"), tool("write_file")], verbose=False)
    final = agent.run("verify and fix result.py")
    assert not agent.changed_files and not agent.claim_unverified
    assert "files changed after" not in final


def test_changed_files_records_resolved_actual_changes_and_resets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "result.py"

    def execute(name, args):
        path.write_text(args["content"])
        return "Successfully wrote result.py"

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Scripted([("write_file", {"path": "result.py", "content": "new"}), "I wrote the file.", "hello"])
    agent = Agent(provider, tools=[tool("write_file")], verbose=False)
    agent.run("produce the output", require_edit=True)
    assert agent.changed_files == {str(path.resolve())}
    assert not agent.edit_not_applied
    agent.run("hello")
    assert agent.changed_files == set()
    assert not agent.claimed_unapplied_edit


def test_verification_only_claim_is_flagged_with_no_tools():
    agent = Agent(Scripted(["All tests pass."]), tools=[], verbose=False)
    assert "unverified" in agent.run("check tests")
    assert agent.claim_unverified
    assert not agent.edit_not_applied


def test_stream_renders_only_appended_correction_not_whole_answer():
    chunks = []
    agent = Agent(Scripted(["All tests pass."]), tools=[], verbose=False,
                  event_callback=lambda event: chunks.append(event) if event["type"] == "stream_chunk" else None)
    final = agent.run("verify the suite", stream=True)
    rendered = "".join(event["content"] for event in chunks)
    assert "unverified" in final and "unverified" in rendered
    assert rendered.count("All tests pass.") == 1
    assert agent.claim_unverified


def test_stream_without_callback_visibly_appends_grounded_git_output(monkeypatch):
    chunks = []
    monkeypatch.setattr(core.display, "print_stream_chunk", chunks.append)
    monkeypatch.setattr(core, "execute_tool", lambda *args: " M source.py")
    agent = Agent(Scripted([("git_status", {}), "The working tree is clean."]),
                  tools=[tool("git_status")], verbose=False)
    final = agent.run("show git status", stream=True)
    rendered = "".join(chunks)
    assert "Actual git output:" in rendered and "source.py" in rendered
    assert rendered.count("The working tree is clean.") == 1
    assert "Actual git output:" in final


def test_force_final_still_reviews_failed_tests(monkeypatch):
    monkeypatch.setattr(core, "execute_tool", lambda *args: "Pytest Results: FAILED\nassert 1 == 2")
    turns = [("run_tests", {})] * 3 + ["Tests pass."] * 3
    agent = Agent(Scripted(turns), tools=[tool("run_tests")], max_iterations=3, verbose=False)
    final = agent.run("verify the suite")
    assert "run_tests last reported FAILED" in final
    assert agent.claim_unverified
    assert any(message.get("metadata", {}).get("internal") for message in agent.memory.get())


def test_edit_nudges_are_internal_and_retain_original_prompt():
    provider = Scripted(["I will make the change.", "No code changes are needed."], local=True)
    memory = ConversationMemory(max_messages=2)
    agent = Agent(provider, tools=[tool("write_file")], memory=memory, max_iterations=3, verbose=False)
    agent.run("produce the requested output", require_edit=True)
    seen = provider.seen[1]
    assert seen[0]["content"] == "produce the requested output"
    assert len([message for message in seen if is_genuine_user_message(message)]) == 1
    assert seen[-1]["metadata"] == {"internal": True}


def test_iteration_limit_stores_terminal_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "execute_tool", lambda *args: "observed file")
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="limit")
    agent = Agent(Scripted([("read_file", {"path": "a.py"})]), tools=[tool("read_file")],
                  memory=memory, max_iterations=1, verbose=False)
    final = agent.run("diagnose this", require_edit=True)
    assert final.startswith("Stopped: maximum tool iterations")
    assert agent.edit_not_applied
    assert agent.unresolved_errors
    assert memory.get()[-1] == {"role": "assistant", "content": final}


def test_direct_provider_rate_limit_does_not_claim_fallback(monkeypatch):
    warnings = []
    monkeypatch.setattr(core.display, "print_warn", warnings.append)
    agent = Agent(Scripted([RateLimitError("direct")]), tools=[], verbose=False)
    with pytest.raises(RateLimitError):
        agent.run("hello")
    assert warnings and "no fallback provider is available" in warnings[-1]
    assert not any("Switching" in warning for warning in warnings)


def test_raw_sdk_message_is_json_safe_in_memory_and_sqlite(tmp_path):
    class SDKMessage:
        def model_dump(self, **kwargs):
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "sdk", "type": "function", "function": {
                    "name": "read_file", "arguments": json.dumps({"path": "a.py"}),
                },
            }]}

    for memory in (ConversationMemory(), SQLiteMemory(tmp_path / "private" / "history.db", session_id="sdk")):
        memory.add("user", "read a.py")
        memory.add_raw(SDKMessage())
        memory.add_raw({"role": "tool", "tool_call_id": "sdk", "content": "source"})
        messages = memory.get()
        assert json.loads(json.dumps(messages)) == messages
        assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"] == "sdk"
        assert messages[1]["content"] == ""


def test_fallback_transition_and_recovery_rebudget_full_history(monkeypatch):
    from nexus_agent_ai.agent import context
    from nexus_agent_ai.providers import fallback_provider

    primary = Scripted([RateLimitError("primary"), "recovered"])
    primary.context_size = 8192
    primary.max_output_tokens = 2048
    local = Scripted(["local answer"], local=True)
    local.context_size = 2048
    local.max_output_tokens = 512
    monkeypatch.setattr(fallback_provider, "FALLBACK_CHAIN", [
        ("primary", lambda: primary), ("local", lambda: local),
    ])
    provider = fallback_provider.FallbackProvider("primary")
    memory = ConversationMemory(max_messages=500)
    memory.replace([
        {"role": role, "content": f"{index} " + "x" * 600}
        for index in range(30) for role in ("user", "assistant")
    ])
    agent = Agent(provider, memory=memory, tools=[], verbose=False)
    agent.system = "Short test instructions."
    assert agent.run("explain this") == "local answer"
    assert agent.context_window == 2048
    assert agent.output_reserve == provider.max_output_tokens == 512
    assert context.history_tokens(local.seen[0]) <= 2048 - 512 - context.estimate_tokens(agent.system)
    assert len(primary.seen[0]) > len(local.seen[0])
    assert provider.before_request is None
    provider._fallback_time = 0
    provider._recovery_cooldown = 0
    assert agent.run("continue") == "recovered"
    assert agent.context_window == 8192
    assert agent.output_reserve == provider.max_output_tokens == 2048
    assert len(primary.seen[1]) > len(local.seen[0])
    assert provider.before_request is None


@pytest.mark.parametrize("query", [
    "give me the full path of that file",
    "where is the file you created",
    "where did you save the script",
    "where did you write that",
    "what is the path of this file",
    "what is the absolute path",
])
def test_path_location_queries_route_to_tools(query):
    assert core._query_needs_tools(query)


def test_system_prompt_contains_resolved_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(Scripted([]), tools=[], verbose=False)
    assert f"Working directory: {tmp_path.resolve()}" in agent.system


def test_system_prompt_rules_run_file_arithmetic_speaker_identity():
    agent = Agent(Scripted([]), tools=[], verbose=False)
    system = agent.system
    # Rule 14: run_file honest limit
    assert "Executing existing files or scripts is disabled for safety" in system
    assert "offer run_code for pure computation or run_tests for project tests" in system
    # Rule 15: speaker identity
    assert "The user is the person typing; never claim the user's name" in system


def test_zero_tool_file_creation_claim_is_corrected_without_retry():
    # Output-claim guard: model claims a concrete file was created although no
    # mutating tool ran; the request itself carries no edit intent keywords.
    agent = Agent(Scripted(["I've created calculator.py with the add function."]), tools=[], verbose=False)
    final = agent.run("make me a small example for addition")
    assert agent.claimed_unapplied_edit
    assert "no file tool modified any file" in final
    assert len(agent.turns if hasattr(agent, "turns") else []) == 0  # no extra provider round-trip


def test_zero_tool_file_claim_does_not_repair_retry():
    provider = Scripted(["Done - math_utils.py has been saved with the helper."])
    agent = Agent(provider, tools=[], verbose=False)
    final = agent.run("set up the helper")
    assert agent.claimed_unapplied_edit
    assert len(provider.seen) == 1  # fail clean: exactly one provider call, no auto-retry


def test_pure_advice_with_filename_is_not_a_false_claim():
    agent = Agent(Scripted(["You can create config.py by adding these lines:"]), tools=[], verbose=False)
    final = agent.run("how do I make a config file?")
    assert not agent.claimed_unapplied_edit
    assert "Correction:" not in final
