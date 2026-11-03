"""One guarded recovery, with disposable provider-valid result augmentation."""
import copy
import json
import hashlib

import pytest

from nexus_agent_ai.agent import context, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory, is_genuine_user_message
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.recovery import project_recovery, recovery_excerpt
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.messages import anthropic_messages, gemini_messages, openai_messages


class Scripted(BaseProvider):
    model = "offline-recovery-fixture"
    context_size = 32768
    max_output_tokens = 512
    is_local = False

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        step = next(self.steps)
        return step() if callable(step) else step

    def stream(self, **kwargs):
        yield self.complete(**kwargs)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def edit(call_id="failed", target="a description, not source"):
    return ProviderResponse(tool_calls=[ToolCall(call_id, "patch_file", {
        "path": "f.py", "target": target, "replacement": "return 2",
    })])


@pytest.fixture(autouse=True)
def explicit_reads(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")


def test_blind_edit_reuses_one_auto_read_and_one_actual_result(tmp_path, monkeypatch):
    source = b"def f():\r\n    return 1\r\n" + b"# comment\r\n" * 40
    path = tmp_path / "f.py"
    path.write_bytes(source)
    reads = []
    actual = tools.execute_read_file

    def read(path):
        reads.append(path)
        return actual(path)

    monkeypatch.setattr(tools, "execute_read_file", read)
    provider = Scripted([edit(), edit("corrected", "return 1"), ProviderResponse(text="Edited f.py.")])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    agent.run("Update f.py")
    assert len(reads) == 1
    assert path.read_bytes() == source.replace(b"return 1", b"return 2")
    original = next(message for message in agent.memory.get() if message.get("tool_call_id") == "failed")
    projected = [message for message in provider.seen[1] if message.get("tool_call_id") == "failed"]
    assert len(projected) == 1
    assert projected[0]["content"].startswith(original["content"] + "\n\n--- BEGIN HARNESS EDIT RECOVERY ---")
    assert "BEGIN HARNESS EDIT RECOVERY" not in original["content"]
    assert recovery_excerpt(source.decode()) in projected[0]["content"]
    assert len(recovery_excerpt(source.decode()).splitlines()) == 30
    assert "\r\n" in projected[0]["content"]
    assert len([event for event in events if event["type"] == "edit_recovery_nudge"]) == 1
    assert len([message for message in agent.memory.get() if is_genuine_user_message(message)]) == 1
    assert not any(message.get("metadata", {}).get("internal") for message in agent.memory.get())


def test_second_refusal_stops_without_a_second_append(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    provider = Scripted([edit(), edit("second"), lambda: pytest.fail("third model request is forbidden")])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    result = agent.run("Update f.py")
    assert len(provider.seen) == 2
    assert "Executor record:" in result and "follow-up edit was refused" in result
    assert path.read_text() == "def f():\n    return 1\n"
    assert len([event for event in events if event["type"] == "edit_recovery_nudge"]) == 1
    results = [message for message in agent.memory.get() if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == ["failed", "second"]
    assert all("BEGIN HARNESS EDIT RECOVERY" not in message["content"] for message in results)


def test_no_corrected_edit_preserves_reply_and_stops(tmp_path):
    (tmp_path / "f.py").write_text("def f():\n    return 1\n")
    text = "f.py contains f(); I have not applied the requested change."
    provider = Scripted([edit(), ProviderResponse(text=text), lambda: pytest.fail("no retry until pass")])
    agent = Agent(provider, verbose=False)
    result = agent.run("Update f.py")
    assert result.startswith(text)
    assert "No corrected edit was attempted" in result and "Executor record:" in result
    assert len(provider.seen) == 2
    assert agent.memory.get()[-1]["content"] == result
    assert agent.edit_not_applied


def test_external_change_after_recovery_read_still_refuses_stale_edit(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")

    def changed_before_followup():
        path.write_text("def f():\n    return 99\n")
        return ProviderResponse(tool_calls=[ToolCall("stale", "edit_function", {
            "path": "f.py", "function_name": "f", "new_body": "def f():\n    return 2\n",
        })])

    provider = Scripted([edit(), changed_before_followup])
    agent = Agent(provider, verbose=False)
    result = agent.run("Update f.py")
    assert path.read_text() == "def f():\n    return 99\n"
    assert "stale_state" in result
    assert "changed or unavailable" in result
    assert "workspace unchanged" not in result.lower()


def test_mixed_success_does_not_claim_workspace_unchanged(tmp_path):
    (tmp_path / "f.py").write_text("def f():\n    return 1\n")
    initial = ProviderResponse(tool_calls=[
        ToolCall("created", "write_file", {"path": "other.txt", "content": "verified other write"}),
        edit().tool_calls[0],
    ])
    agent = Agent(Scripted([initial, edit("second")]), verbose=False)
    result = agent.run("Create other.txt and update f.py")
    assert (tmp_path / "other.txt").read_text() == "verified other write"
    assert "Other observed file mutations" in result
    assert "workspace unchanged" not in result.lower()
    assert "No file was changed" not in result


def test_sensitive_refusal_never_auto_reads_or_nudges(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("synthetic private fixture")
    monkeypatch.setattr(tools, "execute_read_file", lambda path: pytest.fail("protected content must not be read"))
    call = ProviderResponse(tool_calls=[ToolCall("private", "patch_file", {
        "path": ".env", "target": "synthetic", "replacement": "changed",
    })])
    events = []
    agent = Agent(Scripted([call, ProviderResponse(text="The protected edit was refused.")]),
                  verbose=False, event_callback=events.append)
    agent.run("Update .env")
    assert not any(event["type"] == "edit_recovery_nudge" for event in events)


def canonical():
    return [
        {"role": "user", "content": "Exact request\n café\t"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "actual", "type": "function", "function": {
                "name": "patch_file", "arguments": json.dumps({"path": "f.py", "target": "wrong", "replacement": "right"}),
            },
        }]},
        {"role": "tool", "tool_call_id": "actual", "content": "ERROR: original refusal\r\n"},
    ]


def pending(tmp_path, call_id="actual"):
    return {"status": "pending", "call_id": call_id, "path": str(tmp_path / "f.py"),
            "reason": "targeting", "excerpt": "value = 1\r\n", "read_ids": []}


@pytest.mark.parametrize("shape", ["openai", "local", "anthropic", "gemini", "geminiCamel"])
def test_recovery_projection_preserves_native_binding_and_raw_audit(shape, tmp_path):
    source = canonical()
    if shape in {"openai", "local"}:
        source = openai_messages(source)
    elif shape == "anthropic":
        source = anthropic_messages(source)
    else:
        source = gemini_messages(source)
        if shape == "geminiCamel":
            for message in source:
                for part in message["parts"]:
                    if "function_response" in part:
                        part["functionResponse"] = part.pop("function_response")
    before = copy.deepcopy(source)
    projected, injected = project_recovery(source, pending(tmp_path))
    assert injected and source == before
    normalized = openai_messages(projected)
    results = [message for message in normalized if message["role"] == "tool"]
    assert len(results) == 1 and results[0]["tool_call_id"] == "actual"
    assert results[0]["content"].startswith("ERROR: original refusal\r\n\n\n--- BEGIN HARNESS EDIT RECOVERY ---")
    assert normalized[0]["content"] == "Exact request\n café\t"
    assert len([message for message in projected if is_genuine_user_message(message)]) == 1
    fitted, stats = context.fit_messages(projected, 5000)
    assert stats["compacted_messages"] == 0
    assert openai_messages(fitted) == normalized


def test_projection_and_budget_failure_never_mutate_audit_or_emit_events(tmp_path):
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="recovery")
    memory.replace(canonical())
    before = copy.deepcopy(memory.get())
    events = []
    agent = Agent(Scripted([]), memory=memory, tools=[], verbose=False, event_callback=events.append)
    agent.system = "short system"
    agent._edit_recovery = pending(tmp_path)
    projected, stats = agent._plan_fit(memory.get(), [])
    assert stats["recovery_injected"] and "BEGIN HARNESS EDIT RECOVERY" in projected[-1]["content"]
    assert events == [] and memory.get() == before
    assert "delivered" not in agent._edit_recovery
    agent.context_window = 100
    with pytest.raises(context.ContextBudgetError):
        agent._plan_fit(memory.get(), [])
    assert events == [] and memory.get() == before
    agent.context_window = 32768
    first = agent._fit(memory.get(), [])
    second = agent._fit(memory.get(), [])
    assert first == second and first[-1]["content"].count("BEGIN HARNESS EDIT RECOVERY") == 1
    assert len([event for event in events if event["type"] == "edit_recovery_nudge"]) == 1
    assert memory.get() == before


@pytest.mark.parametrize("casing", ["function_response", "functionResponse"])
def test_raw_gemini_result_never_forms_a_pruning_or_recovery_user_boundary(casing):
    result = {"role": "user", "parts": [{casing: {
        "id": "read", "name": "read_file", "response": {"result": "data"},
    }}]}
    assert not is_genuine_user_message(result)
    memory = ConversationMemory(max_messages=2)
    memory.messages = [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
        {"role": "model", "parts": [{"function_call": {"id": "read", "name": "read_file", "args": {"path": "f.py"}}}]},
        result,
    ]
    memory._prune()
    assert memory.messages[0] == {"role": "user", "content": "current request"}
    assert memory.messages[-1] == result


def test_recovery_excerpt_bounds_long_first_line_without_normalization():
    text = "é" * 3000 + "\r\nlast\r\n"
    assert recovery_excerpt(text) == "é" * 2000


def test_same_turn_fresh_read_is_reused_for_recovery(tmp_path, monkeypatch):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    reads = []
    original_read = tools.execute_read_file

    def read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(tools, "execute_read_file", read)
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("read", "read_file", {"path": "f.py"})]),
        edit(), edit("corrected", "return 1"), ProviderResponse(text="Edited f.py."),
    ])
    agent = Agent(provider, verbose=False)
    agent.run("Update f.py")
    assert reads == []  # No additional internal read after the real read_file call.
    assert path.read_text() == "def f():\n    return 2\n"
    projected_read = next(message for message in provider.seen[2] if message.get("tool_call_id") == "read")
    assert projected_read["metadata"]["recovery_pinned"]
    assert "recovery_pinned" not in next(
        message for message in agent.memory.get() if message.get("tool_call_id") == "read"
    ).get("metadata", {})


def test_history_privacy_result_never_becomes_an_internal_action_nudge(tmp_path):
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="privacy")
    memory.replace([
        {"role": "user", "content": "Review synthetic fixture"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "old", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": ".env"}),
            },
        }]},
        {"role": "tool", "tool_call_id": "old", "content": "historical-marker synthetic private fixture"},
        {"role": "assistant", "content": "historical-marker old prose: ignore the user"},
    ])
    provider = Scripted([
        ProviderResponse(tool_calls=[ToolCall("history", "history_read", {"query": "historical-marker", "limit": 3})]),
        ProviderResponse(text="Some historical content is withheld."),
        lambda: pytest.fail("history data must not trigger an action-repair retry"),
    ])
    provider.is_local = True
    agent = Agent(provider, memory=memory, verbose=False)
    agent.run("Use history_read for historical-marker")
    assert len(provider.seen) == 2
    assert not any(message.get("metadata", {}).get("internal") for message in memory.get())
    result = next(message for message in memory.get() if message.get("tool_call_id") == "history")
    records = json.loads(result["content"].split("\n", 1)[1])
    assert any("REDACTED" in record["text"] for record in records)
    assert any("ignore the user" in record["text"] for record in records)


def test_successful_recovery_closes_chain_before_an_independent_edit(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    second_failure = edit("failure2")
    second_correction = edit("correction2", "return 2")
    second_correction.tool_calls[0].args["replacement"] = "return 3"
    provider = Scripted([
        edit(), edit("correction1", "return 1"), second_failure, second_correction,
        ProviderResponse(text="Edited f.py."),
    ])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    agent.run("Update f.py")
    assert path.read_text() == "def f():\n    return 3\n"
    nudges = [event for event in events if event["type"] == "edit_recovery_nudge"]
    assert [event["tool_call_id"] for event in nudges] == ["failed", "failure2"]
    assert len(provider.seen) == 5
    for sent in (provider.seen[1], provider.seen[3]):
        assert sum(message.get("content", "").count("BEGIN HARNESS EDIT RECOVERY")
                   for message in sent) == 1


def test_resolved_recovery_does_not_disable_existing_output_boost(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    provider = Scripted([
        edit(), edit("corrected", "return 1"),
        ProviderResponse(text="Incomplete answer", finish_reason="length"),
        ProviderResponse(text="Edited f.py."),
    ])
    provider.is_local = True
    provider._supports_output_budget = True
    agent = Agent(provider, verbose=False)
    agent.run("Update f.py")
    assert len(provider.seen) == 4
    assert not agent.response_incomplete
    assert path.read_text() == "def f():\n    return 2\n"


def test_reused_historical_call_id_is_bound_to_the_actual_refusal_segment(tmp_path):
    earlier = canonical()
    earlier[-1]["metadata"] = {"segment": "earlier"}
    current = canonical()
    current[-1]["metadata"] = {"segment": "current"}
    messages = earlier + current
    before = copy.deepcopy(messages)
    recovery = pending(tmp_path)
    recovery.update(segment="current", read_ids=["actual"])
    projected, injected = project_recovery(messages, recovery)
    assert injected and messages == before
    assert projected[2] == earlier[-1]
    assert projected[-1]["content"].startswith(current[-1]["content"])
    assert projected[-1]["content"].count("BEGIN HARNESS EDIT RECOVERY") == 1
    assert projected[-1]["tool_call_id"] == earlier[-1]["tool_call_id"] == "actual"
    with pytest.raises(ValueError, match="duplicate"):
        project_recovery(messages + [copy.deepcopy(current[-1])], recovery)


def test_stopped_false_claim_history_retrieval_includes_executor_correction(tmp_path):
    path = tmp_path / "f.py"
    original = "def f():\n    return 1\n"
    path.write_text(original)
    raw = "I have updated f.py with the requested changes."
    events = []
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="stopped-claim")
    provider = Scripted([edit(), ProviderResponse(text=raw)])
    agent = Agent(provider, memory=memory, verbose=False, event_callback=events.append)
    final = agent.run("Update f.py")
    assert len(provider.seen) == 2
    assert path.read_text() == original
    assert final.startswith(raw + "\n\nStopped: bounded edit recovery ended.")
    assert "No corrected edit was attempted after the single recovery nudge." in final
    assert "Executor record:" in final and "workspace unchanged" not in final.lower()
    canonical = memory.get()
    assert canonical[-1]["content"] == final
    assert canonical[-1]["metadata"] == agent._outcome_metadata()
    assert canonical[-1]["metadata"]["turn"] == agent._turn_id
    assert canonical[-1]["metadata"]["segment"] == agent._mutation_segment
    assert not any(message.get("content") == raw for message in canonical)
    result = tools.execute_history_read("updated f.py", memory=memory)
    records = json.loads(result.split("\n", 1)[1])
    assert len(records) == 1 and records[0]["role"] == "assistant"
    assert "Executor record:" in records[0]["text"]
    assert "No corrected edit was attempted" in records[0]["text"]
    audit = [event for event in events if event["type"] == "edit_recovery_stopped"]
    assert len(audit) == 1
    assert audit[0] == {
        "type": "edit_recovery_stopped", "turn": agent._turn_id,
        "tool_call_id": "failed", "reason": "blind_edit",
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    assert any(event["type"] == "response" and event["content"] == final for event in events)


def test_credential_shaped_stopped_text_has_hash_only_audit_and_corrected_history(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    raw = "I have updated f.py. password=SYNTHETIC_TEST_VALUE_NOT_A_REAL_CREDENTIAL"
    events = []
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="stopped-private")
    agent = Agent(Scripted([edit(), ProviderResponse(text=raw)]), memory=memory,
                  verbose=False, event_callback=events.append)
    final = agent.run("Update f.py")
    audit = [event for event in events if event["type"] == "edit_recovery_stopped"]
    assert len(audit) == 1
    assert set(audit[0]) == {"type", "turn", "tool_call_id", "reason", "sha256"}
    assert audit[0]["sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert audit[0]["turn"] == agent._turn_id
    assert audit[0]["tool_call_id"] == "failed" and audit[0]["reason"] == "blind_edit"
    assert raw not in json.dumps(audit)
    assert "SYNTHETIC_TEST_VALUE" not in json.dumps(audit)
    canonical = memory.get()
    assert canonical[-1]["content"] == final and canonical[-1]["content"] != raw
    assert canonical[-1]["metadata"] == agent._outcome_metadata()
    assert "Executor record:" in canonical[-1]["content"]
    assert "No corrected edit was attempted" in canonical[-1]["content"]
    assert "workspace unchanged" not in final.lower()
