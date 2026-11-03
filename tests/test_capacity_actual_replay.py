"""Original run41648699327 inputs: offline replay, not model qualification."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from nexus_agent_ai import budget_replay as replay
from nexus_agent_ai.agent import context, core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import RefusalClass, execution_result
from nexus_agent_ai.agent.result_budget import guard_tool_result
from nexus_agent_ai.providers.base import ToolCall


FIXTURES = Path(__file__).parent / "fixtures" / "capacity_stage_a"


def load_snapshot(row, kind):
    name = f"{row}-{kind}.json"
    origin = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    payload = (FIXTURES / name).read_bytes()
    expected = origin["files"][name]
    assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    assert len(payload) == expected["bytes"]
    snapshot = json.loads(payload)
    assert replay.stable_hash(snapshot["canonical_messages"]) == snapshot["input_hash"]
    return snapshot


@pytest.fixture(autouse=True)
def no_runtime_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Saved-input replay must not initialize inference or execute tools")
    monkeypatch.setattr(Agent, "__init__", forbidden)
    monkeypatch.setattr(core, "execute_tool", forbidden)


@pytest.mark.parametrize("row,kind,tokens", [
    ("filler-2", "failure", None),
    ("filler-2", "latest_success", 1531),
    ("filler-3", "failure", None),
    ("filler-3", "latest_success", 1489),
])
def test_original_capture_replays_exact_error_or_success_without_mutation(row, kind, tokens):
    snapshot = load_snapshot(row, kind)
    before = copy.deepcopy(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "reproduced"
    assert report["attempts_match"] is True
    assert snapshot == before
    if tokens is None:
        assert report["error_match"] is True
        expected = snapshot["result"]["error"]
        assert report["replayed_result"]["error"] == {
            "class": expected["class"], "text": expected["text"],
        }
        assert report["classification"] == ("irreducible" if row == "filler-3" else "unresolved")
    else:
        assert report["token_identical"] is True
        assert report["replayed_result"]["stats"]["tokens"] == tokens
        assert report["hashes"]["captured_fitted"] == report["hashes"]["replayed_fitted"]


@pytest.mark.parametrize("row", ["filler-2", "filler-3"])
def test_actual_failure_links_to_unchanged_preceding_success(row):
    failure = load_snapshot(row, "failure")
    success = load_snapshot(row, "latest_success")
    assert failure["previous_success"]["id"] == success["id"]
    assert failure["previous_success"]["input_hash"] == success["input_hash"]
    prefix = failure["transition"]["common_prefix"]
    assert failure["canonical_messages"][:prefix] == success["canonical_messages"]
    assert failure["transition"]["append_only"] is True


def test_actual_filler3_full_nudge_is_rejected_before_another_request(tmp_path, monkeypatch):
    failure = load_snapshot("filler-3", "failure")
    success = load_snapshot("filler-3", "latest_success")
    original, nudge = failure["canonical_messages"][-2:]
    assert original["role"] == "assistant" and not original.get("tool_calls")
    assert nudge["role"] == "user" and nudge["metadata"] == {"internal": True}
    assert failure["projected_messages"] == failure["canonical_messages"]
    assert [context.message_tokens(message) for message in (original, nudge)] == [91, 183]
    report = replay.replay_snapshot(failure)
    assert report["protected_lower_bound"]["tokens"] == 1661
    assert report["effective_budget_tokens"] == 1564
    assert report["checkpoint_accounting"][-1]["actual_pair_tokens"] == 144

    # Use the approved replay's combined numeric overhead surrogate, not private
    # prompts or invented tool schemas. Its planner/reserve policy is unchanged.
    agent = replay._agent(failure["settings"], failure["session_record"])
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="actual-filler3", max_messages=10000)
    memory.replace(success["canonical_messages"])
    memory.set_session_metadata("SESSION_RECORD", failure["session_record"])
    agent.memory = memory
    agent._mutation_segment = ""
    agent.last_trim = {"original": True}
    wire_calls = {call["id"]: call for message in failure["canonical_messages"]
                  for call in message.get("tool_calls", [])}
    agent._pending_calls = [
        {"call": ToolCall(item["id"], item["name"],
                          json.loads(wire_calls[item["id"]]["function"]["arguments"])),
         "state": item["state"], "result": None}
        for item in failure["pending_calls"]
    ]
    pending_before = copy.deepcopy(agent._pending_calls)
    before = copy.deepcopy(memory.get())
    assert before == success["canonical_messages"]
    actual_plan = agent._plan_fit
    candidates = []
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("Rejected saved nudge must not infer, dispatch, retain, or commit fit state")

    def observe(messages, tools, scale=1.0, **kwargs):
        candidates.append(copy.deepcopy(messages))
        return actual_plan(messages, tools, scale, **kwargs)

    agent.provider.complete = forbidden
    agent.provider.stream = forbidden
    monkeypatch.setattr(agent, "_plan_fit", observe)
    monkeypatch.setattr(agent, "_record_fit", forbidden)
    monkeypatch.setattr(agent, "_audit", forbidden)
    monkeypatch.setattr(memory, "store_artifacts", forbidden)
    monkeypatch.setattr(memory, "retain_tool_output", forbidden)
    monkeypatch.setattr(memory, "set_session_metadata", forbidden)
    assert agent._try_automatic_nudge(
        nudge["content"], assistant_text=original["content"], tools=[],
        scale=failure["settings"]["scale"],
    ) is False
    assert candidates == [failure["canonical_messages"]]
    assert memory.get() == before + [original]
    assert memory.get_session_metadata("SESSION_RECORD") == failure["session_record"]
    assert agent._pending_calls == pending_before
    assert agent.output_reserve == 512 and agent.last_trim == {"original": True}
    assert calls == []
    assert load_snapshot("filler-3", "failure") == failure


def test_actual_filler2_receipt_floor_stays_notexecuted_and_unmodified():
    failure = load_snapshot("filler-2", "failure")
    before = copy.deepcopy(failure)
    receipt = failure["receipt"]
    assert receipt["executed"] is False
    assert receipt["retention_events"] == []
    assert failure["pending_calls"] == [{
        "executed": False, "id": receipt["current_call_id"],
        "name": "read_file", "state": "returned",
    }]
    source = execution_result(receipt["raw_result"], executed=False, outcome=receipt["outcome"])
    template = failure["projected_messages"]
    matches = [index for index, message in enumerate(template)
               if message.get("tool_call_id") == receipt["current_call_id"]]
    assert len(matches) == 1
    index = matches[0]
    memory = replay._ReceiptMemory(receipt)
    probes = []

    def fits(content):
        candidate = copy.deepcopy(template)
        candidate[index]["content"] = str(content)
        result, attempts = replay._plan(failure, candidate)
        probes.append((str(content), result, attempts))
        return result["success"]

    with pytest.raises(context.ContextBudgetError) as raised:
        guard_tool_result("read_file", source, memory=memory, fits=fits)
    error = raised.value
    assert type(error) is context.ContextBudgetError
    assert str(error) == failure["result"]["error"]["text"]
    assert error.refusal_class is RefusalClass.CAPACITY
    assert error.executed is False and error.outcome == receipt["outcome"]
    assert str(source) == receipt["raw_result"] and source.executed is False
    assert len(probes) == 4 and all(not result["success"] for _, result, _ in probes)
    assert [attempt["reserve"] for attempt in probes[-1][2]] == [512, 256]
    assert memory.used == [] and memory.missing_event is False
    assert failure == before
