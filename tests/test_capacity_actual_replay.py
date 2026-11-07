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
    ("filler-3", "failure", None),
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
    # Token constants re-derived under max-view wire estimation (providers
    # .messages.serialized_wire_views): canonical metadata is no longer
    # charged, and each message is charged its largest ACTUAL adapter
    # serialization (OpenAI canonical view, Anthropic content blocks, Gemini
    # parts). Per message the delta vs the old basis is the adapter wrapper
    # (+~8 tokens) minus the removed metadata: nudge 183 -> 182 (wrapper +8,
    # metadata -9), original 91 -> 99 (wrapper +8, no metadata), lower bound
    # 1661 -> 1734 (wrappers, plus the Gemini function_response bound-name
    # duplication on the protected nameless results, charged in JSON-escaped
    # wire form), pair 144 -> 142 (wrapper +16, metadata -18).
    # The behavior contract below is unchanged: a rejected saved nudge must not
    # infer, dispatch, retain, or commit fit state.
    assert [context.message_tokens(message) for message in (original, nudge)] == [99, 182]
    report = replay.replay_snapshot(failure)
    assert report["protected_lower_bound"]["tokens"] == 1734
    assert report["effective_budget_tokens"] == 1564
    assert report["checkpoint_accounting"][-1]["actual_pair_tokens"] == 142

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


def test_success_capture_filler2_now_fails_closed_at_both_reserves():
    """filler-2's capture no longer fits even at the halved reserve.

    Recorded at 1531 tokens under metadata-inclusive estimation. Under
    max-view estimation the adapter wrappers on this tool-call-heavy,
    metadata-light history exceed the removed phantom, so the replay fails
    closed at reserve 512 AND at 256: no over-budget request is sent. That is
    the intended fail-closed behavior, asserted as such; the snapshot is not
    mutated.
    """
    snapshot = load_snapshot("filler-2", "latest_success")
    before = copy.deepcopy(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "mismatch"
    assert snapshot == before
    assert report["token_identical"] is False
    attempts = report["attempts"]
    assert [(a["reserve"], a.get("error", {}).get("class")) for a in attempts] == [
        (512, "ContextBudgetError"), (256, "ContextBudgetError")]


def test_success_capture_filler3_replays_with_wrapper_net_increase():
    """filler-3's capture still replays; the net basis shift is now upward.

    Recorded at 1489 tokens under metadata-inclusive estimation. Under
    max-view estimation this metadata-light capture replays at 1554 tokens:
    the actual adapter wrappers (Anthropic content blocks / Gemini parts)
    plus the Gemini function_response bound-name duplication on its nameless
    stored results exceed the removed phantom bookkeeping. The reserve-halving contract is
    unchanged (512 fails closed, 256 succeeds), genuine-pressure eviction and
    the omission protocol still operate, and the fitted messages are
    byte-identical to the capture - only the accounting changed.
    """
    snapshot = load_snapshot("filler-3", "latest_success")
    before = copy.deepcopy(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "mismatch"
    assert snapshot == before
    assert report["token_identical"] is False
    assert report["hashes"]["captured_fitted"] == report["hashes"]["replayed_fitted"]
    attempts = report["attempts"]
    assert attempts[0]["reserve"] == 512 and attempts[0].get("error", {}).get("class") == "ContextBudgetError"
    assert attempts[-1]["reserve"] == 256 and "stats" in attempts[-1]
    stats = report["replayed_result"]["stats"]
    assert stats["tokens"] == 1554
    assert stats["tokens"] > 1489  # wrappers + name binding exceeded phantom here
    assert stats["omission_tokens"] == 142
    assert stats["evicted_turns"] == 2


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
