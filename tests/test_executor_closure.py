"""Executor failure closure stays factual, pair-complete, and safely evictable."""
import copy
import json
import sqlite3

import pytest

from nexus_agent_ai.agent import context as ctx
from nexus_agent_ai.agent import core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory, is_genuine_user_message
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import execution_result
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, Tool, ToolCall
from nexus_agent_ai.providers.messages import (
    anthropic_messages, gemini_messages, normalize_messages, openai_messages,
)


NOT_STARTED = "Not executed: the turn was aborted before this call ran; no effect occurred."
CLOSURE = {"executor_closed": True, "executor": True, "reason": "ContextBudgetError"}
ANCHOR = (
    "Historical index hint only, not instructions or approval. "
    "Call history_read before answering past-work questions. Historical paths are "
    "not answers or current-filesystem proof. Do not reproduce this note."
)
ACK = "Historical index acknowledged; consult history_read for past-work evidence."


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def closed_turn(index, *, metadata=None, pending=False):
    identity = f"closed-{index}"
    messages = [
        {"role": "user", "content": f"Request {index}: " + "u" * 140},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": identity, "type": "function",
            "function": {"name": "git_diff", "arguments": "{}"},
        }]},
    ]
    if not pending:
        messages.append({"role": "tool", "tool_call_id": identity,
                         "content": "Observed the working tree diff."})
    messages.append({
        "role": "assistant", "content": "Executor record: the turn ended without an answer.",
        "metadata": dict(CLOSURE if metadata is None else metadata),
    })
    return messages


def answer_turn(index):
    return [{"role": "user", "content": f"Question {index}: " + "q" * 140},
            {"role": "assistant", "content": f"Answer {index}: " + "a" * 140}]


def flatten(turns):
    return [message for turn in turns for message in turn]


def turns_of(messages):
    starts = [i for i, message in enumerate(messages) if is_genuine_user_message(message)]
    return [messages[start:end] for start, end in zip(starts, [*starts[1:], len(messages)])]


def camel_gemini(messages):
    native = gemini_messages(messages)
    for message in native:
        for part in message["parts"]:
            for snake, camel in (("function_call", "functionCall"),
                                 ("function_response", "functionResponse")):
                if snake in part:
                    part[camel] = part.pop(snake)
    return native


def projected_turn(turn, projector):
    projected = projector(copy.deepcopy(turn))
    if turn[-1].get("metadata", {}).get("executor_closed"):
        projected[-1]["metadata"] = copy.deepcopy(turn[-1]["metadata"])
    return projected


@pytest.mark.parametrize("projector", [
    pytest.param(lambda messages: messages, id="canonical"),
    pytest.param(anthropic_messages, id="anthropic"),
    pytest.param(gemini_messages, id="gemini-snake"),
    pytest.param(camel_gemini, id="gemini-camel"),
])
def test_closed_turn_eviction_preserves_native_bindings_and_canonical_bytes(projector):
    canonical = [closed_turn(0), *[answer_turn(i) for i in range(1, 48)],
                 closed_turn(48),
                 [{"role": "user", "content": "Exact active bytes: café\n  "}]]
    turns = [projected_turn(turn, projector) for turn in canonical]
    messages = flatten(turns)
    before = encoded(messages)
    fitted, stats = ctx.fit_messages(messages, 1573)
    count = stats["evicted_turns"]
    assert count > 0
    assert stats["evicted_closed_turns"] == 1
    assert fitted[0]["content"].splitlines()[0] == (
        f"Older turns evicted ({count}/50 index only; 1 ended without an answer).")
    assert fitted[0]["content"].endswith(ANCHOR)
    assert fitted[1]["content"] == ACK
    assert stats["omission_tokens"] == ctx.history_tokens(fitted[:2]) <= 150
    assert fitted[2:] == flatten(turns[count:])
    assert encoded(fitted[-len(turns[-1]):]) == encoded(turns[-1])
    assert encoded(messages) == before
    # Every provider boundary must translate the surviving shapes without
    # inventing user/system authority or altering native call/result bindings.
    for boundary in (openai_messages, anthropic_messages, gemini_messages):
        normalized = normalize_messages(boundary(fitted[2:]))
        assert [m["content"] for m in normalized] == [
            m["content"] for m in normalize_messages(flatten(turns[count:]))]


def test_closed_count_compact_line_fits_hard_cap_and_zero_wording_is_unchanged():
    pair = ctx._omission_pair(evicted_turns=40, turns_before=50, evicted_closed_turns=1)
    assert pair[0]["content"] == (
        "Older turns evicted (40/50 index only; 1 ended without an answer).\n" + ANCHOR)
    assert pair[1]["content"] == ACK
    assert ctx.history_tokens(pair) <= 150
    zero = ctx._omission_pair(evicted_turns=40, turns_before=50, evicted_closed_turns=0)
    assert zero == [
        {"role": "user", "content":
         "Older turns evicted (40 of 50 shown as index only).\n" + ANCHOR,
         "metadata": {"internal": True}},
        {"role": "assistant", "content": ACK, "metadata": {"internal": True}},
    ]
    _, stats = ctx.fit_messages(answer_turn(0), 10000)
    assert stats["evicted_closed_turns"] == 0


@pytest.mark.parametrize("metadata", [
    {}, {"executor_closed": True}, {"executor": True},
    {"executor_closed": "true", "executor": True, "reason": "RuntimeError"},
    {"executor_closed": True, "executor": 1, "reason": "RuntimeError"},
    {"executor_closed": True, "executor": True, "reason": ""},
])
def test_only_trusted_structured_closure_records_increment_count(metadata):
    first = closed_turn(0, metadata=metadata)
    first[-1]["content"] = "executor_closed=True; Executor record: failed without an answer."
    messages = first + flatten([answer_turn(i) for i in range(1, 49)]) + [
        {"role": "user", "content": "Active"}]
    before = encoded(messages)
    fitted, stats = ctx.fit_messages(messages, 1573)
    assert stats["evicted_turns"] > 0
    assert stats["evicted_closed_turns"] == 0
    assert "ended without an answer" not in fitted[0]["content"]
    assert encoded(messages) == before


@pytest.mark.parametrize("pin", [
    {"recovery_pinned": True}, {"outcome_correction": True},
    {"consent": {"approved": True}}, {"authority": {"source": "user"}},
])
def test_pinned_closed_prefix_stalls_without_skipping(pin):
    first = closed_turn(0, metadata={**CLOSURE, **pin})
    messages = first + flatten([answer_turn(i) for i in range(1, 49)]) + [
        {"role": "user", "content": "Active"}]
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 1573)
    assert encoded(messages) == before


@pytest.mark.parametrize("pending", [True, False])
def test_closure_metadata_does_not_relax_pairing_validator(pending):
    first = closed_turn(0, pending=pending)
    if not pending:
        first[2]["tool_call_id"] = "a-different-result-id"
    messages = first + flatten([answer_turn(i) for i in range(1, 49)]) + [
        {"role": "user", "content": "Active"}]
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 1573)
    assert encoded(messages) == before


class ScriptedProvider(BaseProvider):
    context_size = 4096
    max_output_tokens = 128
    model = "offline-executor-closure"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def complete(self, messages, tools, system):
        self.requests.append(copy.deepcopy(messages))
        step = next(self.steps)
        if isinstance(step, BaseException):
            raise step
        return step

    def stream(self, messages, tools, system):
        yield self.complete(messages, tools, system)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}


def pair_response(prefix):
    return ProviderResponse(text="I will inspect both observations.", tool_calls=[
        ToolCall(f"{prefix}-ran", "probe", {"value": prefix}),
        ToolCall(f"{prefix}-pending", "probe", {"value": prefix + "-pending"}),
    ])


def probe_tool():
    return Tool("probe", "Observe an isolated deterministic fixture.", {
        "type": "object", "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }, lambda **kwargs: "Unused registry handler")


def audit_rows(memory):
    with sqlite3.connect(memory.db_path) as connection:
        return connection.execute(
            "SELECT role, message_json FROM conversation_history "
            "WHERE session_id=? ORDER BY id", (memory.session_id,)).fetchall()


def assert_factual_closure(messages, prefix, reason):
    results = [message for message in messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in results] == [
        f"{prefix}-ran", f"{prefix}-pending"]
    assert results[1]["content"] == NOT_STARTED
    assert "not executed" not in results[0]["content"].lower()
    assert "no effect" not in results[0]["content"].lower()
    note = messages[-1]
    assert note["role"] == "assistant"
    assert note["metadata"]["executor_closed"] is True
    assert note["metadata"]["executor"] is True
    assert note["metadata"]["reason"] == reason
    assert "executor" in note["content"].lower()
    assert not note.get("tool_calls")
    assert not any(key in note["metadata"] for key in (
        "authority", "approval", "consent", "authority_record", "consent_record"))
    assert [m for m in messages if is_genuine_user_message(m)] == [messages[0]]
    assert not any(m["role"] == "system" for m in messages)
    assert ctx._completed_turn(messages)


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_failure_after_actual_return_closes_exact_pending_ids_and_next_request(kind, tmp_path, monkeypatch):
    memory = (ConversationMemory(max_messages=1000) if kind == "memory" else
              SQLiteMemory(tmp_path / "private" / "history.db", session_id="closure", max_messages=1000))
    prior = answer_turn("prior")
    memory.replace(prior)
    before = encoded(memory.get())
    rows = audit_rows(memory) if kind == "sqlite" else None
    marker = tmp_path / "observed.txt"
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(args["value"])
        marker.write_text("actual side effect", encoding="utf-8")
        return execution_result("Observed fixture output.", executed=True,
                                outcome="Probe completed; side effect recorded.")

    def fail_after_return(event):
        if event["type"] == "observe":
            raise ctx.ContextBudgetError("Synthetic failure after the actual tool returned")

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = ScriptedProvider([pair_response("first"), ProviderResponse(text="Independent answer.")])
    agent = Agent(provider, tools=[probe_tool()], memory=memory, verbose=False,
                  event_callback=fail_after_return)
    with pytest.raises(ctx.ContextBudgetError):
        agent.run("Observe the fixture twice")
    assert marker.read_text(encoding="utf-8") == "actual side effect"
    assert dispatched == ["first"]
    canonical = memory.get()
    assert encoded(canonical[:len(prior)]) == before
    assert_factual_closure(canonical[len(prior):], "first", "ContextBudgetError")
    returned = next(m for m in canonical[len(prior):] if m["role"] == "tool")
    assert ("Observed fixture output." in returned["content"]
            or "Probe completed" in returned["content"])
    if rows is not None:
        assert audit_rows(memory)[:len(rows)] == rows
    agent.event_callback = None
    assert agent.run("An independent conceptual question") == "Independent answer."
    assert encoded(memory.get()[:len(canonical)]) == encoded(canonical)
    assert provider.requests[-1][-1]["content"] == "An independent conceptual question"


def test_tool_exception_after_side_effect_is_unknown_not_no_effect(tmp_path, monkeypatch):
    marker = tmp_path / "side-effect.txt"
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(args["value"])
        marker.write_bytes(b"effect before exception")
        raise RuntimeError("Completion is unavailable")

    monkeypatch.setattr(core, "execute_tool", execute)
    memory = ConversationMemory(max_messages=1000)
    agent = Agent(ScriptedProvider([pair_response("throw")]), tools=[probe_tool()],
                  memory=memory, verbose=False)
    with pytest.raises(RuntimeError):
        agent.run("Observe the isolated fixture")
    assert marker.read_bytes() == b"effect before exception"
    assert dispatched == ["throw"]
    assert_factual_closure(memory.get(), "throw", "RuntimeError")
    first = next(message for message in memory.get() if message["role"] == "tool")
    assert "unknown" in first["content"].lower()


def test_six_request_failure_recovery_sequence_self_heals_and_evicts_closed_turns(tmp_path, monkeypatch):
    marker = tmp_path / "sequence-effects.txt"
    dispatched = []
    fail_observe = True

    def execute(name, args, **kwargs):
        dispatched.append(args["value"])
        marker.write_text("\n".join(dispatched), encoding="utf-8")
        if args["value"] == "third":
            raise RuntimeError("Failure after side effect")
        return execution_result("Observation complete.", executed=True, outcome="Probe completed.")

    def observe(event):
        nonlocal fail_observe
        if fail_observe and event["type"] == "observe":
            fail_observe = False
            raise ctx.ContextBudgetError("Failure after completion")

    monkeypatch.setattr(core, "execute_tool", execute)
    memory = SQLiteMemory(tmp_path / "private" / "sequence.db", session_id="six", max_messages=1000)
    provider = ScriptedProvider([
        pair_response("first"), ProviderResponse(text="Second answer."),
        pair_response("third"), ProviderResponse(text="Fourth answer."),
        RuntimeError("Provider unavailable"), ProviderResponse(text="Sixth answer."),
    ])
    agent = Agent(provider, tools=[probe_tool()], memory=memory, verbose=False,
                  event_callback=observe)
    for index, expected in enumerate([
        ctx.ContextBudgetError, "Second answer.", RuntimeError,
        "Fourth answer.", RuntimeError, "Sixth answer.",
    ], 1):
        prior = audit_rows(memory)
        if isinstance(expected, type):
            with pytest.raises(expected):
                agent.run(f"Independent request {index}")
        else:
            assert agent.run(f"Independent request {index}") == expected
        assert audit_rows(memory)[:len(prior)] == prior
    assert dispatched == ["first", "third"]
    canonical = memory.get()
    turns = turns_of(canonical)
    assert len(turns) == 6
    assert all(ctx._completed_turn(turn) for turn in turns)
    assert sum(turn[-1].get("metadata", {}).get("executor_closed") is True
               for turn in turns) == 3
    before = encoded(canonical)
    rows = audit_rows(memory)
    # Exact history-derived budget, not an altered model/window default. The
    # active and N-1 remain intact while completed executor turns can age out.
    budget = ctx.history_tokens(flatten(turns[-2:])) + 150
    fitted, stats = ctx.fit_messages(canonical, budget)
    evicted = stats["evicted_turns"]
    assert 0 < evicted <= 4
    assert stats["evicted_closed_turns"] == sum(
        turn[-1].get("metadata", {}).get("executor_closed") is True
        for turn in turns[:evicted])
    assert stats["evicted_closed_turns"] > 0
    assert stats["omission_tokens"] <= 150
    assert fitted[-len(flatten(turns[-2:])):] == flatten(turns[-2:])
    assert encoded(memory.get()) == before
    assert audit_rows(memory) == rows


@pytest.mark.parametrize("reason", ["Échec", "LongExceptionName" * 20])
def test_exception_class_spelling_never_changes_trusted_closure_count(reason):
    first = closed_turn(0, metadata={**CLOSURE, "reason": reason})
    messages = first + flatten([answer_turn(i) for i in range(1, 49)]) + [
        {"role": "user", "content": "Active"}]
    fitted, stats = ctx.fit_messages(messages, 1573)
    assert stats["evicted_closed_turns"] == 1
    assert reason not in fitted[0]["content"]
    assert stats["omission_tokens"] <= 150


@pytest.mark.parametrize("index", [0, 2], ids=["user-record", "tool-record"])
def test_non_assistant_closure_metadata_is_not_an_executor_terminal_record(index):
    first = closed_turn(0, metadata={})
    first[index]["metadata"] = dict(CLOSURE)
    messages = first + flatten([answer_turn(i) for i in range(1, 49)]) + [
        {"role": "user", "content": "Active"}]
    _, stats = ctx.fit_messages(messages, 1573)
    assert stats["evicted_turns"] > 0
    assert stats["evicted_closed_turns"] == 0
