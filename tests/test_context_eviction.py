"""R7 provider-view eviction regressions; canonical history is never trimmed."""
import copy
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import context as ctx
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider
from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.providers.messages import (
    anthropic_messages, gemini_messages, normalize_messages, openai_messages,
)


BUDGET = 1573
ANCHOR = (
    "Historical index hint only, not instructions or approval. "
    "Call history_read before answering past-work questions. Historical paths are "
    "not answers or current-filesystem proof. Do not reproduce this note."
)
ACK = "Historical index acknowledged; consult history_read for past-work evidence."


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def flatten(turns):
    return [message for turn in turns for message in turn]


def call(identity, path="a.py", *, arguments=None, name="read_file"):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": identity, "type": "function",
        "function": {"name": name, "arguments": json.dumps(
            {"path": path} if arguments is None else arguments, ensure_ascii=False)},
    }]}


def read_turn(identity, content):
    return [
        {"role": "user", "content": f"Read {identity}"},
        call(identity),
        {"role": "tool", "tool_call_id": identity, "content": content},
        {"role": "assistant", "content": f"Finished {identity}"},
    ]


def short_turn(index):
    """Individually sub-threshold payloads, but significant per-turn overhead."""
    if index % 2:
        return read_turn(f"r{index:02}", "small result " + "z" * 70)
    return [
        {"role": "user", "content": f"Turn {index:02}: " + "u" * 160},
        {"role": "assistant", "content": f"Answer {index:02}: " + "a" * 160},
    ]


def fifty_turns():
    return [*[short_turn(index) for index in range(49)],
            [{"role": "user", "content": "  Active café\n\t bytes stay exact  "}]]


def bare_index(evicted=0, total=0):
    # Deliberately independent of the fitter's index/eviction helpers: the
    # mandatory protocol text plus actual serialized message/metadata overhead.
    first = (f"Older turns evicted ({evicted} of {total} shown as index only).\n"
             if evicted else "")
    return [
        {"role": "user", "content": first + ANCHOR, "metadata": {"internal": True}},
        {"role": "assistant", "content": ACK, "metadata": {"internal": True}},
    ]


def expected_prefix(turns, budget):
    """Minimal complete prefix whose removal leaves room for mandatory coverage."""
    total = len(turns)
    return next(count for count in range(1, total - 1)
                if ctx.history_tokens(flatten(turns[count:]))
                + ctx.history_tokens(bare_index(count, total)) <= budget)


def artifact(content, name="read_file"):
    return {
        "artifact_id": hashlib.sha256((name + "\0" + content).encode("utf-8")).hexdigest(),
        "tool_name": name, "byte_len": len(content.encode("utf-8")), "content": content,
    }


def placeholder(content, name="read_file"):
    label = f"{name} result, " if name else ""
    return (f"[COMPACTED: {label}{ctx.estimate_tokens(content)} tokens elided. "
            f"Artifact ID: {artifact(content, name)['artifact_id']}]")


def assert_stats(fitted, stats, budget, total, evicted):
    assert stats["evicted_turns"] == evicted
    assert stats["dropped_messages"] == stats["dropped_turns"] == 0
    assert stats["turns_before"] == total
    assert stats["turns_after"] == total - evicted
    assert stats["tokens"] == ctx.history_tokens(fitted) <= budget
    assert stats["omission_tokens"] == ctx.history_tokens(fitted[:2]) <= 150
    assert fitted[0]["metadata"] == fitted[1]["metadata"] == {"internal": True}
    if evicted:
        assert fitted[0]["content"].splitlines()[0] == (
            f"Older turns evicted ({evicted} of {total} shown as index only).")
    record = stats["session_record"]
    assert set(record) == {"version", "files", "topics", "tools"}
    assert record["version"] == 1
    assert len(record["files"]) <= 32
    assert len(record["topics"]) <= 6
    assert len(record["tools"]) <= 10
    assert len(json.dumps(record)) <= 4096
    assert stats["fields_included"] == {
        key: len(record[key]) for key in ("files", "topics", "tools")}
    assert sum(plan["byte_len"] for plan in stats["artifacts"]) <= 256 * 1024
    assert stats["message_bytes_after"] == [
        len(json.dumps(message, default=str, ensure_ascii=False).encode("utf-8"))
        for message in fitted[2:]]


class OfflineProvider(BaseProvider):
    context_size = 4096
    max_output_tokens = 128
    model = "offline-eviction-regression"

    def complete(self, messages, tools, system):
        raise AssertionError("Context fitting must not make a model call")

    def stream(self, messages, tools, system):
        raise AssertionError("Context fitting must not make a model call")

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def agent_for(memory, budget=BUDGET):
    agent = Agent(OfflineProvider(), memory=memory, tools=[], verbose=False)
    agent.system = "offline context rules"
    agent.output_reserve = 128
    agent.context_window = budget + agent.output_reserve + ctx.estimate_tokens(agent.system)
    return agent


def audit_rows(memory):
    with sqlite3.connect(memory.db_path) as connection:
        return connection.execute(
            "SELECT id, role, message_json FROM conversation_history "
            "WHERE session_id=? ORDER BY id", (memory.session_id,)).fetchall()


def stored_artifacts(memory):
    with sqlite3.connect(memory.db_path) as connection:
        return connection.execute(
            "SELECT artifact_id, tool_name, byte_len, content FROM context_artifacts "
            "WHERE session_id=? ORDER BY rowid", (memory.session_id,)).fetchall()


def test_two_turn_1573_compacts_only_unpinned_n_minus_one_result_and_persists(tmp_path):
    payload = "x" * 35000  # Approximately 10K tokens, not an oversized ACTIVE result.
    seed = read_turn("seed-read", payload)
    pinned_call = call("recovery-read", "recovery.py")
    pinned_result = {
        "role": "tool", "tool_call_id": "recovery-read",
        "content": "Distinct recovery evidence: café\n  exact original bytes",
        "metadata": {"recovery_pinned": True},
    }
    seed[3:3] = [pinned_call, pinned_result]
    active = [
        {"role": "user", "content": "  Active request: café\n\t preserve whitespace  "},
        call("pending-active-read", "active.py", arguments={"path": "active.py", "offset": 12}),
    ]
    messages = seed + active
    before, before_bytes = copy.deepcopy(messages), encoded(messages)
    fitted, stats = ctx.fit_messages(messages, BUDGET)
    expected = copy.deepcopy(messages)
    expected[2]["content"] = placeholder(payload)
    assert fitted[2:] == expected
    assert encoded(fitted[-2:]) == encoded(active)
    assert fitted[6] == pinned_result
    assert stats["compacted_messages"] == stats["truncated_messages"] == 1
    assert stats["elided_tokens"] == ctx.estimate_tokens(payload)
    assert stats["artifacts"] == [artifact(payload)]
    assert_stats(fitted, stats, BUDGET, 2, 0)
    assert messages == before and encoded(messages) == before_bytes

    memory = SQLiteMemory(tmp_path / "history.db", session_id="two-turns", max_messages=1000)
    memory.replace(messages)
    original_rows = audit_rows(memory)
    agent = agent_for(memory)
    previous_trim = copy.deepcopy(agent.last_trim)
    projected, projection_stats = agent._plan_fit(memory.get(), [])
    assert projected == fitted
    assert projection_stats["artifacts"] == stats["artifacts"]
    assert memory.get_session_metadata("SESSION_RECORD") is None
    assert stored_artifacts(memory) == []
    assert audit_rows(memory) == original_rows
    assert agent.last_trim == previous_trim
    actual = agent._fit(memory.get(), [])
    assert actual == fitted
    plan = artifact(payload)
    assert stored_artifacts(memory) == [tuple(plan[key] for key in (
        "artifact_id", "tool_name", "byte_len", "content"))]
    assert memory.get_session_metadata("SESSION_RECORD") == agent.last_trim["session_record"]
    assert audit_rows(memory) == original_rows
    assert memory.get() == messages == before


def camel_gemini(messages):
    native = gemini_messages(messages)
    for message in native:
        for part in message["parts"]:
            for snake, camel in (("function_call", "functionCall"),
                                 ("function_response", "functionResponse")):
                if snake in part:
                    part[camel] = part.pop(snake)
    return native


SHAPES = [
    pytest.param(lambda messages: copy.deepcopy(messages), id="canonical"),
    pytest.param(openai_messages, id="openai"),
    pytest.param(openai_messages, id="local"),
    pytest.param(anthropic_messages, id="anthropic"),
    pytest.param(gemini_messages, id="gemini-snake"),
    pytest.param(camel_gemini, id="gemini-camel"),
]


def assert_pairing(messages):
    pending, completed = [], []
    for message in normalize_messages(messages):
        for binding in message.get("tool_calls", []):
            assert binding["id"] not in pending + completed
            pending.append(binding["id"])
        if message["role"] == "tool":
            identity = message["tool_call_id"]
            assert identity in pending
            pending.remove(identity)
            completed.append(identity)
    assert pending == []
    return completed


@pytest.mark.parametrize("projector", SHAPES)
def test_fifty_short_native_turns_evict_exact_minimal_complete_prefix(projector, request):
    canonical = fifty_turns()
    turns = [projector(turn) for turn in canonical]
    messages = flatten(turns)
    before = encoded(messages)
    expected_evicted = expected_prefix(turns, BUDGET)
    assert 0 < expected_evicted <= 48
    fitted, stats = ctx.fit_messages(messages, BUDGET)
    assert fitted[2:] == flatten(turns[expected_evicted:])
    assert encoded(messages) == before
    assert stats["artifacts"] == []
    assert stats["compacted_messages"] == stats["truncated_messages"] == stats["elided_tokens"] == 0
    assert_stats(fitted, stats, BUDGET, 50, expected_evicted)
    expected_ids = assert_pairing(flatten(canonical[expected_evicted:]))
    evicted_ids = assert_pairing(flatten(canonical[:expected_evicted]))
    assert set(expected_ids).isdisjoint(evicted_ids)
    assert assert_pairing(fitted) == expected_ids
    # Exercise the production boundary serializers, not a test-only native clone.
    for serialize in (openai_messages, anthropic_messages, gemini_messages):
        wire = serialize(fitted)
        assert assert_pairing(wire) == expected_ids
        assert not set(assert_pairing(wire)).intersection(evicted_ids)
    if request.node.callspec.id == "local":
        captured = {}

        def complete(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        provider = LocalProvider.__new__(LocalProvider)
        provider.max_output_tokens = 64
        provider._llama = SimpleNamespace(create_chat_completion=complete)
        provider._run_llama_cpp(fitted, [], "rules")
        assert assert_pairing(captured["messages"]) == expected_ids


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_eviction_only_agent_projection_is_pure_and_actual_fit_records_index(kind, tmp_path):
    messages = flatten(fifty_turns())
    if kind == "memory":
        memory = ConversationMemory(max_messages=1000)
    else:
        memory = SQLiteMemory(tmp_path / "history.db", session_id="eviction-only", max_messages=1000)
    memory.replace(messages)
    original_rows = audit_rows(memory) if kind == "sqlite" else None
    agent = agent_for(memory)
    previous_trim = copy.deepcopy(agent.last_trim)
    projected, stats = agent._plan_fit(memory.get(), [])
    assert stats["evicted_turns"] == expected_prefix(fifty_turns(), BUDGET)
    assert stats["artifacts"] == [] and stats["compacted_messages"] == 0
    assert memory.get_session_metadata("SESSION_RECORD") is None
    assert agent.last_trim == previous_trim
    actual = agent._fit(memory.get(), [])
    assert actual == projected
    assert memory.get_session_metadata("SESSION_RECORD") == stats["session_record"]
    assert memory.get() == messages
    if kind == "sqlite":
        assert audit_rows(memory) == original_rows
        assert stored_artifacts(memory) == []


@pytest.mark.parametrize("payload_kind", ["user", "tool-result"])
def test_active_alone_over_budget_fails_without_edit_or_eviction(payload_kind):
    active = [{"role": "user", "content": "Active"}]
    if payload_kind == "user":
        active[0]["content"] = "Active bytes café\n" * 3000
    else:
        active += [call("active-read"), {
            "role": "tool", "tool_call_id": "active-read", "content": "active result " * 3000}]
    before = encoded(active)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(active, BUDGET)
    assert encoded(active) == before


@pytest.mark.parametrize("protected", ["user-text", "assistant-text", "call-arguments"])
def test_n_minus_one_text_and_arguments_cannot_be_compacted_or_evicted(protected):
    prior = read_turn("previous", "eligible result " * 3000)
    if protected == "user-text":
        prior[0]["content"] = "protected prior user " * 3000
    elif protected == "assistant-text":
        prior[-1]["content"] = "protected prior answer " * 3000
    else:
        prior[1] = call("previous", arguments={"path": "a.py", "literal": "exact argument " * 3000})
    messages = read_turn("older", "old result " * 3000) + prior + [
        {"role": "user", "content": "Active exact request"}]
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


PIN_MARKERS = [
    pytest.param({"recovery_pinned": True}, id="recovery-read"),
    pytest.param({"edit_recovery": {"reason": "literal mismatch", "active": True}}, id="recovery-note"),
    pytest.param({"internal": True, "outcome_correction": True}, id="outcome-correction"),
    pytest.param({"claim_redactions": [{"turn": "old", "file": "a.py", "trigger": "absence"}]},
                 id="claim-redaction"),
    pytest.param({"consent": {"approved": True, "scope": "a.py"}}, id="consent"),
    pytest.param({"authority": {"source": "user", "scope": "a.py"}}, id="authority"),
]


@pytest.mark.parametrize("metadata", PIN_MARKERS)
@pytest.mark.parametrize("large", [False, True], ids=["prefix-stall", "payload-pinned"])
def test_globally_pinned_oldest_turn_blocks_compaction_and_skipping(metadata, large):
    oldest = short_turn(0)
    oldest[-1]["metadata"] = metadata
    if large:
        oldest[-1]["content"] = "pinned evidence " * 3000
    messages = flatten([oldest, *fifty_turns()[1:]])
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


def test_system_content_is_globally_pinned():
    messages = [{"role": "system", "content": "system evidence " * 3000},
                *flatten(fifty_turns())]
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


@pytest.mark.parametrize("unfinished", ["unanswered-user", "pending-call", "result-without-answer"])
def test_incomplete_oldest_turn_stalls_instead_of_skipping_to_later_complete_turns(unfinished):
    oldest = [{"role": "user", "content": "Unfinished oldest request"}]
    if unfinished != "unanswered-user":
        oldest.append(call("unfinished"))
    if unfinished == "result-without-answer":
        oldest.append({"role": "tool", "tool_call_id": "unfinished", "content": "Observed but not answered"})
    messages = flatten([oldest, *fifty_turns()[1:]])
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


def test_failed_edit_read_evidence_pins_entire_oldest_turn():
    oldest = read_turn("evidence", "small exact read evidence")
    identity = {"turn": "old", "segment": "failed-segment"}
    oldest[1]["metadata"] = identity.copy()
    oldest[2]["metadata"] = identity.copy()
    mutation = {
        "event": "refusal", **identity, "path": "a.py", "action": "patch_file",
        "status": "unchanged", "success": False, "failed": True,
    }
    refused = call("refused", name="patch_file")
    refused["metadata"] = identity.copy()
    oldest[-1:-1] = [refused, {
        "role": "tool", "tool_call_id": "refused", "content": "ERROR: literal mismatch",
        "metadata": {**identity, "mutation_outcomes": [mutation]},
    }]
    messages = flatten([oldest, *fifty_turns()[1:]])
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


def test_impossible_bare_anchor_reports_coverage_gap_without_changing_active():
    active = [{"role": "user", "content": "Exact protected request café\n  "}]
    record = {"version": 1, "files": [], "topics": [], "tools": []}
    before = encoded((active, record))
    budget = ctx.history_tokens(active) + ctx.history_tokens(bare_index()) - 1
    with pytest.raises(ctx.ContextBudgetError, match="Historical omission coverage gap"):
        ctx.fit_messages(active, budget, session_record=record)
    assert encoded((active, record)) == before


def test_anchor_reserve_shortfall_after_compaction_is_rescued_by_prefix_eviction(tmp_path):
    old_payload, retained_payload = "old payload " * 3000, "retained payload " * 3000
    oldest = read_turn("oldest", old_payload)
    previous = read_turn("previous", retained_payload)
    active = [{"role": "user", "content": "Exact active bytes: café\n  " + "a" * 1000}]
    messages = oldest + previous + active
    all_compacted = copy.deepcopy(messages)
    all_compacted[2]["content"] = placeholder(old_payload)
    all_compacted[len(oldest) + 2]["content"] = placeholder(retained_payload)
    # Everything fits before mandatory index overhead; the bare pair misses by
    # exactly 75 tokens even after every eligible result has been compacted.
    budget = ctx.history_tokens(all_compacted) + ctx.history_tokens(bare_index()) - 75
    assert ctx.history_tokens(all_compacted) <= budget
    assert ctx.history_tokens(all_compacted) + ctx.history_tokens(bare_index()) == budget + 75
    before = encoded(messages)
    fitted, stats = ctx.fit_messages(messages, budget)
    expected = copy.deepcopy(previous + active)
    expected[2]["content"] = placeholder(retained_payload)
    assert fitted[2:] == expected
    assert_stats(fitted, stats, budget, 3, 1)
    assert stats["compacted_messages"] == stats["truncated_messages"] == 1
    assert stats["artifacts"] == [artifact(retained_payload)]
    assert stats["elided_tokens"] == ctx.estimate_tokens(retained_payload)
    assert encoded(messages) == before

    memory = SQLiteMemory(tmp_path / "history.db", session_id="anchor-rescue", max_messages=1000)
    memory.replace(messages)
    original_rows = audit_rows(memory)
    agent = agent_for(memory, budget)
    assert agent._fit(memory.get(), []) == fitted
    plan = artifact(retained_payload)
    assert stored_artifacts(memory) == [tuple(plan[key] for key in (
        "artifact_id", "tool_name", "byte_len", "content"))]
    assert memory.get_session_metadata("SESSION_RECORD") == stats["session_record"]
    assert audit_rows(memory) == original_rows


def test_overcap_old_completed_result_evicts_without_artifact_plan_or_dangling_id(tmp_path):
    payload = "x" * (ctx.ARTIFACT_SESSION_MAX_BYTES + 1)
    oldest = read_turn("overcap-oldest", payload)
    previous = short_turn(1)
    active = [{"role": "user", "content": "Exact active request café\n  "}]
    messages = oldest + previous + active
    before = encoded(messages)
    fitted, stats = ctx.fit_messages(messages, BUDGET)
    assert fitted[2:] == previous + active
    assert_stats(fitted, stats, BUDGET, 3, 1)
    assert stats["artifacts"] == []
    assert stats["compacted_messages"] == stats["truncated_messages"] == stats["elided_tokens"] == 0
    assert assert_pairing(fitted) == ["r01"]
    assert "overcap-oldest" not in encoded(fitted).decode("utf-8")
    assert encoded(messages) == before

    memory = SQLiteMemory(tmp_path / "history.db", session_id="overcap-old", max_messages=1000)
    memory.replace(messages)
    original_rows = audit_rows(memory)
    agent = agent_for(memory)
    assert agent._fit(memory.get(), []) == fitted
    assert stored_artifacts(memory) == []
    assert memory.get_session_metadata("SESSION_RECORD") == stats["session_record"]
    assert audit_rows(memory) == original_rows


def test_overcap_n_minus_one_result_cannot_be_evicted_or_claimed_as_artifact():
    previous = read_turn("overcap-previous", "x" * (ctx.ARTIFACT_SESSION_MAX_BYTES + 1))
    messages = short_turn(0) + previous + [{"role": "user", "content": "Active"}]
    before = encoded(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, BUDGET)
    assert encoded(messages) == before


@pytest.mark.parametrize("operation", ["eviction-only", "compaction-only"])
def test_actual_agent_fit_emits_exact_cli_warning_protocol(operation, tmp_path, monkeypatch):
    memory = SQLiteMemory(tmp_path / "history.db", session_id="warning", max_messages=1000)
    if operation == "eviction-only":
        messages = flatten(fifty_turns())
    else:
        messages = read_turn("seed", "old result " * 3000) + [
            {"role": "user", "content": "Exact active request"}]
    memory.replace(messages)
    agent = agent_for(memory)
    warnings = []
    monkeypatch.setattr("nexus_agent_ai.agent.core.display.print_warn", warnings.append)
    fitted = agent._fit(memory.get(), [])
    stats = agent.last_trim
    assert stats["tokens"] == ctx.history_tokens(fitted) <= BUDGET
    if operation == "eviction-only":
        assert stats["evicted_turns"] > 0 and stats["compacted_messages"] == 0
        optional_clause = f"evicted {stats['evicted_turns']} older turn(s), "
    else:
        assert stats["evicted_turns"] == 0 and stats["compacted_messages"] == 1
        optional_clause = ""
    assert warnings == [
        f"Context is full: kept {stats['turns_after']} of {stats['turns_before']} turn(s) "
        f"(compacted {stats['compacted_messages']} old content block(s), "
        f"{optional_clause}~{stats['elided_tokens']} payload tokens elided, "
        f"~{stats['tokens']} tokens preserved) "
        "for this request. Canonical history is unchanged."
    ]
    assert memory.get() == messages
