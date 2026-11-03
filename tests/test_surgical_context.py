import copy
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import context as ctx
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.messages import openai_messages, anthropic_messages, gemini_messages
from nexus_agent_ai.providers.local_provider import LocalProvider


def call(name="read_file", path="src/a.py", identity="read", metadata=None):
    message = {"role": "assistant", "content": "", "tool_calls": [{
        "id": identity, "type": "function",
        "function": {"name": name, "arguments": json.dumps({"path": path})},
    }]}
    if metadata:
        message["metadata"] = metadata
    return message


def recent():
    return [{"role": "user", "content": "Prior request"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "  Current café\n\t "}]


def history(content="é" * 12000):
    return [{"role": "user", "content": "Read src/a.py"}, call(),
            {"role": "tool", "tool_call_id": "read", "content": content},
            {"role": "assistant", "content": "Observed contents"}, *recent()]


def plan(content, name="read_file"):
    return {"artifact_id": hashlib.sha256((name + "\0" + content).encode("utf-8")).hexdigest(),
            "tool_name": name, "byte_len": len(content.encode("utf-8")), "content": content}


def test_exact_placeholder_utf8_bytes_token_accounting_and_pure_deterministic_plan():
    messages = history()
    original = copy.deepcopy(messages)
    fitted, stats = ctx.fit_messages(messages, 700)
    artifact = plan(messages[2]["content"])
    expected = (f"[COMPACTED: read_file result, {ctx.estimate_tokens(messages[2]['content'])} "
                f"tokens elided. Artifact ID: {artifact['artifact_id']}]")
    assert fitted[4]["content"] == expected
    assert stats["artifacts"] == [artifact]
    assert stats["artifacts"][0]["byte_len"] == 24000
    assert stats["elided_tokens"] == ctx.estimate_tokens(messages[2]["content"]) >= 0
    assert stats["message_bytes_before"] == [len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in messages]
    assert stats["message_bytes_after"] == [len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in fitted[2:]]
    assert stats["tokens"] == ctx.history_tokens(fitted) <= 700
    assert fitted[-3:] == messages[-3:]
    assert messages == original
    assert ctx.fit_messages(messages, 700) == (fitted, stats)


@pytest.mark.parametrize("projector", [lambda x: x, openai_messages, anthropic_messages, gemini_messages])
@pytest.mark.parametrize("tool_turn", ["older", "previous"])
def test_each_message_shape_keeps_structure_ids_arguments_and_protected_text(projector, tool_turn):
    source = history("payload " * 3000)
    if tool_turn == "previous":
        source = source[:4] + source[-1:]
    messages = projector(source)
    before = copy.deepcopy(messages)
    fitted, stats = ctx.fit_messages(messages, 900)
    assert stats["compacted_messages"] == 1
    assert stats["turns_before"] == stats["turns_after"] == (3 if tool_turn == "older" else 2)
    assert stats["evicted_turns"] == stats["dropped_messages"] == stats["dropped_turns"] == 0
    assert len(fitted[2:]) == len(messages)
    # Restoring just the eligible string yields the original native containers.
    restored = copy.deepcopy(fitted[2:])
    calls = {"read": ("read_file", ["src/a.py"])}
    slots = [(owner, key) for message in restored for owner, key, _ in ctx._payloads(message, calls)
             if owner[key].startswith("[COMPACTED:")]
    assert len(slots) == 1
    slots[0][0][slots[0][1]] = stats["artifacts"][0]["content"]
    assert restored == messages == before


def test_anthropic_nested_result_keeps_nontext_blocks_and_result_attributes():
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read", "is_error": False,
        "content": [{"type": "text", "text": "z" * 18000}, {"type": "image", "source": {"type": "base64", "data": "keep"}}]}]}
    messages = [{"role": "user", "content": "Read"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "read", "name": "read_file", "input": {"path": "a.py"}}]},
                result, *recent()]
    fitted, stats = ctx.fit_messages(messages, 800)
    nested = fitted[4]["content"][0]
    assert nested["content"][0]["text"].startswith("[COMPACTED: read_file result,")
    assert nested["content"][1] == result["content"][0]["content"][1]
    assert nested["tool_use_id"] == "read" and nested["is_error"] is False
    assert stats["artifacts"][0]["content"] == "z" * 18000


def test_local_consumer_receives_original_pairing_with_compacted_result_only():
    fitted, stats = ctx.fit_messages(history(), 800)
    captured = {}
    def complete(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    provider = LocalProvider.__new__(LocalProvider)
    provider.max_output_tokens = 64
    provider._llama = SimpleNamespace(create_chat_completion=complete)
    provider._run_llama_cpp(fitted, [], "rules")
    sent = captured["messages"]
    assert sent[4]["tool_calls"][0]["id"] == sent[5]["tool_call_id"] == "read"
    assert sent[4]["tool_calls"][0]["function"]["arguments"] == history()[1]["tool_calls"][0]["function"]["arguments"]
    assert stats["artifacts"][0]["artifact_id"] in sent[5]["content"]


def test_non_tool_oversized_text_uses_exact_non_tool_placeholder():
    content = "old prose " * 2000
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": "Done"}, *recent()]
    fitted, stats = ctx.fit_messages(messages, 600)
    artifact = plan(content, "")
    assert stats["artifacts"] == [artifact]
    assert fitted[2]["content"] == f"[COMPACTED: {ctx.estimate_tokens(content)} tokens elided. Artifact ID: {artifact['artifact_id']}]"


@pytest.mark.parametrize("index", [4, 5, 6, 7])
def test_last_two_genuine_turns_keep_user_and_assistant_text_full_even_when_heavy(index):
    messages = history("old " * 3000) + [{"role": "assistant", "content": "Current answer"}]
    messages[index]["content"] = "protected " * 3000
    original = copy.deepcopy(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 800)
    assert messages == original


def test_refused_same_segment_read_is_pinned_but_other_segment_read_is_eligible():
    messages = history()
    identity = {"turn": "turn", "segment": "segment"}
    messages[1]["metadata"] = identity.copy()
    messages[2]["metadata"] = identity.copy()
    refusal = {"event": "event", "turn": "turn", "segment": "segment", "path": "src/a.py",
               "action": "patch_file", "status": "unchanged", "success": False, "failed": True}
    messages.insert(3, {"role": "tool", "tool_call_id": "edit", "content": "ERROR: refused",
                        "metadata": {**identity, "mutation_outcomes": [refusal]}})
    before = copy.deepcopy(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 900)
    assert messages == before
    messages[1]["metadata"]["segment"] = "earlier-segment"
    messages[2]["metadata"]["segment"] = "earlier-segment"
    fitted, stats = ctx.fit_messages(messages, 900)
    assert stats["compacted_messages"] == 1
    assert fitted[4]["content"].startswith("[COMPACTED:")


def test_old_outcome_correction_is_never_replaced():
    messages = history()
    correction = {"role": "assistant", "content": "correction " * 3000,
                  "metadata": {"internal": True, "outcome_correction": True}}
    messages.insert(4, correction)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 900)


@pytest.mark.parametrize("metadata", [
    {"recovery_pinned": True},
    {"edit_recovery": {"reason": "literal mismatch", "active": True}},
])
def test_mid_chain_recovery_evidence_is_not_compacted_even_in_older_turn(metadata):
    messages = history()
    messages[2]["metadata"] = metadata
    before = copy.deepcopy(messages)
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 900)
    assert messages == before


def test_oversized_artifact_cannot_produce_a_dangling_claim():
    messages = history("x" * (ctx.ARTIFACT_SESSION_MAX_BYTES + 1))
    messages = messages[:4] + messages[-1:]
    with pytest.raises(ctx.ContextBudgetError):
        ctx.fit_messages(messages, 800)


def test_artifacts_cap_is_exact_deterministic_session_scoped_and_atomic(tmp_path):
    path = tmp_path / "history.db"
    memory = SQLiteMemory(path, session_id="one")
    other = SQLiteMemory(path, session_id="two")
    first, second, third = (plan(char * (128 * 1024)) for char in "abc")
    memory.store_artifacts([first, second])
    other.store_artifacts([first])
    memory.store_artifacts([third])
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT artifact_id, byte_len FROM context_artifacts WHERE session_id='one' ORDER BY rowid").fetchall()
        assert rows == [(second["artifact_id"], second["byte_len"]), (third["artifact_id"], third["byte_len"])]
        assert sum(row[1] for row in rows) == 256 * 1024
    with pytest.raises(ValueError):
        memory.store_artifacts([plan("d" * (256 * 1024 + 1))])
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_artifacts WHERE session_id='one'").fetchone()[0] == 2
    memory.clear()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT session_id FROM context_artifacts").fetchall() == [("two",)]
    SQLiteMemory.delete_session("two", db_path=path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_artifacts").fetchone()[0] == 0


def test_history_literal_keywords_recency_exact_excerpt_and_select_only(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    memory = SQLiteMemory(path, session_id="one")
    messages = [{"role": "system", "content": "needle secret"},
                {"role": "user", "content": "older needle exact"},
                call(), {"role": "tool", "tool_call_id": "read", "content": "prefix " + "x" * 2500 + " needle exact suffix"},
                {"role": "assistant", "content": "needle elsewhere exact"},
                {"role": "user", "content": "newer needle exact"},
                {"role": "user", "content": "needle exact internal", "metadata": {"internal": True}}]
    memory.replace(messages)
    other = SQLiteMemory(path, session_id="two")
    other.add("user", "needle exact other session")
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT * FROM conversation_history").fetchall()
    statements = []
    real_connect = sqlite3.connect
    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(sqlite3, "connect", connect)
    entries = memory.history_entries("needle exact", 10)
    assert entries[0]["content"] == "newer needle exact"
    assert entries[-1]["content"] == "needle elsewhere exact"
    tool = next(entry for entry in entries if entry["role"] == "tool")
    assert tool["content"] in messages[3]["content"] and len(tool["content"]) <= 2000
    assert "needle exact" in tool["content"]
    assert tool["paths"] == ["src/a.py"] and tool["provenance_known"] is True
    assert all(entry["role"] != "system" and "internal" not in entry["content"] for entry in entries)
    assert all(entry["entry_id"] and entry["timestamp"] for entry in entries)
    assert memory.history_entries("", 1)[0]["content"] == "newer needle exact"
    assert statements and all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
    with real_connect(path) as conn:
        assert conn.execute("SELECT * FROM conversation_history").fetchall() == before
