import json
import pytest

from nexus_agent_ai.agent import context as ctx
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse


def turn(i, size=40):
    return [{"role": "user", "content": f"question {i} " + "x" * size},
            {"role": "assistant", "content": f"answer {i} " + "y" * size}]


def history(n, size=40):
    out = []
    for i in range(n):
        out += turn(i, size)
    return out


def test_estimate_is_conservative():
    assert ctx.estimate_tokens("a" * 350) >= 100


def test_fit_noop_when_under_budget():
    msgs = history(3)
    fitted, stats = ctx.fit_messages(msgs, 10_000)
    assert fitted == msgs and stats["dropped_messages"] == 0


def test_fit_drops_oldest_whole_turns_and_keeps_latest():
    msgs = history(10, size=400)
    budget = ctx.history_tokens(msgs) // 3
    fitted, stats = ctx.fit_messages(msgs, budget)
    assert ctx.history_tokens(fitted) <= budget
    assert fitted[-1] == msgs[-1]
    assert fitted[0]["role"] == "user"  # never starts mid-turn
    assert stats["dropped_messages"] == len(msgs) - len(fitted) > 0


def test_fit_keeps_tool_pairs_together():
    msgs = [
        {"role": "user", "content": "old " + "x" * 2000},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "new question"},
    ]
    fitted, _ = ctx.fit_messages(msgs, 60)
    assert fitted[0]["content"] == "new question"


def test_fit_truncates_giant_single_message():
    big = "HEAD" + ("z" * 40_000) + "TAIL"
    msgs = [{"role": "user", "content": big}]
    fitted, stats = ctx.fit_messages(msgs, 1000)
    text = fitted[0]["content"]
    assert ctx.history_tokens(fitted) <= 1000
    assert text.startswith("HEAD") and text.endswith("TAIL") and "truncated" in text
    assert stats["truncated_messages"] >= 1
    assert msgs[0]["content"] == big  # stored history not mutated


def test_fit_truncates_big_tool_result_in_latest_turn():
    msgs = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "A" * 30_000},
    ]
    fitted, _ = ctx.fit_messages(msgs, 1500)
    assert ctx.history_tokens(fitted) <= 1500
    assert "truncated" in fitted[2]["content"]


def test_compact_summarizes_old_turns_keeps_recent():
    msgs = history(6)
    new, info = ctx.compact_messages(msgs, keep_turns=2)
    assert info["turns_summarized"] == 4
    assert new[0]["role"] == "user" and "compacted" in new[0]["content"]
    assert "question 0" in new[0]["content"]
    assert new[1]["role"] == "assistant"
    assert new[2:] == msgs[-4:]
    assert ctx.history_tokens(new) < ctx.history_tokens(msgs)


def test_compact_noop_on_short_history():
    msgs = history(2)
    new, info = ctx.compact_messages(msgs, keep_turns=2)
    assert new == msgs and info["removed_messages"] == 0


def test_compact_twice_keeps_earlier_requests():
    msgs = history(6)
    once, _ = ctx.compact_messages(msgs, keep_turns=2)
    once += turn(7) + turn(8)
    twice, _ = ctx.compact_messages(once, keep_turns=2)
    assert "question 0" in twice[0]["content"]


def test_meter_text():
    assert "50%" in ctx.meter(2048, 4096)
    assert "/compact" in ctx.meter(3900, 4096)
    assert "tokens" in ctx.meter(500, None)


def test_overflow_detection():
    err = RuntimeError('Local model request failed (HTTP 400): {"type":"exceed_context_size_error"}')
    assert ctx.is_context_overflow(err)
    assert not ctx.is_context_overflow(RuntimeError("connection refused"))


def test_memory_replace_in_memory_and_sqlite(tmp_path):
    m = ConversationMemory()
    m.replace(history(2))
    assert len(m.get()) == 4
    db = SQLiteMemory(db_path=tmp_path / "h.db", session_id="s")
    db.replace(history(3))
    assert len(db.get()) == 6
    db.replace([{"role": "user", "content": "only"}])
    assert db.get() == [{"role": "user", "content": "only"}]


class RecordingProvider(BaseProvider):
    context_size = 1024
    model = "stub"

    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def complete(self, messages, tools, system):
        self.calls.append(list(messages))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("Local model request failed (HTTP 400): exceed_context_size_error")
        return ProviderResponse(text="ok", input_tokens=1, output_tokens=1)

    def stream(self, messages, tools, system):
        yield self.complete(messages, tools, system)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def test_agent_trims_request_but_keeps_stored_history():
    prov = RecordingProvider()
    mem = ConversationMemory(max_messages=500)
    mem.replace(history(40, size=300))
    agent = Agent(provider=prov, memory=mem, verbose=False, tools=[])
    agent.run("final question")
    sent = prov.calls[0]
    assert ctx.history_tokens(sent) < 1024
    assert sent[-1]["content"] == "final question"
    assert len(mem.get()) == 40 * 2 + 2  # nothing was deleted from storage


def test_agent_retries_once_after_overflow():
    prov = RecordingProvider(fail_first=True)
    mem = ConversationMemory(max_messages=500)
    mem.replace(history(4, size=100))
    agent = Agent(provider=prov, memory=mem, verbose=False, tools=[])
    assert agent.run("again") == "ok"
    assert len(prov.calls) == 2
    assert ctx.history_tokens(prov.calls[1]) <= ctx.history_tokens(prov.calls[0])


def test_agent_does_not_retry_unrelated_errors():
    class Broken(RecordingProvider):
        def complete(self, messages, tools, system):
            self.calls.append(1)
            raise RuntimeError("connection refused")
    prov = Broken()
    agent = Agent(provider=prov, memory=ConversationMemory(), verbose=False, tools=[])
    with pytest.raises(RuntimeError):
        agent.run("hi")
    assert len(prov.calls) == 1


def test_agent_compact_and_usage():
    prov = RecordingProvider()
    mem = ConversationMemory(max_messages=500)
    mem.replace(history(8, size=200))
    agent = Agent(provider=prov, memory=mem, verbose=False, tools=[])
    used = agent.context_used()
    before, after, info = agent.compact()
    assert info["removed_messages"] > 0 and after < before
    assert agent.context_used() < used
