"""Provider-overflow retry: real handler integration through Agent.run.

The retry after an endpoint context rejection must send a strictly smaller,
still-feasible history budget, happen at most once, and fail accurately when
the fixed overhead cannot fit at all. It must never retry at a window scale
that is statically below the fixed overhead and minimum output reserve.
"""
import copy
import json

import pytest

from nexus_agent_ai.agent import context, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, Tool, ToolCall


class Scripted(BaseProvider):
    model = "offline-overflow-retry"
    context_size = 32768
    max_output_tokens = 512
    is_local = True
    _supports_output_budget = True

    def __init__(self, steps, *, context_size=None, max_output_tokens=None):
        self.steps = list(steps)
        self.seen = []
        if context_size is not None:
            self.context_size = context_size
        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens

    def complete(self, messages, tools, system):
        if getattr(self, "before_request", None):
            messages, system = self.before_request(self, messages, tools, system)
        self.seen.append(copy.deepcopy(messages))
        assert self.steps, "overflow handling made an unexpected provider request"
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step() if callable(step) else step

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": str(result)}


def overflow():
    return RuntimeError("exceed_context_size: prompt is too long")


def text(value):
    return ProviderResponse(text=value, tool_calls=[])


def tool(description="Observe a deterministic fixture."):
    return Tool("probe", description, {
        "type": "object", "properties": {"path": {"type": "string"}},
    }, lambda **kwargs: "unused registry handler")


def make_agent(provider, offered, tmp_path, *, system="Use the supplied observations; do not invent execution."):
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="overflow-retry")
    agent = Agent(provider, tools=offered, memory=memory, verbose=False)
    agent.system = system
    return agent


def fill_history(agent, turns=6, size=900):
    for index in range(turns):
        agent.memory.add("user", f"earlier question {index} " + "context " * (size // 8))
        agent.memory.add("assistant", f"earlier answer {index} " + "detail " * (size // 8))


def sent_tokens(provider, index):
    return context.history_tokens(provider.seen[index])


def record_dispatch_caps(provider):
    """Capture the effective output reserve applied to each dispatched request."""
    caps = []
    original_complete = provider.complete

    def recording_complete(messages, tools=None, system=None):
        caps.append(getattr(provider, "_request_output_cap", "unset"))
        return original_complete(messages, tools, system)

    provider.complete = recording_complete
    return caps


def executor_closures(agent):
    return [m for m in agent.memory.get()
            if m["role"] == "assistant" and m.get("metadata", {}).get("executor_closed")]


def test_overflow_retry_sends_strictly_smaller_feasible_history_and_completes(tmp_path):
    provider = Scripted([overflow(), text("final answer")], context_size=4096)
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    result = agent.run("current question")
    assert result == "final answer"
    assert len(provider.seen) == 2, "the overflow retry must issue exactly one second request"
    assert sent_tokens(provider, 1) < sent_tokens(provider, 0), "retry must send a strictly smaller payload"
    assert executor_closures(agent) == []
    final = agent.memory.get()[-1]
    assert final["role"] == "assistant" and final["content"] == "final answer"


def test_overflow_retry_when_window_cannot_fit_fixed_overhead_fails_accurately_without_request(tmp_path):
    # Fixed overhead + minimum reserve already exceed the window: no request is
    # sent, no misleading retry warning, and the accurate budget failure stands.
    big_system = "instructions " * 400  # ~1200 tokens of system alone
    provider = Scripted([], context_size=600)
    agent = make_agent(provider, [tool()], tmp_path, system=big_system)
    with pytest.raises(context.ContextBudgetError):
        agent.run("current question")
    assert provider.seen == [], "no over-budget request may be sent"
    assert executor_closures(agent), "the accurate capacity failure is recorded honestly"


def test_overflow_retry_happens_at_most_once_then_original_failure_stands(tmp_path):
    provider = Scripted([overflow(), overflow()], context_size=4096)
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    with pytest.raises(RuntimeError, match="exceed_context_size"):
        agent.run("current question")
    assert len(provider.seen) == 2, "the retry is bounded to a single attempt"
    assert executor_closures(agent), "the unresolved overflow is closed honestly"


def test_overflow_retry_at_minimum_reserve_still_shrinks_and_completes(tmp_path):
    provider = Scripted([overflow(), text("reserved answer")], context_size=4096, max_output_tokens=256)
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    assert agent.run("current question") == "reserved answer"
    assert len(provider.seen) == 2
    assert sent_tokens(provider, 1) < sent_tokens(provider, 0)


def test_no_overflow_single_request_unchanged(tmp_path):
    provider = Scripted([text("plain answer")])
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent)
    assert agent.run("current question") == "plain answer"
    assert len(provider.seen) == 1, "no overflow means no retry and no changed behaviour"


def test_overflow_retry_never_scales_below_fixed_overhead(tmp_path):
    # Regression for the r7 probe failure shape: 4096 window with ~2.7k tokens of
    # fixed overhead, where the old 0.6 window scale (2457) was statically below
    # overhead + minimum reserve (3012) and the retry could never succeed.
    provider = Scripted([overflow(), text("recovered")], context_size=4096)
    big_system = "system instructions " * 400  # ~2300 tokens of fixed overhead
    agent = make_agent(provider, [tool()], tmp_path, system=big_system)
    fill_history(agent)
    assert agent.run("current question") == "recovered"
    assert len(provider.seen) == 2, "a feasible retry must actually reach the provider"
    assert sent_tokens(provider, 1) < sent_tokens(provider, 0)


def test_overflow_retry_without_possible_reduction_sends_nothing_and_fails_accurately(tmp_path):
    # Finding 1: protected content cannot shrink (no evictable history), so the
    # retry would be byte-identical. No second request may be sent.
    provider = Scripted([overflow(), text("never sent")], context_size=4096)
    agent = make_agent(provider, [tool()], tmp_path)
    with pytest.raises(context.ContextBudgetError, match="cannot be reduced"):
        agent.run("current question")
    assert len(provider.seen) == 1, "an unchanged payload must not be resent"
    assert executor_closures(agent), "the accurate capacity failure is recorded honestly"


def test_overflow_retry_with_adaptive_reserve_reduction_still_shrinks(tmp_path):
    # Finding 1 adaptive bypass: window 4096, ~3.6k fixed overhead, so the first
    # fit only gets a positive budget after the adaptive reserve drops 512 -> 256.
    # The old retry derived its cap from the full window (history_cap None), so it
    # could resend the identical rejected request. The corrected policy derives the
    # retry cap from the REJECTED fit's actual budget/reserve; when even the minimal
    # conversation cannot fit below the rejected payload, the accurate capacity
    # failure stands and NO unchanged request is resent.
    probe = tool()
    overhead_target = 3100
    big_system = "s" * int(context.CHARS_PER_TOKEN * (overhead_target - context.tools_tokens([probe]) - 1))
    provider = Scripted([overflow(), text("recovered")], context_size=4096)
    agent = make_agent(provider, [probe], tmp_path, system=big_system)
    fill_history(agent, turns=2)
    caps = record_dispatch_caps(provider)
    with pytest.raises(context.ContextBudgetError):
        agent.run("current question")
    assert len(provider.seen) == 1, "the rejected request must never be resent unchanged"
    assert caps == [256], (
        "the rejected request was dispatched at the adaptively reduced 256 reserve, "
        "so the retry cap is derived from the actual 256-reserve fit, not the nominal 512")


def test_retry_then_tool_result_uses_normal_policy_and_preserves_receipt(tmp_path):
    # Finding 2: the reduced policy is bounded to the single resent request;
    # result preflight/admission and the following request use the normal policy.
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("receipt-value\n", encoding="utf-8")

    call = ToolCall("call-1", "count_file_text", {"path": "fixture.txt", "literal": "receipt"})
    provider = Scripted([overflow(),
                         ProviderResponse(text="", tool_calls=[call]),
                         text("done")], context_size=4096)
    agent = make_agent(provider, [tools.COUNT_FILE_TEXT_TOOL], tmp_path)
    fill_history(agent, turns=20)
    caps = record_dispatch_caps(provider)
    rejected_fits = []
    actual_fit = agent._fit

    def capture_rejected_fit(messages, tools, scale=1.0, **kwargs):
        fitted = actual_fit(messages, tools, scale, **kwargs)
        if not kwargs.get("overflow_retry", False) and agent._last_outbound_fit is not None:
            rejected_fits.append(dict(agent._last_outbound_fit))
        return fitted

    agent._fit = capture_rejected_fit
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert agent.run("Count how many times 'receipt' appears in fixture.txt") == "done"
    finally:
        os.chdir(cwd)
    assert len(provider.seen) == 3
    # The rejected fit and the retry both dispatched at the same effective reserve
    # (512; no adaptive reduction was needed).
    assert caps == [512, 512, 512]
    # Derive the expected retry cap from the REJECTED fit's actual budget: the
    # post-tool request must be fitted under the normal policy, i.e. its fitted
    # history strictly exceeds the retry cap, while the retry itself respected it.
    retry_cap = max(1, int(rejected_fits[0]["budget"] * 0.6))
    assert sent_tokens(provider, 1) <= retry_cap
    assert sent_tokens(provider, 2) > retry_cap, (
        "the post-tool request must fit the normal budget, not the retry cap")
    # The provider-facing next request carries the exact assistant call binding
    # and the receipt facts equal to the canonical record.
    canonical = [m for m in agent.memory.get() if m["role"] == "tool"]
    assert len(canonical) == 1
    assert canonical[0]["tool_call_id"] == "call-1"
    assert canonical[0]["metadata"]["executed"] is True
    outbound_calls = [tc for m in provider.seen[2] if m["role"] == "assistant"
                      for tc in m.get("tool_calls", [])]
    assert len(outbound_calls) == 1
    assert outbound_calls[0]["id"] == "call-1"
    assert outbound_calls[0]["function"]["name"] == "count_file_text"
    assert json.loads(outbound_calls[0]["function"]["arguments"]) == {"path": "fixture.txt", "literal": "receipt"}
    outbound_tool = [m for m in provider.seen[2] if m["role"] == "tool"]
    assert len(outbound_tool) == 1
    assert outbound_tool[0]["tool_call_id"] == outbound_calls[0]["id"]
    assert outbound_tool[0]["content"] == canonical[0]["content"]
    # The receipt carries the fixture's substantive count, not just a status word.
    parsed = json.loads(canonical[0]["content"])
    assert parsed["status"] == "complete"
    assert parsed["count"] == 1, "one whole-word 'receipt' in 'receipt-value'"


def test_retry_pending_multi_call_closure_preserves_all_bindings(tmp_path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("receipt-value\n", encoding="utf-8")

    calls = [ToolCall("call-a", "count_file_text", {"path": "fixture.txt", "literal": "receipt"}),
             ToolCall("call-b", "count_file_text", {"path": "fixture.txt", "literal": "value"})]
    provider = Scripted([overflow(),
                         ProviderResponse(text="", tool_calls=calls),
                         text("done")], context_size=4096)
    agent = make_agent(provider, [tools.COUNT_FILE_TEXT_TOOL], tmp_path)
    fill_history(agent, turns=20)
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert agent.run("Count how many times 'receipt' appears in fixture.txt") == "done"
    finally:
        os.chdir(cwd)
    receipts = {m["tool_call_id"]: m for m in agent.memory.get() if m["role"] == "tool"}
    assert set(receipts) == {"call-a", "call-b"}
    assert all(m["metadata"]["executed"] is True for m in receipts.values())


def test_retry_multi_call_abort_closes_pending_call_honestly(tmp_path, monkeypatch):
    # The turn aborts while the second call of a pending multi-call batch has
    # started but not completed: the closure must record the first call's exact
    # receipt and an honest not-completed record for the interrupted one.
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("receipt-value\n", encoding="utf-8")

    from nexus_agent_ai.agent import core as core_mod
    original_execute = core_mod.execute_tool

    def aborting_execute(name, args, **kwargs):
        if args.get("literal") == "value":
            raise RuntimeError("executor died")
        return original_execute(name, args, **kwargs)

    monkeypatch.setattr(core_mod, "execute_tool", aborting_execute)
    calls = [ToolCall("call-a", "count_file_text", {"path": "fixture.txt", "literal": "receipt"}),
             ToolCall("call-b", "count_file_text", {"path": "fixture.txt", "literal": "value"})]
    provider = Scripted([overflow(),
                         ProviderResponse(text="", tool_calls=calls)], context_size=4096)
    agent = make_agent(provider, [tools.COUNT_FILE_TEXT_TOOL], tmp_path)
    fill_history(agent, turns=20)
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="executor died"):
            agent.run("Count how many times 'receipt' appears in fixture.txt")
    finally:
        os.chdir(cwd)
    tool_messages = [m for m in agent.memory.get() if m["role"] == "tool"]
    call_a = [m for m in tool_messages if m["tool_call_id"] == "call-a"]
    call_b = [m for m in tool_messages if m["tool_call_id"] == "call-b"]
    assert len(call_a) == 1 and len(call_b) == 1, "each call is recorded exactly once"
    assert call_a[0]["metadata"]["executed"] is True
    assert '"status":"complete"' in call_a[0]["content"]
    # The interrupted call is closed with executor provenance and UNKNOWN
    # completion/effects - not a definite executed True/False assertion.
    assert call_b[0]["metadata"]["executor"] is True
    assert call_b[0]["metadata"]["executed"] is None
    assert "Execution began" in call_b[0]["content"]
    assert "Completion and effects are unknown" in call_b[0]["content"]
    # The turn closure identifies the actual abort reason.
    closures = executor_closures(agent)
    assert len(closures) == 1
    assert "executor aborted (RuntimeError)" in closures[0]["content"]


def test_managed_before_request_path_retries_reduced_and_restores_callback(tmp_path):
    provider = Scripted([overflow(), text("managed answer")], context_size=4096)

    def original_callback(active, full_messages, tools, system):
        return full_messages, system

    provider.before_request = original_callback  # marks the provider as managed
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    assert agent.run("current question") == "managed answer"
    assert len(provider.seen) == 2
    assert sent_tokens(provider, 1) < sent_tokens(provider, 0)
    assert provider.before_request is original_callback, "the original callback slot is restored"


def test_managed_callback_restored_when_retry_raises(tmp_path):
    provider = Scripted([overflow(), overflow()], context_size=4096)

    def original_callback(active, full_messages, tools, system):
        return full_messages, system

    provider.before_request = original_callback
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    with pytest.raises(RuntimeError, match="exceed_context_size"):
        agent.run("current question")
    assert len(provider.seen) == 2, "the bounded retry was attempted exactly once"
    assert provider.before_request is original_callback, (
        "the original callback slot is restored even when the retry raises")


def test_stream_path_overflow_retry_reduces_and_completes(tmp_path):
    provider = Scripted([overflow(), text("streamed answer")], context_size=4096)
    agent = make_agent(provider, [tool()], tmp_path)
    fill_history(agent, turns=20)
    events = []
    agent.event_callback = events.append
    assert agent.run("current question", stream=True) == "streamed answer"
    assert len(provider.seen) == 2
    assert sent_tokens(provider, 1) < sent_tokens(provider, 0)
    chunks = [e for e in events if e["type"] == "stream_chunk"]
    assert chunks == [{"type": "stream_chunk", "content": "streamed answer"}], (
        "exactly one stream chunk from the successful retry; nothing emitted for the rejected request")
    responses = [e for e in events if e["type"] == "response"]
    assert len(responses) == 1 and responses[0]["content"] == "streamed answer"
