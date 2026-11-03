"""R9 uses actual serialized omission costs, without changing window policy."""
import copy
import math
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import context as ctx, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import execution_result
from nexus_agent_ai.agent.result_budget import guard_tool_result
from nexus_agent_ai.providers.base import BaseProvider, ToolCall
from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.providers.messages import (
    anthropic_messages, gemini_messages, normalize_messages, openai_messages,
)
from nexus_agent_ai.providers.messages import assistant_message


BUDGET = 1573
RECORD = {"version": 1, "files": [], "topics": [], "tools": []}


class Offline(BaseProvider):
    model = "offline-r9-budget-fixture"
    context_size = 4096
    max_output_tokens = 512
    is_local = False

    def complete(self, **kwargs):
        raise AssertionError("Budget planning must not call a model")

    def stream(self, **kwargs):
        raise AssertionError("Budget planning must not call a model")

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def fixture_agent(tmp_path, *, record=False, closed=False, request="  Read sample.py. café\n\t"):
    memory = SQLiteMemory(tmp_path / "state" / "history.db", session_id="precedence")
    old_final = {"role": "assistant", "content": "Recorded."}
    if closed:
        old_final["metadata"] = {
            "executor_closed": True, "executor": True, "reason": "ContextBudgetError",
        }
    current = ToolCall("current", "read_file", {"path": "sample.py"})
    for message in [
        {"role": "user", "content": "Old topic."}, old_final,
        {"role": "user", "content": "Previous topic."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "user", "content": request},
        assistant_message("", [current]),
    ]:
        memory.add_raw(message)
    if record:
        memory.set_session_metadata("SESSION_RECORD", RECORD)
    agent = Agent(Offline(), memory=memory, tools=[tools.READ_FILE_TOOL], verbose=False)
    wanted = 4096 - 512 - BUDGET - ctx.tools_tokens(agent.tools)
    agent.system = "s" * math.ceil((wanted - 1) * ctx.CHARS_PER_TOKEN)
    assert 4096 - 512 - ctx.estimate_tokens(agent.system) - ctx.tools_tokens(agent.tools) == BUDGET
    pending = {"call": current, "state": "pending", "result": None}
    agent._pending_calls = [pending]
    return agent, pending


def maximal_payload(agent, pending, budget):
    low, high = 0, budget * 4
    while low < high:
        middle = (low + high + 1) // 2
        view = agent._prospective_result_messages(pending, "x" * middle)
        if ctx.history_tokens(view) <= budget:
            low = middle
        else:
            high = middle - 1
    assert ctx.history_tokens(agent._prospective_result_messages(pending, "x" * low)) == budget
    return "x" * low


@pytest.mark.parametrize("closed", [False, True])
def test_first_eviction_floor_is_reserved_before_result_admission(tmp_path, closed):
    agent, pending = fixture_agent(tmp_path, closed=closed)
    raw = maximal_payload(agent, pending, BUDGET)
    before = copy.deepcopy(agent.memory.get())
    # It fits only before the first omission transition; the evictable old turn
    # cannot pay for the actual count-line + anchor + acknowledgement pair.
    assert not agent._tool_result_fits(pending, raw, agent.tools)
    result = guard_tool_result(
        "read_file", execution_result(raw, executed=True, outcome="read_file completed."),
        memory=agent.memory,
        fits=lambda text: agent._tool_result_fits(pending, text, agent.tools),
    )
    assert "Outcome: read_file completed." in result
    assert "retained as artifact" in result
    assert agent.memory.get() == before
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None
    fitted, stats = agent._plan_fit(agent._prospective_result_messages(pending, result), agent.tools)
    assert stats["request_window"] == 4096 and stats["reserve"] == 512
    assert stats["tokens"] == ctx.history_tokens(fitted) <= BUDGET


def test_existing_record_and_large_result_preserve_verbatim_pair(tmp_path):
    agent, pending = fixture_agent(tmp_path, record=True)
    raw = execution_result("large output\n" * 2000, executed=True, outcome="read_file completed.")
    result = guard_tool_result(
        "read_file", raw, memory=agent.memory,
        fits=lambda text: agent._tool_result_fits(pending, text, agent.tools),
    )
    original = agent.memory.get()
    fitted, stats = agent._plan_fit(agent._prospective_result_messages(pending, result), agent.tools)
    assert fitted[0]["content"] == ctx._omission_pair()[0]["content"]
    assert fitted[1] == ctx._omission_pair()[1]
    assert fitted[0]["metadata"] == {"internal": True}
    assert fitted[-1]["content"] == result
    assert fitted[-1]["tool_call_id"] == "current"
    assert fitted[2:-1] == original
    assert fitted[-1]["metadata"]["executed"] is True
    assert ctx.history_tokens(fitted[:2]) <= ctx.OMISSION_MAX_TOKENS
    assert stats["tokens"] + stats["reserve"] + ctx.estimate_tokens(agent.system) + ctx.tools_tokens(agent.tools) <= 4096
    assert agent.memory.get() == original


def test_request_bindings_and_minimum_pair_true_floor_still_fails(tmp_path):
    minimum_pair = ctx.history_tokens(ctx._omission_pair())
    binding = assistant_message("", [ToolCall("current", "read_file", {"path": "sample.py"})])
    minimum_receipt = "Outcome: read_file completed.\nOutput discarded."
    current = [
        {"role": "user", "content": ""}, binding,
        {"role": "tool", "tool_call_id": "current", "content": minimum_receipt},
    ]
    low, high = 0, BUDGET * 4
    while low < high:
        middle = (low + high + 1) // 2
        current[0]["content"] = "q" * middle
        if ctx.history_tokens(current) <= BUDGET - minimum_pair + 1:
            low = middle
        else:
            high = middle - 1
    current[0]["content"] = "q" * low
    assert ctx.history_tokens(current) < BUDGET
    assert minimum_pair + ctx.history_tokens(current) == BUDGET + 1
    agent, pending = fixture_agent(tmp_path, record=True, request=current[0]["content"])
    original = agent.memory.get()
    prospective = agent._prospective_result_messages(pending, minimum_receipt)
    with pytest.raises(ctx.ContextBudgetError, match="Historical omission coverage gap"):
        agent._plan_fit(prospective, agent.tools)
    assert not agent._tool_result_fits(pending, minimum_receipt, agent.tools)
    assert agent.memory.get() == original


@pytest.mark.parametrize("closed", [False, True])
def test_preflight_integer_payload_boundary_includes_exact_pair_cost(tmp_path, closed):
    agent, pending = fixture_agent(tmp_path, closed=closed)
    messages = agent._prospective_result_messages(pending, "")
    floor = ctx.history_tokens(ctx._omission_pair(
        evicted_turns=1, turns_before=3, evicted_closed_turns=int(closed),
    ))
    available = BUDGET - ctx.history_tokens(messages[2:]) - floor
    assert agent._tool_result_fits(pending, "", agent.tools, result_tokens=available)
    assert not agent._tool_result_fits(pending, "", agent.tools, result_tokens=available + 1)
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("shape", ["openai", "anthropic", "gemini", "gemini-camel", "local"])
def test_first_eviction_tax_precedes_payload_and_preserves_provider_pairing(tmp_path, closed, shape):
    agent, pending = fixture_agent(tmp_path, closed=closed)
    before = copy.deepcopy(agent.memory.get())
    pair = ctx._omission_pair(evicted_turns=1, turns_before=3, evicted_closed_turns=int(closed))
    floor = ctx.history_tokens(pair)
    prefix_tokens = ctx.history_tokens(before[:2])
    # Full history fits the raw budget, but not the mandatory transition tax.
    # Evicting exactly the oldest complete turn pays that tax without touching
    # the active request, call arguments, or result.
    payload = maximal_payload(agent, pending, BUDGET - floor + prefix_tokens // 2)
    prospective = agent._prospective_result_messages(pending, payload)
    assert ctx.history_tokens(prospective) < BUDGET
    fitted, stats = agent._plan_fit(prospective, agent.tools)
    assert stats["evicted_turns"] == 1
    assert stats["evicted_closed_turns"] == int(closed)
    assert fitted[:2] == pair
    assert fitted[2:] == prospective[2:]
    assert stats["omission_tokens"] == floor <= ctx.OMISSION_MAX_TOKENS
    assert stats["tokens"] == ctx.history_tokens(fitted) <= BUDGET
    assert agent.memory.get() == before
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None
    assert fitted[-3]["content"].encode("utf-8") == before[-2]["content"].encode("utf-8")

    if shape == "local":
        captured = {}

        def complete(**kwargs):
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        provider = LocalProvider.__new__(LocalProvider)
        provider.max_output_tokens = 512
        provider._llama = SimpleNamespace(create_chat_completion=complete)
        provider._run_llama_cpp(fitted, [], "rules")
        wire = captured["messages"][1:]
    else:
        projector = {"openai": openai_messages, "anthropic": anthropic_messages,
                     "gemini": gemini_messages, "gemini-camel": gemini_messages}[shape]
        wire = projector(fitted)
        if shape == "gemini-camel":
            for message in wire:
                for part in message.get("parts", []):
                    for snake, camel in (("function_call", "functionCall"),
                                         ("function_response", "functionResponse")):
                        if snake in part:
                            part[camel] = part.pop(snake)
    normalized = normalize_messages(wire)
    assert [(message["role"], message.get("content")) for message in normalized[:2]] == [
        (message["role"], message["content"]) for message in pair
    ]
    calls = [binding["id"] for message in normalized for binding in message.get("tool_calls", [])]
    results = [message["tool_call_id"] for message in normalized if message["role"] == "tool"]
    assert calls == results == ["current"]
    assert next(message for message in normalized if message["role"] == "tool")["content"] == payload


def test_unused_omission_headroom_does_not_create_eviction_state(tmp_path):
    agent, pending = fixture_agent(tmp_path)
    before = copy.deepcopy(agent.memory.get())
    fitted, stats = agent._plan_fit(agent._prospective_result_messages(pending, "small result"), agent.tools)
    assert fitted == before + [{"role": "tool", "tool_call_id": "current", "content": "small result"}]
    assert stats["evicted_turns"] == stats["omission_tokens"] == 0
    assert stats["session_record"] is None
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None
    assert agent.memory.get() == before


def test_stat_preflight_refuses_output_that_would_spend_first_eviction_tax(tmp_path):
    agent, pending = fixture_agent(tmp_path)
    messages = agent._prospective_result_messages(pending, "")
    floor = ctx.history_tokens(ctx._omission_pair(evicted_turns=1, turns_before=3))
    available = BUDGET - ctx.history_tokens(messages[2:]) - floor
    (tmp_path / "sample.py").write_bytes(b"x" * ((available + 1) * 4))
    refused = agent._preflight_tool_result(pending, agent.tools, 1.0)
    assert refused.executed is False
    assert str(tmp_path / "sample.py") not in tools._FILE_STATE
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None


def projected_margin_fixture(tmp_path, evicted, prefix_closed, next_closed, record):
    memory = SQLiteMemory(tmp_path / "projected" / "history.db", session_id="r10")
    history = []
    for index in range(evicted + 2):
        final = {"role": "assistant", "content": "Recorded."}
        if index < prefix_closed or (index == evicted and next_closed):
            final["metadata"] = {
                "executor_closed": True, "executor": True, "reason": "ContextBudgetError",
            }
        history.extend([{"role": "user", "content": f"Earlier {index}."}, final])
    current = ToolCall("projected", "read_file", {"path": "sample.py"})
    history.extend([
        {"role": "user", "content": "  Read sample.py. café\n\t"},
        assistant_message("", [current]),
    ])
    for message in history:
        memory.add_raw(message)
    if record:
        memory.set_session_metadata("SESSION_RECORD", RECORD)
    agent = Agent(Offline(), memory=memory, tools=[tools.READ_FILE_TOOL], verbose=False)
    wanted = 4096 - 512 - BUDGET - ctx.tools_tokens(agent.tools)
    agent.system = "s" * math.ceil((wanted - 1) * ctx.CHARS_PER_TOKEN)
    pending = {"call": current, "state": "pending", "result": None}
    agent._pending_calls = [pending]
    pair = ctx._omission_pair(
        evicted_turns=evicted, turns_before=evicted + 3,
        evicted_closed_turns=prefix_closed,
    )
    remaining = ctx.history_tokens(pair)
    target = BUDGET - remaining + ctx.history_tokens(history[:2 * evicted])
    payload = maximal_payload(agent, pending, target)
    return agent, pending, payload, pair


@pytest.mark.parametrize("record", [False, True], ids=["first-record", "existing-record"])
@pytest.mark.parametrize("evicted,prefix_closed,next_closed,extra", [
    (1, 0, True, 1), (9, 0, False, 1), (5, 2, False, 0),
], ids=["closed-suffix", "count-digit", "actual-closed-count"])
def test_projected_pair_razor_margin_preserves_active_content(
        tmp_path, record, evicted, prefix_closed, next_closed, extra):
    agent, pending, payload, current_pair = projected_margin_fixture(
        tmp_path, evicted, prefix_closed, next_closed, record)
    original = copy.deepcopy(agent.memory.get())
    prospective = agent._prospective_result_messages(pending, payload)
    assert (ctx.history_tokens(prospective[2 * evicted:])
            + ctx.history_tokens(current_pair)) == BUDGET
    fitted, stats = agent._plan_fit(prospective, agent.tools)
    count = evicted + extra
    closed = prefix_closed + int(next_closed and extra > 0)
    expected_pair = ctx._omission_pair(
        evicted_turns=count, turns_before=evicted + 3, evicted_closed_turns=closed)
    assert stats["evicted_turns"] == count
    assert stats["evicted_closed_turns"] == closed
    assert fitted[:2] == expected_pair
    assert fitted[2:] == prospective[2 * count:]
    assert fitted[-3:] == prospective[-3:]
    assert stats["tokens"] == ctx.history_tokens(fitted) <= BUDGET
    assert stats["omission_tokens"] == ctx.history_tokens(expected_pair) <= ctx.OMISSION_MAX_TOKENS
    assert agent.memory.get() == original
    assert agent.memory.get_session_metadata("SESSION_RECORD") == (RECORD if record else None)


@pytest.mark.parametrize("evicted,next_closed", [(1, True), (9, False)],
                         ids=["closed-suffix", "count-digit"])
def test_admission_projects_next_eviction_pair_and_exact_receipt_capacity(tmp_path, evicted, next_closed):
    agent, pending, payload, _ = projected_margin_fixture(tmp_path, evicted, 0, next_closed, True)
    original = copy.deepcopy(agent.memory.get())
    result = guard_tool_result(
        "read_file", execution_result(payload, executed=True, outcome="read_file completed."),
        memory=agent.memory,
        fits=lambda text: agent._tool_result_fits(pending, text, agent.tools))
    fitted, stats = agent._plan_fit(agent._prospective_result_messages(pending, result), agent.tools)
    assert result == payload
    assert stats["evicted_turns"] == evicted + 1
    assert fitted[-1]["tool_call_id"] == "projected"
    assert fitted[-1]["content"] == payload
    pair = ctx._omission_pair(
        evicted_turns=evicted + 1, turns_before=evicted + 3,
        evicted_closed_turns=int(next_closed))
    empty = agent._prospective_result_messages(pending, "")
    capacity = BUDGET - ctx.history_tokens(empty[2 * (evicted + 1):]) - ctx.history_tokens(pair)
    assert agent._tool_result_fits(pending, "", agent.tools, result_tokens=capacity)
    assert not agent._tool_result_fits(pending, "", agent.tools, result_tokens=capacity + 1)
    assert agent.memory.get() == original


def test_projected_reservation_without_pressure_does_not_create_record(tmp_path):
    agent, pending, _, _ = projected_margin_fixture(tmp_path, 9, 2, True, False)
    original = copy.deepcopy(agent.memory.get())
    prospective = agent._prospective_result_messages(pending, "small result")
    fitted, stats = agent._plan_fit(prospective, agent.tools)
    assert fitted == prospective
    assert stats["evicted_turns"] == stats["omission_tokens"] == 0
    assert stats["session_record"] is None
    assert agent.memory.get_session_metadata("SESSION_RECORD") is None
    assert agent.memory.get() == original


def test_projected_pair_does_not_suppress_irreducible_receipt_fatal(tmp_path):
    minimum_pair = ctx.history_tokens(ctx._omission_pair())
    minimum_receipt = "Outcome: read_file completed.\nOutput discarded."
    binding = assistant_message("", [ToolCall("current", "read_file", {"path": "sample.py"})])
    current = [
        {"role": "user", "content": ""}, binding,
        {"role": "tool", "tool_call_id": "current", "content": minimum_receipt},
    ]
    low, high = 0, BUDGET * 4
    while low < high:
        middle = (low + high + 1) // 2
        current[0]["content"] = "q" * middle
        if ctx.history_tokens(current) <= BUDGET - minimum_pair + 1:
            low = middle
        else:
            high = middle - 1
    current[0]["content"] = "q" * low
    assert ctx.history_tokens(current[:-1]) + minimum_pair <= BUDGET
    assert ctx.history_tokens(current) + minimum_pair == BUDGET + 1
    agent, pending = fixture_agent(tmp_path, record=True, request=current[0]["content"])
    original = copy.deepcopy(agent.memory.get())
    with pytest.raises(ctx.ContextBudgetError, match="truthful outcome and output-omission receipt cannot fit"):
        guard_tool_result(
            "read_file", execution_result("x" * 20000, executed=True, outcome="read_file completed."),
            memory=agent.memory,
            fits=lambda text: agent._tool_result_fits(pending, text, agent.tools))
    assert agent.memory.get() == original
    assert agent.memory.get_session_metadata("SESSION_RECORD") == RECORD


def test_assistant_arguments_between_fits_remain_protected(tmp_path):
    agent, pending, payload, _ = projected_margin_fixture(tmp_path, 1, 0, True, True)
    prospective = agent._prospective_result_messages(pending, payload)
    fitted, _ = agent._plan_fit(prospective, agent.tools)
    assert ctx.history_tokens(fitted) <= BUDGET
    call = ToolCall("later", "run_code", {"code": "print('" + "x" * 600 + "')"})
    later_binding = assistant_message("", [call])
    assert ctx.message_tokens(later_binding) < 512
    after = prospective + [
        later_binding, {"role": "tool", "tool_call_id": "later", "content": "Output discarded."},
    ]
    before = copy.deepcopy(after)
    with pytest.raises(ctx.ContextBudgetError):
        agent._plan_fit(after, agent.tools)
    assert after == before
