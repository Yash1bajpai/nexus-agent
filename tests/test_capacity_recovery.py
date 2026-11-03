"""Optional recovery uses the real budget policy and preserves executor truth."""
import copy
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import context, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory, is_genuine_user_message
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import RefusalClass, execution_result, refusal
from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, Tool, ToolCall
from nexus_agent_ai.providers.messages import assistant_message


class Scripted(BaseProvider):
    model = "offline-capacity-recovery"
    context_size = 32768
    max_output_tokens = 512
    is_local = True

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []
        self.offered = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        self.offered.append([tool.name for tool in tools])
        assert self.steps, "optional recovery made an unexpected provider request"
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


def response(name, args=None, identity="actual", text=""):
    return ProviderResponse(text=text, tool_calls=[ToolCall(identity, name, args or {})])


def probe(name="probe"):
    return Tool(name, "Observe a deterministic fixture.", {
        "type": "object", "properties": {
            "path": {"type": "string"}, "value": {"type": "string"},
        },
    }, lambda **kwargs: "unused registry handler")


def make_agent(provider, offered, **kwargs):
    agent = Agent(provider, tools=offered, verbose=False, **kwargs)
    agent.system = "Use the supplied observations; do not invent execution."
    return agent


def encoded(messages):
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()


def audit_rows(memory):
    with sqlite3.connect(memory.db_path) as connection:
        return connection.execute(
            "SELECT role, message_json FROM conversation_history WHERE session_id=? ORDER BY id",
            (memory.session_id,),
        ).fetchall()


def assert_closed(agent):
    messages = agent.memory.get()
    closure = messages[-1]
    assert closure["role"] == "assistant"
    metadata = closure["metadata"]
    assert metadata["executor_closed"] is True
    assert metadata["executor"] is True
    assert isinstance(metadata["reason"], str) and metadata["reason"]
    assert "executor" in closure["content"].lower()
    assert "output limit" not in closure["content"].lower()
    assert "workspace unchanged" not in closure["content"].lower()
    assert not closure.get("tool_calls")
    assert context._completed_turn(messages)
    assert not any(message["role"] == "system" for message in messages)
    return closure


@pytest.fixture(autouse=True)
def explicit_tools_only(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    monkeypatch.setenv("NEXUS_AGENT_AUTO_CHECK", "0")


@pytest.mark.parametrize("missing_component", ["assistant", "nudge"])
def test_rejected_probe_fits_exact_prospective_candidate_without_side_effects(
    tmp_path, monkeypatch, missing_component,
):
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="prospective")
    memory.add("user", "Exact active request: café\r\n  ")
    provider = Scripted([])
    offered = [probe()]
    agent = make_agent(provider, offered, memory=memory)
    agent._mutation_segment = "observed-segment"
    agent._mutation_outcomes = {"write": {
        "event": "observed-write", "turn": agent._turn_id, "segment": agent._mutation_segment,
        "path": str(tmp_path / "observed.txt"), "action": "write_file", "status": "changed",
        "success": True, "failed": False,
    }}
    original = "Exact model bytes: café\r\n  " + "a" * 1200
    nudge = "[SYSTEM] Retry the operation using the observed state. " + "n" * 800
    assistant = assistant_message(original, [])
    assistant["metadata"] = agent._outcome_metadata()
    internal = {"role": "user", "content": nudge, "metadata": {"internal": True}}
    candidate = copy.deepcopy(memory.get()) + [assistant, internal]
    smaller = copy.deepcopy(candidate)
    if missing_component == "assistant":
        del smaller[-2]
    else:
        smaller.pop()
    overhead = context.estimate_tokens(agent.system) + context.tools_tokens(offered)
    _, complete_stats = agent._plan_fit(candidate, offered)
    agent.context_window = overhead + agent.output_reserve + complete_stats["tokens"] - 1
    agent._plan_fit(smaller, offered)
    with pytest.raises(context.ContextBudgetError):
        agent._plan_fit(candidate, offered)
    before = copy.deepcopy(memory.get())
    rows = audit_rows(memory)
    previous_trim = copy.deepcopy(agent.last_trim)
    seen = []
    actual_plan = agent._plan_fit

    def observe_plan(messages, tools, scale=1.0, **kwargs):
        seen.append(copy.deepcopy(messages))
        # The planner is explicitly allowed a disposable, deeply isolated view.
        messages[0].setdefault("metadata", {})["probe_only"] = True
        return actual_plan(messages, tools, scale, **kwargs)

    monkeypatch.setattr(agent, "_plan_fit", observe_plan)
    monkeypatch.setattr(agent, "_record_fit", lambda *a, **k: pytest.fail("probe committed fit state"))
    monkeypatch.setattr(agent, "_audit", lambda *a, **k: pytest.fail("probe emitted audit events"))
    monkeypatch.setattr(memory, "store_artifacts", lambda *a, **k: pytest.fail("probe stored artifacts"))
    monkeypatch.setattr(memory, "set_session_metadata", lambda *a, **k: pytest.fail("probe changed SESSION_RECORD"))
    monkeypatch.setattr(core, "execute_tool", lambda *a, **k: pytest.fail("probe executed a tool"))
    assert agent._try_automatic_nudge(nudge, assistant_text=original, tools=offered) is False
    assert seen == [candidate]
    assert memory.get() == before + [assistant]
    assert encoded(memory.get()[:len(before)]) == encoded(before)
    assert audit_rows(memory)[:len(rows)] == rows
    assert agent.last_trim == previous_trim
    assert provider.seen == []


def test_admitted_probe_includes_recovery_record_scale_tools_and_pending_boost(tmp_path, monkeypatch):
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="projected")
    call = ToolCall("failed", "patch_file", {"path": "f.py", "target": "wrong", "replacement": "right"})
    memory.replace([
        {"role": "user", "content": "Exact request\r\n café  "},
        assistant_message("", [call]),
        {"role": "tool", "tool_call_id": "failed", "content": "ERROR: Original refusal\r\n",
         "metadata": {"executed": False, "refusal_class": "targeting", "outcome": "Not executed."}},
    ])
    record = {"version": 1, "files": ["prior.py"], "topics": [], "tools": []}
    memory.set_session_metadata("SESSION_RECORD", record)
    provider = Scripted([])
    provider._supports_output_budget = True
    provider._request_output_cap = 333
    offered = [tools.PATCH_FILE_TOOL]
    agent = make_agent(provider, offered, memory=memory)
    agent._output_boost = 2048
    agent._edit_recovery = {
        "status": "pending", "call_id": "failed", "path": str(tmp_path / "f.py"),
        "reason": "targeting", "excerpt": "value = 1\r\n", "read_ids": [],
    }
    original = "I have not changed f.py.\r\n café  "
    nudge = "[SYSTEM] Use the source observation to correct the edit.\r\n  "
    before = copy.deepcopy(memory.get())
    rows = audit_rows(memory)
    recovery_before = copy.deepcopy(agent._edit_recovery)
    trim_before = copy.deepcopy(agent.last_trim)
    seen = []
    actual_plan = agent._plan_fit

    def observe_plan(messages, tools, scale=1.0, **kwargs):
        fitted, stats = actual_plan(messages, tools, scale, **kwargs)
        seen.append((copy.deepcopy(messages), list(tools), scale, copy.deepcopy(fitted), stats))
        return fitted, stats

    monkeypatch.setattr(agent, "_plan_fit", observe_plan)
    monkeypatch.setattr(agent, "_record_fit", lambda *a, **k: pytest.fail("probe committed fit state"))
    monkeypatch.setattr(agent, "_audit", lambda *a, **k: pytest.fail("probe emitted audit events"))
    monkeypatch.setattr(memory, "store_artifacts", lambda *a, **k: pytest.fail("probe stored artifacts"))
    monkeypatch.setattr(memory, "set_session_metadata", lambda *a, **k: pytest.fail("probe changed SESSION_RECORD"))
    assert agent._try_automatic_nudge(nudge, assistant_text=original, tools=offered, scale=0.75) is True
    assert len(seen) == 1
    candidate, planned_tools, scale, fitted, stats = seen[0]
    assert candidate == before + [assistant_message(original, []), {
        "role": "user", "content": nudge, "metadata": {"internal": True},
    }]
    assert planned_tools == offered and scale == 0.75
    assert stats["request_window"] == int(agent.context_window * 0.75)
    assert stats["reserve"] == 2048
    assert stats["recovery_injected"] is True
    assert "Retained file paths: prior.py" in fitted[0]["content"]
    projected_result = next(m for m in fitted if m.get("tool_call_id") == "failed")
    assert projected_result["content"].startswith(before[-1]["content"] + "\n\n--- BEGIN HARNESS EDIT RECOVERY ---")
    assert memory.get() == candidate
    assert audit_rows(memory)[:len(rows)] == rows
    assert memory.get_session_metadata("SESSION_RECORD") == record
    assert agent._edit_recovery == recovery_before
    assert agent._output_boost == 2048
    assert agent.output_reserve == provider.max_output_tokens == 512
    assert provider._request_output_cap == 333
    assert agent.last_trim == trim_before
    assert provider.seen == []


def test_nonbudget_probe_exception_preserves_original_and_propagates(monkeypatch):
    agent = make_agent(Scripted([]), [probe()])
    agent.memory.add("user", "Observe the fixture")
    before = copy.deepcopy(agent.memory.get())
    original = "Exact generated text\r\n café  "
    failure = ValueError("unrelated planner failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(agent, "_plan_fit", fail)
    with pytest.raises(ValueError) as raised:
        agent._try_automatic_nudge("[SYSTEM] Retry", assistant_text=original, tools=agent.tools)
    assert raised.value is failure
    assert agent.memory.get() == before + [assistant_message(original, [])]
    assert agent.provider.seen == []


@pytest.mark.parametrize("kind", ["tool_error", "edit", "test_honesty"])
def test_rejected_optional_nudge_keeps_originals_and_closes_without_extra_request(kind, tmp_path, monkeypatch):
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id=kind)
    memory.replace([{"role": "user", "content": "Earlier request"},
                    {"role": "assistant", "content": "Earlier exact answer\r\n café  "}])
    prior = copy.deepcopy(memory.get())
    prior_rows = audit_rows(memory)
    prefix = "All tests pass.\r\n" if kind == "test_honesty" else "I cannot complete this yet.\r\n"
    original = prefix + "exact generated text café  \r\n" * 500
    result_text = "Pytest Results: FAILED\nassert 1 == 2" if kind == "test_honesty" else "ERROR: temporary target failure\r\n"
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append((name, copy.deepcopy(args)))
        return execution_result(result_text, executed=kind == "test_honesty", outcome="Fixture observation recorded.")

    monkeypatch.setattr(core, "execute_tool", execute)
    offered = [tools.WRITE_FILE_TOOL] if kind == "edit" else [probe() if kind == "tool_error" else tools.RUN_TESTS_TOOL]
    steps = [] if kind == "edit" else [response(offered[0].name)]
    provider = Scripted(steps + [ProviderResponse(text=original)])
    provider.context_size = 4096
    agent = make_agent(provider, offered, memory=memory)
    final = agent.run("Fix the requested output" if kind == "edit" else "Check fixture.txt", require_edit=kind == "edit")
    assert len(provider.seen) == (1 if kind == "edit" else 2)
    assert len(dispatched) == (0 if kind == "edit" else 1)
    messages = memory.get()
    assert messages[:len(prior)] == prior
    assert audit_rows(memory)[:len(prior_rows)] == prior_rows
    originals = [m for m in messages if m["role"] == "assistant" and m.get("content") == original]
    assert len(originals) == 1
    if kind != "edit":
        tool_result = next(m for m in messages if m.get("tool_call_id") == "actual")
        assert tool_result["content"].startswith(result_text)
        assert tool_result["metadata"]["executed"] is (kind == "test_honesty")
    assert not any(m.get("metadata", {}).get("internal") for m in messages)
    assert agent.unresolved_errors
    assert not agent.response_incomplete
    assert_closed(agent)
    assert "output limit" not in final.lower()
    if kind == "test_honesty":
        assert agent.claim_unverified


@pytest.mark.parametrize("typed", [True, False])
def test_capacity_no_progress_uses_source_type_not_identical_error_prose(typed, monkeypatch):
    text = "ERROR: Not executed: result exceeds the capacity context budget.\r\n"
    first = refusal(RefusalClass.CAPACITY, text) if typed else text
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(copy.deepcopy(args))
        if args.get("value") == "initial":
            return first
        return execution_result("Recovered observation.", executed=True, outcome="Observation completed.")

    monkeypatch.setattr(core, "execute_tool", execute)
    original = "The observation was refused.\r\n café  "
    steps = [response("probe", {"value": "initial"}), ProviderResponse(text=original)]
    if not typed:
        steps += [response("probe", {"value": "corrected"}, "corrected"), ProviderResponse(text="Observation complete.")]
    provider = Scripted(steps)
    agent = make_agent(provider, [probe()])
    agent.run("Inspect fixture.txt")
    assert len(provider.seen) == (2 if typed else 4)
    assert len(dispatched) == (1 if typed else 2)
    assert any(m["role"] == "assistant" and m.get("content") == original for m in agent.memory.get())
    first_result = next(m for m in agent.memory.get() if m.get("tool_call_id") == "actual")
    assert first_result["content"] == text
    if typed:
        assert first_result["metadata"]["refusal_class"] == "capacity"
        assert first_result["metadata"]["executed"] is False
        assert agent.unresolved_errors
        assert_closed(agent)
        assert not any(m.get("metadata", {}).get("internal") for m in agent.memory.get())
    else:
        assert first_result.get("metadata", {}).get("refusal_class") != "capacity"
        assert not agent.unresolved_errors
        assert any(m.get("metadata", {}).get("internal") for m in agent.memory.get())


def test_real_oversized_read_preflight_is_capacity_and_never_enters_handler(tmp_path, monkeypatch):
    (tmp_path / "large.txt").write_text("x" * 200000)
    monkeypatch.setattr(tools.READ_FILE_TOOL, "execute", lambda **k: pytest.fail("capacity-refused read executed"))
    provider = Scripted([response("read_file", {"path": "large.txt"}),
                         ProviderResponse(text="The file could not be read within the context budget.")])
    provider.context_size = 4096
    agent = make_agent(provider, [tools.READ_FILE_TOOL])
    agent.run("Read large.txt")
    assert len(provider.seen) == 2
    result = next(m for m in agent.memory.get() if m["role"] == "tool")
    assert "the file was not read" in result["content"]
    assert result["metadata"]["refusal_class"] == "capacity"
    assert result["metadata"]["executed"] is False
    assert str(tmp_path / "large.txt") not in tools._FILE_STATE
    assert agent.unresolved_errors
    assert_closed(agent)


def test_mixed_capacity_and_other_errors_retry_only_actionable_error(monkeypatch):
    capacity_text = "ERROR: capacity observation cannot fit."
    other_text = "ERROR: fix the syntax for the capacity-named target."
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append((name, copy.deepcopy(args)))
        if args["path"] == "large.txt":
            return refusal(RefusalClass.CAPACITY, capacity_text)
        if args["value"] == "bad":
            return refusal(RefusalClass.SYNTAX, other_text)
        return execution_result("Corrected observation.", executed=True, outcome="Observation completed.")

    monkeypatch.setattr(core, "execute_tool", execute)
    batch = ProviderResponse(tool_calls=[
        ToolCall("capacity", "probe", {"path": "large.txt", "value": "bad"}),
        ToolCall("syntax", "probe", {"path": "capacity-named.txt", "value": "bad"}),
    ])
    provider = Scripted([batch, ProviderResponse(text="The observations are incomplete."),
                         response("probe", {"path": "capacity-named.txt", "value": "fixed"}, "fixed"),
                         ProviderResponse(text="The second observation succeeded; the large one is unavailable.")])
    agent = make_agent(provider, [probe()])
    agent.run("Inspect large.txt and capacity-named.txt")
    assert len(provider.seen) == 4 and len(dispatched) == 3
    nudge = provider.seen[2][-1]
    assert nudge["metadata"] == {"internal": True}
    assert other_text in nudge["content"]
    assert capacity_text not in nudge["content"]
    assert capacity_text in agent.unresolved_errors
    assert other_text not in agent.unresolved_errors
    assert_closed(agent)


def test_capacity_does_not_block_model_selected_alternative_or_later_user_turn(monkeypatch):
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(copy.deepcopy(args))
        if args["value"] == "large":
            return refusal(RefusalClass.CAPACITY, "ERROR: large observation cannot fit.")
        return execution_result("Small observation.", executed=True, outcome="Observation completed.")

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Scripted([
        response("probe", {"path": "same.txt", "value": "large"}, "large"),
        response("probe", {"path": "same.txt", "value": "small"}, "small"),
        ProviderResponse(text="The smaller observation succeeded."),
        response("probe", {"path": "same.txt", "value": "large"}, "later"),
        ProviderResponse(text="That observation still cannot fit."),
    ])
    agent = make_agent(provider, [probe()])
    agent.run("Inspect same.txt")
    assert not agent.unresolved_errors
    assert len(provider.seen) == 3
    assert not any(m.get("metadata", {}).get("executor_closed") for m in agent.memory.get())
    before = copy.deepcopy(agent.memory.get())
    agent.run("Try the original observation for same.txt again")
    assert len(provider.seen) == 5 and len(dispatched) == 3
    assert agent.memory.get()[:len(before)] == before
    assert agent.unresolved_errors
    assert_closed(agent)


def test_normal_edit_nudge_still_allows_actual_write(tmp_path):
    original = "I will prepare the requested file."
    provider = Scripted([ProviderResponse(text=original),
                         response("write_file", {"path": "created.txt", "content": "actual content"}),
                         ProviderResponse(text="Created created.txt.")])
    agent = make_agent(provider, [tools.WRITE_FILE_TOOL])
    agent.run("Create created.txt", require_edit=True)
    assert (tmp_path / "created.txt").read_text() == "actual content"
    assert len(provider.seen) == 3
    assert provider.seen[1][-2]["content"] == original
    assert provider.seen[1][-1]["metadata"] == {"internal": True}
    assert not agent.unresolved_errors
    assert agent.changed_files == {str(tmp_path / "created.txt")}


def test_normal_test_honesty_nudge_still_runs_passing_followup(monkeypatch):
    results = iter(["Pytest Results: FAILED\nassert 1 == 2", "Pytest Results: PASSED"])
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(name)
        return execution_result(next(results), executed=True, outcome="Fixture test process completed.")

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Scripted([
        response("run_tests"), ProviderResponse(text="All tests pass."),
        response("run_tests", {"path": "tests"}, "rechecked"), ProviderResponse(text="All tests pass."),
    ])
    agent = make_agent(provider, [tools.RUN_TESTS_TOOL], guardrails=True)
    agent.run("Run the tests")
    assert dispatched == ["run_tests", "run_tests"]
    assert len(provider.seen) == 4
    assert provider.seen[2][-1]["metadata"] == {"internal": True}
    assert "FAILED" in provider.seen[2][-1]["content"]
    assert not agent.claim_unverified and not agent.unresolved_errors


@pytest.mark.parametrize("failure", ["target", "syntax", "stale"])
def test_normal_source_edit_recovery_remains_available(failure, tmp_path):
    path = tmp_path / "f.py"
    path.write_text("def f():\n    return 1\n")
    read = response("read_file", {"path": "f.py"}, "read")
    corrected = response("patch_file", {"path": "f.py", "target": "return 1", "replacement": "return 2"}, "corrected")
    if failure == "target":
        failed = response("patch_file", {"path": "f.py", "target": "missing source", "replacement": "return 2"}, "failed")
    elif failure == "syntax":
        failed = response("patch_file", {"path": "f.py", "target": "return 1", "replacement": "return ("}, "failed")
    else:
        def failed():
            path.write_text("def f():\n    return 10\n")
            return response("patch_file", {"path": "f.py", "target": "return 1", "replacement": "return 2"}, "failed")
        corrected.tool_calls[0].args["target"] = "return 10"
    provider = Scripted([read, failed, corrected, ProviderResponse(text="Updated f.py.")])
    agent = make_agent(provider, [tools.READ_FILE_TOOL, tools.PATCH_FILE_TOOL])
    agent.run("Update f.py", require_edit=True)
    assert path.read_text() == "def f():\n    return 2\n"
    assert len(provider.seen) == 4
    failed_result = next(m for m in agent.memory.get() if m.get("tool_call_id") == "failed")
    assert failed_result["metadata"]["refusal_class"] != "capacity"
    assert not agent.unresolved_errors
    assert not agent.edit_not_applied
    assert len([m for m in agent.memory.get() if is_genuine_user_message(m)]) == 1


@pytest.mark.parametrize("executed", [False, True])
def test_minimum_receipt_failure_keeps_exact_result_and_closes_pending_ids(executed, tmp_path, monkeypatch):
    full = "ERROR: Fixture result\r\n café  " + "raw result\n" * 1000
    result = execution_result(full, executed=executed, outcome="Fixture action completed." if executed else "Fixture action not executed.")
    dispatched = []

    def execute(name, args, **kwargs):
        dispatched.append(name)
        return result

    monkeypatch.setattr(core, "execute_tool", execute)
    original = "Exact generated assistant café\r\n  " * 700
    batch = ProviderResponse(text=original, tool_calls=[
        ToolCall("returned", "run_tests", {}), ToolCall("pending", "probe", {"value": "must not execute"}),
    ])
    provider = Scripted([batch])
    provider.context_size = 4096
    memory = SQLiteMemory(tmp_path / "private" / "history.db", session_id="minimum")
    agent = make_agent(provider, [tools.RUN_TESTS_TOOL, probe()], memory=memory, guardrails=False)
    with pytest.raises(context.ContextBudgetError) as raised:
        agent.run("Observe both fixture.txt observations")
    assert type(raised.value) is context.ContextBudgetError
    assert str(raised.value) == (
        "The tool's truthful outcome and output-omission receipt cannot fit the current context. "
        "Execution status is preserved; no over-budget request was sent."
    )
    assert raised.value.refusal_class is RefusalClass.CAPACITY
    assert len(provider.seen) == 1 and dispatched == ["run_tests"]
    messages = memory.get()
    assert next(m for m in messages if m.get("tool_calls"))["content"] == original
    results = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["returned", "pending"]
    assert results[0]["content"] == full
    assert results[0]["metadata"]["executed"] is executed
    assert results[0]["metadata"]["outcome"] == result.outcome
    assert results[1]["metadata"]["executed"] is False
    assert "Not executed" in results[1]["content"]
    assert_closed(agent)
    with sqlite3.connect(memory.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM context_artifacts").fetchone()[0] == 0


def test_mixed_success_and_capacity_closure_acknowledges_actual_write(tmp_path, monkeypatch):
    actual_execute = core.execute_tool

    def execute(name, args, **kwargs):
        if name == "probe":
            return refusal(RefusalClass.CAPACITY, "ERROR: observation cannot fit.")
        return actual_execute(name, args, **kwargs)

    monkeypatch.setattr(core, "execute_tool", execute)
    original = "Created actual.txt; the other observation was unavailable.\r\n café  "
    provider = Scripted([
        ProviderResponse(tool_calls=[
            ToolCall("write", "write_file", {"path": "actual.txt", "content": "actual side effect"}),
            ToolCall("refused", "probe", {}),
        ]), ProviderResponse(text=original),
    ])
    agent = make_agent(provider, [tools.WRITE_FILE_TOOL, probe()])
    final = agent.run("Create actual.txt and inspect the fixture", require_edit=True)
    assert (tmp_path / "actual.txt").read_text() == "actual side effect"
    assert agent.changed_files == {str(tmp_path / "actual.txt")}
    assert len(provider.seen) == 2
    assert not agent.edit_not_applied
    assert "No file was changed" not in final and "workspace unchanged" not in final.lower()
    original_message = next(m for m in agent.memory.get() if m.get("content") == original)
    assert original_message["metadata"]["mutation_outcomes"]
    results = {m["tool_call_id"]: m for m in agent.memory.get() if m["role"] == "tool"}
    assert results["write"]["metadata"]["executed"] is True
    assert results["refused"]["metadata"]["executed"] is False
    assert_closed(agent)


@pytest.mark.parametrize("stream", [False, True])
def test_capacity_no_progress_is_nonzero_in_cli_without_extra_inference(tmp_path, monkeypatch, stream):
    (tmp_path / "large.txt").write_text("x" * 300000)
    provider = Scripted([
        response("read_file", {"path": "large.txt"}),
        ProviderResponse(text="The read was refused; I cannot complete that observation."),
    ])
    instances = []
    actual_agent = cli.Agent

    def capture_agent(*args, **kwargs):
        instance = actual_agent(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(cli, "Agent", capture_agent)
    monkeypatch.setattr(cli, "get_provider_instance", lambda *a, **k: (provider, "scripted"))
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(tools.READ_FILE_TOOL, "execute", lambda **k: pytest.fail("refused read handler ran"))
    args = ["chat", "Read large.txt", "-p", "local"]
    if not stream:
        args.append("--no-stream")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert len(provider.seen) == 2
    assert instances[0].unresolved_errors
    assert not instances[0].response_incomplete
    assert_closed(instances[0])


@pytest.mark.parametrize("minimum_fits", [False, True])
def test_adaptive_probe_uses_normal_reserve_floor_without_consuming_boost(minimum_fits, monkeypatch):
    provider = Scripted([])
    provider._supports_output_budget = True
    provider._request_output_cap = 400
    agent = make_agent(provider, [probe()])
    agent.memory.add("user", "Observe the exact active fixture.")
    original = "The observation needs another attempt.\r\n café  "
    nudge = "[SYSTEM] Correct the observation using the same available tools."
    candidate = copy.deepcopy(agent.memory.get()) + [
        assistant_message(original, []),
        {"role": "user", "content": nudge, "metadata": {"internal": True}},
    ]
    overhead = context.estimate_tokens(agent.system) + context.tools_tokens(agent.tools)
    agent.context_window = overhead + context.history_tokens(candidate) + (256 if minimum_fits else 255)
    agent._output_boost = 2048
    reserves = []
    actual_plan = agent._plan_fit
    before = copy.deepcopy(agent.memory.get())
    previous_trim = copy.deepcopy(agent.last_trim)

    def observe_plan(messages, tools, scale=1.0, **kwargs):
        fitted, stats = actual_plan(messages, tools, scale, **kwargs)
        reserves.append(stats["reserve"])
        return fitted, stats

    monkeypatch.setattr(agent, "_plan_fit", observe_plan)
    accepted = agent._try_automatic_nudge(nudge, assistant_text=original, tools=agent.tools)
    assert accepted is minimum_fits
    assert reserves == ([256] if minimum_fits else [])
    assert agent.memory.get() == (candidate if minimum_fits else before + [assistant_message(original, [])])
    assert agent.output_reserve == provider.max_output_tokens == 512
    assert provider._request_output_cap == 400
    assert agent._output_boost == 2048
    assert agent.last_trim == previous_trim
    assert provider.seen == []
