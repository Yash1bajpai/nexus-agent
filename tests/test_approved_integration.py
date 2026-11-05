import copy
import os

import pytest

from nexus_agent_ai.agent import context, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.agent.refusals import RefusalClass, RunnerStatus, execution_result, refusal, runner_result
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


class Transcript(BaseProvider):
    model = "approved-integration-stub"
    is_local = False
    context_size = 16384
    max_output_tokens = 512

    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append((copy.deepcopy(messages), [tool.name for tool in tools]))
        step = self.steps.pop(0)
        if isinstance(step, ProviderResponse):
            return step
        if isinstance(step, tuple):
            return ProviderResponse(tool_calls=[ToolCall(f"c{len(self.seen)}", step[0], step[1])])
        return ProviderResponse(text=step)

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": result}


@pytest.fixture(autouse=True)
def explicit_file_reads(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")


def patch():
    return "patch_file", {"path": "f.py", "target": "absent", "replacement": "x = 2"}


@pytest.mark.parametrize("kind", list(RefusalClass))
@pytest.mark.parametrize("edit_tool", ["patch_file", "edit_function", "add_function"])
def test_typed_refusal_matrix_controls_offering_and_dispatch_only_by_class(tmp_path, monkeypatch, kind, edit_tool):
    path = tmp_path / "f.py"
    path.write_text("x = 1\n")
    original_bytes = path.read_bytes()
    calls = []
    real_dispatch = tools.execute_tool

    def execute(name, args, **kwargs):
        if name == edit_tool:
            # Deliberately identical wording: taxonomy, not strings, determines the gate.
            return refusal(kind, "ERROR: neutral refusal message")
        if name == "run_code" and kwargs.get("gated_reason") is not None:
            return real_dispatch(name, args, **kwargs)
        calls.append(name)
        return "STDOUT:\n42"

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Transcript([
        (edit_tool, {"path": "f.py", "function_name": "f", "new_body": "def f():\n    return 2\n"}),
        ProviderResponse(tool_calls=[
            ToolCall("code", "run_code", {"code": "print(42)"}),
            ToolCall("corrected", edit_tool, {
                "path": "f.py", "function_name": "f", "new_body": "def f():\n    return 2\n",
            }),
        ]), "No changes were made.",
    ])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    agent.run("Fix f.py")
    gates = {RefusalClass.TARGETING, RefusalClass.SYNTAX, RefusalClass.CONTENT_LOSS,
             RefusalClass.AMBIGUITY, RefusalClass.BLIND_EDIT}
    gated = kind in gates
    assert ("run_code" not in provider.seen[1][1]) is gated
    assert ("run_code" not in calls) is gated
    audit = [event for event in events if event["type"] == "tool_gated"]
    assert [event["enforced_at"] for event in audit] == (["offered", "dispatch"] if gated else [])
    assert all(event["reason"] == kind.value and event["tool"] == "run_code" for event in audit)
    assert path.read_bytes() == original_bytes


def test_untyped_error_text_is_not_used_to_guess_gate_taxonomy(tmp_path, monkeypatch):
    (tmp_path / "f.py").write_text("x = 1\n")
    executed = []

    def execute(name, args):
        executed.append(name)
        return "ERROR: Python syntax validation failed" if name == "patch_file" else "STDOUT:\n42"

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Transcript([patch(), ("run_code", {"code": "print(42)"}), "No changes were made."])
    Agent(provider, verbose=False).run("Fix f.py")
    assert executed == ["patch_file", "run_code"]
    assert "run_code" in provider.seen[1][1]


@pytest.mark.parametrize("code_first", [False, True])
def test_same_batch_order_gate_and_readonly_tools_are_deterministic(tmp_path, monkeypatch, code_first):
    (tmp_path / "f.py").write_text("x = 1\n")
    original_text = (tmp_path / "f.py").read_bytes().decode("utf-8")
    tools.execute_read_file("f.py")
    code_calls = []
    monkeypatch.setattr(tools.RUN_CODE_TOOL, "execute", lambda **kwargs: code_calls.append(kwargs) or "STDOUT:\n42")
    edit = ToolCall("edit", *patch())
    code = ToolCall("code", "run_code", {"code": "print(42)"})
    read = ToolCall("read", "read_file", {"path": "f.py"})
    batch = [code, edit, read] if code_first else [edit, code, read]
    provider = Transcript([ProviderResponse(tool_calls=batch), "No changes were made."])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    agent.run("Fix f.py")
    assert len(code_calls) == int(code_first)
    results = {m["tool_call_id"]: m["content"] for m in agent.memory.get() if m["role"] == "tool"}
    assert set(results) == {"edit", "code", "read"}
    assert results["read"] == original_text
    assert results["code"].startswith("STDOUT:" if code_first else "ERROR:")
    assert "run_code" not in provider.seen[1][1]
    assert any(event["type"] == "tool_gated" and event["enforced_at"] == "offered" for event in events)


def test_every_hidden_gated_call_has_its_own_dispatch_audit(tmp_path, monkeypatch):
    path = tmp_path / "f.py"
    path.write_text("x = 1\n")
    tools.execute_read_file("f.py")
    executed = []
    monkeypatch.setattr(tools.RUN_CODE_TOOL, "execute", lambda **kwargs: executed.append(kwargs))
    provider = Transcript([
        ProviderResponse(tool_calls=[
            ToolCall("edit", *patch()),
            ToolCall("hidden1", "run_code", {"code": "print(1)"}),
            ToolCall("hidden2", "run_code", {"code": "print(2)"}),
        ]),
        "The edit failed.",
    ])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    agent.run("Fix f.py")
    dispatches = [event for event in events if event["type"] == "tool_gated"
                  and event["enforced_at"] == "dispatch"]
    results = {message["tool_call_id"]: message["content"]
               for message in agent.memory.get() if message["role"] == "tool"}
    assert len(dispatches) == 2
    assert all(event["reason"] == RefusalClass.TARGETING.value for event in dispatches)
    assert all(results[key].startswith("ERROR:") for key in ("hidden1", "hidden2"))
    assert executed == []
    assert path.read_text() == "x = 1\n"


def test_gate_restores_at_next_user_turn(tmp_path, monkeypatch):
    (tmp_path / "f.py").write_text("x = 1\n")
    tools.execute_read_file("f.py")
    calls = []
    completed_receipt = runner_result(
        execution_result("STDOUT:\n42", executed=True, outcome="Code completed (exit code 0)."),
        tool="run_code",
        status=RunnerStatus.COMPLETED,
    )
    monkeypatch.setattr(tools.RUN_CODE_TOOL, "execute", lambda **kwargs: calls.append(kwargs) or completed_receipt)
    provider = Transcript([patch(), "No changes were made.", ("run_code", {"code": "print(42)"}), "42"])
    agent = Agent(provider, verbose=False)
    agent.run("Fix f.py")
    assert "run_code" in [tool.name for tool in agent.tools]
    assert agent.run("Use run_code to print 42") == "42"
    assert len(calls) == 1
    assert "run_code" in provider.seen[2][1]


def test_function_edit_checks_fresh_pre_call_hash_and_does_not_gate_stale_state(tmp_path, monkeypatch):
    path = tmp_path / "f.py"
    original = b"def f():\n    return 1\n"
    external = b"def f():\n    return 3\n"
    path.write_bytes(original)
    tools.execute_read_file("f.py")
    stat = path.stat()
    real_dispatch = tools.execute_tool
    states = []
    refused_classes = []

    def execute(name, args, **kwargs):
        if name == "edit_function":
            states.append(kwargs["expected_state"])
            path.write_bytes(external)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        result = real_dispatch(name, args, **kwargs)
        if name == "edit_function":
            refused_classes.append(result.refusal_class)
        return result

    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Transcript([
        ("edit_function", {"path": "f.py", "function_name": "f", "new_body": "def f():\n    return 2\n"}),
        "No changes were made.",
    ])
    events = []
    Agent(provider, verbose=False, event_callback=events.append).run("Fix f.py")
    assert states and states[0][0] == "file"
    assert path.read_bytes() == external
    assert not path.with_name("f.py.bak").exists()
    assert "run_code" in provider.seen[1][1]
    assert not any(event["type"] == "tool_gated" for event in events)
    assert refused_classes == [RefusalClass.STALE_STATE]


@pytest.mark.parametrize("persistent", [False, True])
def test_readonly_semantic_claim_uses_only_failed_executor_facts_after_resume(tmp_path, persistent):
    path = tmp_path / "calculator.py"
    path.write_text("def add(a, b):\n    return a + b\n")
    tools.execute_read_file("calculator.py")
    memory = (SQLiteMemory(tmp_path / "private" / "audit.db", session_id="facts")
              if persistent else ConversationMemory())
    first = Agent(Transcript([
        ("patch_file", {"path": "calculator.py", "target": "absent", "replacement": "def subtract(a,b): return a-b"}),
        "No changes were made.",
    ]), memory=memory, verbose=False)
    first.run("Edit calculator.py")
    audit = copy.deepcopy(memory.get())
    if persistent:
        memory = SQLiteMemory(tmp_path / "private" / "audit.db", session_id="facts")
    text = "The subtract function has been successfully added to calculator.py."
    events = []
    second = Agent(Transcript([text]), memory=memory, verbose=False, event_callback=events.append)
    result = second.run("What does calculator.py contain?")
    assert result.startswith(text)
    assert "Executor record:" in result
    assert "failed edit segment(s)" in result and "none recorded" in result
    assert "absent" not in result and "claim above" not in result and "false" not in result
    assert memory.get()[:len(audit)] == audit
    assert memory.get()[-1]["content"] == text
    corrected = [event for event in events if event["type"] == "claim_corrected"]
    assert len(corrected) == 1 and os.path.normcase(os.path.abspath(corrected[0]["file"])) == os.path.normcase(os.path.abspath(path))
    assert "text" not in corrected[0]


def test_truthful_post_create_and_unknown_readonly_claims_are_untouched(tmp_path):
    first = Transcript([("write_file", {"path": "calculator.py", "content": "x = 1\n"}), "I created calculator.py."])
    memory = ConversationMemory()
    Agent(first, memory=memory, verbose=False).run("Create calculator.py")
    for text in ("Here is your updated calculator.py", "calculator.py contains add()", "Here is your updated unknown.py"):
        events = []
        result = Agent(Transcript([text]), memory=memory, tools=[], verbose=False,
                       event_callback=events.append).run("Describe the prior session")
        assert result == text
        assert not any(event["type"] == "claim_corrected" for event in events)


def test_session_record_actual_fit_audits_but_projection_is_pure(tmp_path):
    messages = [
        {"role": "user", "content": "Review tests"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "edit", "type": "function", "function": {
                "name": "edit_function", "arguments": '{"path":"old.py","function_name":"f","new_body":"def f(): return 1"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "edit", "content": "old output " * 2000},
        {"role": "assistant", "content": "Earlier discussion"},
        {"role": "user", "content": "Intervening request"},
        {"role": "assistant", "content": "Intervening answer"},
        {"role": "user", "content": "Exact current request\n café  "},
    ]
    memory = SQLiteMemory(tmp_path / "history.db", session_id="actual-fit", max_messages=100)
    memory.replace(messages)
    audit = copy.deepcopy(memory.get())
    provider = Transcript([])
    provider.context_size = 1000
    provider.max_output_tokens = 128
    events = []
    agent = Agent(provider, memory=memory, tools=[], verbose=False, event_callback=events.append)
    agent.system = "s"
    projection = agent.context_breakdown()
    assert memory.get_session_metadata("SESSION_RECORD") is None and events == []
    fitted = agent._fit(memory.get(), [])
    retained = memory.get_session_metadata("SESSION_RECORD")
    assert retained and retained["files"] == ["old.py"]
    assert "Retained file paths: old.py" in fitted[0]["content"]
    assert fitted[-1]["content"].encode() == messages[-1]["content"].encode()
    assert context.history_tokens(fitted[:2]) <= 150
    assert projection["projection"] == context.estimate_tokens("s") + context.history_tokens(fitted) + 128
    injected = [event for event in events if event["type"] == "session_record_injected"]
    assert len(injected) == 1 and injected[0]["fields_included"]["files"] == 1
    assert memory.get() == audit
    before = memory.get_session_metadata("SESSION_RECORD")
    agent.context_breakdown()
    assert memory.get_session_metadata("SESSION_RECORD") == before
    assert len(events) == 1


def test_private_executor_ledger_does_not_displace_provider_input_budget(tmp_path):
    from nexus_agent_ai.agent.outcomes import mutation_metadata
    record = {
        "event": "e1", "turn": "t1", "segment": "s1", "path": str(tmp_path / "old.py"),
        "action": "write_file", "status": "changed", "success": True, "failed": False,
    }
    current = {"role": "user", "content": "Current request\n café  "}
    original = [
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier answer",
         "metadata": {"turn": "t1", "segment": "s1", **mutation_metadata([record])}},
        current,
    ]
    audit = copy.deepcopy(original)
    wire = [{"role": m["role"], "content": m["content"]} for m in original]
    budget = context.history_tokens(wire)
    fitted, stats = context.fit_messages(original, budget)
    assert fitted == wire
    assert fitted[-1] == current
    assert stats["tokens"] == budget and stats["dropped_turns"] == 0
    assert stats["session_record"] is None
    assert original == audit
