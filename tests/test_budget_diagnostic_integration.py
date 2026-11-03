"""Real planner/receipt hooks replay only an explicitly isolated synthetic turn."""
import copy
import json
import math
from pathlib import Path

import pytest

from nexus_agent_ai.agent import budget_diagnostics as diagnostics, context, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.budget_replay import replay_snapshot
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.messages import assistant_message


BUDGET = 1573
RECORD = {"version": 1, "files": [], "topics": [], "tools": []}


class Scripted(BaseProvider):
    model = "offline-r11-fixture"
    context_size = 4096
    max_output_tokens = 512
    is_local = False

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def complete(self, messages, tools, system):
        self.seen.append(copy.deepcopy(messages))
        step = next(self.steps)
        return step() if callable(step) else step

    def stream(self, **kwargs):
        yield self.complete(**kwargs)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    scope = diagnostics.create_scope(tmp_path / "r11-isolated")
    monkeypatch.setenv("HOME", scope["home"])
    monkeypatch.setenv("USERPROFILE", scope["home"])
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(Path(scope["home"]) / ".nexus-agent"))
    monkeypatch.chdir(scope["workspace"])
    Path("nexus_cfg_v9.yaml").write_text("owner: zephyr-labs\nrevision: 9\nmode: quarantine\n")
    Path("evict_filler.txt").write_text(("Filler record alpha beta gamma delta epsilon zeta. " * 120 + "\n") * 6)
    return scope


def agent_for(scope, steps, offered, *, prior=False):
    memory = SQLiteMemory(Path(scope["db_path"]), session_id=diagnostics.SESSION_ID)
    if prior:
        memory.add_raw({"role": "user", "content": "earlier-fixture " * 230})
        memory.add_raw({"role": "assistant", "content": "Earlier synthetic observation."})
        memory.set_session_metadata("SESSION_RECORD", RECORD)
    provider = Scripted(steps)
    agent = Agent(provider, tools=offered, memory=memory, verbose=False)
    wanted = 4096 - 512 - BUDGET - context.tools_tokens(offered)
    agent.system = "s" * math.ceil((wanted - 1) * context.CHARS_PER_TOKEN)
    return agent


def snapshot(scope, filename):
    return json.loads((Path(scope["output"]) / "seed" / filename).read_text(encoding="utf-8"))


def fills_empty_receipt_margin(agent, call):
    """A bounded model response; no extra planner/model/tool calls to tune it."""
    canonical = agent.memory.get()
    pair_tokens = context.history_tokens(context._omission_pair())
    low, high = 0, 512 * 3
    while low < high:
        middle = (low + high + 1) // 2
        candidate = canonical + [assistant_message("m" * middle, [call]),
            {"role": "tool", "tool_call_id": call.id, "content": ""}]
        if context.history_tokens(candidate) + pair_tokens <= BUDGET - 1:
            low = middle
        else:
            high = middle - 1
    binding = assistant_message("m" * low, [call])
    assert context.message_tokens(binding) < 512
    return ProviderResponse(text="m" * low, tool_calls=[call])


@pytest.mark.parametrize("enabled", [False, True], ids=["ordinary-env-only", "explicit-scope"])
def test_real_success_capture_is_token_identical_without_extra_calls(isolated, monkeypatch, enabled):
    agent = agent_for(isolated, [ProviderResponse(text="Synthetic observation.")], [])
    monkeypatch.setenv("NEXUS_BUDGET_DIAGNOSTICS", isolated["scope_path"])
    monkeypatch.setattr(core, "execute_tool", lambda *args, **kwargs: pytest.fail("Unexpected tool execution"))
    if enabled:
        with diagnostics.activate_scope(Path(isolated["scope_path"]), "seed"):
            answer = agent.run(diagnostics.frozen_argv("seed")[-1])
    else:
        answer = agent.run(diagnostics.frozen_argv("seed")[-1])
    assert answer == "Synthetic observation."
    assert len(agent.provider.seen) == 1
    if not enabled:
        assert list(Path(isolated["output"]).iterdir()) == []
        return
    saved = snapshot(isolated, "latest_success.json")
    original = copy.deepcopy(agent.memory.get())
    assert saved["canonical_messages"] == saved["projected_messages"] == agent.provider.seen[0]
    report = replay_snapshot(saved)
    assert report["status"] == "reproduced" and report["token_identical"] is True
    assert report["hashes"]["canonical_input"] == diagnostics.stable_hash(saved["canonical_messages"])
    assert agent.memory.get() == original
    assert len(agent.provider.seen) == 1


def test_real_coverage_gap_links_preceding_fit_and_exact_appended_payload(isolated, monkeypatch):
    call = ToolCall("directory", "list_directory", {"path": "."})
    agent = agent_for(isolated, [lambda: fills_empty_receipt_margin(agent, call)],
                      [tools.LIST_DIRECTORY_TOOL], prior=True)
    executed = []
    execute = core.execute_tool

    def counted(name, args, **kwargs):
        executed.append(name)
        return execute(name, args, **kwargs)

    monkeypatch.setattr(core, "execute_tool", counted)
    with diagnostics.activate_scope(Path(isolated["scope_path"]), "seed"):
        with pytest.raises(context.ContextBudgetError) as error:
            agent.run(diagnostics.frozen_argv("seed")[-1])
    assert str(error.value).startswith("Historical omission coverage gap:")
    failed = snapshot(isolated, "failure.json")
    success = snapshot(isolated, "latest_success.json")
    assert failed["stage"] == "outbound_post_result_fit"
    assert failed["pending_calls"] == [{
        "name": "list_directory", "id": "directory", "state": "recorded", "executed": True,
    }]
    assert failed["result"]["error"]["text"] == str(error.value)
    assert failed["previous_success"]["id"] == success["id"]
    assert failed["transition"]["append_only"] is True
    appended = failed["transition"]["new_messages"]
    assistant = next(item for item in appended if item["role"] == "assistant")
    exact = failed["canonical_messages"][assistant["index"]]
    assert exact["tool_calls"][0]["id"] == call.id
    assert assistant["hash"] == diagnostics.stable_hash(exact)
    assert executed == ["list_directory"] and len(agent.provider.seen) == 1
    original = copy.deepcopy(agent.memory.get())
    monkeypatch.setattr(core, "execute_tool", lambda *args, **kwargs: pytest.fail("Replay executed a tool"))
    report = replay_snapshot(failed)
    assert report["status"] == "reproduced" and report["error_match"] is True
    assert report["classification"] == "irreducible"
    assert report["transition"]["sole_cause_established"] is False
    assert replay_snapshot(success)["token_identical"] is True
    assert agent.memory.get() == original
    assert len(agent.provider.seen) == 1


def test_real_receipt_floor_replays_after_normal_executor_closure(isolated):
    call = ToolCall("output", "run_code", {"code": "print('x' * 20000)"})
    agent = agent_for(isolated, [lambda: fills_empty_receipt_margin(agent, call)],
                      [tools.RUN_CODE_TOOL], prior=True)
    with diagnostics.activate_scope(Path(isolated["scope_path"]), "seed"):
        with pytest.raises(context.ContextBudgetError, match="truthful outcome and output-omission receipt"):
            agent.run(diagnostics.frozen_argv("seed")[-1])
    failed = snapshot(isolated, "failure.json")
    assert failed["kind"] == "receipt" and failed["real_turn_failure"] is True
    assert failed["receipt"]["current_call_id"] == call.id
    assert failed["receipt"]["executed"] is True
    assert failed["receipt"]["raw_result"].startswith("STDOUT:")
    original = copy.deepcopy(agent.memory.get())
    assert original[-1]["metadata"]["executor_closed"] is True
    report = replay_snapshot(failed)
    assert report["status"] == "reproduced" and report["error_match"] is True
    assert agent.memory.get() == original
    assert len(agent.provider.seen) == 1


def test_recovery_projection_is_captured_exactly_without_changing_audit(isolated):
    source = "def untouched():\n    return 1\n"
    Path("untouched.py").write_text(source)
    call = ToolCall("refused", "patch_file", {
        "path": "untouched.py", "target": "return 1", "replacement": "return 2"})
    agent = agent_for(isolated, [ProviderResponse(tool_calls=[call]),
        ProviderResponse(text="No corrected edit attempted.")], [tools.PATCH_FILE_TOOL])
    with diagnostics.activate_scope(Path(isolated["scope_path"]), "seed"):
        agent.run(diagnostics.frozen_argv("seed")[-1])
    saved = snapshot(isolated, "latest_success.json")
    assert saved["settings"]["recovery_injected"] is True
    assert saved["canonical_messages"] != saved["projected_messages"]
    canonical_tool = next(message for message in saved["canonical_messages"] if message["role"] == "tool")
    recorded = next(message for message in agent.memory.get() if message["role"] == "tool")
    assert recorded["content"] == canonical_tool["content"]
    assert Path("untouched.py").read_text() == source
    assert replay_snapshot(saved)["token_identical"] is True
    assert len(agent.provider.seen) == 2


def test_real_adaptive_attempt_errors_and_success_replay_exactly(isolated):
    agent = agent_for(isolated, [ProviderResponse(text="Synthetic observation.")], [])
    agent.provider.is_local = True
    agent.provider._supports_output_budget = True
    agent.memory.add_raw({"role": "user", "content": "fixture " * 125})
    agent.memory.add_raw({"role": "assistant", "content": "Earlier observation."})
    agent.memory.set_session_metadata("SESSION_RECORD", RECORD)
    wanted = 4096 - 512 - 500
    agent.system = "s" * math.ceil((wanted - 1) * context.CHARS_PER_TOKEN)
    with diagnostics.activate_scope(Path(isolated["scope_path"]), "seed"):
        assert agent.run(diagnostics.frozen_argv("seed")[-1]) == "Synthetic observation."
    saved = snapshot(isolated, "latest_success.json")
    assert [attempt["reserve"] for attempt in saved["attempts"]] == [512, 256]
    assert saved["attempts"][0]["error"]["class"] == "ContextBudgetError"
    assert saved["result"]["stats"]["reserve"] == 256
    report = replay_snapshot(saved)
    assert report["status"] == "reproduced" and report["attempts_match"] is True
    assert report["token_identical"] is True
    assert len(agent.provider.seen) == 1
