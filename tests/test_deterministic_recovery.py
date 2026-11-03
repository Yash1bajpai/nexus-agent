import copy
import json

import pytest

from nexus_agent_ai.agent import context, core, tools
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory, is_genuine_user_message
from nexus_agent_ai.agent.outcomes import fitted_claim_view, mutation_metadata
from nexus_agent_ai.agent.persistence import SQLiteMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall
from nexus_agent_ai.providers.messages import anthropic_messages, gemini_messages, openai_messages


class Scripted(BaseProvider):
    model = "recovery-transcript"
    is_local = False

    def __init__(self, steps, before_request=None):
        self.steps = list(steps)
        self.seen = []
        self.hook = before_request

    def complete(self, messages, tools, system):
        if self.hook:
            self.hook(len(self.seen))
        self.seen.append(copy.deepcopy(messages))
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
def no_implicit_attachment(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")


def outcome_metadata(path_statuses, *, turn="t1", segment="s1"):
    events = [
        {"event": f"{turn}-{segment}-{index}", "turn": turn, "segment": segment,
         "path": path, "action": "patch_file", "status": status,
         "success": status == "changed", "failed": status == "unchanged"}
        for index, (path, status) in enumerate(path_statuses.items())
    ]
    return {"turn": turn, "segment": segment, **mutation_metadata(events)}

def bad_patch(path="b.py"):
    return "patch_file", {"path": path, "target": "nonexistent source", "replacement": "x = 2"}


def read_b(tmp_path):
    path = tmp_path / "b.py"
    path.write_text("x = 1\n", encoding="utf-8")
    tools.execute_read_file("b.py")
    return path


def test_second_identical_failure_ends_turn_without_another_model_request(tmp_path):
    path = read_b(tmp_path)
    provider = Scripted([bad_patch(), bad_patch(), "I updated b.py."])
    events = []
    agent = Agent(provider, verbose=False, event_callback=events.append)
    final = agent.run("Fix b.py")
    assert len(provider.seen) == 2 and provider.steps == ["I updated b.py."]
    assert "Stopped: repeated identical tool failure" in final
    assert "target-mismatch" in final and "no observed target-file state change" in final
    assert path.read_bytes() == b"x = 1\n"
    assert len([event for event in events if event["type"] == "observe"]) == 2
    assert agent.unresolved_errors


def test_successful_actual_call_resets_failure_evidence(tmp_path):
    read_b(tmp_path)
    (tmp_path / "other.txt").write_text("other")
    failed_write = ("write_file", {"path": "b.py", "content": "def broken(:"})
    provider = Scripted([failed_write, ("read_file", {"path": "other.txt"}), failed_write, failed_write, "unused"])
    final = Agent(provider, verbose=False).run("Fix b.py")
    assert len(provider.seen) == 4
    assert "repeated identical tool failure" in final


def test_changed_actual_file_hash_does_not_stop_on_second_different_state(tmp_path, monkeypatch):
    path = read_b(tmp_path)
    monkeypatch.setattr(core, "execute_tool", lambda *args: "ERROR: Target content not found in 'b.py'.")

    def change_state(index):
        if index == 1:
            path.write_text("x = 10\n", encoding="utf-8")

    provider = Scripted([bad_patch(), bad_patch(), bad_patch(), "unused"], change_state)
    final = Agent(provider, verbose=False).run("Fix b.py")
    assert len(provider.seen) == 3
    assert "repeated identical tool failure" in final
    assert path.read_bytes() == b"x = 10\n"


def test_later_independent_user_request_is_not_permanently_blocked(tmp_path):
    read_b(tmp_path)
    provider = Scripted([bad_patch(), bad_patch(), bad_patch(), "No changes were made."])
    agent = Agent(provider, verbose=False)
    assert "repeated identical tool failure" in agent.run("Fix b.py")
    final = agent.run("Fix b.py")
    assert len(provider.seen) == 4
    assert "Stopped: repeated identical tool failure" not in final
    assert "No changes were made" in final


def test_blind_refusal_and_target_mismatch_are_different_failures(tmp_path):
    (tmp_path / "b.py").write_text("x = 1\n")
    provider = Scripted([bad_patch(), bad_patch(), bad_patch(), "unused"])
    final = Agent(provider, verbose=False).run("Fix b.py")
    assert len(provider.seen) == 2
    assert "DATA: RAW file content" in provider.seen[1][-1]["content"]
    assert "repeated identical tool failure" not in final
    assert "targeting" in final
    assert len(provider.steps) == 2


def test_mixed_write_repeat_stop_never_claims_no_files_changed(tmp_path):
    read_b(tmp_path)
    provider = Scripted([
        ("write_file", {"path": "a.py", "content": "a = 1\n"}),
        bad_patch(), bad_patch(), "unused",
    ])
    agent = Agent(provider, verbose=False)
    final = agent.run("Update a.py and b.py")
    assert (tmp_path / "a.py").read_bytes() == b"a = 1\n"
    assert agent.changed_files == {str(tmp_path / "a.py")}
    assert "Other observed file mutations" in final
    assert "no file was changed" not in final.lower()
    assert not agent.edit_not_applied


def test_repeat_stop_pairs_unexecuted_batch_calls_without_running_them(tmp_path):
    read_b(tmp_path)
    name, args = bad_patch()
    batch = ProviderResponse(tool_calls=[
        ToolCall("first", name, args), ToolCall("second", name, args),
        ToolCall("not-run", "write_file", {"path": "should_not_exist.py", "content": "x = 1\n"}),
    ])
    agent = Agent(Scripted([batch]), verbose=False)
    final = agent.run("Fix b.py")
    messages = agent.memory.get()
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert set(results) == {"first", "second", "not-run"}
    assert results["not-run"].startswith("ERROR: Not executed:")
    assert not (tmp_path / "should_not_exist.py").exists()
    assert "repeated identical tool failure" in final


@pytest.mark.parametrize("persistent", [False, True])
def test_legacy_mixed_outcomes_redact_only_false_claims_in_next_fitted_view(tmp_path, persistent):
    read_b(tmp_path)
    raw = "I updated a.py.\r\nI updated b.py.\nI updated unknown.py."
    provider = Scripted([
        ("write_file", {"path": "a.py", "content": "a = 1\n"}), bad_patch(), raw,
        "Please restate missing details.",
    ])
    memory = (SQLiteMemory(tmp_path / "private" / "audit.db", session_id="claims")
              if persistent else ConversationMemory())
    agent = Agent(provider, memory=memory, verbose=False)
    agent.run("Update a.py and b.py")
    # Pre-follow-up records retain raw prose: no migration is performed. Exercise
    # their existing fitted-view redaction separately from corrected new stops.
    legacy = memory.get()
    legacy[-1]["content"] = raw
    memory.replace(legacy)
    audit = copy.deepcopy(memory.get())
    original = audit[-1]
    assert original["content"].encode() == raw.encode()
    statuses = {item["path"]: item["status"] for item in original["metadata"]["mutation_outcomes"]}
    assert statuses == {str(tmp_path / "b.py"): "unchanged"}

    agent.run("What happened?")
    fitted = provider.seen[-1]
    assistant = next(m for m in fitted if m["role"] == "assistant" and "unknown.py" in m.get("content", ""))
    assert "I updated a.py." in assistant["content"]
    assert "I updated b.py." not in assistant["content"]
    assert "I updated unknown.py." in assistant["content"]
    corrections = [m for m in fitted if m.get("metadata", {}).get("outcome_correction")]
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction["role"] == "user" and not is_genuine_user_message(correction)
    assert "no executor write to" in correction["content"] and str(tmp_path / "b.py") in correction["content"]
    assert str(tmp_path / "a.py") not in correction["content"]
    assert "unknown.py" not in correction["content"]
    assert "not a new request or approval" in correction["content"]
    assert memory.get()[:len(audit)] == audit


@pytest.mark.parametrize("claim", [
    "I updated a.py and b.py.",
    "I updated a.py and I fixed b.py.",
    "I updated `a.py`. I updated `b.py`.",
])
def test_shared_or_separate_predicates_keep_successful_file_claim(tmp_path, claim):
    metadata = outcome_metadata({str(tmp_path / "a.py"): "changed", str(tmp_path / "b.py"): "unchanged"})
    original = [{"role": "user", "content": "Update both files"},
                {"role": "assistant", "content": claim, "metadata": metadata}]
    before = copy.deepcopy(original)
    fitted = fitted_claim_view(original)
    assert "a.py" in fitted[1]["content"]
    assert "b.py" not in fitted[1]["content"]
    assert fitted[-1]["metadata"]["outcome_correction"]
    assert original == before


@pytest.mark.parametrize("claim", [
    "I will update b.py.", "You can update b.py.", "I did not update b.py.",
    "For example: I updated b.py.", '"I updated b.py."',
    "```python\n# I updated b.py.\n```", "I updated unknown.py.",
    "I updated a.py using b.py.",
])
def test_unknown_advice_examples_and_negations_are_not_invented_contradictions(tmp_path, claim):
    messages = [{"role": "user", "content": "I updated b.py."}, {
        "role": "assistant", "content": claim,
        "metadata": outcome_metadata({str(tmp_path / "b.py"): "unchanged"}),
    }]
    assert fitted_claim_view(messages) == messages


def test_generic_correction_without_verified_outcomes_cannot_redact():
    messages = [{"role": "user", "content": "Create b.py"}, {
        "role": "assistant", "content": "I created b.py.\n\nWarning: no file was changed.",
    }]
    assert fitted_claim_view(messages) == messages


def paired_claim_history(tmp_path):
    return [
        {"role": "user", "content": "Inspect both files"},
        {"role": "assistant", "content": "I updated b.py.", "metadata": {"turn": "t1", "segment": "s1"}, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
            {"id": "b", "type": "function", "function": {"name": "patch_file", "arguments": '{"path":"b.py","target":"x","replacement":"y"}'}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "observed source"},
        {"role": "tool", "tool_call_id": "b", "content": "ERROR: Target content not found", "metadata": outcome_metadata({str(tmp_path / "b.py"): "unchanged"})},
        {"role": "assistant", "content": "End of turn."},
    ]


def test_redaction_correction_keeps_tool_pairing_and_provider_roles(tmp_path):
    messages = paired_claim_history(tmp_path)
    audit = copy.deepcopy(messages)
    fitted, stats = context.fit_messages(messages, 2000)
    assert [m["role"] for m in fitted] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert "I updated b.py" not in fitted[1]["content"]
    assert stats["correction_tokens"] == context.message_tokens(fitted[-1]) > 0
    assert stats["tokens"] == context.history_tokens(fitted) <= 2000
    assert stats["turns_before"] == stats["turns_after"] == 1
    assert messages == audit
    translated = openai_messages(fitted)
    assert [m["tool_call_id"] for m in translated if m["role"] == "tool"] == ["a", "b"]
    assert all("metadata" not in m and m["role"] != "system" for m in translated)
    anthropic = anthropic_messages(fitted)
    assert anthropic[2]["content"][0]["tool_use_id"] == "a"
    assert anthropic[2]["content"][1]["tool_use_id"] == "b"
    gemini = gemini_messages(fitted)
    responses = [p["function_response"] for m in gemini for p in m["parts"] if "function_response" in p]
    assert [(r["id"], r["name"]) for r in responses] == [("a", "read_file"), ("b", "patch_file")]
    with pytest.raises(context.ContextBudgetError):
        context.fit_messages(messages, stats["tokens"] - 1)
    assert messages == audit


def test_omission_and_projection_include_correction_cost_without_audit_changes(tmp_path):
    messages = paired_claim_history(tmp_path)
    current = {"role": "user", "content": "Current request must remain exact."}
    messages.append(current)
    memory = ConversationMemory()
    memory.replace(messages)
    messages = memory.get()  # Use canonical history, not the pre-normalized fixture.
    before = copy.deepcopy(messages)
    provider = Scripted([])
    provider.context_size = 2000
    provider.max_output_tokens = 128
    agent = Agent(provider, memory=memory, tools=[], verbose=False)
    agent.system = "s"
    breakdown = agent.context_breakdown()
    fitted, stats = context.fit_messages(messages, 2000 - 128 - context.estimate_tokens("s"))
    assert breakdown["projection"] == context.estimate_tokens("s") + context.history_tokens(fitted) + 128
    assert stats["correction_tokens"] > 0
    assert fitted[-1] == current
    assert memory.get() == before and messages == before

    budget = context.history_tokens([current]) + 150
    with pytest.raises(context.ContextBudgetError):
        context.fit_messages(messages, budget)
    assert messages == before


def test_metadata_is_bounded_and_ambiguous_observations_remain_unknown(tmp_path):
    outcomes = {str(tmp_path / f"f{i}.py"): "unchanged" for i in range(50)}
    metadata = {"mutation_outcomes": outcome_metadata(outcomes)["mutation_outcomes"]}
    assert len(metadata["mutation_outcomes"]) <= 8
    assert len(json.dumps(metadata, ensure_ascii=False)) <= 2048
    unknown = outcome_metadata({str(tmp_path / "b.py"): "unknown"})
    messages = [{"role": "assistant", "content": "I updated b.py.", "metadata": unknown}]
    assert fitted_claim_view(messages) == messages


def test_success_string_and_unchanged_bytes_do_not_prove_typed_failed_edit(tmp_path, monkeypatch):
    read_b(tmp_path)
    monkeypatch.setattr(core, "execute_tool", lambda *args: "Successfully patched b.py")
    provider = Scripted([bad_patch(), "I updated b.py."])
    agent = Agent(provider, verbose=False)
    agent.run("Update b.py")
    raw = agent.memory.get()
    assert raw[-1]["content"] == "I updated b.py."
    assert raw[-1]["metadata"]["mutation_outcomes"][0]["status"] == "unchanged"
    fitted = agent._fit(raw, [])
    assert fitted[-1]["content"] == "I updated b.py."
    assert not any(m.get("metadata", {}).get("outcome_correction") for m in fitted)


def test_later_failed_segment_is_not_blanket_protected_by_old_mutation(tmp_path):
    provider = Scripted([
        ("write_file", {"path": "a.py", "content": "a = 1\n"}),
        bad_patch("a.py"), "I updated a.py.",
    ])
    agent = Agent(provider, verbose=False)
    agent.run("Update a.py")
    audit = agent.memory.get()
    records = audit[-1]["metadata"]["mutation_outcomes"]
    assert len(records) == 1 and records[0]["path"] == str(tmp_path / "a.py")
    assert records[0]["status"] == "unchanged" and records[0]["failed"]
    fitted = agent._fit(audit, [])
    assert "I updated a.py." not in fitted[-2]["content"]
    assert fitted[-1]["metadata"]["outcome_correction"]


def test_outcomes_from_a_later_turn_do_not_rewrite_earlier_evidence(tmp_path):
    path = str(tmp_path / "b.py")
    messages = [
        {"role": "user", "content": "First request"},
        {"role": "assistant", "content": "I updated b.py.",
         "metadata": outcome_metadata({path: "unchanged"})},
        {"role": "user", "content": "Independent later request"},
        {"role": "assistant", "content": "I updated b.py.",
         "metadata": outcome_metadata({path: "changed"}, turn="t2", segment="s2")},
    ]
    audit = copy.deepcopy(messages)
    fitted = fitted_claim_view(messages)
    assert "I updated b.py." not in fitted[1]["content"]
    assert fitted[2]["metadata"]["outcome_correction"]
    assert fitted[3:] == messages[2:]
    assert messages == audit
