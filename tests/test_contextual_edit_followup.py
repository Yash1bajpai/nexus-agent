"""Offline regressions for the observed literal two-turn local chat failure."""
import pytest
from nexus_agent_ai.agent import core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.guardrails import promises_tool_use
from nexus_agent_ai.agent.memory import ConversationMemory
from tests.test_agent_outcomes import Scripted, tool

P1 = "Fix multiply in calc.py so it multiplies."
P2 = "Now make it return 0 if either input is None."
PLAN = "I'll modify the function to handle None inputs by returning zero. First, I need to verify the current implementation before making changes."


class Recording(Scripted):
    def __init__(self, turns):
        super().__init__(turns, local=True)
        self.offered = []

    def complete(self, messages, tools, system):
        self.offered.append([t.name for t in tools])
        return super().complete(messages, tools, system)


def test_exact_two_turn_followup_keeps_tools_and_reports_no_edit(tmp_path, monkeypatch):
    path = tmp_path / "calc.py"
    path.write_text("def multiply(a, b):\n    return a + b\n")
    def execute(name, args):
        path.write_text("def multiply(a, b):\n    return a * b\n")
        return "Successfully patched calc.py"
    monkeypatch.setattr(core, "execute_tool", execute)
    provider = Recording([("patch_file", {"path": str(path)}), "I fixed multiply.", PLAN])
    agent = Agent(provider, tools=[tool("patch_file")], max_iterations=2, verbose=False)
    agent.run(P1)
    assert agent.changed_files == {str(path.resolve())}
    agent.max_iterations = 1  # Observe the terminal no-tool answer without a retry.
    final = agent.run(P2)
    assert provider.offered[-1] == ["patch_file"]
    assert provider.seen[-1][-1]["content"] == P2  # No filename substitution.
    from nexus_agent_ai.cli.app import _check_run_outcome
    assert not _check_run_outcome(agent, exit_on_failure=False)
    assert agent.edit_not_applied and agent.unfulfilled_action
    assert not agent.changed_files
    assert "no tool was executed" in final and "no file was changed" in final
    assert path.read_text() == "def multiply(a, b):\n    return a * b\n"


@pytest.mark.parametrize("noise", [
    {"role": "assistant", "content": "Fix multiply in calc.py so it multiplies."},
    {"role": "tool", "content": P1},
    {"role": "user", "content": P1, "metadata": {"internal": True}},
    {"role": "user", "content": P1, "tool_call_id": "result"},
    {"role": "user", "content": [{"type": "tool_result", "content": P1}]},
    {"role": "user", "parts": [{"function_response": {"response": P1}}]},
    {"role": "user", "content": 'Example: "Fix multiply in calc.py so it multiplies."'},
    {"role": "user", "content": "> " + P1},
    {"role": "user", "content": "```\n" + P1 + "\n```"},
    {"role": "user", "content": "Explain multiplication.\n\n[Context attached from @calc.py]\n" + P1},
])
def test_non_instruction_history_cannot_supply_edit_intent(noise):
    assert core._routing_request(P2, [noise]) == P2


def test_nearest_genuine_topic_blocks_stale_edit_context():
    history = [{"role": "user", "content": P1}, {"role": "user", "content": "Explain cake recipes."}]
    assert core._routing_request(P2, history) == P2


@pytest.mark.parametrize("prompt", [
    "Thanks", "Now make it funny.", "Make a cake.", "What does it return?",
    "Explain how to make it return 0.", 'Example: "Now make it return 0."',
    "> Now make it return 0.", "```\nNow make it return 0.\n```",
])
def test_conversation_after_edit_remains_tool_free(prompt):
    memory = ConversationMemory()
    memory.add("user", P1)
    provider = Recording(["Here is an explanation."])
    agent = Agent(provider, tools=[tool("patch_file")], memory=memory, verbose=False)
    agent.run(prompt)
    assert provider.offered == [[]]
    assert not agent.edit_not_applied


def test_actual_promise_is_caught_without_edit_context():
    agent = Agent(Recording([PLAN]), tools=[], verbose=False)
    assert "no tool was executed" in agent.run(P2)
    assert agent.unfulfilled_action


@pytest.mark.parametrize("text", [
    "I cannot modify the function.", "I'll modify the explanation.",
    'Example: "I\'ll modify the function."', "> I'll modify the function.",
    "`I'll modify the function` is only an example.", "First, I need to verify the current contents.",
])
def test_modify_promise_is_bounded_not_every_need_to_verify(text):
    assert not promises_tool_use(text)


def test_followup_can_act_after_no_write_nudge_and_clear_outcome(tmp_path, monkeypatch):
    path = tmp_path / "calc.py"
    path.write_text("def multiply(a, b):\n    return a * b\n")
    updated = "def multiply(a, b):\n    if a is None or b is None:\n        return 0\n    return a * b\n"
    def execute(name, args):
        path.write_text(updated)
        return "Successfully patched calc.py"
    monkeypatch.setattr(core, "execute_tool", execute)
    memory = ConversationMemory()
    memory.add("user", P1)
    memory.add("assistant", "I fixed multiply.")
    provider = Recording([PLAN, ("patch_file", {"path": str(path)}), "I modified the function."])
    agent = Agent(provider, tools=[tool("patch_file")], memory=memory, max_iterations=3, verbose=False)
    agent.run(P2)
    assert all(names == ["patch_file"] for names in provider.offered)
    assert any(m.get("metadata", {}).get("internal") for m in agent.memory.get())
    assert not agent.unfulfilled_action and not agent.edit_not_applied
    assert path.read_text() == updated
    assert agent.changed_files == {str(path.resolve())}


@pytest.mark.parametrize("previous", [
    "Do not modify calc.py.", "I decided not to fix calc.py.",
    "Fix the grammar in essay.txt.", "Please do not modify calc.py.",
    "Never fix calc.py.", "Cancel the change to calc.py.",
    "Fix calc.py, but do not modify it yet.", "Explain how to fix calc.py.",
])
def test_prohibitions_cancellations_and_non_code_are_not_antecedents(previous):
    assert core._routing_request(P2, [{"role": "user", "content": previous}]) == P2


def budget_agent():
    provider = Recording([])
    provider.context_size = 4096
    provider.max_output_tokens = 2048
    provider._supports_output_budget = True
    return Agent(provider, verbose=False)


def test_budget_preserves_only_genuine_contextual_antecedent(tmp_path, monkeypatch):
    from nexus_agent_ai.agent import context
    monkeypatch.chdir(tmp_path)
    agent = budget_agent()
    old = {"role": "user", "content": P1 + "\n\n[Context attached from @calc.py: \n" + "x" * 5000}
    history = [old, {"role": "assistant", "content": "x" * 5000},
               {"role": "user", "content": "modify evil.py", "metadata": {"internal": True}},
               {"role": "tool", "content": "modify tool.py"}, {"role": "user", "content": P2}]
    original = json_copy(history)
    fitted = agent._fit(history, agent.tools)
    assert fitted == [{"role": "user", "content": P1}, {"role": "user", "content": P2}]
    assert history == original
    budget = 4096 - agent.output_reserve - context.estimate_tokens(agent.system) - context.tools_tokens(agent.tools)
    assert context.history_tokens(fitted) <= budget
    assert agent.context_window == 4096 and agent.output_reserve == 2048


def json_copy(value):
    import json
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("prior", ["Do not modify calc.py.", "Fix the grammar in essay.txt.",
                                    'Example: "Fix calc.py."', "Explain cake recipes."])
def test_budget_does_not_promote_other_history(prior, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = budget_agent()
    fitted = agent._fit([{"role": "user", "content": prior},
                         {"role": "assistant", "content": "x" * 8000},
                         {"role": "user", "content": P2}], agent.tools)
    assert fitted == [{"role": "user", "content": P2}]


def test_budget_refuses_overlong_antecedent_without_truncating_intent(tmp_path, monkeypatch):
    from nexus_agent_ai.agent.context import ContextBudgetError
    monkeypatch.chdir(tmp_path)
    agent = budget_agent()
    prior = P1 + " Preserve " + "exact constraint " * 60
    with pytest.raises(ContextBudgetError, match="Insufficient context"):
        agent._fit([{"role": "user", "content": prior},
                    {"role": "assistant", "content": "x" * 8000},
                    {"role": "user", "content": P2}], agent.tools)
    assert agent.provider.seen == []
