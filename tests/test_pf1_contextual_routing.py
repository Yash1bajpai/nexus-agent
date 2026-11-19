"""PF-1 regressions: contextual follow-up edits must keep tools in local mode.

Release-line defect: Smart Tool Routing stripped all tools from contextual
follow-up edit requests when the antecedent request started with
Create/Write/Generate/Build/Make, or when the follow-up did not match the
narrow "make it return" rewrite shape. Both gaps left use_tools=[] and
effective_max_iter=1, so the model's correct patch_file/write_file call was
rejected with "Tool 'patch_file' is not allowed in this session".
"""
import pytest
from nexus_agent_ai.agent import core
from nexus_agent_ai.agent.core import Agent
from tests.test_agent_outcomes import Scripted, tool

FIX_P1 = "Fix multiply in calc.py so it multiplies."
RETURN_FOLLOWUP = "Now make it return 0 if either input is None."
REWRITE_PREFIX = "Modify the previously requested code. "


class Recording(Scripted):
    def __init__(self, turns):
        super().__init__(turns, local=True)
        self.offered = []

    def complete(self, messages, tools, system):
        self.offered.append([t.name for t in tools])
        return super().complete(messages, tools, system)


def offered_on_followup(p1, p2):
    provider = Recording(["Done.", "I made the change."])
    agent = Agent(provider, tools=[tool("patch_file")], max_iterations=1, verbose=False)
    agent.run(p1)
    agent.run(p2)
    return provider.offered[-1]


@pytest.mark.parametrize("antecedent", [
    "Create scores.py that computes the average of a list.",
    "Write calc.py with a multiply function.",
    "Generate a parser in parser.py.",
    "Build the CLI entry point in cli.py.",
    "Make calc.py expose a multiply function.",
])
def test_creation_antecedents_enable_contextual_rewrite(antecedent):
    followup = "Now make it return 0 when the list is empty."
    routed = core._routing_request(followup, [{"role": "user", "content": antecedent}])
    assert routed == REWRITE_PREFIX + followup
    assert offered_on_followup(antecedent, followup) == ["patch_file"]


@pytest.mark.parametrize("followup", [
    "Now round the result to 2 decimals.",
    "Now handle the case where the list is empty.",
    "Also log the result.",
    "Use it to compute the total.",
])
def test_non_return_followup_shapes_enable_contextual_rewrite(followup):
    routed = core._routing_request(followup, [{"role": "user", "content": FIX_P1}])
    assert routed == REWRITE_PREFIX + followup
    assert offered_on_followup(FIX_P1, followup) == ["patch_file"]


def test_fix_antecedent_with_return_followup_still_rewrites():
    routed = core._routing_request(RETURN_FOLLOWUP, [{"role": "user", "content": FIX_P1}])
    assert routed == REWRITE_PREFIX + RETURN_FOLLOWUP
    assert offered_on_followup(FIX_P1, RETURN_FOLLOWUP) == ["patch_file"]


@pytest.mark.parametrize("antecedent", [
    "Create a cake recipe.",
    "Write an essay about gardening.",
    "Do not create scores.py.",
    "Never write calc.py.",
    "Please stop generating parser.py.",
    "Cancel making cli.py.",
])
def test_non_code_or_negated_creation_is_not_an_antecedent(antecedent):
    followup = "Now make it return 0 when the list is empty."
    assert core._routing_request(followup, [{"role": "user", "content": antecedent}]) == followup


@pytest.mark.parametrize("prompt", [
    "Now make it funny.",
    "What does it return?",
    "Explain how to round the result.",
    'Example: "Now round the result to 2 decimals."',
    "> Now round the result to 2 decimals.",
    "```\nNow round the result to 2 decimals.\n```",
])
def test_non_imperative_after_edit_remains_tool_free(prompt):
    routed = core._routing_request(prompt, [{"role": "user", "content": FIX_P1}])
    assert routed == prompt
    assert offered_on_followup(FIX_P1, prompt) == []
