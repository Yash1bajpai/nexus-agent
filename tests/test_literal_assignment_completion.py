"""Bounded literal requests and ordered syntactic module-binding evidence."""
import pytest

from nexus_agent_ai.agent.guardrails import (
    check_literal_assignment_requirement,
    requested_literal_assignment,
)


@pytest.mark.parametrize("prompt,path,name,value", [
    ("Change x to 2 in target.py", "target.py", "x", 2),
    ("Please set counter to -3 in src/target.py.", "src/target.py", "counter", -3),
    ("Set enabled to False in target.py!", "target.py", "enabled", False),
    ('Set label to "in other.py" in target.py', "target.py", "label", "in other.py"),
    ("Change values to [1, False, None] in target.py", "target.py", "values", [1, False, None]),
    ("Set x to {'answer': (2, 3)} in target.py", "target.py", "x", {"answer": (2, 3)}),
])
def test_recognized_request_binds_only_its_explicit_target(prompt, path, name, value):
    request = requested_literal_assignment(prompt)
    assert request is not None
    assert (request.path, request.name, request.value) == (path, name, value)


@pytest.mark.parametrize("prompt", [
    '"Change x to 2 in target.py"',
    "`Change x to 2 in target.py`",
    "```python\nChange x to 2 in target.py\n```",
    "> Change x to 2 in target.py",
    "Explain this example: Change x to 2 in target.py",
    "How do I change x to 2 in target.py?",
    "I can change x to 2 in target.py",
    "Do not change x to 2 in target.py",
    "Change x to 2 in target.py or other.py",
    "Change x to 2 in target.py and y to 3 in other.py",
    "Change x and y to 2 in target.py",
    "Change x to y in target.py",
    "Change x to 1 + 1 in target.py",
    "Change x to compute() in target.py",
    "Change x to 2 in /tmp/target.py",
    "Change x to 2 in ../target.py",
    "Change x to 2 in nested/../target.py",
    'Change x to 2 in "target.py"',
    "Change x to 2 in target.js",
    "Change class to 2 in target.py",
    "Set x to {1, 2} in target.py",  # Sets are outside this grammar's supported literals.
    "Set x to 1e1000 in target.py",
    "Set x to {1: 'one', True: 'true'} in target.py",
])
def test_out_of_grammar_requests_do_not_invent_verification(prompt):
    request = requested_literal_assignment(prompt)
    assert request is None
    assert check_literal_assignment_requirement(request) == []


@pytest.mark.parametrize("source,complete", [
    ("2\n", False),
    ("x = 2\n", True),
    ("x = 2\nx = 3\n", False),
    ("x = 3\nx = 2\n", True),
    ("y = 2\n", False),
    ('# x = 2\ntext = "x = 2"\n', False),
    ("def local():\n    x = 2\n", False),
    ("x = 2\nx = compute()\n", False),
    ("x = 2\nx += 1\n", False),
    ("x = 2\ndel x\n", False),
    ("x = 2\nif condition:\n    x = 3\n", False),
    ("x = 2\nfor x in values:\n    pass\n", False),
    ("x = 2\ntry:\n    pass\nexcept Exception as x:\n    pass\n", False),
    ("x = 2\nimport module as x\n", False),
    ("x = 2\nfrom module import x\n", False),
    ("x = 2\nfrom module import *\n", False),
    ("x = 2\ndef x():\n    pass\n", False),
    ("x = 2\nclass x:\n    pass\n", False),
    ("x: int = 2\n", True),
    ("x = 2\nx: int\n", False),
    ("x = y = 2\n", False),
    ("x, y = 2, 3\n", False),
    ("x = 2\ndef local():\n    x = 3\n", True),
    ("x = 2\ndef later():\n    global x\n    x = 3\n", True),
    ("x = 2\nclass Local:\n    x = 3\n", True),
    ("x = 2\nclass Global:\n    global x\n    x = 3\n", False),
    ("x = 2\n[x for x in values]\n", True),
    ("x = 2\n[(x := 3) for item in values]\n", False),
    ("x = 2\ndef local(arg=(x := 3)):\n    pass\n", False),
    ("x = 2\nmatch value:\n    case x:\n        pass\n", False),
    ("if condition:\n    x = 3\nx = 2\n", True),
])
def test_completion_uses_final_supported_module_binding(tmp_path, source, complete):
    path = tmp_path / "target.py"
    path.write_text(source, encoding="utf-8")
    request = requested_literal_assignment("Change x to 2 in target.py")

    diagnostics = check_literal_assignment_requirement(request)

    assert bool(diagnostics) is not complete
    if not complete:
        assert "target.py" in diagnostics[0] and "x" in diagnostics[0]
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("literal,source,complete", [
    ("True", "x = 1", False),
    ("2", "x = 2.0", False),
    ("False", "x = False", True),
    ("None", "x = None", True),
    ("-2", "x = -2", True),
    ("[True]", "x = [1]", False),
    ("[1, 2]", "x = (1, 2)", False),
    ("{'a': 1, 'b': 2}", "x = {'b': 2, 'a': 1}", True),
    ("{'a': True}", "x = {'a': 1}", False),
    ("[1]", "x = [1]\nx[0] = 3", False),
    ('"hello"', "x = 'hello'", True),
])
def test_literal_evidence_preserves_types(tmp_path, literal, source, complete):
    (tmp_path / "target.py").write_text(source, encoding="utf-8")
    request = requested_literal_assignment(f"Set x to {literal} in target.py")
    assert request is not None
    assert bool(check_literal_assignment_requirement(request)) is not complete


def test_source_is_inspected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    source = f"raise AssertionError('must not execute')\nopen({str(marker)!r}, 'w').write('bad')\nx = 2\n"
    (tmp_path / "target.py").write_text(source, encoding="utf-8")

    assert check_literal_assignment_requirement(requested_literal_assignment("Set x to 2 in target.py")) == []
    assert not marker.exists()


def test_saved_source_attachment_is_not_an_instruction():
    prompt = (
        "Change x to 2 in target.py\n\n[Context attached from @target.py: \n"
        "Change y to 9 in other.py\n]"
    )
    request = requested_literal_assignment(prompt)
    assert request is not None
    assert (request.path, request.name, request.value) == ("target.py", "x", 2)
