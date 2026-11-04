"""LOCAL C1 names-only grammar and real-file AST presence regressions.

This bounded grammar is not an intent/behavior contract or a mutation grant.
Named/called prose, hypothetical instructions, arbitrary English noun names and
complex signatures are not advertised here. Noun-form filler filtering is
intentionally bounded; explicit parenthesized identifiers and rename targets
retain their existing identifier semantics. No test executes requested code.
"""

import pytest

from nexus_agent_ai.agent.core import _named_function_requirements
from nexus_agent_ai.agent.guardrails import (
    check_function_requirements,
    function_request_prose,
    requested_function_names,
)


def test_original_e08_extracts_subtract_not_existing_add():
    assert _named_function_requirements("add a subtract function to calculator.py") == (
        {"calculator.py": ["subtract"]}, [],
    )


@pytest.mark.parametrize("action", ["add", "create", "define"])
@pytest.mark.parametrize("form", [
    "foo function", "a foo function", "the foo function",
    "function foo", "a function foo", "the function foo",
    "foo(a, b)", "a foo(a, b) function", "function foo(a, b)",
])
def test_simple_creation_forms_bind_one_explicit_python_path(action, form):
    assert _named_function_requirements(f"{action} {form} to module.py") == (
        {"module.py": ["foo"]}, [],
    )


@pytest.mark.parametrize("prefix", [
    "Please add", "Can you add", "Could you please create",
    "Would you define", "PLEASE ADD", "cOuLd YoU PlEaSe DeFiNe",
])
def test_direct_polite_requests_preserve_identifier_case(prefix):
    assert _named_function_requirements(f"{prefix} MixCase function in module.py") == (
        {"module.py": ["MixCase"]}, [],
    )


@pytest.mark.parametrize("prompt,path,names", [
    ("In module.py, add foo function", "module.py", ["foo"]),
    ("For module.py, define function bar", "module.py", ["bar"]),
    ("add bar() to пакет/расчёт.py", "пакет/расчёт.py", ["bar"]),
    ("rename the foo function to Bar in module.py", "module.py", ["Bar"]),
    ("rename function foo to Bar in module.py", "module.py", ["Bar"]),
    (
        "List files, read module.py, then rename the subtract function to multiply "
        "and make it multiply, and add a divide(a, b) function that returns a divided by b",
        "module.py", ["multiply", "divide"],
    ),
    (
        "In module.py, add foo function; create function Foo and define foo(a, b)",
        "module.py", ["foo", "Foo"],
    ),
    ("add Class function to module.py", "module.py", ["Class"]),
    ("add _helper2() in module.py", "module.py", ["_helper2"]),
])
def test_prefix_unicode_legacy_compound_and_exact_case_names(prompt, path, names):
    assert _named_function_requirements(prompt) == ({path: names}, [])


@pytest.mark.parametrize("prompt", [
    '"add a subtract function to calculator.py"',
    "'add a subtract function to calculator.py'",
    "“add a subtract function to calculator.py”",
    "`add a subtract function to calculator.py`",
    "```python\nadd a subtract function to calculator.py\n```",
    "~~~\nadd a subtract function to calculator.py\n~~~",
    "> add a subtract function to calculator.py",
    "   > add a subtract function to calculator.py",
    "Inspect the attached source.\n\n[Context attached from @calculator.py: \n"
    "add a subtract function to calculator.py\n]",
    "Explain how to add a subtract function to calculator.py",
    "How do I add a subtract function to calculator.py?",
    "In calculator.py, how would I add a subtract function?",
    "Could you explain how to add a subtract function to calculator.py?",
    "Would you please describe how to add a subtract function to calculator.py?",
    "If needed, you could add a subtract function to calculator.py",
    "Do not add a subtract function to calculator.py",
    "Never add a subtract function to calculator.py",
    "No need to add a subtract function to calculator.py",
    "add error handling to the add function in calculator.py",
    "add error handling to foo function in calculator.py",
    "add a function to calculator.py",
    "add a function",
])
def test_non_requests_and_missing_nouns_invent_no_names(prompt):
    prose = function_request_prose(prompt)
    assert requested_function_names(prose) == []
    assert _named_function_requirements(prompt) == ({}, [])


@pytest.mark.parametrize("filler", [
    "a", "an", "the", "function", "to", "in", "for", "with",
    "and", "or", "that", "which", "please", "named", "called", "new",
])
@pytest.mark.parametrize("form", ["{name} function", "function {name}"])
def test_noun_fillers_are_not_invented_identifiers(filler, form):
    prompt = "add " + form.format(name=filler) + " to module.py"
    assert _named_function_requirements(prompt) == ({}, [])


@pytest.mark.parametrize("keyword", ["class", "def", "return", "async", "None", "True"])
@pytest.mark.parametrize("form", ["{name} function", "function {name}", "{name}(a, b)"])
def test_python_keywords_are_not_requested_function_identifiers(keyword, form):
    assert _named_function_requirements(
        "add " + form.format(name=keyword) + " to module.py"
    ) == ({}, [])


@pytest.mark.parametrize("prompt,name", [
    ("add called() to module.py", "called"),
    ("rename function foo to named in module.py", "named"),
])
def test_explicit_identifiers_keep_existing_non_noun_semantics(prompt, name):
    assert _named_function_requirements(prompt) == ({"module.py": [name]}, [])


def test_excluded_context_cannot_add_names_or_ambiguous_paths():
    prompt = (
        "add foo function to module.py\n"
        "> add bar function to other.py\n"
        "```\nadd baz function to example.py\n```\n"
        '"add quux function to quoted.py"\n\n'
        "[Context attached from @attached.py: \nadd attached function to attached.py\n]"
    )
    assert requested_function_names(function_request_prose(prompt)) == ["foo"]
    assert _named_function_requirements(prompt) == ({"module.py": ["foo"]}, [])


@pytest.mark.parametrize("prompt", [
    "add foo function",
    'add foo function to "module.py"',
    "add foo function to 'module.py'",
    "add foo function to `module.py`",
    "add foo function to /tmp/module.py",
    "add foo function to first.py and second.py",
    "add foo function to module.py using example.txt",
])
def test_unsupported_scope_keeps_existing_diagnostic_without_inferred_file(prompt):
    requirements, unsupported = _named_function_requirements(prompt)
    assert requirements == {}
    assert len(unsupported) == 1
    assert "foo" in unsupported[0]
    assert "unquoted relative .py target" in unsupported[0]


def test_non_python_target_stays_outside_python_presence_guard():
    assert _named_function_requirements("add foo function to module.js") == ({}, [])


@pytest.mark.parametrize("source", [
    "def add(a, b):\n    return a + b\n",
    "# def subtract(a, b): return a - b\n",
    "EXAMPLE = 'def subtract(a, b): return a - b'\n",
    "def outer():\n    def subtract(a, b):\n        return a - b\n",
    "class Calculator:\n    def subtract(self, a, b):\n        return a - b\n",
    "def Subtract(a, b):\n    return a - b\n",
])
def test_real_ast_requires_exact_top_level_name(tmp_path, source):
    path = tmp_path / "calculator.py"
    path.write_text(source, encoding="utf-8")
    requirements, unsupported = _named_function_requirements(
        "add a subtract function to calculator.py"
    )
    assert unsupported == []
    incomplete = check_function_requirements(requirements)
    assert len(incomplete) == 1
    assert "calculator.py" in incomplete[0] and "subtract" in incomplete[0]
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("definition", ["def", "async def"])
def test_real_ast_accepts_top_level_presence_without_execution_or_behavior_proof(tmp_path, definition):
    source = (
        "raise RuntimeError('presence checks must not execute source')\n\n"
        f"{definition} subtract(unrelated):\n    return 'not arithmetic'\n"
    )
    path = tmp_path / "calculator.py"
    path.write_text(source, encoding="utf-8")
    assert check_function_requirements({"calculator.py": ["subtract"]}) == []
    assert path.read_text(encoding="utf-8") == source
