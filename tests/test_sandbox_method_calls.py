"""Regression tests: sandboxed run_code must allow pure-data method calls.

The AST gate previously rejected ALL ast.Attribute nodes and required every
callee to be a bare name from _SAFE_CALLS, so even "hello".split() was blocked
— run_code could not execute real Python. Attributes are now allowed when the
attribute name passes _attr_name_is_safe (no blocked dunders / forbidden
reflection attrs), and calls may target safe methods or user-defined functions
(their bodies are validated by the same walk). All escape vectors stay blocked.
"""
import pytest

from nexus_agent_ai.agent.tools import _sandbox_check, execute_run_code


SAFE_SNIPPETS = [
    'x = "hello world".split()',
    "items = []\nitems.append(1)\nitems.append(2)\nprint(items)",
    'd = {"a": 1}\nprint(d.get("a"))',
    's = "  Hi There  "\nprint(s.strip().lower())',
    'parts = ["a", "b"]\nprint("-".join(parts))',
    "nums = [3, 1, 2]\nnums.sort()\nprint(nums)",
    'words = ["apple", "fig", "banana"]\nprint(max(words, key=len))',
    'cleaned = [w.strip() for w in "a, b ,c ".split(",")]\nprint(cleaned)',
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib(10))",
    "def add(x, y):\n    return x + y\nprint(add(2, 3))",
]

BLOCKED_SNIPPETS = [
    # Dunder graph-walk escapes (each hop is an attribute the gate must reject)
    "[].__class__.__subclasses__()",
    '"x".__class__',
    "type.__bases__",
    "print.__self__",
    "print.__code__",
    "().__globals__",
    "exit.__builtins__",
    # Forbidden reflection attributes from the pinned table
    "def g():\n    yield 1\nframe = g().gi_frame",
    # Non-Name/Attribute callees stay rejected
    'funcs = {"a": print}\nfuncs["a"]()',
    "(lambda: 1)()",
    # Blocked builtins by name (shadowing a blocked name is also rejected)
    'open("x.txt")',
    "def open(p):\n    return p",
    # Imports of any kind
    "import math\nmath.sqrt(4)",
    # Dynamic construction of a blocked name
    'g=getattr\nu="__"+"class__"\nprint(g((), u))',
]


@pytest.mark.parametrize("code", SAFE_SNIPPETS)
def test_safe_method_calls_allowed(code):
    err = _sandbox_check(code)
    assert err is None, f"Expected safe snippet to pass, got: {err}"


@pytest.mark.parametrize("code", BLOCKED_SNIPPETS)
def test_escape_vectors_still_blocked(code):
    err = _sandbox_check(code)
    assert err is not None, f"Expected snippet to be blocked, but was allowed: {code}"


def test_run_code_executes_method_calls_end_to_end():
    code = 'd = {"score": 41}\nprint(sorted(d.keys()))\nprint(" ".join(["ok", "42"]))'
    result = execute_run_code(code)
    assert result.startswith("STDOUT:")
    assert "score" in result
    assert "ok 42" in result


def test_run_code_blocks_dunder_escape_at_runtime():
    result = execute_run_code("x = [].__class__")
    assert result.startswith(("SANDBOX BLOCK", "ERROR:"))
