import sys
from types import ModuleType

import pytest

from nexus_agent_ai.agent.tools import execute_tool


QUERY = "python downloads"
RESULTS = [
    {
        "title": "Python downloads",
        "href": "https://www.python.org/downloads/",
        "body": "Download Python releases.",
    },
    {
        "title": "Python documentation",
        "href": "https://docs.python.org/3/",
        "body": "Read the Python documentation.",
    },
]
IRRELEVANT = [{"title": "Garden plants", "href": "https://example.com/garden", "body": "Flowers."}]
FORMATTED = (
    "Search results for: 'python downloads'\n\n"
    "1. Python downloads\n"
    "   URL: https://www.python.org/downloads/\n"
    "   Summary: Download Python releases.\n\n"
    "2. Python documentation\n"
    "   URL: https://docs.python.org/3/\n"
    "   Summary: Read the Python documentation.\n"
)


@pytest.fixture
def backends(monkeypatch):
    """Replace both optional modules; None models an unavailable import."""
    def install(modern, legacy):
        calls = []

        def fake_backend(name, outcome):
            class FakeDDGS:
                def text(self, query, max_results):
                    calls.append(name)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome
            return FakeDDGS

        for name, outcome in (("ddgs", modern), ("duckduckgo_search", legacy)):
            if outcome is None:
                monkeypatch.setitem(sys.modules, name, None)
            else:
                module = ModuleType(name)
                module.DDGS = fake_backend(name, outcome)
                monkeypatch.setitem(sys.modules, name, module)
        return calls
    return install


@pytest.mark.parametrize(
    "modern,legacy,expected_calls",
    [
        pytest.param(RESULTS, None, ["ddgs"], id="modern-ok-control"),
        pytest.param(None, RESULTS, ["duckduckgo_search"], id="legacy-import-fallback-control"),
        pytest.param(ValueError("No results found."), RESULTS, ["ddgs", "duckduckgo_search"], id="execution-fallback-control"),
    ],
)
def test_success_formatting_and_fallback(backends, modern, legacy, expected_calls):
    calls = backends(modern, legacy)
    result = execute_tool("search_web", {"query": QUERY})
    assert result == FORMATTED
    assert calls == expected_calls


@pytest.mark.parametrize(
    "modern,legacy,category,expected_calls",
    [
        pytest.param([], RESULTS, "empty", ["ddgs"], id="modern-empty-no-fallback"),
        pytest.param(IRRELEVANT, RESULTS, "irrelevant", ["ddgs"], id="modern-irrelevant-no-fallback"),
        pytest.param(None, [], "empty", ["duckduckgo_search"], id="legacy-empty"),
        pytest.param(None, IRRELEVANT, "irrelevant", ["duckduckgo_search"], id="legacy-irrelevant"),
        pytest.param(ValueError("No results found."), [], "empty", ["ddgs", "duckduckgo_search"], id="fallback-empty"),
        pytest.param(ValueError("No results found."), IRRELEVANT, "irrelevant", ["ddgs", "duckduckgo_search"], id="fallback-irrelevant"),
    ],
)
def test_empty_and_irrelevant_are_distinct_errors(backends, modern, legacy, category, expected_calls):
    calls = backends(modern, legacy)
    result = execute_tool("search_web", {"query": QUERY})
    assert result.startswith(f"ERROR: search_web [{category}]:")
    assert QUERY in result
    assert calls == expected_calls
    assert "Search results for:" not in result
    assert "rate-limit" not in result
    assert "network" not in result.lower()
    if category == "empty":
        assert "returned no results" in result
        assert "underlying cause is unknown" in result
    else:
        assert "did not pass the relevance check" in result


@pytest.mark.parametrize(
    "modern,legacy,category,exception_classes,expected_calls",
    [
        pytest.param(
            None, None, "backend_error",
            ["ModuleNotFoundError"],
            [], id="both-imports-unavailable",
        ),
        pytest.param(
            None, ValueError("No results found."), "backend_error",
            ["ModuleNotFoundError", "ValueError"],
            ["duckduckgo_search"], id="legacy-execution-error",
        ),
        pytest.param(
            ValueError("No results found."), None, "backend_error",
            ["ValueError", "ModuleNotFoundError"],
            ["ddgs"], id="modern-execution-error",
        ),
        pytest.param(
            ValueError("No results found."), RuntimeError("backend failed"), "both_failed",
            ["ValueError", "RuntimeError"],
            ["ddgs", "duckduckgo_search"], id="both-executions-failed",
        ),
    ],
)
def test_backend_errors_report_categories_and_exception_classes(backends, modern, legacy, category, exception_classes, expected_calls):
    calls = backends(modern, legacy)
    result = execute_tool("search_web", {"query": QUERY})
    assert result.startswith(f"ERROR: search_web [{category}]:")
    assert QUERY in result
    for exception_class in exception_classes:
        assert exception_class in result
    assert calls == expected_calls
    assert "underlying cause is unknown" in result
    assert "rate-limit" not in result
    assert "network" not in result.lower()
    assert "Offline/Cached" not in result




@pytest.mark.parametrize("phase", ["import", "execution"])
def test_search_errors_never_expose_exception_messages(monkeypatch, backends, capsys, phase):
    from nexus_agent_ai.agent.core import Agent
    from tests.test_real_user_failures import Transcript

    # Synthetic credentials, never real ones. Either library may put these in
    # exception messages, which become both tool observations and final errors.
    credentials = ("leak-user", "leak-password", "leak-token")
    message = "request https://leak-user:leak-password@example.invalid/?token=leak-token failed"
    if phase == "import":
        class BrokenModule(ModuleType):
            def __getattr__(self, name):
                if name == "DDGS":
                    raise ImportError(message)
                raise AttributeError(name)

        for backend in ("ddgs", "duckduckgo_search"):
            monkeypatch.setitem(sys.modules, backend, BrokenModule(backend))
        category, exception_class = "backend_error", "ImportError"
    else:
        backends(RuntimeError(message), ValueError(message))
        category, exception_class = "both_failed", "RuntimeError"

    result = execute_tool("search_web", {"query": QUERY})
    events = []
    agent = Agent(Transcript([
        ("search_web", {"query": QUERY}), "I could not retrieve live results.",
    ]), max_iterations=2, verbose=False, event_callback=events.append)
    final = agent.run("Search the web for Python downloads")
    visible = "\n".join((result, final, str(agent.memory.get()), str(events)))
    captured = capsys.readouterr()
    visible += captured.out + captured.err

    assert result.startswith(f"ERROR: search_web [{category}]:")
    assert exception_class in result
    assert agent.unresolved_errors
    assert message not in visible
    for credential in credentials:
        assert credential not in visible
