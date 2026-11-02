import pytest
from pathlib import Path
from nexus_agent_ai.agent.tools import execute_read_file, execute_list_directory, execute_tool

def test_read_file_success(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello Programmer Assistant!", encoding="utf-8")

    result = execute_read_file(str(test_file))
    assert result == "Hello Programmer Assistant!"

def test_read_file_not_found():
    result = execute_read_file("non_existent_random_file_123.txt")
    assert result.startswith("ERROR: File not found")

def test_list_directory_success(tmp_path: Path):
    (tmp_path / "file1.py").write_text("print(1)")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.py").write_text("print(2)")

    result = execute_list_directory(str(tmp_path))
    assert "[FILE] file1.py" in result
    assert "[DIR] subdir/" in result
    assert "[FILE] file2.py" in result

def test_list_directory_not_found():
    result = execute_list_directory("non_existent_folder_999")
    assert result.startswith("ERROR: Directory not found")

def test_search_web(monkeypatch):
    import sys
    from types import ModuleType

    calls = []

    class FakeDDGS:
        def text(self, query, max_results):
            calls.append((query, max_results))
            return [{
                "title": "Python programming",
                "href": "https://www.python.org/",
                "body": "The Python programming language.",
            }]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

    res = execute_tool("search_web", {"query": "python programming"})
    assert calls == [("python programming", 5)]
    assert res == (
        "Search results for: 'python programming'\n\n"
        "1. Python programming\n"
        "   URL: https://www.python.org/\n"
        "   Summary: The Python programming language.\n"
    )

def test_write_file_success(tmp_path: Path):
    target = tmp_path / "new_dir" / "out.txt"
    res = execute_tool("write_file", {"path": str(target), "content": "Test content 123"})
    assert "Successfully wrote" in res
    assert target.read_text(encoding="utf-8") == "Test content 123"

def test_run_code_success():
    res = execute_tool("run_code", {"code": "print(2 * 21)"})
    assert "STDOUT:" in res
    assert "42" in res


def test_run_code_fstring_allowed_and_escapes_blocked():
    res = execute_tool("run_code", {"code": 'print(f"{2+2}")'})
    assert "STDOUT:" in res
    assert "4" in res

    for bad in [
        'print(f"{__import__(\'os\')}")',
        'print(f"{open(\'foo\')}")',
        'print(f"{eval(\'1+1\')}")',
    ]:
        res_bad = execute_tool("run_code", {"code": bad})
        assert "SANDBOX BLOCK" in res_bad
def test_git_status_tool(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    res = execute_tool("git_status", {})
    assert "git status" in res or "Working tree clean" in res or "Changed Files" in res

def test_execute_tool_dispatcher():
    res = execute_tool("read_file", {"path": "invalid_path_456.txt"})
    assert "ERROR: File not found" in res

    res_unknown = execute_tool("fake_tool", {})
    assert "ERROR: Unknown tool" in res_unknown

def test_get_readonly_tools():
    from nexus_agent_ai.agent.tools import get_readonly_tools
    ro_tools = get_readonly_tools()
    names = [t.name for t in ro_tools]
    assert "read_file" in names
    assert "list_directory" in names
    assert "write_file" not in names
    assert "patch_file" not in names
    assert "run_code" not in names
    assert "run_tests" not in names
    assert "git_commit" not in names

def test_patch_file_success(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text("def hello():\n    print('old')\n", encoding="utf-8")
    execute_tool("read_file", {"path": str(target)})
    res = execute_tool("patch_file", {
        "path": str(target),
        "target": "print('old')",
        "replacement": "print('new')"
    })
    assert "Successfully patched" in res
    assert "print('new')" in target.read_text(encoding="utf-8")
    assert "print('old')" not in target.read_text(encoding="utf-8")

def test_patch_file_not_found(tmp_path: Path):
    res = execute_tool("patch_file", {
        "path": str(tmp_path / "missing.py"),
        "target": "foo",
        "replacement": "bar"
    })
    assert res.startswith("ERROR: File not found")

def test_patch_file_target_not_found(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text("x = 10\n", encoding="utf-8")
    execute_tool("read_file", {"path": str(target)})
    res = execute_tool("patch_file", {
        "path": str(target),
        "target": "y = 20",
        "replacement": "y = 30"
    })
    assert "ERROR: Target content not found" in res

def test_patch_file_multiple_occurrences(tmp_path: Path):
    target = tmp_path / "multi.txt"
    target.write_text("item item item", encoding="utf-8")
    execute_tool("read_file", {"path": str(target)})
    # Default: multiple blocked
    res_blocked = execute_tool("patch_file", {
        "path": str(target),
        "target": "item",
        "replacement": "widget"
    })
    assert "ERROR: Target content found 3 times" in res_blocked

    # With allow_multiple=True
    res_allowed = execute_tool("patch_file", {
        "path": str(target),
        "target": "item",
        "replacement": "widget",
        "allow_multiple": True
    })
    assert "Successfully patched" in res_allowed
    assert target.read_text(encoding="utf-8") == "widget widget widget"

def test_patch_file_empty_target(tmp_path: Path):
    target = tmp_path / "empty_test.txt"
    target.write_text("hello", encoding="utf-8")
    res = execute_tool("patch_file", {
        "path": str(target),
        "target": "",
        "replacement": "new"
    })
    assert "ERROR: Target string to replace cannot be empty." in res

def test_patch_file_sensitive_path(tmp_path: Path):
    secret = tmp_path / ".env"
    secret.write_text("SECRET=123", encoding="utf-8")
    res = execute_tool("patch_file", {
        "path": str(secret),
        "target": "123",
        "replacement": "456"
    })
    assert "ERROR: Security Blocked" in res

def test_run_tests_success(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    test_file = tmp_path / "test_mini.py"
    test_file.write_text("def test_ok(): assert 1 + 1 == 2\n", encoding="utf-8")
    res = execute_tool("run_tests", {"path": str(test_file), "args": "-q"})
    assert "PASSED" in res
    assert "1 passed" in res

def test_run_tests_rejected_unsafe_args(monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    res = execute_tool("run_tests", {"args": "--override-ini=bad"})
    assert "ERROR: Unsupported or unsafe pytest argument" in res

def test_run_tests_nonexistent_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    res = execute_tool("run_tests", {"path": str(tmp_path / "nonexistent_test.py")})
    assert "ERROR: Target test path does not exist" in res

def test_run_tests_maxfail_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    # Invalid maxfail
    res_bad = execute_tool("run_tests", {"args": "--maxfail=abc"})
    assert "ERROR: Invalid --maxfail value" in res_bad

    res_zero = execute_tool("run_tests", {"args": "--maxfail 0"})
    assert "ERROR: Invalid --maxfail value" in res_zero

    # Valid maxfail
    test_file = tmp_path / "test_maxfail.py"
    test_file.write_text("def test_one(): assert 1 == 1\n", encoding="utf-8")
    res_good = execute_tool("run_tests", {"path": str(test_file), "args": "-q --maxfail=1"})
    assert "PASSED" in res_good



def test_patch_miss_on_small_file_shows_whole_file(tmp_path, monkeypatch):
    from nexus_agent_ai.agent import tools as T
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a - b\n")
    T.reset_file_tracking()
    T.execute_read_file("a.py")
    out = T.execute_patch_file("a.py", "zzzzzzzzzz qqqqqqqqq", "x")
    assert "Target content not found" in out and "return a - b" in out


def test_run_code_refusal_points_to_run_tests():
    from nexus_agent_ai.agent import tools as T
    out = T.execute_run_code("import os\nprint(os.getcwd())")
    assert "run_tests" in out


# --- f-string pre/post-eval bounds in sandboxed run_code (design-review MODIFY) ---

@pytest.mark.parametrize("code", [
    'print(f"{1:{10**9}}")',            # exponent-driven width -> memory DoS
    'print(f"{\'x\':>999999999}")',     # huge width literal
    'print(f"{3.14:.10000000f}")',      # huge precision literal
    'x = 1\nprint(f"{x:{2:{3}}}")',     # format spec nested deeper than one level
    'w = 10**9\nprint(f"{1:{w}}")',    # computed width - unboundable pre-eval
    'w = 1000 * 1000\nprint(f"{1:{w}}")',  # multiplication-generated width

    'print(f"{f\'{f"{f"{1}"}"}\'}")',   # JoinedStr nesting depth 4
    'print(f"{__import__(\'os\').system(\'echo pwned\')}")',  # escape via FormattedValue.value
])
def test_run_code_fstring_dos_and_escape_blocked(code):
    from nexus_agent_ai.agent.tools import execute_run_code
    result = execute_run_code(code)
    assert result.startswith("SANDBOX BLOCK:"), result[:200]
    assert "pwned" not in result


@pytest.mark.parametrize("code,expected", [
    ('name = "world"\nprint(f"hello {name}")', "hello world"),
    ('x = 7\nprint(f"{x:>100}")', "7"),                  # width 100 fine
    ('print(f"{f\'{f"{1}"}\'}")', "1"),                  # nesting depth 3 allowed
    ('x = 42\nprint(f"{x=}")', "x=42"),  # self-documenting fine
])
def test_run_code_fstring_legit_forms_still_work(code, expected):
    from nexus_agent_ai.agent.tools import execute_run_code
    result = execute_run_code(code)
    assert not result.startswith("SANDBOX BLOCK:"), result[:200]
    assert expected in result


def test_run_code_output_capped_at_one_mib():
    from nexus_agent_ai.agent.tools import execute_run_code
    result = execute_run_code('print("x" * 2_000_000)')
    assert "output truncated at 1 MiB" in result
    assert len(result) < 1_200_000



def test_run_code_memory_ceiling_kills_huge_allocation():
    # Real resource-failure proof: a ~600MB allocation exceeds the 512MB child
    # address-space limit and dies as MemoryError instead of hanging or OOM-ing
    # the host. The ceiling is a POSIX-only control verified on Linux (see
    # _child_rlimits_supported): Android/Termux and Windows deliberately run
    # the child without rlimits, so this test runs only where the control is
    # actually enforced; the omission on other platforms is pinned by
    # test_run_code_rlimit_dispatch_matches_supported_boundary below.
    from nexus_agent_ai.agent.tools import _child_rlimits_supported, execute_run_code
    if not _child_rlimits_supported():
        pytest.skip("child rlimits not applied on this platform (Android/Termux or Windows); "
                    "memory ceiling is a desktop-POSIX-only control")
    result = execute_run_code('x = "y" * 600_000_000\nprint(len(x))')
    assert "600000000" not in result
    assert "ERROR" in result or "MemoryError" in result or "STDERR" in result


def test_run_code_rlimit_dispatch_matches_supported_boundary(monkeypatch):
    # The dispatch itself is the contract: rlimits on plain POSIX, omitted on
    # Android/Termux (ANDROID_ROOT set) and on non-POSIX. Pinning the omission
    # keeps the uncovered-platform gap explicit instead of silently skipped.
    from nexus_agent_ai.agent import tools
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.setattr(tools.os, "name", "posix")
    assert tools._child_rlimits_supported() is True
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    assert tools._child_rlimits_supported() is False
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.setattr(tools.os, "name", "nt")
    assert tools._child_rlimits_supported() is False


def test_run_code_execution_works_where_rlimits_omitted(monkeypatch):
    # Where rlimits are deliberately omitted (simulated Android dispatch), the
    # remaining controls must still let ordinary code execute: timeout,
    # stripped environment and output caps stay active.
    from nexus_agent_ai.agent import tools
    if tools.os.name != "posix":
        pytest.skip("dispatch simulation requires a posix host")
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    assert tools._child_rlimits_supported() is False
    result = tools.execute_run_code('x = "y" * 1_000\nprint(len(x))')
    assert "1000" in result


def test_run_code_repr_of_large_object_output_truncated():
    from nexus_agent_ai.agent.tools import execute_run_code
    result = execute_run_code('x = "y" * 2_000_000\nprint(f"{x!r}")')
    assert "output truncated at 1 MiB" in result
