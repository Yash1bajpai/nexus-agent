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

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_search_web():
    res = execute_tool("search_web", {"query": "python programming"})
    assert "Search results for:" in res or "ERROR:" in res or "No web search results found" in res or "[Live Search Warning]:" in res

def test_write_file_success(tmp_path: Path):
    target = tmp_path / "new_dir" / "out.txt"
    res = execute_tool("write_file", {"path": str(target), "content": "Test content 123"})
    assert "Successfully wrote" in res
    assert target.read_text(encoding="utf-8") == "Test content 123"

def test_run_code_success():
    res = execute_tool("run_code", {"code": "print(2 * 21)"})
    assert "STDOUT:" in res
    assert "42" in res

def test_git_status_tool():
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
    res = execute_tool("patch_file", {
        "path": str(target),
        "target": "y = 20",
        "replacement": "y = 30"
    })
    assert "ERROR: Target content not found" in res

def test_patch_file_multiple_occurrences(tmp_path: Path):
    target = tmp_path / "multi.txt"
    target.write_text("item item item", encoding="utf-8")
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

def test_run_tests_success(tmp_path: Path):
    test_file = tmp_path / "test_mini.py"
    test_file.write_text("def test_ok(): assert 1 + 1 == 2\n", encoding="utf-8")
    res = execute_tool("run_tests", {"path": str(test_file), "args": "-q"})
    assert "PASSED" in res
    assert "1 passed" in res

def test_run_tests_rejected_unsafe_args():
    res = execute_tool("run_tests", {"args": "--override-ini=bad"})
    assert "ERROR: Unsupported or unsafe pytest argument" in res

def test_run_tests_nonexistent_path(tmp_path: Path):
    res = execute_tool("run_tests", {"path": str(tmp_path / "nonexistent_test.py")})
    assert "ERROR: Target test path does not exist" in res

