import pytest
from pathlib import Path
from src.agent.tools import execute_read_file, execute_list_directory, execute_tool

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
    assert "📄 file1.py" in result
    assert "📁 subdir/" in result
    assert "📄 file2.py" in result

def test_list_directory_not_found():
    result = execute_list_directory("non_existent_folder_999")
    assert result.startswith("ERROR: Directory not found")

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
