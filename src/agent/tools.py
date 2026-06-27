import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from src.providers.base import Tool
from src.utils.config import CODE_EXECUTION_TIMEOUT

def execute_read_file(path: str) -> str:
    """Read the contents of any file and return it as a string."""
    try:
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except Exception as e:
        return f"ERROR: Could not read file: {str(e)}"

def execute_write_file(path: str, content: str) -> str:
    """Write string content to a file (creating parent directories if needed)."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"ERROR: Permission denied writing to: {path}"
    except Exception as e:
        return f"ERROR: Could not write file: {str(e)}"

def execute_list_directory(path: str = ".") -> str:
    """List all files and folders in a directory (up to 2 levels deep)."""
    try:
        root_path = Path(path).resolve()
        if not root_path.exists() or not root_path.is_dir():
            return f"ERROR: Directory not found or not a directory: {path}"

        ignore_dirs = {".git", "__pycache__", "node_modules", "venv", ".env", ".pytest_cache", "dist", "build"}
        output_lines = [f"Directory tree for: {root_path.name}"]

        root_depth = len(root_path.parts)

        for current_root, dirs, files in os.walk(root_path):
            current_path = Path(current_root)
            depth = len(current_path.parts) - root_depth

            if depth >= 2:
                dirs[:] = []  # Do not recurse deeper than 2 levels
                continue

            # Filter out ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            indent = "    " * depth
            rel_path = current_path.relative_to(root_path)
            if rel_path != Path("."):
                output_lines.append(f"{indent}[DIR] {current_path.name}/")
                file_indent = indent + "    "
            else:
                file_indent = ""

            for file in sorted(files):
                if file not in ignore_dirs:
                    output_lines.append(f"{file_indent}[FILE] {file}")

        return "\n".join(output_lines)
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except Exception as e:
        return f"ERROR: Could not list directory: {str(e)}"

def execute_run_code(code: str, language: str = "python") -> str:
    """Run Python code snippet in subprocess with timeout."""
    if language.lower() not in ["python", "py"]:
        return f"ERROR: Running language '{language}' is not supported yet. Only Python is supported."

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(code)
            temp_path = tf.name

        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT
        )
        try:
            os.unlink(temp_path)
        except Exception:
            pass

        out = res.stdout.strip()
        err = res.stderr.strip()
        output = ""
        if out:
            output += f"STDOUT:\n{out}\n"
        if err:
            output += f"STDERR:\n{err}\n"
        if not output:
            output = "[Code executed successfully with no output]"
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"ERROR: Code execution timed out after {CODE_EXECUTION_TIMEOUT} seconds."
    except Exception as e:
        return f"ERROR: Code execution failed: {str(e)}"

def execute_search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        import warnings
        orig_warn = warnings.warn
        try:
            warnings.warn = lambda msg, *a, **kw: None if "duckduckgo_search" in str(msg) else orig_warn(msg, *a, **kw)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=5)
        finally:
            warnings.warn = orig_warn
        if not results:
            return f"No web search results found for query: {query}"
        formatted = [f"Search results for: '{query}'\n"]
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            href = r.get("href", "")
            body = r.get("body", "")
            formatted.append(f"{idx}. {title}\n   URL: {href}\n   Summary: {body}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"ERROR: Web search failed: {str(e)}"

def execute_git_status() -> str:
    """Run git status and git diff stats to inspect repository state."""
    try:
        status_res = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=False)
        diff_res = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, check=False)

        status_out = status_res.stdout.strip()
        diff_out = diff_res.stdout.strip()

        output = []
        if status_out:
            output.append(f"Changed Files (git status --short):\n{status_out}")
        else:
            output.append("Working tree clean (no uncommitted changes).")

        if diff_out:
            output.append(f"\nDiff Statistics:\n{diff_out}")

        return "\n".join(output)
    except Exception as e:
        return f"ERROR: Could not get git status: {str(e)}"

READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read the contents of any file and return it as a string. Use this before answering any question about a specific file.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to read.",
            }
        },
        "required": ["path"],
    },
    execute=execute_read_file,
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description="Write or overwrite contents of a file on disk. Use this when creating new code files or making edits.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to write to.",
            },
            "content": {
                "type": "string",
                "description": "The full text content to write into the file.",
            }
        },
        "required": ["path", "content"],
    },
    execute=execute_write_file,
)

LIST_DIRECTORY_TOOL = Tool(
    name="list_directory",
    description="List all files and folders in a directory (2 levels deep). Use at the start of any project-level question to understand structure.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Use '.' for current directory.",
            }
        },
        "required": ["path"],
    },
    execute=execute_list_directory,
)

RUN_CODE_TOOL = Tool(
    name="run_code",
    description="Execute a snippet of Python code safely and capture stdout/stderr output. Use for calculation, testing, or verifying logic.",
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code snippet to execute.",
            },
            "language": {
                "type": "string",
                "description": "Programming language (default: 'python').",
            }
        },
        "required": ["code"],
    },
    execute=execute_run_code,
)

SEARCH_WEB_TOOL = Tool(
    name="search_web",
    description="Search the web for real-time information, documentation, error solutions, or library releases.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search keywords or query string.",
            }
        },
        "required": ["query"],
    },
    execute=execute_search_web,
)

GIT_STATUS_TOOL = Tool(
    name="git_status",
    description="Inspect current Git repository status (modified files and diff summary).",
    input_schema={
        "type": "object",
        "properties": {},
    },
    execute=execute_git_status,
)

def get_all_tools() -> List[Tool]:
    """Return all available tools."""
    return [
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        LIST_DIRECTORY_TOOL,
        RUN_CODE_TOOL,
        SEARCH_WEB_TOOL,
        GIT_STATUS_TOOL,
    ]

def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool by name with given arguments."""
    for t in get_all_tools():
        if t.name == name:
            try:
                return t.execute(**args)
            except TypeError as e:
                return f"ERROR: Invalid arguments for tool '{name}': {str(e)}"
            except Exception as e:
                return f"ERROR: Execution failed for tool '{name}': {str(e)}"
    return f"ERROR: Unknown tool '{name}'"
