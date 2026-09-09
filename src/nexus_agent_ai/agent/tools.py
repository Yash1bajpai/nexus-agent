import os
import sys
import tempfile
import subprocess
import ast
from pathlib import Path
from typing import Any, Dict, List
from ..providers.base import Tool
from ..utils.config import CODE_EXECUTION_TIMEOUT

def _validate_workspace_path(path: str | Path) -> Path | str:
    """Validate that path is inside the current workspace."""
    p = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        if p.is_relative_to(cwd):
            return p
    except AttributeError:
        try:
            if os.path.commonpath([str(p), str(cwd)]) == str(cwd):
                return p
        except ValueError:
            pass
    return f"ERROR: Security Sandbox Access Denied: Path '{path}' is outside the project workspace ({cwd})."

def _safe_git_run(args: List[str]) -> subprocess.CompletedProcess:
    """Execute a git command safely with timeouts, blocked stdin, and captured output."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=15.0,
        check=False
    )

def execute_read_file(path: str) -> str:
    """Read the contents of any file and return it as a string."""
    try:
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated

        # Do not let an agent copy common credential/config files into prompts.
        filename_lower = p.name.lower()
        sensitive_names = {".git", "config", "credentials", "secrets", ".npmrc", ".pypirc", "netrc"}
        sensitive_suffixes = (".pem", ".key", ".pfx", ".p12", ".crt", ".token", ".secret")
        if (filename_lower.startswith(".env") or filename_lower in sensitive_names
                or any(part.lower() in sensitive_names for part in p.parts)
                or any(filename_lower.endswith(ext) for ext in sensitive_suffixes)
                or any(marker in filename_lower for marker in ("credential", "secret", "token", "password", "passwd"))
                or "id_rsa" in filename_lower):
            return f"ERROR: Security Blocked: Access to secret/credential file '{p.name}' is restricted."

        with open(p, "r", encoding="utf-8", errors="replace") as f:
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
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated
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
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        root_path = validated
        if not root_path.exists() or not root_path.is_dir():
            return f"ERROR: Directory not found or not a directory: {path}"

        ignore_dirs = {".git", "__pycache__", "node_modules", "venv", ".env", ".pytest_cache", "dist", "build"}
        output_lines = [f"Directory tree for: {root_path.name}"]

        root_depth = len(root_path.parts)

        for current_root, dirs, files in os.walk(root_path):
            current_path = Path(current_root)
            depth = len(current_path.parts) - root_depth

            if depth >= 3:
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

# Modules that are too dangerous to allow inside agent-generated run_code snippets.
# Use read_file / write_file tools for file I/O; use git_status / git_diff for shell work.
_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "socket", "socketserver", "urllib", "urllib3", "http", "httpx", "requests",
    "pickle", "shelve", "ctypes", "multiprocessing", "sys", "platform", "operator",
    "pathlib", "pty", "asm", "cffi", "signal", "importlib", "runpy", "builtins", "io", "codecs",
    "gc", "warnings", "pkgutil", "types", "marshal", "smtplib", "ftplib", "telnetlib",
    "threading", "asyncio", "socketio",
    "linecache", "xmlrpc", "xmlrpc.client", "xmlrpc.server",
    "winreg", "webbrowser", "email", "json.tool",
    "zipfile", "tarfile", "tempfile", "glob", "fileinput", "filecmp",
    "compileall", "zipimport", "pkg_resources",
    "inspect", "traceback", "__future__",
})

_FORBIDDEN_ATTRIBUTES = frozenset({
    "gi_frame", "gi_code", "f_builtins", "f_globals", "f_locals",
    "cr_frame", "cr_code", "ag_frame", "ag_code", "get_objects", "get_referents",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
})

# Dunders that are safe to use in sandboxed code (e.g. __name__ == "__main__").
_SAFE_DUNDERS = frozenset({
    "__name__", "__main__", "__file__", "__doc__", "__init__",
    "__str__", "__repr__", "__len__", "__eq__", "__hash__",
    "__bool__", "__iter__", "__next__", "__call__", "__contains__",
    "__enter__", "__exit__", "__getattr__", "__setattr__",
})

# Dunders that are explicitly blocked (sandbox escape mechanisms).
_BLOCKED_DUNDERS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__import__", "__code__",
    "__func__", "__self__", "__weakref__", "__module__",
})

_SAFE_CALLS = frozenset({
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "len", "list", "map", "max", "min", "print", "range", "repr", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
})


def _attr_name_is_safe(attr: str) -> bool:
    """Attribute-name gate for sandboxed run_code.

    Pure-data method calls (s.split(), d.get(), items.append()) are safe because
    imports are blocked, so names can only resolve to literals, user-defined
    functions, and non-dangerous builtins. The remaining risk is dunder/reflection
    attributes, which walk object graphs back to dangerous machinery — those stay
    blocked via the explicit tables.
    """
    if attr in _FORBIDDEN_ATTRIBUTES or attr in _BLOCKED_DUNDERS:
        return False
    if attr.startswith("__") and attr not in _SAFE_DUNDERS:
        return False
    return True


def _sandbox_check(code: str) -> str | None:
    """
    AST-based static analysis. Returns an error string if forbidden
    constructs or indirect import bypass mechanisms are detected, or None if safe.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"ERROR: Code has a syntax error: {e}"

    violations = []
    allowed_nodes = (
        ast.Module, ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Name, ast.Load,
        ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp,
        ast.SetComp, ast.DictComp, ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp,
        ast.Compare, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
        ast.Return, ast.FunctionDef, ast.arguments, ast.arg, ast.Call, ast.keyword,
        ast.IfExp, ast.Subscript, ast.Slice, ast.Index, ast.Assert, ast.Lambda,
        ast.Attribute,
        ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    )
    blocked_names = {"__import__", "eval", "exec", "compile", "open", "getattr", "setattr",
                     "delattr", "globals", "locals", "vars", "dir", "help", "input",
                     "breakpoint", "exit", "quit", "license", "credits", "copyright"}
    # Calls to user-defined functions are safe: their bodies are validated by this
    # same walk. Blocked-name shadowing (e.g. `def open(...)`) is rejected outright
    # so blocked identifiers are never callable, period.
    defined_funcs = {
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name not in blocked_names and not n.name.startswith("__")
    }
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            violations.append(f"{type(node).__name__} is not allowed in sandboxed run_code")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(f"{type(node).__name__} is not allowed in sandboxed run_code")
        if isinstance(node, ast.Name) and (node.id in blocked_names or (node.id.startswith("__") and node.id not in {"__name__", "__file__"})):
            violations.append(f"{node.id} is not available in sandboxed run_code")
        if isinstance(node, ast.FunctionDef) and (node.name in blocked_names or node.name.startswith("__")):
            violations.append(f"function name '{node.name}' may not shadow sandboxed identifiers")
        if isinstance(node, ast.Attribute) and not _attr_name_is_safe(node.attr):
            violations.append(f"attribute '{node.attr}' is not allowed in sandboxed run_code")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_CALLS and node.func.id not in defined_funcs:
                    violations.append(f"function '{node.func.id}' is not an approved pure function for sandboxed run_code")
            elif isinstance(node.func, ast.Attribute):
                if not _attr_name_is_safe(node.func.attr):
                    violations.append(f"method '{node.func.attr}' is not allowed in sandboxed run_code")
            else:
                violations.append("only approved pure functions or safe method calls are allowed in sandboxed run_code")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ("__" in node.value and node.value not in {"__main__"}) or any(word in node.value.lower() for word in ("import", "subprocess", "system", "password", "token")):
                violations.append("sensitive reflection or command text is not allowed in sandboxed run_code")

    if violations:
        bullet_list = "\n  - ".join(violations)
        return (
            f"SANDBOX BLOCK: The following forbidden constructs were detected:\n"
            f"  - {bullet_list}\n\n"
            f"run_code is for pure computation only (math, algorithms, data processing).\n"
            f"For file I/O -> use read_file / write_file tools.\n"
            f"For shell commands -> use git_status / git_diff tools.\n"
            f"Existing scripts cannot be executed by the agent."
        )
    return None

def execute_run_code(code: str, language: str = "python") -> str:
    """Run Python code snippet in a sandboxed subprocess with AST analysis + timeout."""
    if language.lower() not in ["python", "py"]:
        return f"ERROR: Running language '{language}' is not supported yet. Only Python is supported."

    # AST sandbox check before any execution
    sandbox_error = _sandbox_check(code)
    if sandbox_error:
        return sandbox_error

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(code)
            temp_path = tf.name

        try:
            safe_env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
            preexec_fn = None
            # Android/Termux applies process resource limits differently; the
            # subprocess timeout and stripped environment remain active there.
            if os.name == "posix" and not os.getenv("ANDROID_ROOT"):
                def _limit_child_resources():
                    try:
                        import resource
                        resource.setrlimit(resource.RLIMIT_CPU, (CODE_EXECUTION_TIMEOUT, CODE_EXECUTION_TIMEOUT + 1))
                        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                        resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
                        if hasattr(resource, "RLIMIT_NOFILE"):
                            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                    except (ImportError, OSError, ValueError):
                        pass
                preexec_fn = _limit_child_resources
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CODE_EXECUTION_TIMEOUT,
                cwd=tempfile.gettempdir(),
                env=safe_env,
                preexec_fn=preexec_fn,
            )
        finally:
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
    """Search the web using DuckDuckGo with relevance validation and resilient fallback."""
    def _get_curated_fallback(query: str) -> str:
        """Return a curated fallback when DuckDuckGo is unavailable or returns off-topic results."""
        return f"Live web search is currently unavailable or rate-limited for query: '{query}'."

    def _is_relevant(results: list, query: str) -> bool:
        """Check if DuckDuckGo results are actually relevant to the query."""
        if not results:
            return False
        query_keywords = set(w.lower() for w in query.split() if len(w) > 3)
        if not query_keywords:
            return True
        hits = 0
        for r in results[:3]:
            combined = (r.get("title", "") + " " + r.get("body", "") + " " + r.get("href", "")).lower()
            if any(kw in combined for kw in query_keywords):
                hits += 1
        return hits >= 1

    try:
        import warnings
        orig_warn = warnings.warn
        results = []
        try:
            warnings.warn = lambda msg, *a, **kw: None if "duckduckgo_search" in str(msg) or "ddgs" in str(msg) else orig_warn(msg, *a, **kw)
            try:
                from ddgs import DDGS
                results = DDGS().text(query, max_results=5)
            except Exception:
                try:
                    from duckduckgo_search import DDGS
                    results = DDGS().text(query, max_results=5)
                except Exception:
                    results = []
        except Exception:
            results = []
        finally:
            warnings.warn = orig_warn

        if not results or not _is_relevant(results, query):
            return f"[Live Search Warning]: {_get_curated_fallback(query)}"

        formatted = [f"Search results for: '{query}'\n"]
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            href = r.get("href", "")
            body = r.get("body", "")
            formatted.append(f"{idx}. {title}\n   URL: {href}\n   Summary: {body}\n")
        return "\n".join(formatted)
    except Exception:
        return f"[Live Search Warning]: {_get_curated_fallback(query)}"

def execute_git_status() -> str:
    """Run git status and git diff stats to inspect repository state."""
    try:
        status_res = _safe_git_run(["git", "status", "--short"])
        if status_res.returncode != 0:
            return "ERROR: Not inside a Git repository."

        diff_res = _safe_git_run(["git", "diff", "--stat"])

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

def execute_git_diff() -> str:
    """Return full git diff of staged changes (or HEAD diff if nothing staged)."""
    try:
        staged = _safe_git_run(["git", "diff", "--staged"])
        diff_text = staged.stdout.strip()

        if not diff_text:
            unstaged = _safe_git_run(["git", "diff", "HEAD"])
            diff_text = unstaged.stdout.strip()

        if not diff_text:
            return "No changes to commit. Working tree is clean and nothing is staged."

        lines = diff_text.split("\n")
        if len(lines) > 500:
            diff_text = "\n".join(lines[:500]) + f"\n\n[... diff truncated at 500 lines, {len(lines)} total ...]"

        return diff_text
    except Exception as e:
        return f"ERROR: Could not read git diff: {str(e)}"

def execute_git_commit(message: str) -> str:
    """Run git commit with the given message. Auto-stages tracked modified files if nothing is staged."""
    try:
        if not message or not message.strip():
            return "ERROR: Commit message cannot be empty."

        staged_check = _safe_git_run(["git", "diff", "--cached", "--name-only"])
        if not staged_check.stdout.strip():
            # Auto-stage tracked modified files per tool schema description
            _safe_git_run(["git", "add", "-u"])
            staged_check = _safe_git_run(["git", "diff", "--cached", "--name-only"])

        if not staged_check.stdout.strip():
            return "ERROR: No staged changes found to commit. Use git_status to review untracked/modified files."

        result = _safe_git_run(["git", "commit", "-m", message])
        if result.returncode == 0:
            return f"Committed successfully:\n{result.stdout.strip()}"
        else:
            return f"ERROR: git commit failed:\n{result.stderr.strip() or result.stdout.strip()}"
    except Exception as e:
        return f"ERROR: Could not commit: {str(e)}"

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

GIT_DIFF_TOOL = Tool(
    name="git_diff",
    description="Get the full git diff of staged changes (or working-tree changes if nothing is staged). Use before generating a commit message.",
    input_schema={
        "type": "object",
        "properties": {},
    },
    execute=execute_git_diff,
)

GIT_COMMIT_TOOL = Tool(
    name="git_commit",
    description="Commit staged changes with a given message. Auto-stages tracked modified files if nothing is staged.",
    input_schema={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The commit message to use.",
            }
        },
        "required": ["message"],
    },
    execute=execute_git_commit,
)

def execute_run_file(path: str) -> str:
    """Reject unrestricted script execution."""
    return (
        "ERROR: Running existing files is disabled because it would execute arbitrary code "
        "with the user's permissions. Use run_code for pure computation instead."
    )

RUN_FILE_TOOL = Tool(
    name="run_file",
    description="Disabled for security: existing files must not be executed with the user's full permissions.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the .py file to execute.",
            }
        },
        "required": ["path"],
    },
    execute=execute_run_file,
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
        GIT_DIFF_TOOL,
        GIT_COMMIT_TOOL,
    ]

def get_readonly_tools() -> List[Tool]:
    """Return read-only inspection tools (safe for code review without modifying disk)."""
    return [
        READ_FILE_TOOL,
        LIST_DIRECTORY_TOOL,
        SEARCH_WEB_TOOL,
        GIT_STATUS_TOOL,
        GIT_DIFF_TOOL,
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
