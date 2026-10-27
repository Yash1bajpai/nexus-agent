import os
import sys
import tempfile
import subprocess
import ast
import stat
from pathlib import Path
from typing import Any, Dict, List
from ..providers.base import Tool
from ..utils.config import CODE_EXECUTION_TIMEOUT
from ..utils.private_files import copy_file_permissions

def _validate_workspace_path(path: str | Path) -> Path | str:
    """Validate that path is inside the current workspace."""
    p = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except Exception:
        home = None

    # Agent tools must never edit their own grants, history, or configuration, even
    # when a project overlaps a state directory. Check resolved paths for aliases.
    state_dirs = [home / ".nexus-agent"] if home is not None else []
    if os.environ.get("NEXUS_AGENT_HOME"):
        state_dirs.append(Path(os.environ["NEXUS_AGENT_HOME"]))
    for state_dir in state_dirs:
        state_root = state_dir.resolve()
        if p == state_root or state_root in p.parents:
            return "ERROR: Security Sandbox Access Denied: Nexus state directories are protected."

    # Boundary safeguard: When running in home directory (Global Mode), prevent direct operations
    # on the home root itself or root dotfiles/configuration (e.g. ~/.bashrc, ~/.gitconfig).
    if home and cwd == home:
        if p == home:
            return f"ERROR: Security Sandbox Access Denied: Direct targeting of the root home directory is forbidden."
        if p.parent == home and p.name.startswith("."):
            return f"ERROR: Security Sandbox Access Denied: Root configuration file '{p.name}' in home directory is protected."

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

def _safe_git_run(args: List[str], *, execution: bool = False) -> subprocess.CompletedProcess:
    """Inspect Git without inherited helpers; explicit writes retain normal Git behavior."""
    kwargs = {}
    if not execution:
        safe_env = {
            key: os.environ[key] for key in
            ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP")
            if key in os.environ
        }
        safe_env.update({
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_LAZY_FETCH": "1", "GIT_PAGER": "cat",
            "GIT_ATTR_NOSYSTEM": "1",
        })
        kwargs["env"] = safe_env
        options = [
            "--no-pager", "-c", "core.fsmonitor=false",
            "-c", f"core.hooksPath={os.devnull}", "-c", "diff.external=",
            "-c", f"core.attributesFile={os.devnull}", "-c", "submodule.recurse=false",
            "-c", "core.quotePath=true", "-c", "status.relativePaths=false",
            "-c", "diff.renameLimit=0",
        ]
        location = args[1:3] if len(args) > 2 and args[1] == "-C" else []
        # --no-textconv does not disable clean/process filters used when Git reads
        # the worktree. Enumerate configuration without reading worktree content,
        # then explicitly neutralize every filter driver before inspection.
        drivers = subprocess.run(
            [args[0], *options, *location, "config", "--null", "--name-only",
             "--get-regexp", r"^filter\."],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=15.0, check=False, env=safe_env,
        )
        if drivers.returncode not in (0, 1):
            raise ValueError("Could not safely inspect Git filter configuration.")
        for key in set(drivers.stdout.split("\0")):
            if key:
                value = "false" if key.lower().endswith(".required") else ""
                options.extend(["-c", f"{key}={value}"])
        args = [args[0], *options, *args[1:]]
    return subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=15.0, check=False, **kwargs,
    )

# Files the model has read (or written) in this process, with their state at that time.
# Edits are only allowed on files whose current state matches, so a patch is always
# based on content the model has actually seen.
_FILE_STATE: Dict[str, tuple] = {}


def _file_state(p: Path):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _remember_file(p: Path) -> None:
    state = _file_state(p)
    if state is not None:
        _FILE_STATE[str(p)] = state


def reset_file_tracking() -> None:
    """Forget which files were read (used by tests and /clear)."""
    _FILE_STATE.clear()


def _check_read_before_edit(p: Path, path: str) -> str | None:
    """Return an error string unless the existing file was read and is unchanged since."""
    if not p.exists():
        return None
    known = _FILE_STATE.get(str(p))
    if known is None:
        return (f"ERROR: Read '{path}' with read_file before editing it, "
                "so the edit is based on its current content.")
    if known != _file_state(p):
        return (f"ERROR: '{path}' changed on disk since you last read it. "
                "Read it again with read_file, then retry the edit.")
    return None


def execute_read_file(path: str) -> str:
    """Read the contents of any file and return it as a string."""
    try:
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated

        # Do not let an agent copy common credential/config files into prompts.
        if _is_sensitive_path(p):
            return f"ERROR: Security Blocked: Access to secret/credential file '{p.name}' is restricted."

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        _remember_file(p)
        return text
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except PermissionError:
        return f"ERROR: Permission denied: {path}"
    except Exception as e:
        return f"ERROR: Could not read file: {str(e)}"

_SENSITIVE_NAMES = frozenset({
    ".git", ".env", "config", "credentials", "secrets", ".npmrc", ".pypirc", "netrc",
    ".ssh", ".aws", ".azure", ".kube", ".gnupg", ".bashrc", ".zshrc", ".profile",
    ".bash_profile", "authorized_keys", "known_hosts", "id_rsa", "id_ed25519",
})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".crt", ".token", ".secret")
_SENSITIVE_MARKERS = ("credential", "secret", "token", "password", "passwd")

def _is_sensitive_path(p: Path) -> bool:
    """Check if a path points to sensitive credentials or configuration."""
    filename_lower = p.name.lower()
    return (
        filename_lower.startswith(".env")
        or filename_lower in _SENSITIVE_NAMES
        or any(part.lower() in _SENSITIVE_NAMES for part in p.parts)
        or any(filename_lower.endswith(ext) for ext in _SENSITIVE_SUFFIXES)
        or any(marker in filename_lower for marker in _SENSITIVE_MARKERS)
        or "id_rsa" in filename_lower
        or "id_ed25519" in filename_lower
    )

def _validate_python_content(path: Path, content: str) -> str | None:
    """Reject malformed Python before changing a file; never execute it."""
    if path.suffix.lower() == ".py":
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            return f"ERROR: Python syntax validation failed at line {exc.lineno}: {exc.msg}. File unchanged."
    return None

def _atomic_file_update(p: Path, content: str) -> bool:
    """Prepare both files fully before publishing; never truncate a live inode."""
    old = p.read_bytes() if p.exists() else None
    if old is not None and b"\r\n" in old and b"\n" not in old.replace(b"\r\n", b""):
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    elif old is not None and b"\n" in old and b"\r\n" not in old:
        content = content.replace("\r\n", "\n")
    encoded = content.encode("utf-8")  # Reject surrogates before touching either file.
    if encoded == old:
        return False
    backup = p.with_name(p.name + ".bak")
    if old is not None:
        validated = _validate_workspace_path(backup)
        if isinstance(validated, str):
            raise PermissionError(validated)
        if backup.is_symlink() or (backup.exists() and
                                   (not backup.is_file() or backup.stat().st_nlink != 1)):
            raise PermissionError("Unsafe backup destination; file unchanged.")
    pending = []

    def prepare(destination: Path, data: bytes) -> Path:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=p.parent)
        temporary = Path(name)
        pending.append(temporary)
        with os.fdopen(fd, "wb") as stream:
            if old is not None:
                copy_file_permissions(p, temporary)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary

    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        new_file = prepare(p, encoded)
        if old is not None:
            new_backup = prepare(backup, old)
            os.replace(new_backup, backup)
        os.replace(new_file, p)
    finally:
        for temporary in pending:
            if temporary.exists():
                if os.name == "nt":
                    os.chmod(temporary, stat.S_IWRITE | stat.S_IREAD)
                temporary.unlink()
    return True

def execute_write_file(path: str, content: str) -> str:
    """Write string content to a file (creating parent directories if needed)."""
    try:
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated

        if _is_sensitive_path(p):
            return f"ERROR: Security Blocked: Access to secret/credential file '{p.name}' is restricted."

        unread = _check_read_before_edit(p, path)
        if unread:
            return unread

        syntax_error = _validate_python_content(p, content)
        if syntax_error:
            return syntax_error

        existed = p.exists()
        if not _atomic_file_update(p, content):
            return f"No changes: '{path}' already has the requested content."
        _remember_file(p)
        backup_msg = f" (backup saved to {p.name}.bak)" if existed else ""
        return f"Successfully wrote {len(content)} characters to {path}{backup_msg}"
    except PermissionError:
        return f"ERROR: Permission denied writing to: {path}"
    except Exception as e:
        return f"ERROR: Could not write file: {str(e)}"

def _find_loose_match(content: str, target: str):
    """Find a unique span equal to target when trailing whitespace is ignored. Returns (start, end) or None."""
    target_lines = [ln.rstrip() for ln in target.replace("\r\n", "\n").split("\n")]
    while target_lines and target_lines[-1] == "":
        target_lines.pop()
    if not target_lines or not any(target_lines):
        return None
    content_lines = content.splitlines(keepends=True)
    stripped = [ln.rstrip("\r\n").rstrip() for ln in content_lines]
    n = len(target_lines)
    hits = [i for i in range(len(stripped) - n + 1) if stripped[i:i + n] == target_lines]
    if len(hits) != 1:
        return None
    i = hits[0]
    start = sum(len(ln) for ln in content_lines[:i])
    end = start + sum(len(ln) for ln in content_lines[i:i + n])
    last = content_lines[i + n - 1]
    if not target.endswith(("\n", "\r\n")) and last.endswith(("\r\n", "\n")):
        end -= len(last) - len(last.rstrip("\r\n"))
    return start, end


def _target_not_found_message(path: str, content: str, target: str) -> str:
    """Explain the miss and show the closest real lines so the model can retry accurately."""
    import difflib
    msg = (f"ERROR: Target content not found in '{path}'. "
           "Ensure the target matches existing file content exactly, including whitespace and line breaks.")
    lines = content.splitlines()
    t_lines = target.splitlines() or [target]
    n = max(1, len(t_lines))
    flat_target = "\n".join(t_lines)
    best_ratio, best_i = 0.0, -1
    for i in range(0, max(1, len(lines) - n + 1)):
        sm = difflib.SequenceMatcher(None, "\n".join(lines[i:i + n]), flat_target)
        if sm.real_quick_ratio() < 0.4 or sm.quick_ratio() < 0.4:
            continue
        ratio = sm.ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    if best_i >= 0 and best_ratio >= 0.4:
        snippet = "\n".join(f"{best_i + k + 1}: {ln}" for k, ln in enumerate(lines[best_i:best_i + min(n, 12)]))
        msg += f"\nClosest text in the file (lines {best_i + 1}-{best_i + min(n, 12)}):\n{snippet}\nCopy the target from these lines exactly and retry."
    elif len(lines) <= 40:
        # Nothing similar: show the whole (small) file so the model copies real text instead of inventing it.
        numbered = "\n".join(f"{k + 1}: {ln}" for k, ln in enumerate(lines))
        msg += f"\nThe whole file ({len(lines)} lines):\n{numbered}\nCopy the target from these lines exactly and retry."
    return msg


def execute_patch_file(path: str, target: str, replacement: str, allow_multiple: bool = False) -> str:
    """Replace target text with replacement text in an existing file."""
    try:
        validated = _validate_workspace_path(path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated

        if _is_sensitive_path(p):
            return f"ERROR: Security Blocked: Access to secret/credential file '{p.name}' is restricted."

        if not p.exists():
            return f"ERROR: File not found: {path}"
        if p.is_dir():
            return f"ERROR: Path is a directory, not a file: {path}"
        if not target:
            return "ERROR: Target string to replace cannot be empty."

        unread = _check_read_before_edit(p, path)
        if unread:
            return unread

        with open(p, "r", encoding="utf-8", newline="") as f:
            content = f.read()

        note = ""
        # Line-ending mismatch (CRLF file, LF target) is the most common harmless miss.
        if "\r\n" in content and "\r\n" not in target and "\n" in target:
            target = target.replace("\n", "\r\n")
            replacement = replacement.replace("\r\n", "\n").replace("\n", "\r\n")

        count = content.count(target)
        if count == 0:
            loose = _find_loose_match(content, target)
            if loose is not None:
                start, end = loose
                new_content = content[:start] + replacement + content[end:]
                note = " Target matched after ignoring trailing whitespace."
                count = 1
            else:
                return _target_not_found_message(path, content, target)
        elif count > 1 and not allow_multiple:
            return (
                f"ERROR: Target content found {count} times in '{path}'. "
                f"Provide more surrounding context to make the target unique, or set allow_multiple=True."
            )
        else:
            new_content = content.replace(target, replacement)
        syntax_error = _validate_python_content(p, new_content)
        if syntax_error:
            return syntax_error

        if not _atomic_file_update(p, new_content):
            return f"No changes: '{path}' already has the requested content."
        _remember_file(p)
        backup_msg = f" (backup saved to {p.name}.bak)"
        return f"Successfully patched '{path}': replaced {count} occurrence(s){backup_msg}.{note}"
    except PermissionError:
        return f"ERROR: Permission denied patching: {path}"
    except Exception as e:
        return f"ERROR: Could not patch file: {str(e)}"

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
            dirs[:] = [
                d for d in dirs if d not in ignore_dirs
                and isinstance(_validate_workspace_path(current_path / d), Path)
                and not _is_sensitive_path(current_path / d)
            ]

            indent = "    " * depth
            rel_path = current_path.relative_to(root_path)
            if rel_path != Path("."):
                output_lines.append(f"{indent}[DIR] {current_path.name}/")
                file_indent = indent + "    "
            else:
                file_indent = ""

            for file in sorted(files):
                child = current_path / file
                if (file not in ignore_dirs and not file.endswith(".bak")
                        and isinstance(_validate_workspace_path(child), Path)
                        and not _is_sensitive_path(child)):
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
            f"Existing scripts cannot be executed by run_code. To run the project's tests use the run_tests tool."
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
    """Return formatted results (ok), or an ERROR with a stable search category.

    Empty and irrelevant responses are final outcomes, not backend exceptions.
    backend_error means imports failed or only one backend could execute and failed;
    both_failed means both backends raised during execution.
    """

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

    from importlib import import_module

    failures = []
    execution_failures = []
    for backend in ("ddgs", "duckduckgo_search"):
        try:
            DDGS = import_module(backend).DDGS
        except Exception as exc:
            failures.append(f"{backend}: import failed ({type(exc).__name__})")
            continue

        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return (
                    f"ERROR: search_web [empty]: The {backend} backend returned no results "
                    f"for query: '{query}'. The underlying cause is unknown."
                )
            if not _is_relevant(results, query):
                return (
                    f"ERROR: search_web [irrelevant]: Results from {backend} did not pass "
                    f"the relevance check for query: '{query}'."
                )

            formatted = [f"Search results for: '{query}'\n"]
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                href = r.get("href", "")
                body = r.get("body", "")
                formatted.append(f"{idx}. {title}\n   URL: {href}\n   Summary: {body}\n")
            return "\n".join(formatted)
        except Exception as exc:
            execution_failures.append(backend)
            # Exception messages may contain credential-bearing URLs or tokens.
            failures.append(f"{backend}: execution failed ({type(exc).__name__})")

    category = "both_failed" if len(execution_failures) == 2 else "backend_error"
    return (
        f"ERROR: search_web [{category}]: No live search results could be retrieved "
        f"for query: '{query}'. " + "; ".join(failures)
        + ". The underlying cause is unknown."
    )

def _git_root() -> Path:
    result = _safe_git_run(["git", "rev-parse", "--show-toplevel"])
    if result.returncode:
        raise ValueError("Not inside a Git repository.")
    return Path(result.stdout.strip()).resolve()


def _git_path_allowed(root: Path, name: str) -> bool:
    path = root / name
    return (isinstance(_validate_workspace_path(path), Path)
            and not _is_sensitive_path(path)
            and not _is_sensitive_path(path.resolve()))


def _git_status_entries(root: Path) -> tuple[list[str], bool, bool]:
    result = _safe_git_run(["git", "-C", str(root), "status", "--porcelain=v1", "-z",
                            "--untracked-files=normal", "--ignore-submodules=all"])
    if result.returncode:
        raise ValueError("Git status inspection failed.")
    fields = iter(result.stdout.split("\0"))
    visible, hidden, untracked = [], False, False
    for field in fields:
        if not field:
            continue
        code, name = field[:2], field[3:]
        names = [name]
        if "R" in code or "C" in code:
            names.append(next(fields))
        if not all(_git_path_allowed(root, item) for item in names):
            hidden = True
            continue
        untracked = untracked or code == "??"
        label = repr(name) if len(names) == 1 else f"{names[1]!r} -> {name!r}"
        visible.append(f"{code} {label}")
    return visible, hidden, untracked


def execute_git_status() -> str:
    """Report only workspace-visible Git changes, without running repository helpers."""
    try:
        root = _git_root()
        visible, hidden, _ = _git_status_entries(root)
        if visible:
            output = "Changed Files (git status --short):\n" + "\n".join(visible)
        elif hidden:
            output = "No visible changes in this workspace."
        else:
            output = "Working tree clean (no uncommitted changes)."
        paths, _, _ = _git_diff_paths(root, [])
        if paths:
            stats = _safe_git_run([
                "git", "-C", str(root), "diff", *_GIT_DIFF_FLAGS, "--stat",
                "--no-renames", "--", *[f":(literal){name}" for name in paths],
            ])
            if stats.returncode:
                raise ValueError("Git diff statistics inspection failed.")
            if stats.stdout.strip():
                output += "\n\nDiff Statistics:\n" + stats.stdout.strip()
        if hidden:
            output += "\n[Protected or outside-workspace changes omitted.]"
        return output
    except ValueError as exc:
        return f"ERROR: Could not get git status: {exc}"
    except Exception as exc:
        return f"ERROR: Could not get git status ({type(exc).__name__})."


_GIT_DIFF_FLAGS = [
    "--no-ext-diff", "--no-textconv", "--ignore-submodules=all",
    "--no-relative", "--no-color", "--src-prefix=a/", "--dst-prefix=b/",
]


def _git_diff_paths(root: Path, revisions: list[str]) -> tuple[list[str], bool, bool]:
    result = _safe_git_run([
        "git", "-C", str(root), "diff", *_GIT_DIFF_FLAGS, *revisions,
        "--name-status", "-z", "--find-renames",
    ])
    if result.returncode:
        raise ValueError("Git diff inspection failed.")
    fields = iter(result.stdout.split("\0"))
    paths, hidden, changed = [], False, False
    for status in fields:
        if not status:
            continue
        changed = True
        names = [next(fields)]
        if status[0] in ("R", "C"):
            names.append(next(fields))
        if all(_git_path_allowed(root, name) for name in names):
            paths.extend(names)
        else:
            hidden = True
    return list(dict.fromkeys(paths)), hidden, changed


def execute_git_diff() -> str:
    """Show staged or working changes only for permitted workspace paths."""
    try:
        root = _git_root()
        revisions = ["--cached"]
        paths, hidden, changed = _git_diff_paths(root, revisions)
        if not changed:
            head = _safe_git_run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", "HEAD"])
            if head.returncode not in (0, 1):
                raise ValueError("Could not inspect Git HEAD.")
            revisions = ["HEAD"] if head.returncode == 0 else []
            paths, hidden, changed = _git_diff_paths(root, revisions)
        if not paths:
            if hidden:
                return "Changes exist, but protected or outside-workspace changes are omitted."
            visible, status_hidden, untracked = _git_status_entries(root)
            if untracked:
                return "Untracked files exist (not included in git diff):\n" + "\n".join(visible)
            if visible or status_hidden:
                return "Changes exist, but no visible textual diff is available in this workspace."
            return "No changes to commit. Working tree is clean and nothing is staged."
        result = _safe_git_run([
            "git", "-C", str(root), "diff", *_GIT_DIFF_FLAGS, *revisions,
            "--no-renames", "--", *[f":(literal){name}" for name in paths],
        ])
        if result.returncode:
            raise ValueError("Git diff inspection failed.")
        diff_text = result.stdout.strip()
        if not diff_text:
            return "Changes exist, but no textual diff is available."
        lines = diff_text.splitlines()
        if len(lines) > 500:
            diff_text = "\n".join(lines[:500]) + f"\n\n[... diff truncated at 500 lines, {len(lines)} total ...]"
        if hidden:
            diff_text += "\n[Protected or outside-workspace changes omitted.]"
        return diff_text
    except ValueError as exc:
        return f"ERROR: Could not read git diff: {exc}"
    except Exception as exc:
        return f"ERROR: Could not read git diff ({type(exc).__name__})."

def execute_git_commit(message: str, approved: bool = False) -> str:
    """Run git commit after explicit command confirmation or opt-in."""
    if not approved and os.environ.get("NEXUS_ALLOW_GIT_COMMIT") != "1":
        return "ERROR: Git commit is disabled for autonomous tools. Use nexus-agent commit for explicit review."
    try:
        if not message or not message.strip():
            return "ERROR: Commit message cannot be empty."

        staged_check = _safe_git_run(["git", "diff", "--cached", "--name-only"])
        if not staged_check.stdout.strip():
            # Auto-stage tracked modified files per tool schema description
            _safe_git_run(["git", "add", "-u"], execution=True)
            staged_check = _safe_git_run(["git", "diff", "--cached", "--name-only"])

        if not staged_check.stdout.strip():
            return "ERROR: No staged changes found to commit. Use git_status to review untracked/modified files."

        result = _safe_git_run(["git", "commit", "-m", message], execution=True)
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

PATCH_FILE_TOOL = Tool(
    name="patch_file",
    description="Replace target code or text block with new replacement text in an existing file. Prefer this over write_file when editing existing files to avoid losing unchanged code or truncating files.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to patch.",
            },
            "target": {
                "type": "string",
                "description": "The exact string or block of code to find and replace. Must match the existing file content exactly.",
            },
            "replacement": {
                "type": "string",
                "description": "The new replacement code or text.",
            },
            "allow_multiple": {
                "type": "boolean",
                "description": "Whether to allow replacing multiple occurrences if found (default: false).",
            },
        },
        "required": ["path", "target", "replacement"],
    },
    execute=execute_patch_file,
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

def execute_run_tests(path: str = "", args: str = "") -> str:
    """Execute pytest in the workspace with safety controls, timeout, and output formatting."""
    from .consent import execution_allowed
    allowed, why = execution_allowed()
    if not allowed:
        return ("ERROR: Project test execution is not approved. Pytest imports arbitrary project code "
                "with your user permissions; it is NOT sandboxed. " + why)
    import importlib.util
    if importlib.util.find_spec("pytest") is None:
        return (f"ERROR: pytest is not installed for the Python Nexus runs on. Install it with: "
                f"\"{sys.executable}\" -m pip install pytest")
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts="]

    # Target path validation
    if path and path.strip():
        clean_path = path.strip()
        validated = _validate_workspace_path(clean_path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            return validated
        p = validated
        if not p.exists():
            return f"ERROR: Target test path does not exist: {clean_path}"
        cmd.append(str(p))

    # Safe args whitelist
    if args and args.strip():
        safe_flags = {"-v", "-q", "-x", "-s", "--disable-warnings", "--tb=short", "--tb=line", "--tb=no"}
        import shlex
        import re
        try:
            tokens = shlex.split(args)
        except Exception:
            tokens = args.split()

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in safe_flags:
                cmd.append(token)
                i += 1
            elif token == "-k" and i + 1 < len(tokens):
                k_val = tokens[i + 1]
                if re.match(r'^[a-zA-Z0-9_\s\(\)]+$', k_val):
                    cmd.extend(["-k", k_val])
                else:
                    return f"ERROR: Invalid or unsafe -k filter expression: '{k_val}'"
                i += 2
            elif token.startswith("-k="):
                if re.match(r'^-k=[a-zA-Z0-9_\s\(\)]+$', token):
                    cmd.append(token)
                else:
                    return f"ERROR: Invalid or unsafe -k filter expression: '{token}'"
                i += 1
            elif token.startswith("--maxfail="):
                val = token.split("=", 1)[1]
                if val.isdigit() and int(val) > 0:
                    cmd.append(token)
                else:
                    return f"ERROR: Invalid --maxfail value: '{val}'. Must be a positive integer."
                i += 1
            elif token == "--maxfail" and i + 1 < len(tokens):
                val = tokens[i + 1]
                if val.isdigit() and int(val) > 0:
                    cmd.extend(["--maxfail", val])
                else:
                    return f"ERROR: Invalid --maxfail value: '{val}'. Must be a positive integer."
                i += 2
            else:
                return f"ERROR: Unsupported or unsafe pytest argument: '{token}'. Allowed flags: -v, -q, -x, -s, -k <expr>, --disable-warnings, --tb=short, --maxfail=N."

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45.0,
            cwd=str(Path.cwd()),
            stdin=subprocess.DEVNULL,
            env={**{key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TMPDIR") if key in os.environ},
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        exit_code = proc.returncode

        combined = []
        if stdout:
            combined.append(stdout)
        if stderr:
            combined.append(stderr)
        full_output = "\n".join(combined)

        if len(full_output) > 3000:
            full_output = full_output[:1500] + "\n\n... [truncated] ...\n\n" + full_output[-1500:]

        status_str = "PASSED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
        return f"=== Pytest Results: {status_str} ===\n{full_output}"
    except subprocess.TimeoutExpired:
        return "ERROR: Pytest execution timed out after 45 seconds."
    except Exception as e:
        return f"ERROR: Failed to run pytest: {str(e)}"

RUN_TESTS_TOOL = Tool(
    name="run_tests",
    description="Run the pytest test suite in the workspace to verify fixes, test passes, or regressions. Optionally specify a test file/path or filter flags like -k.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional test file or directory path (e.g. 'tests/' or 'tests/test_tools.py'). Defaults to running the project test suite.",
            },
            "args": {
                "type": "string",
                "description": "Optional safe pytest flags (e.g. '-v', '-k test_name', '-x', '--tb=short').",
            },
        },
    },
    execute=execute_run_tests,
)

def get_all_tools() -> List[Tool]:
    """Return inspection/editing tools; project execution is opt-in."""
    tools = [
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        PATCH_FILE_TOOL,
        LIST_DIRECTORY_TOOL,
        RUN_CODE_TOOL,
        SEARCH_WEB_TOOL,
        GIT_STATUS_TOOL,
        GIT_DIFF_TOOL,
    ]
    from .consent import offer_run_tests
    if offer_run_tests():
        tools.append(RUN_TESTS_TOOL)
    if os.environ.get("NEXUS_ALLOW_GIT_COMMIT") == "1":
        tools.append(GIT_COMMIT_TOOL)
    return tools

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
