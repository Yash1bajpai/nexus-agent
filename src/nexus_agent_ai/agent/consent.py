"""Explicit, per-folder permission to run the project's tests (pytest runs the project's code, unsandboxed).

Three ways to be allowed, none automatic:
  - NEXUS_ALLOW_PROJECT_EXECUTION=1 in the environment (existing behaviour, applies everywhere),
  - the user answers "once" at the prompt (this folder, until the program exits, nothing saved),
  - the user answers "always" or runs `nexus-agent trust add` (this exact folder, saved until revoked).
Nothing is ever allowed without a person answering or setting the variable. A non-interactive run is refused.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

_session_allowed: set = set()
_session_denied: set = set()
_ask: Callable[[str], str] = input  # replaced in tests


def trust_file() -> Path:
    return Path(os.environ.get("NEXUS_AGENT_HOME", str(Path.home() / ".nexus-agent"))) / "trusted_folders.json"


def folder_key(path: "str | Path | None" = None) -> str:
    return os.path.normcase(str(Path(path or Path.cwd()).resolve()))


def list_trusted() -> List[str]:
    try:
        data = json.loads(trust_file().read_text(encoding="utf-8"))
        return [str(x) for x in data.get("folders", [])]
    except Exception:
        return []


def _save(folders: List[str]) -> None:
    f = trust_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps({"folders": sorted(set(folders))}, indent=2), encoding="utf-8")
    os.replace(tmp, f)


def trust(path: "str | Path | None" = None) -> str:
    key = folder_key(path)
    _save(list_trusted() + [key])
    return key


def revoke(path: "str | Path | None" = None, all_folders: bool = False) -> int:
    """Remove one folder (or all). Returns how many entries were removed."""
    _session_allowed.clear() if all_folders else _session_allowed.discard(folder_key(path))
    cur = list_trusted()
    if all_folders:
        _save([])
        return len(cur)
    key = folder_key(path)
    new = [c for c in cur if c != key]
    _save(new)
    return len(cur) - len(new)


def _interactive() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def execution_allowed_without_asking() -> bool:
    """True only if a person already granted it (env variable, 'once' this session, or a saved folder). Never prompts."""
    return (os.environ.get("NEXUS_ALLOW_PROJECT_EXECUTION") == "1"
            or folder_key() in _session_allowed or folder_key() in list_trusted())


def auto_check_enabled() -> bool:
    """Automatic test run after each successful edit: needs BOTH an explicit opt-in and an existing grant."""
    return os.environ.get("NEXUS_AGENT_AUTO_CHECK") == "1" and execution_allowed_without_asking()


def execution_allowed() -> Tuple[bool, str]:
    if os.environ.get("NEXUS_ALLOW_PROJECT_EXECUTION") == "1":
        return True, "environment variable"
    key = folder_key()
    if key in _session_allowed:
        return True, "allowed once for this session"
    if key in list_trusted():
        return True, "remembered for this folder"
    if key in _session_denied:
        return False, "You declined to run tests in this folder in this session."
    if not _interactive():
        return False, (f"Running tests executes this project's code and is not approved for {key}. "
                       "In a terminal, Nexus asks you. Otherwise run `nexus-agent trust add` in this folder, "
                       "or set NEXUS_ALLOW_PROJECT_EXECUTION=1 for trusted code.")
    print(f"\nNexus wants to run pytest in:\n  {key}\n"
          "This runs the project's own code on your computer, with your permissions, not sandboxed.\n"
          "  [o] once  - this folder, until Nexus exits (nothing saved)\n"
          "  [a] always - this exact folder only, saved; undo with `nexus-agent trust revoke`\n"
          "  [N] no")
    try:
        ans = _ask("Allow? [o/a/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans in ("o", "once"):
        _session_allowed.add(key)
        return True, "allowed once"
    if ans in ("a", "always"):
        trust(key)
        return True, "remembered"
    _session_denied.add(key)
    return False, "You declined to run tests in this folder."


def offer_run_tests() -> bool:
    """Should the model be told run_tests exists? Yes if the variable is set, this folder is remembered, or a
    person is at the terminal (they get asked when it is called). Never in a non-interactive run."""
    if os.environ.get("NEXUS_ALLOW_PROJECT_EXECUTION") == "1":
        return True
    return folder_key() in list_trusted() or _interactive()
