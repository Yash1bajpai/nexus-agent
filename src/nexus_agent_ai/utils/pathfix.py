"""Optional, consented fix for "nexus-agent-ai: command not found".

pip installs the command into a Scripts/bin folder but never edits PATH. When that folder is not on PATH the bare
command fails, while `python -m nexus_agent_ai` always works. This module finds the folder and, only when the user
says yes, adds it to their USER PATH (Windows: HKCU\\Environment; macOS/Linux: one marked line in a shell profile).
Nothing here runs without an explicit yes, and it never touches system-wide settings.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

MARK = "# added by nexus-agent-ai setup-path"


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expandvars(p.strip().strip('"')))) if p.strip() else ""


def _path_entries(path_value: Optional[str] = None) -> List[str]:
    value = os.environ.get("PATH", "") if path_value is None else path_value
    return [_norm(p) for p in value.split(os.pathsep) if p.strip()]


def install_script_dir() -> Optional[str]:
    """The folder holding the nexus-agent-ai script for this Python, or None when it cannot be found."""
    from .doctor import _script_dirs
    exe = "nexus-agent-ai.exe" if os.name == "nt" else "nexus-agent-ai"
    return next((d for d in _script_dirs() if (Path(d) / exe).exists()), None)


def folder_missing_from_path(path_value: Optional[str] = None) -> Optional[str]:
    """The script folder when it exists and is not on PATH; None when the bare command should already work."""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return None  # inside a virtual environment: the user activates it; never edit PATH (or the registry) for a venv
    d = install_script_dir()
    if d and _norm(d) not in _path_entries(path_value):
        return d
    return None


def _profile_for_shell() -> Path:
    shell = os.path.basename(os.environ.get("SHELL", ""))  # fish and others fall back to ~/.profile (read by login shells)
    home = Path.home()
    if shell == "zsh":
        return home / ".zshrc"
    if shell == "bash":
        return home / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    return home / ".profile"


def _add_posix(folder: str) -> str:
    profile = _profile_for_shell()
    line = f'export PATH="{folder}:$PATH"'
    existing = profile.read_text(encoding="utf-8", errors="replace") if profile.exists() else ""
    if folder in existing:
        return f"{profile} already mentions {folder}. Open a new terminal."
    with open(profile, "a", encoding="utf-8") as fh:
        fh.write(("\n" if existing and not existing.endswith("\n") else "") + f"{MARK}\n{line}\n")
    return f"Added to {profile}:\n    {line}\nOpen a new terminal (or run: source {profile}) and nexus-agent-ai will work from any folder.\nTo undo: delete those two lines (the '{MARK}' comment and the export line) from {profile}."


def _add_windows(folder: str) -> str:
    import winreg  # type: ignore
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, kind = "", winreg.REG_EXPAND_SZ
        if _norm(folder) in [_norm(p) for p in current.split(";") if p.strip()]:
            return "Your user PATH already contains that folder. Open a new terminal."
        new = (current.rstrip(";") + ";" if current else "") + folder
        winreg.SetValueEx(key, "Path", 0, kind if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) else winreg.REG_EXPAND_SZ, new)
    try:  # tell running programs (new terminals) that the environment changed
        import ctypes
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 3000, None)
    except Exception:
        pass
    return f"Added {folder} to your user PATH. Open a NEW terminal window and nexus-agent-ai will work from any folder.\nTo undo: Settings > System > About > Advanced system settings > Environment Variables > user Path, and delete that entry."


def add_to_user_path(folder: str) -> str:
    return _add_windows(folder) if os.name == "nt" else _add_posix(folder)


def explain(folder: str) -> str:
    where = "your user PATH (a registry setting for your account only)" if os.name == "nt" else f"your shell profile ({_profile_for_shell()})"
    return (f"The command is installed in {folder}, which is not on PATH, so typing nexus-agent-ai in a new terminal would fail. "
            f"I can add that folder to {where}. Nothing else is changed.")


def offer(ask=input, say=print) -> bool:
    """Ask once. Returns True when PATH was changed. Default is no."""
    folder = folder_missing_from_path()
    if not folder:
        return False
    say("")
    say(explain(folder))
    try:
        answer = ask("  Add it now? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer not in ("y", "yes"):
        say("  Skipped. You can run it as: python -m nexus_agent_ai   (or later: python -m nexus_agent_ai setup-path)")
        return False
    try:
        say("  " + add_to_user_path(folder).replace("\n", "\n  "))
        return True
    except Exception as e:  # never break setup over this
        say(f"  Could not change PATH ({e}). Run it as: python -m nexus_agent_ai")
        return False
