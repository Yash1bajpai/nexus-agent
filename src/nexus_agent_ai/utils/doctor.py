"""System checks behind `nexus-agent doctor`. Nothing here downloads a model."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import List, Tuple

from . import model_select

Check = Tuple[str, str, str]  # (status "ok" | "warn" | "fail" | "info", label, detail)


def run_checks(context_size: int = 4096, probe_network: bool = True) -> List[Check]:
    checks: List[Check] = []

    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        checks.append(("ok", "Python", f"{ver} on {platform.system()} {platform.machine()}"))
    else:
        checks.append(("fail", "Python", f"{ver}; nexus-agent needs 3.11 or newer"))

    total = None
    try:
        import psutil
        total = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    avail = model_select.available_ram_gb()
    if avail is None:
        checks.append(("warn", "RAM", "could not measure free RAM"))
    else:
        total_txt = f" of {total} GB" if total else ""
        status = "ok" if avail >= 3 else "warn"
        checks.append((status, "RAM", f"{avail} GB free{total_txt}"))

    disk = model_select.free_disk_gb()
    if disk is None:
        checks.append(("warn", "Disk", "could not measure free space"))
    else:
        checks.append(("ok" if disk >= 3 else "warn", "Disk", f"{disk} GB free in your home folder"))

    choice = model_select.pick_model(avail, context_size)
    cached = model_select.is_cached(choice["repo"], choice["filename"])
    state = "downloaded" if cached else "not downloaded yet"
    checks.append(("ok", "Model fit", f"{choice['label']} ({choice['filename']}, ~{choice['size_gb']} GB, {state}). {choice['reason']}"))

    if os.getenv("NEXUS_AGENT_MODEL_REPO") or os.getenv("NEXUS_AGENT_MODEL_FILENAME"):
        checks.append(("info", "Model override", f"{os.getenv('NEXUS_AGENT_MODEL_REPO', '(default repo)')} / {os.getenv('NEXUS_AGENT_MODEL_FILENAME', '(default file)')}"))

    try:
        from ..providers.local_provider import _SERVER_DIR
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        found = any(Path(_SERVER_DIR).rglob(exe)) if Path(_SERVER_DIR).exists() else False
        checks.append(("ok" if found else "info", "Inference engine", "installed" if found else "not installed yet (downloads ~50 MB on first local run)"))
    except Exception:
        checks.append(("info", "Inference engine", "status unknown"))

    if probe_network:
        speed = model_select.probe_speed_mbps(choice["repo"], choice["filename"])
        if speed is None:
            checks.append(("warn", "Network", "could not reach huggingface.co (blocked or offline)"))
        else:
            checks.append(("ok" if speed >= 0.5 else "warn", "Network", model_select.format_eta(choice["size_gb"], speed) + " for the chosen model"))

    import shutil
    import sysconfig
    if shutil.which("nexus-agent-ai"):
        checks.append(("ok", "Command", "nexus-agent-ai is on PATH"))
    else:
        checks.append(("warn", "Command", _path_advice()))

    keys = [k for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY") if os.getenv(k)]
    checks.append(("info", "Cloud keys", ", ".join(keys) if keys else "none set (local mode works without them)"))

    env_file = Path.home() / ".nexus-agent" / ".env"
    if env_file.exists() and os.name != "nt":
        mode = env_file.stat().st_mode & 0o777
        if mode & 0o077:
            checks.append(("warn", "Key file", f"{env_file} is readable by other users (mode {oct(mode)}). Fix: chmod 600 {env_file}"))
        else:
            checks.append(("ok", "Key file", f"{env_file} is private (mode {oct(mode)})"))

    home = Path.home() / ".nexus-agent"
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-test"
        probe.write_text("x")
        probe.unlink()
        checks.append(("ok", "Data folder", f"{home} is writable"))
    except Exception as exc:
        checks.append(("fail", "Data folder", f"{home} is not writable ({exc})"))
    return checks


def _script_dirs():
    """Folders where pip may have put the console script for this Python (environment, then user install)."""
    import sysconfig
    dirs = []
    for scheme in (None, "nt_user" if os.name == "nt" else "posix_user"):
        try:
            d = sysconfig.get_path("scripts", scheme) if scheme else sysconfig.get_path("scripts")
        except Exception:
            d = None
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def _path_advice() -> str:
    """Why the bare command is not found and the exact, copy-pasteable fix. pip cannot edit PATH itself."""
    exe = "nexus-agent-ai.exe" if os.name == "nt" else "nexus-agent-ai"
    where = next((d for d in _script_dirs() if (Path(d) / exe).exists()), None)
    run_as = "python -m nexus_agent_ai"
    if where is None:
        return f"nexus-agent-ai was not found on PATH or in this Python's script folders. Run it as: {run_as}"
    if os.name == "nt":
        fix = f'add {where} to your user PATH (Settings > System > About > Advanced system settings > Environment Variables), then open a new terminal'
    else:
        fix = f'add this line to your shell profile (~/.profile, ~/.zshrc or ~/.bashrc): export PATH="{where}:$PATH"'
    return f"installed in {where}, which is not on PATH (pip does not change PATH). Run it as: {run_as}  or run: {run_as} setup-path (asks first, then adds the folder to your user PATH)  or {fix}"
