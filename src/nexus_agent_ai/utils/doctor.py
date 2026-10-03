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
    if shutil.which("nexus-agent"):
        checks.append(("ok", "Command", "nexus-agent is on PATH"))
    else:
        scheme = "nt_user" if os.name == "nt" else "posix_user"
        try:
            scripts = sysconfig.get_path("scripts", scheme)
        except Exception:
            scripts = "your Python Scripts folder"
        checks.append(("warn", "Command", f"nexus-agent is not on PATH. Run it as: python -m nexus_agent_ai   (or add {scripts} to PATH)"))

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
