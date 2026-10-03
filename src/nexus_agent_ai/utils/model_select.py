"""Pick a local model that fits the machine and confirm big downloads first.

Sizing uses *available* RAM (not total) and keeps headroom for the OS and the
context window. Nothing here downloads anything by itself.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

SMALL_MODEL = {
    "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
    "filename": "LFM2.5-1.2B-Instruct-Q4_0.gguf",
    "size_gb": 0.7,
    "label": "1.2B small",
}

# Largest first. size_gb is the approximate file size (RAM use is about the same).
BIG_MODELS: List[Dict] = [
    {"repo": "LiquidAI/LFM2.5-2.6B-GGUF", "filename": "LFM2.5-2.6B-Q6_K.gguf", "size_gb": 2.2, "label": "2.6B large"},
    {"repo": "LiquidAI/LFM2.5-2.6B-GGUF", "filename": "LFM2.5-2.6B-Q5_K_M.gguf", "size_gb": 1.9, "label": "2.6B medium"},
    {"repo": "LiquidAI/LFM2.5-2.6B-GGUF", "filename": "LFM2.5-2.6B-Q4_0.gguf", "size_gb": 1.5, "label": "2.6B compact"},
]

HEADROOM_GB = 1.0          # kept free for the OS and other programs
CONTEXT_GB_PER_4K = 0.4    # rough KV-cache + buffers cost per 4096 tokens


def available_ram_gb() -> Optional[float]:
    """Free RAM in GB right now, or None when it cannot be measured."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 2)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullAvailPhys / (1024 ** 3), 2)
        except Exception:
            pass
    return None


def free_disk_gb(path: Optional[Path] = None) -> Optional[float]:
    import shutil
    target = Path(path) if path else Path.home()
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return round(shutil.disk_usage(target).free / (1024 ** 3), 2)
    except Exception:
        return None


def required_ram_gb(model: Dict, context_size: int = 4096) -> float:
    return round(model["size_gb"] + CONTEXT_GB_PER_4K * (context_size / 4096) + HEADROOM_GB, 2)


def pick_model(available_gb: Optional[float], context_size: int = 4096, android: bool = False) -> Dict:
    """Return the biggest model that fits with headroom, else the small one.

    The result carries a one-line "reason" explaining the choice.
    """
    if available_gb is None:
        choice = dict(SMALL_MODEL)
        choice["reason"] = "Could not measure free RAM, so the small model was chosen to be safe."
        return choice
    candidates = BIG_MODELS[1:] if android else BIG_MODELS  # no Q6_K on phones
    for model in candidates:
        need = required_ram_gb(model, context_size)
        if available_gb >= need:
            choice = dict(model)
            choice["reason"] = f"{available_gb:.1f} GB RAM free; {model['label']} needs about {need:.1f} GB with headroom."
            return choice
    smallest_big = required_ram_gb(candidates[-1], context_size)
    choice = dict(SMALL_MODEL)
    choice["reason"] = (
        f"Only {available_gb:.1f} GB RAM free; the next model up needs about {smallest_big:.1f} GB "
        "with headroom, so the small model was chosen."
    )
    return choice


def is_cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
        return os.path.isfile(path)
    except Exception:
        return False


def probe_speed_mbps(repo: str, filename: str, timeout: float = 6.0, probe_bytes: int = 1_000_000) -> Optional[float]:
    """Measure real download speed (MB/s) by fetching a small range of the model file."""
    try:
        import urllib.request
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(repo_id=repo, filename=filename)
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{probe_bytes - 1}"})
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(probe_bytes)
        elapsed = max(time.time() - start, 1e-3)
        return len(data) / elapsed / 1_000_000
    except Exception:
        return None


def format_eta(size_gb: float, speed_mbps: Optional[float]) -> str:
    if not speed_mbps or speed_mbps <= 0:
        return "speed unknown"
    seconds = size_gb * 1000 / speed_mbps
    if seconds < 90:
        eta = f"{int(seconds)} sec"
    elif seconds < 5400:
        eta = f"{int(round(seconds / 60))} min"
    else:
        eta = f"{seconds / 3600:.1f} h"
    return f"{speed_mbps:.1f} MB/s, about {eta}"


def confirm_download(
    choice: Dict,
    context_size: int = 4096,
    *,
    ask: Optional[Callable[[str], str]] = None,
    interactive: Optional[bool] = None,
    speed_probe: Callable[..., Optional[float]] = probe_speed_mbps,
    out: Callable[[str], None] = print,
) -> Optional[Dict]:
    """Show size, speed and the small/large options, then ask before downloading.

    Returns the model dict to download, or None if the user declined.
    NEXUS_AGENT_ASSUME_YES=1 or a non-interactive terminal skips the question
    (the size line is still printed).
    """
    if interactive is None:
        interactive = sys.stdin.isatty() and os.environ.get("NEXUS_AGENT_ASSUME_YES", "") not in ("1", "true", "yes")
    speed = speed_probe(choice["repo"], choice["filename"])
    out(f"Model not downloaded yet: {choice['label']} ({choice['filename']}, ~{choice['size_gb']} GB, {format_eta(choice['size_gb'], speed)}).")
    out(f"  {choice['reason']}")
    out("  The download resumes if interrupted; run the same command again.")
    disk = free_disk_gb()
    if disk is not None and disk < choice["size_gb"] + 0.5:
        out(f"  Warning: only {disk:.1f} GB free disk space.")
    if not interactive:
        return choice
    ask = ask or input
    options = {"s": SMALL_MODEL}
    prompt = "  Download [y]es"
    if choice["filename"] != SMALL_MODEL["filename"]:
        prompt += f" / [s]mall instead (~{SMALL_MODEL['size_gb']} GB, uses less RAM)"
    prompt += " / [n]o: "
    try:
        answer = ask(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if answer in ("", "y", "yes"):
        return choice
    if answer in options and choice["filename"] != SMALL_MODEL["filename"]:
        small = dict(SMALL_MODEL)
        small["reason"] = "Small model chosen by you."
        return small
    return None
