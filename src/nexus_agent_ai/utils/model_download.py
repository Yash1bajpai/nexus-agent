"""Resumable model download using only the standard library.

Replaces huggingface_hub for the one thing nexus needs from it (fetching one GGUF file).
That keeps the install pure Python with no compiled dependencies (PyYAML etc.).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

HF_BASE = os.environ.get("NEXUS_AGENT_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1024 * 256


def model_url(repo: str, filename: str) -> str:
    return f"{HF_BASE}/{repo}/resolve/main/{filename}"


def models_dir() -> Path:
    return Path(os.environ.get("NEXUS_AGENT_HOME", str(Path.home() / ".nexus-agent"))) / "models"


def local_path(repo: str, filename: str) -> Path:
    return models_dir() / repo.replace("/", "--") / filename


def find_cached(repo: str, filename: str) -> Optional[str]:
    """Return the path of an already-downloaded model, or None. Also finds files an older
    nexus version stored in the Hugging Face cache (only if huggingface_hub happens to be installed)."""
    p = local_path(repo, filename)
    if p.is_file() and p.stat().st_size > 0:
        return str(p)
    try:
        from huggingface_hub import hf_hub_download  # optional, legacy cache only
        legacy = hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
        if os.path.isfile(legacy):
            return os.path.realpath(legacy)
    except Exception:
        pass
    return None


def _total_size(resp, start: int) -> Optional[int]:
    cr = resp.headers.get("Content-Range")  # "bytes 100-999/1000"
    if cr and "/" in cr:
        try:
            return int(cr.rsplit("/", 1)[1])
        except ValueError:
            return None
    cl = resp.headers.get("Content-Length")
    return int(cl) + start if cl and cl.isdigit() else None


def download(repo: str, filename: str, retries: int = 8, timeout: float = 30.0,
             progress: Optional[Callable[[int, Optional[int]], None]] = None,
             url: Optional[str] = None) -> str:
    """Download to ~/.nexus-agent/models/..., resuming a partial `.part` file and retrying on
    timeouts or dropped connections. Returns the final path. Raises RuntimeError after `retries` failures."""
    final = local_path(repo, filename)
    if final.is_file() and final.stat().st_size > 0:
        return str(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    url = url or model_url(repo, filename)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "nexus-agent-ai"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if have and getattr(resp, "status", 200) != 206:
                    have = 0  # server ignored Range; restart
                total = _total_size(resp, have)
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if progress:
                            progress(have, total)
            if total is not None and have != total:
                raise OSError(f"incomplete download ({have} of {total} bytes)")
            os.replace(part, final)
            return str(final)
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # already complete
                os.replace(part, final)
                return str(final)
            if e.code in (401, 403, 404):
                raise RuntimeError(f"Download refused (HTTP {e.code}) for {url}") from e
            last_err = e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
        if attempt < retries:
            print(f"Download interrupted ({last_err}); resuming from {have // 1_000_000} MB "
                  f"(try {attempt + 1}/{retries})...", file=sys.stderr)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"Download failed after {retries} tries: {last_err}. "
                       "Run the same command again to resume from the saved partial file.")


def console_progress():
    state = {"last": 0.0}

    def cb(have: int, total: Optional[int]) -> None:
        now = time.time()
        if now - state["last"] < 1.0 and not (total and have >= total):
            return
        state["last"] = now
        if total:
            sys.stderr.write(f"\r  {have / 1e6:,.0f} / {total / 1e6:,.0f} MB ({100 * have / total:.0f}%)   ")
        else:
            sys.stderr.write(f"\r  {have / 1e6:,.0f} MB   ")
        sys.stderr.flush()
    return cb
