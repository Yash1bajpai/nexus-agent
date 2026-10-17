"""Resumable model download using only the standard library.

Replaces huggingface_hub for the one thing nexus needs from it (fetching one GGUF file).
That keeps the install pure Python with no compiled dependencies (PyYAML etc.).
"""
from __future__ import annotations

import hashlib
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
    nexus version stored in the Hugging Face cache (read directly from the cache folders)."""
    p = local_path(repo, filename)
    if p.is_file() and p.stat().st_size > 0:
        return str(p)
    return _find_in_hf_cache(repo, filename)


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _find_in_hf_cache(repo: str, filename: str) -> Optional[str]:
    """Look for the file in the Hugging Face cache layout used by older nexus versions
    (models--org--name/snapshots/<rev>/file). Reads the folders directly; needs no extra package."""
    try:
        snaps = _hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
        for rev in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            cand = rev / filename
            if cand.is_file() and cand.stat().st_size > 0:
                return os.path.realpath(cand)
    except Exception:
        pass
    return None


def _hf_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k, "").strip():
            return os.environ[k].strip()
    try:
        home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        t = (Path(home) / "token").read_text().strip()
        return t or None
    except Exception:
        return None


class _KeepAuthOnlyOnSameHost(urllib.request.HTTPRedirectHandler):
    """Follow redirects (Hugging Face sends files from a CDN) but never forward the token to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            from urllib.parse import urlsplit
            if urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
                for h in ("Authorization", "authorization"):
                    new.headers.pop(h, None)
                    new.unredirected_hdrs.pop(h, None)
        return new


def _opener():
    return urllib.request.build_opener(_KeepAuthOnlyOnSameHost)


def _auth_headers(url: str) -> dict:
    """Token only for Hugging Face hosts (or the configured NEXUS_AGENT_HF_ENDPOINT), never a random mirror."""
    from urllib.parse import urlsplit
    host = urlsplit(url).netloc
    allowed = {urlsplit(HF_BASE).netloc, "huggingface.co"}
    tok = _hf_token()
    return {"Authorization": f"Bearer {tok}"} if tok and host in allowed else {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def expected_digest(url: str, timeout: float = 15.0) -> "tuple[Optional[int], Optional[str]]":
    """Best-effort (size, sha256) of the real file from Hugging Face's X-Linked-* headers on the first
    response. (None, None) for hosts that do not send them; then only the size check from the download applies."""
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "nexus-agent-ai", **_auth_headers(url)})
        try:
            resp = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:  # the 302 carries the headers
            resp = e
        size = resp.headers.get("X-Linked-Size")
        etag = (resp.headers.get("X-Linked-ETag") or "").strip('"').lower()
        sha = etag if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag) else None
        return (int(size) if size and size.isdigit() else None), sha
    except Exception:
        return None, None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    want_size, want_sha = expected_digest(url)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "nexus-agent-ai", **_auth_headers(url)})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with _opener().open(req, timeout=timeout) as resp:
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
            if want_size is not None and have != want_size:
                raise OSError(f"size {have} does not match expected {want_size}")
            if want_sha is not None and _sha256(part) != want_sha:
                part.unlink(missing_ok=True)  # corrupt: do not resume from bad bytes
                have = 0
                raise OSError("SHA-256 mismatch, discarded the partial file")
            os.replace(part, final)
            return str(final)
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():  # already complete
                os.replace(part, final)
                return str(final)
            if e.code in (401, 403, 404):
                raise RuntimeError(f"Download refused (HTTP {e.code}) for {url}. For a private or gated model, set HF_TOKEN to a Hugging Face access token that can read it.") from e
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
