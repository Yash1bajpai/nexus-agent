"""Resumable model download using only the standard library.

Replaces huggingface_hub for the one thing nexus needs from it (fetching one GGUF file).
That keeps the install pure Python with no compiled dependencies (PyYAML etc.).
"""
from __future__ import annotations

import hashlib
import os
import sys
import re
import time
import urllib.error
import urllib.request
from pathlib import Path, PureWindowsPath
from typing import Callable, Optional

HF_BASE = os.environ.get("NEXUS_AGENT_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1024 * 256


def model_url(repo: str, filename: str) -> str:
    return f"{HF_BASE}/{repo}/resolve/main/{filename}"


def models_dir() -> Path:
    return Path(os.environ.get("NEXUS_AGENT_HOME", str(Path.home() / ".nexus-agent"))) / "models"


def local_path(repo: str, filename: str) -> Path:
    """Keep repository and nested HF filenames inside the configured model cache."""
    for value in (repo, filename):
        if (not isinstance(value, str) or not value or "\\" in value or ":" in value
                or "\0" in value or value.startswith("/") or PureWindowsPath(value).drive
                or any(p in ("", ".", "..") or p.endswith((" ", "."))
                       for p in value.split("/"))):
            raise ValueError("Repository and filename must be safe relative HF paths")
    if len(repo.split("/")) > 2:
        raise ValueError("Repository must be a name or owner/name")
    root = models_dir().resolve()
    repository = root / repo.replace("/", "--")
    candidate = repository / filename
    if (not repository.resolve().is_relative_to(root)
            or not candidate.resolve().is_relative_to(repository.resolve())):
        raise ValueError("Model path escapes its cache directory through a symlink")
    return candidate


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
    """Read optional size/SHA-256 from the first Hugging Face X-Linked-* headers.

    Missing metadata means only advertised HTTP framing can be checked; with no
    size or digest headers, download integrity remains best-effort.
    """
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
        resp.close()
        return (int(size) if size and size.isdigit() else None), sha
    except Exception:
        return None, None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _response_sizes(resp, start: int) -> tuple[Optional[int], Optional[int]]:
    """Validate range framing and return (full size, response body size)."""
    length = resp.headers.get("Content-Length")
    if length is not None and not length.isdigit():
        raise ValueError("Invalid Content-Length")
    body_size = int(length) if length is not None else None
    if getattr(resp, "status", 200) == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                             resp.headers.get("Content-Range", ""))
        if not match:
            raise ValueError("Missing or invalid Content-Range")
        first, last = int(match[1]), int(match[2])
        total = None if match[3] == "*" else int(match[3])
        if first != start or last < first or (total is not None and last >= total):
            raise ValueError("Content-Range does not match the requested bytes")
        range_size = last - first + 1
        if body_size is not None and body_size != range_size:
            raise ValueError("Content-Length does not match Content-Range")
        return total, range_size
    if resp.headers.get("Content-Range") is not None:
        raise ValueError("Unexpected Content-Range on a full response")
    return body_size, body_size


def _integrity_error(path: Path, size: Optional[int], sha: Optional[str]) -> Optional[str]:
    actual = path.stat().st_size
    if not actual:
        return "empty model file"
    if size is not None and actual != size:
        return f"size {actual} does not match expected {size}"
    if sha is not None and _sha256(path) != sha:
        return "SHA-256 mismatch"
    return None


def download(repo: str, filename: str, retries: int = 8, timeout: float = 30.0,
             progress: Optional[Callable[[int, Optional[int]], None]] = None,
             url: Optional[str] = None) -> str:
    """Resume/retry into a `.part` file, validating advertised sizes and SHA-256.

    Without digest headers, integrity is best-effort: HTTP size/range framing can
    detect truncation but cannot detect same-size corruption. KeyboardInterrupt
    propagates, preserving the partial file for the next invocation.
    """
    final = local_path(repo, filename)
    part = local_path(repo, filename + ".part")
    url = url or model_url(repo, filename)
    want_size, want_sha = expected_digest(url)
    if final.is_file():
        if _integrity_error(final, want_size, want_sha) is None:
            return str(final)
        final.unlink()
    final.parent.mkdir(parents=True, exist_ok=True)
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
                total, body_size = _response_sizes(resp, have)
                if (getattr(resp, "status", 200) == 206 and total is None
                        and want_size is None and want_sha is None):
                    raise ValueError("Cannot establish completion from an unknown range total")
                before = have
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if progress:
                            progress(have, total)
            if body_size is not None and have - before != body_size:
                raise OSError(f"incomplete response ({have - before} of {body_size} bytes)")
            if total is not None and have != total:
                raise OSError(f"incomplete download ({have} of {total} bytes)")
            error = _integrity_error(part, want_size, want_sha)
            if error:
                part.unlink(missing_ok=True)
                have = 0
                raise OSError(error + ", discarded the partial file")
            os.replace(part, final)
            return str(final)
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():
                # A rejected range is not proof of completion. Check every known
                # piece of metadata, including the server's unsatisfied-range size.
                match = re.fullmatch(r"bytes \*/(\d+)", e.headers.get("Content-Range", ""))
                range_size = int(match[1]) if match else None
                error = _integrity_error(part, want_size, want_sha)
                if error is None and range_size is not None and have != range_size:
                    error = f"partial size {have} does not match server size {range_size}"
                if error is None and (range_size is not None or want_size is not None or want_sha is not None):
                    os.replace(part, final)
                    e.close()
                    return str(final)
                part.unlink(missing_ok=True)
                have = 0
                last_err = OSError(error or "Cannot validate completion from HTTP 416")
            elif e.code in (401, 403, 404):
                e.close()
                raise RuntimeError(f"Download refused (HTTP {e.code}) for {url}. For a private or gated model, set HF_TOKEN to a Hugging Face access token that can read it.") from e
            else:
                last_err = e
            e.close()
        except ValueError as e:
            part.unlink(missing_ok=True)  # malformed ranges must restart, not append
            have = 0
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
