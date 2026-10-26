"""Interrupted engine download, private-repo token handling, old Hugging Face cache discovery."""
import hashlib
import http.server
import io
import os
import threading
import urllib.request
import zipfile

import pytest

from nexus_agent_ai.providers import local_provider as lp
from nexus_agent_ai.utils import model_download as md


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("llama-server", "#!/bin/sh\n")
        z.writestr("llama-server.exe", "MZ")
    return buf.getvalue()


def test_interrupted_engine_download_leaves_nothing_and_next_run_recovers(tmp_path, monkeypatch):
    data = _zip_bytes()
    monkeypatch.setattr(lp, "_SERVER_DIR", tmp_path / "srv")
    monkeypatch.setenv("NEXUS_AGENT_LLAMA_SERVER_SHA256", hashlib.sha256(data).hexdigest())
    calls = {"n": 0}

    def fake_retrieve(url, filename):
        calls["n"] += 1
        with open(filename, "wb") as f:
            if calls["n"] == 1:
                f.write(data[: len(data) // 2])
                raise KeyboardInterrupt  # hard stop (Ctrl+C / killed): no cleanup code runs for it
            f.write(data)

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    prov = lp.LocalProvider()
    with pytest.raises(KeyboardInterrupt):
        prov._download_llama_server()
    srv = tmp_path / "srv"
    assert not (srv / "llama-server.zip").exists()          # the old bug: half a zip under the final name
    exe = prov._download_llama_server()                       # next launch downloads fresh and verifies
    assert calls["n"] == 2 and os.path.isfile(exe)


def test_legacy_hf_cache_found_without_huggingface_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    snap = tmp_path / "hub" / "models--Org--Repo" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    blob = tmp_path / "hub" / "models--Org--Repo" / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"gguf-bytes")
    try:
        os.symlink(blob, snap / "m.gguf")
    except OSError:
        (snap / "m.gguf").write_bytes(b"gguf-bytes")
    found = md.find_cached("Org/Repo", "m.gguf")
    assert found and open(found, "rb").read() == b"gguf-bytes"
    assert not md.local_path("Org/Repo", "m.gguf").exists()  # legacy discovery does not copy the model
    assert md.find_cached("Org/Other", "m.gguf") is None


class _Rec(http.server.BaseHTTPRequestHandler):
    seen = {}
    redirect_to = None
    body = b"secret-model-bytes" * 100
    require_auth = False
    seen_ranges = {}

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        _Rec.seen[self.server.server_port] = self.headers.get("Authorization")
        _Rec.seen_ranges[self.server.server_port] = self.headers.get("Range")
        if (_Rec.require_auth and self.server.server_port == _Rec.first
                and self.headers.get("Authorization") != "Bearer hf_testtoken"):
            self.send_response(401)
            self.end_headers()
            return
        if self.server.server_port == _Rec.first and _Rec.redirect_to:
            self.send_response(302)
            self.send_header("Location", _Rec.redirect_to)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(_Rec.body)))
        self.end_headers()
        self.wfile.write(_Rec.body)


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Rec)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_token_sent_to_hf_host_but_not_forwarded_to_cdn(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken")
    a, b = _serve(), _serve()
    try:
        _Rec.seen.clear()
        _Rec.first = a.server_port
        _Rec.redirect_to = f"http://127.0.0.1:{b.server_port}/file"
        monkeypatch.setattr(md, "HF_BASE", f"http://127.0.0.1:{a.server_port}")
        p = md.download("o/r", "m.gguf", url=f"http://127.0.0.1:{a.server_port}/o/r/resolve/main/m.gguf", retries=1)
        assert open(p, "rb").read() == _Rec.body
        assert _Rec.seen[a.server_port] == "Bearer hf_testtoken"
        assert _Rec.seen[b.server_port] is None            # CDN host never sees the token
    finally:
        a.shutdown(); b.shutdown()


def test_token_not_sent_to_unrelated_host(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken")
    a = _serve()
    try:
        _Rec.seen.clear()
        _Rec.first = a.server_port
        _Rec.redirect_to = None
        monkeypatch.setattr(md, "HF_BASE", "https://huggingface.co")
        md.download("o/r", "m.gguf", url=f"http://127.0.0.1:{a.server_port}/x", retries=1)
        assert _Rec.seen[a.server_port] is None
    finally:
        a.shutdown()


@pytest.mark.parametrize("redirect", [False, True])
def test_speed_probe_authenticates_and_isolates_redirect_token(monkeypatch, redirect):
    from nexus_agent_ai.utils import model_select
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken")
    a, b = _serve(), _serve()
    try:
        _Rec.seen.clear()
        _Rec.seen_ranges.clear()
        _Rec.first = a.server_port
        _Rec.require_auth = True
        _Rec.redirect_to = f"http://127.0.0.1:{b.server_port}/file" if redirect else None
        monkeypatch.setattr(md, "HF_BASE", f"http://127.0.0.1:{a.server_port}")
        speed = model_select.probe_speed_mbps("private/model", "nested/m.gguf", probe_bytes=100)
        assert speed is not None and speed > 0
        assert _Rec.seen[a.server_port] == "Bearer hf_testtoken"
        assert _Rec.seen_ranges[a.server_port] == "bytes=0-99"
        if redirect:
            assert _Rec.seen[b.server_port] is None
            assert _Rec.seen_ranges[b.server_port] == "bytes=0-99"
        else:
            assert b.server_port not in _Rec.seen
    finally:
        _Rec.require_auth = False
        _Rec.redirect_to = None
        a.shutdown()
        b.shutdown()
