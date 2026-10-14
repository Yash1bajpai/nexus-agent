"""Resumable stdlib downloader, tested against a real local HTTP server (supports Range)."""
import http.server
import threading

import pytest

from nexus_agent_ai.utils import model_download

DATA = bytes(range(256)) * 4000  # ~1 MB


class _H(http.server.BaseHTTPRequestHandler):
    drop_first = {"n": 0}

    def log_message(self, *a):
        pass

    ignore_range = False
    sha = None

    def do_HEAD(self):
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:1/x")
        if _H.sha:
            self.send_header("X-Linked-ETag", '"' + _H.sha + '"')
            self.send_header("X-Linked-Size", str(len(DATA)))
        self.end_headers()

    def do_GET(self):
        rng = None if _H.ignore_range else self.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = DATA[start:]
        self.send_response(206 if rng else 200)
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(DATA)-1}/{len(DATA)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _H.drop_first["n"] > 0 and not rng:
            _H.drop_first["n"] -= 1
            self.wfile.write(body[: len(body) // 3])  # cut the connection mid-file
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/m.gguf"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset():
    _H.ignore_range, _H.sha = False, None
    yield


def test_server_ignoring_range_restarts_cleanly(server):
    _H.drop_first["n"] = 0
    _H.ignore_range = True
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name("m.gguf.part").write_bytes(b"garbage" * 100)
    p = model_download.download("a/b", "m.gguf", url=server)
    assert open(p, "rb").read() == DATA


def test_sha256_checked_when_advertised(server):
    import hashlib
    _H.drop_first["n"] = 0
    _H.sha = hashlib.sha256(DATA).hexdigest()
    p = model_download.download("a/b", "m.gguf", url=server)
    assert open(p, "rb").read() == DATA


def test_sha256_mismatch_is_rejected(server):
    _H.drop_first["n"] = 0
    _H.sha = "0" * 64
    with pytest.raises(RuntimeError):
        model_download.download("a/b", "m.gguf", url=server, retries=2)
    assert model_download.find_cached("a/b", "m.gguf") is None


def test_plain_download(server):
    _H.drop_first["n"] = 0
    p = model_download.download("a/b", "m.gguf", url=server)
    assert open(p, "rb").read() == DATA
    assert model_download.find_cached("a/b", "m.gguf") == p


def test_resumes_after_dropped_connection(server):
    _H.drop_first["n"] = 1
    p = model_download.download("a/b", "m.gguf", url=server, retries=3)
    assert open(p, "rb").read() == DATA


def test_resumes_existing_part_file(server):
    _H.drop_first["n"] = 0
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name("m.gguf.part").write_bytes(DATA[:1000])
    p = model_download.download("a/b", "m.gguf", url=server)
    assert open(p, "rb").read() == DATA
