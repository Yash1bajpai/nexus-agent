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
    advertise_size = False
    range_mode = None
    seen_ranges = []

    def do_HEAD(self):
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:1/x")
        if _H.sha:
            self.send_header("X-Linked-ETag", '"' + _H.sha + '"')
        if _H.sha or _H.advertise_size:
            self.send_header("X-Linked-Size", str(len(DATA)))
        self.end_headers()

    def do_GET(self):
        rng = None if _H.ignore_range else self.headers.get("Range")
        _H.seen_ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        if rng and start >= len(DATA):
            self.send_response(416)
            if _H.range_mode != "missing":
                self.send_header("Content-Range", f"bytes */{len(DATA)}")
            self.end_headers()
            return
        body = DATA[start:]
        self.send_response(206 if rng else 200)
        if rng:
            first = start + 1 if _H.range_mode == "wrong-start" else start
            self.send_header("Content-Range", f"bytes {first}-{len(DATA)-1}/{len(DATA)}")
        length = len(body) + 1 if rng and _H.range_mode == "wrong-length" else len(body)
        self.send_header("Content-Length", str(length))
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
    _H.ignore_range, _H.sha, _H.advertise_size, _H.range_mode = False, None, False, None
    _H.drop_first["n"] = 0
    _H.seen_ranges = []
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


@pytest.mark.parametrize("partial", [b"x" * len(DATA), DATA + b"extra"], ids=["same-size-corrupt", "oversized"])
def test_invalid_partial_416_never_promoted(server, partial):
    import hashlib
    _H.sha = hashlib.sha256(DATA).hexdigest()
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    part = final.with_name(final.name + ".part")
    part.write_bytes(partial)
    with pytest.raises(RuntimeError):
        model_download.download("a/b", "m.gguf", url=server, retries=1)
    assert not final.exists()
    assert not part.exists()
    assert model_download.find_cached("a/b", "m.gguf") is None


@pytest.mark.parametrize("partial", [b"x" * len(DATA), DATA + b"extra"], ids=["same-size-corrupt", "oversized"])
def test_invalid_partial_416_restarts_and_recovers(server, monkeypatch, partial):
    import hashlib
    monkeypatch.setattr(model_download.time, "sleep", lambda _: None)
    _H.sha = hashlib.sha256(DATA).hexdigest()
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".part").write_bytes(partial)
    path = model_download.download("a/b", "m.gguf", url=server, retries=2)
    assert final.read_bytes() == DATA
    assert path == str(final)
    assert _H.seen_ranges == [f"bytes={len(partial)}-", None]


@pytest.mark.parametrize("range_mode", [None, "missing"])
def test_valid_completed_partial_416_recovers(server, range_mode):
    import hashlib
    _H.sha = hashlib.sha256(DATA).hexdigest()
    _H.range_mode = range_mode
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    part = final.with_name(final.name + ".part")
    part.write_bytes(DATA)
    assert model_download.download("a/b", "m.gguf", url=server, retries=1) == str(final)
    assert final.read_bytes() == DATA
    assert not part.exists()
    assert _H.seen_ranges == [f"bytes={len(DATA)}-"]


def test_416_without_completion_evidence_restarts(server, monkeypatch):
    monkeypatch.setattr(model_download.time, "sleep", lambda _: None)
    _H.range_mode = "missing"
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".part").write_bytes(DATA + b"extra")
    model_download.download("a/b", "m.gguf", url=server, retries=2)
    assert final.read_bytes() == DATA
    assert _H.seen_ranges == [f"bytes={len(DATA) + 5}-", None]


@pytest.mark.parametrize("mode", ["wrong-start", "wrong-length"])
def test_inconsistent_range_headers_cannot_complete(server, mode):
    _H.range_mode = mode
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".part").write_bytes(DATA[:1000])
    with pytest.raises(RuntimeError):
        model_download.download("a/b", "m.gguf", url=server, retries=1)
    assert not final.exists()
    assert model_download.find_cached("a/b", "m.gguf") is None


def test_existing_final_validated_against_advertised_digest(server):
    import hashlib
    _H.sha = hashlib.sha256(DATA).hexdigest()
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.write_bytes(b"x" * len(DATA))
    model_download.download("a/b", "m.gguf", url=server, retries=1)
    assert final.read_bytes() == DATA


@pytest.mark.parametrize(("repo", "filename"), [
    ("../escape", "m.gguf"), ("/absolute", "m.gguf"), ("C:/repo", "m.gguf"),
    ("org\\repo", "m.gguf"), ("org/../repo", "m.gguf"),
    ("a/b", "../escape.gguf"), ("a/b", "/escape.gguf"), ("a/b", "C:/escape.gguf"),
    ("a/b", "C:escape.gguf"), ("a/b", "sub\\escape.gguf"),
    ("a/b", "sub/../../escape.gguf"), ("a/b", ""), ("", "m.gguf"),
])
def test_model_paths_reject_traversal(tmp_path, monkeypatch, repo, filename):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        model_download.local_path(repo, filename)
    with pytest.raises(ValueError):
        model_download.find_cached(repo, filename)


def test_nested_hf_filename_preserved_and_downloaded(server):
    final = model_download.local_path("a/b", "quantized/q4/m.gguf")
    assert final.parts[-4:] == ("a--b", "quantized", "q4", "m.gguf")
    assert model_download.download("a/b", "quantized/q4/m.gguf", url=server) == str(final)
    assert final.read_bytes() == DATA


@pytest.mark.parametrize("link_at", ["repository", "subdir", "file", "part"])
def test_model_paths_reject_symlink_escape(tmp_path, monkeypatch, link_at):
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    outside = tmp_path / "outside"
    outside.mkdir()
    root = model_download.models_dir()
    root.mkdir(parents=True)
    repository = root / "a--b"
    if link_at == "repository":
        link, target = repository, outside
    else:
        repository.mkdir()
        if link_at == "subdir":
            link, target = repository / "nested", outside
        else:
            target = outside / "m.gguf"
            target.write_bytes(b"untouched")
            link = repository / ("m.gguf.part" if link_at == "part" else "m.gguf")
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    filename = "nested/m.gguf" if link_at == "subdir" else "m.gguf"
    with pytest.raises(ValueError):
        if link_at == "part":
            model_download.download("a/b", filename, retries=1)
        else:
            model_download.local_path("a/b", filename)
    if target.is_file():
        assert target.read_bytes() == b"untouched"


def test_416_checks_known_size_without_sha(server):
    _H.advertise_size = True
    final = model_download.local_path("a/b", "m.gguf")
    final.parent.mkdir(parents=True)
    final.with_name(final.name + ".part").write_bytes(DATA + b"extra")
    with pytest.raises(RuntimeError):
        model_download.download("a/b", "m.gguf", url=server, retries=1)
    assert not final.exists()


def test_transfer_cancel_preserves_partial_for_next_invocation(server):
    def cancel(have, total):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        model_download.download("a/b", "m.gguf", url=server, progress=cancel)
    final = model_download.local_path("a/b", "m.gguf")
    part = final.with_name(final.name + ".part")
    partial = part.read_bytes()
    assert 0 < len(partial) < len(DATA)
    assert partial == DATA[:len(partial)]
    assert not final.exists()
    model_download.download("a/b", "m.gguf", url=server, retries=1)
    assert final.read_bytes() == DATA
    assert not part.exists()
