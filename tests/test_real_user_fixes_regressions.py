import io
import os
import ssl
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nexus_agent_ai.agent import context as ctx
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse
from nexus_agent_ai.providers.capabilities import _LOCAL_DEFAULTS, ModelLimits, resolve_limits
from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.utils import doctor, model_download as md


class DummyResponse:
    def __init__(self, data=b"chunk of data"):
        self._io = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}
        self.status = 200

    def read(self, amt=-1):
        return self._io.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyProvider(BaseProvider):
    model = "dummy"
    context_size = 8192
    max_output_tokens = 2048

    def complete(self, *a, **k):
        return ProviderResponse(text="ok")

    def stream(self, *a, **k):
        yield "ok"

    def format_tool_result_message(self, *a, **k):
        return {}


# ---------------------------------------------------------------------------
# Fix 7: SSL cert verification with certifi retry
# ---------------------------------------------------------------------------

def test_model_download_ssl_cert_retry_success(tmp_path, monkeypatch):
    """Mocked urlopen raising SSLCertVerificationError on first call, then success."""
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "nexus"))
    monkeypatch.setattr(md, "expected_digest", lambda url: (None, None))

    attempts = []
    real_create_default_context = ssl.create_default_context
    ctx_calls = []

    def tracking_create_default_context(*args, **kwargs):
        ctx_calls.append(kwargs.get("cafile", args[2] if len(args) > 2 else None))
        return real_create_default_context(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", tracking_create_default_context)

    def fake_open(self, req, timeout=30.0):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            raise ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED: certificate verify failed")
        return DummyResponse(b"valid model payload")

    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)

    out = md.download("LiquidAI/test-repo", "model.gguf")
    assert Path(out).read_bytes() == b"valid model payload"
    assert len(attempts) == 2  # retried once
    # Default trust (system store, cafile=None) first; certifi bundle only on the cert-failure retry.
    import certifi
    assert ctx_calls[0] is None
    assert certifi.where() in ctx_calls[1:]


def test_model_download_ssl_cert_retry_wrapped_urlerror(tmp_path, monkeypatch):
    """urllib wraps handshake failures in URLError(reason=SSLCertVerificationError):
    the cert retry must fire on the wrapped path too (URLError is OSError, not ValueError)."""
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "nexus"))
    monkeypatch.setattr(md, "expected_digest", lambda url: (None, None))

    attempts = []

    def fake_open(self, req, timeout=30.0):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            inner = ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED: certificate verify failed")
            raise urllib.error.URLError(inner)
        return DummyResponse(b"valid model payload")

    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)

    out = md.download("LiquidAI/test-repo", "model.gguf")
    assert Path(out).read_bytes() == b"valid model payload"
    assert len(attempts) == 2  # wrapped cert error still triggered exactly one retry


def test_model_download_ssl_cert_failure_actionable_guidance(tmp_path, monkeypatch):
    """Mocked urlopen failing twice with SSLCertVerificationError produces actionable guidance."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "nexus"))
    monkeypatch.setattr(md, "expected_digest", lambda url: (None, None))

    def fake_open(self, req, timeout=30.0):
        raise ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: certificate verify failed")

    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)

    with pytest.raises(RuntimeError) as exc:
        md.download("LiquidAI/test-repo", "model.gguf")

    err = str(exc.value)
    assert "SSL certificate verification failed" in err
    assert "pip install certifi" in err or "SSL_CERT_FILE" in err
    assert "firewall" not in err.lower()
    assert "no internet" not in err.lower()


def test_engine_download_ssl_cert_retry_and_actionable_guidance(tmp_path, monkeypatch):
    """Engine download on repeated SSL cert error prints actionable guidance, not generic firewall."""
    provider = LocalProvider()
    import nexus_agent_ai.providers.local_provider as lp_mod
    monkeypatch.setattr(lp_mod, "_get_llama_server_info", lambda: ("https://github.com/test/llama-server.zip", "llama-server"))
    monkeypatch.setenv("NEXUS_AGENT_LLAMA_SERVER_SHA256", "a" * 64)

    import nexus_agent_ai.providers.local_provider as lp
    server_dir = tmp_path / "server"
    monkeypatch.setattr(lp, "_SERVER_DIR", server_dir)
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: None)

    call_count = [0]

    def fake_open(self, *a, **kw):
        call_count[0] += 1
        raise ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: certificate verify failed")

    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)

    with pytest.raises(RuntimeError) as exc:
        provider._download_llama_server()

    err = str(exc.value)
    assert call_count[0] >= 2, "Expected certifi retry"
    assert "SSL certificate verification failed" in err
    assert "pip install certifi" in err or "SSL_CERT_FILE" in err
    assert "College/corporate firewall" not in err
    assert "No internet connection" not in err


# ---------------------------------------------------------------------------
# Fix 8: llama-server exit code 3221225781 MSVC guidance & doctor probe
# ---------------------------------------------------------------------------

def test_llama_server_exit_code_3221225781_actionable_guidance(monkeypatch, tmp_path):
    """Exit code 3221225781 (0xC0000135) provides MSVC redistributable link."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    provider = LocalProvider()
    monkeypatch.setattr(provider, "_download_llama_server", lambda: str(tmp_path / "llama-server"))

    class FakeProc:
        returncode = 3221225781
        def poll(self):
            return 3221225781

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(provider, "_is_server_alive", lambda: False)

    with pytest.raises(RuntimeError) as exc:
        provider._start_llama_server("dummy.gguf")

    err = str(exc.value)
    assert "3221225781" in err
    assert "Install Microsoft Visual C++ Redistributable x64" in err
    assert "https://aka.ms/vs/17/release/vc_redist.x64.exe" in err


def test_doctor_windows_msvc_dll_probe(monkeypatch, tmp_path):
    """Doctor reports MSVC runtime state on Windows platform."""
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/nexus-agent-ai")

    # Both DLLs present
    sys32 = tmp_path / "win" / "System32"
    sys32.mkdir(parents=True)
    (sys32 / "vcruntime140.dll").write_bytes(b"x")
    (sys32 / "msvcp140.dll").write_bytes(b"x")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "win"))
    checks_ok = doctor.run_checks(probe_network=False)
    msvc_ok = [c for c in checks_ok if c[1] == "MSVC Runtime"]
    assert len(msvc_ok) == 1
    assert msvc_ok[0][0] == "ok"
    assert "vcruntime140.dll and msvcp140.dll present" in msvc_ok[0][2]

    # DLLs missing
    (sys32 / "vcruntime140.dll").unlink()
    (sys32 / "msvcp140.dll").unlink()
    checks_missing = doctor.run_checks(probe_network=False)
    msvc_warn = [c for c in checks_missing if c[1] == "MSVC Runtime"]
    assert len(msvc_warn) == 1
    assert msvc_warn[0][0] == "warn"
    assert "Install Microsoft Visual C++ Redistributable x64" in msvc_warn[0][2]
    assert "https://aka.ms/vs/17/release/vc_redist.x64.exe" in msvc_warn[0][2]


# ---------------------------------------------------------------------------
# Local context defaults remain measured 4096/512, with RAM note and meter breakdown
# ---------------------------------------------------------------------------

def test_local_defaults_remain_4096_with_512_reserve():
    assert _LOCAL_DEFAULTS == ModelLimits(4096, 512)
    assert resolve_limits("local", "custom.gguf").context_size == 4096
    assert resolve_limits("local", "custom.gguf").max_output_tokens == 512


def test_context_breakdown_and_meter():
    prov = DummyProvider()
    agent = Agent(prov, tools=[], verbose=False)
    bk = agent.context_breakdown()
    for key in ("overhead", "system", "tools", "history", "reserve", "total", "window"):
        assert key in bk

    text = ctx.meter(agent.context_used(), agent.context_window, breakdown=bk)
    assert "breakdown:" in text
    assert "overhead" in text
    assert "history" in text
    assert "reserve" in text
    assert "window" in text


def test_search_web_full_sentence_query_cleaned_and_retried(monkeypatch):
    """A small model copies the whole prompt as the query; engines then return
    pages about search itself. The tool must retry once with the search-intent
    phrasing stripped instead of declaring the results irrelevant."""
    import sys
    from types import ModuleType
    from nexus_agent_ai.agent.tools import execute_tool

    calls = []

    class FakeDDGS:
        def text(self, query, max_results=5):
            calls.append(query)
            if "llama" in query and "search" not in query.lower():
                return [{
                    "title": "Releases: ggml-org/llama.cpp - GitHub",
                    "href": "https://github.com/ggml-org/llama.cpp/releases",
                    "body": "LLM inference in C/C++.",
                }]
            return [{"title": "Google Search", "href": "https://search.google/", "body": "Search anything."}]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

    out = execute_tool("search_web", {"query": "search the web for the latest llama.cpp release"})
    assert "llama.cpp" in out and "github.com/ggml-org/llama.cpp/releases" in out
    assert not out.startswith("ERROR")
    assert calls == [
        "search the web for the latest llama.cpp release",
        "the latest llama.cpp release",
    ]


def test_search_web_still_irrelevant_after_cleaning_stays_honest(monkeypatch):
    """If the cleaned retry is also irrelevant, keep the honest ERROR verdict."""
    import sys
    from types import ModuleType
    from nexus_agent_ai.agent.tools import execute_tool

    class FakeDDGS:
        def text(self, query, max_results=5):
            return [{"title": "Garden plants", "href": "https://example.com/garden", "body": "Flowers."}]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

    out = execute_tool("search_web", {"query": "search the web for the latest llama.cpp release"})
    assert out.startswith("ERROR") and "irrelevant" in out


def test_search_web_prefix_strip_preserves_user_terms(monkeypatch):
    """The prefix strip must only drop leading search-intent phrasing and keep
    quoted phrases, operators, URLs and user terms verbatim."""
    import sys
    from types import ModuleType
    from nexus_agent_ai.agent.tools import execute_tool

    calls = []

    class FakeDDGS:
        def text(self, query, max_results=5):
            calls.append(query)
            if "site:github.com" in query and "search" not in query.lower():
                return [{
                    "title": 'llama.cpp releases "exact phrase"',
                    "href": "https://github.com/ggml-org/llama.cpp/releases",
                    "body": "llama.cpp release notes.",
                }]
            return [{"title": "Bing", "href": "https://www.bing.com/", "body": "Search engine."}]

    module = ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

    out = execute_tool("search_web", {"query": 'search the web for "exact phrase" llama.cpp site:github.com'})
    assert not out.startswith("ERROR")
    # the retry kept the quoted phrase, the operator and the user term verbatim
    assert calls[-1] == '"exact phrase" llama.cpp site:github.com'
