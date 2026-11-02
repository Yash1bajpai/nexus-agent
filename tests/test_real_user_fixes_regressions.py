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

    def fake_open(self, req, timeout=30.0):
        # Inspect SSL context used by the opener
        ctx_used = getattr(self, "_context", None)
        attempts.append(ctx_used)
        if len(attempts) == 1:
            raise ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: certificate verify failed")
        return DummyResponse(b"valid model payload")

    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)

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
    monkeypatch.setattr(provider, "_get_llama_server_info", lambda: ("https://github.com/test/llama-server.zip", "llama-server"))
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


def test_doctor_windows_msvc_dll_probe(monkeypatch):
    """Doctor reports MSVC runtime state on Windows platform."""
    monkeypatch.setattr(doctor.sys, "platform", "win32")

    # Both DLLs present
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    checks_ok = doctor.run_checks(probe_network=False)
    msvc_ok = [c for c in checks_ok if c[1] == "MSVC Runtime"]
    assert len(msvc_ok) == 1
    assert msvc_ok[0][0] == "ok"
    assert "vcruntime140.dll and msvcp140.dll present" in msvc_ok[0][2]

    # DLLs missing
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    checks_missing = doctor.run_checks(probe_network=False)
    msvc_warn = [c for c in checks_missing if c[1] == "MSVC Runtime"]
    assert len(msvc_warn) == 1
    assert msvc_warn[0][0] == "warn"
    assert "Install Microsoft Visual C++ Redistributable x64" in msvc_warn[0][2]
    assert "https://aka.ms/vs/17/release/vc_redist.x64.exe" in msvc_warn[0][2]


# ---------------------------------------------------------------------------
# Fix 9: Local context defaults (8192, 2048), RAM note, and meter breakdown
# ---------------------------------------------------------------------------

def test_local_defaults_raised_to_8192():
    assert _LOCAL_DEFAULTS == ModelLimits(8192, 2048)
    assert resolve_limits("local", "custom.gguf").context_size == 8192
    assert resolve_limits("local", "custom.gguf").max_output_tokens == 2048


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
