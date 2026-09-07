"""Regression tests: the llama-server engine must be checked and installed at
the initial stage (onboarding wizard + local provider startup) so the LFM
model never fails at query time due to a missing engine."""
import builtins

import nexus_agent_ai.providers.local_provider as lp
from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.cli.app import get_provider_instance


class _StubLocalProvider:
    """Stands in for LocalProvider so no real model/HTTP work happens."""
    instances = []

    def __init__(self, model_id=None, filename=None):
        self.model_id = model_id
        self.filename = filename or "LFM2.5-2.6B-Q6_K.gguf"
        self.setup_calls = 0
        _StubLocalProvider.instances.append(self)

    def setup_model(self, verify_download=False):
        self.setup_calls += 1
        return f"/fake/models/{self.filename}"


def _reset_stubs(monkeypatch, engine_calls):
    _StubLocalProvider.instances = []
    monkeypatch.setattr(lp, "LocalProvider", _StubLocalProvider)
    monkeypatch.setattr(lp, "ensure_llama_server_binary",
                        lambda verbose=True: engine_calls.append(verbose) or "/fake/llama-server")


# ── ensure_llama_server_binary itself ────────────────────────────────────────

def test_ensure_llama_server_binary_downloads_when_missing(monkeypatch):
    """Engine missing + not ARM → must attempt the download and return its path."""
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: None)
    monkeypatch.setattr(lp, "system_machine_is_arm_linux", lambda: False)
    downloads = []
    monkeypatch.setattr(lp.LocalProvider, "_download_llama_server",
                        lambda self: downloads.append(1) or "/installed/llama-server")
    result = lp.ensure_llama_server_binary(verbose=False)
    assert result == "/installed/llama-server"
    assert downloads == [1]


def test_ensure_llama_server_binary_failure_is_quiet_but_soft(monkeypatch):
    """Startup pre-flight (verbose=False) must never raise on download failure."""
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: None)
    monkeypatch.setattr(lp, "system_machine_is_arm_linux", lambda: False)
    monkeypatch.setattr(lp.LocalProvider, "_download_llama_server",
                        lambda self: (_ for _ in ()).throw(RuntimeError("network blocked")))
    assert lp.ensure_llama_server_binary(verbose=False) is None


# ── provider startup (`-p local`, default provider, unknown fallback) ───────

def test_local_provider_startup_preflights_engine(monkeypatch):
    """`-p local` must check/install the engine before the agent loop runs."""
    engine_calls = []
    _reset_stubs(monkeypatch, engine_calls)
    prov, resolved = get_provider_instance("local")
    assert engine_calls == [False]  # quiet pre-flight ran exactly once
    assert prov.setup_calls == 1
    assert resolved.startswith("Local (")


def test_unknown_provider_falls_back_to_local_with_engine_check(monkeypatch):
    engine_calls = []
    _reset_stubs(monkeypatch, engine_calls)
    prov, resolved = get_provider_instance("does-not-exist")
    assert engine_calls == [False]
    assert resolved.startswith("Local (")


# ── onboarding wizard ─────────────────────────────────────────────────────────

def test_onboarding_installs_engine_even_when_model_declined(monkeypatch):
    """Declining the model download must NOT skip the llama-server install."""
    engine_calls = []
    _reset_stubs(monkeypatch, engine_calls)
    monkeypatch.setattr(onboarding, "detect_system_specs",
                        lambda: {"ram_gb": 16, "cpu_cores": 8, "gpu": None,
                                 "cpu_name": "Test CPU", "avx2": False})
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    onboarding._step_local_model_setup()
    assert engine_calls == [False]  # engine install attempted (not consent-gated)
    assert all(inst.setup_calls == 0 for inst in _StubLocalProvider.instances)


def test_onboarding_engine_and_model_both_installed_on_consent(monkeypatch):
    engine_calls = []
    _reset_stubs(monkeypatch, engine_calls)
    monkeypatch.setattr(onboarding, "detect_system_specs",
                        lambda: {"ram_gb": 16, "cpu_cores": 8, "gpu": None,
                                 "cpu_name": "Test CPU", "avx2": False})
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    onboarding._step_local_model_setup()
    assert engine_calls == [False]
    assert [inst.setup_calls for inst in _StubLocalProvider.instances] == [1]


# ── Ubuntu zip layout regression (found via Kaggle clean-start test) ─────────
# The b7075 ubuntu-x64 release nests the binary under build/bin/ — two levels
# deep. The old finder only checked flat and one-level nesting, so the engine
# "installed" but was then reported missing on every Linux machine.

def test_find_llama_server_exe_recurses_deep_layouts(monkeypatch, tmp_path):
    """build/bin/llama-server (ubuntu-x64 layout) must be found."""
    import platform as _plat
    from nexus_agent_ai.providers import local_provider as lp

    engine_dir = tmp_path / "llama-server"
    nested = engine_dir / "build" / "bin"
    nested.mkdir(parents=True)
    (nested / "llama-server").write_bytes(b"#!/bin/sh\n")
    (nested / "libggml-base.so").write_bytes(b"lib")

    monkeypatch.setattr(lp, "_SERVER_DIR", engine_dir)
    monkeypatch.setattr(_plat, "system", lambda: "Linux")
    monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
    found = lp._find_llama_server_exe()
    assert found == str(nested / "llama-server")


def test_download_llama_server_extracts_and_finds_nested_binary(monkeypatch, tmp_path):
    """End-to-end _download_llama_server with a synthetic ubuntu-layout zip:
    digest check passes, extraction is zip-slip safe, and the deeply nested
    binary is located (previously raised 'binary not found after extraction')."""
    import hashlib
    import zipfile
    import platform as _plat
    from nexus_agent_ai.providers import local_provider as lp

    engine_dir = tmp_path / "llama-server"
    engine_dir.mkdir()
    zip_path = engine_dir / "llama-server.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("build/bin/llama-server", "#!/bin/sh\necho ok")
        zf.writestr("build/bin/libggml-base.so", "lib")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    monkeypatch.setattr(lp, "_SERVER_DIR", engine_dir)
    monkeypatch.setattr(_plat, "system", lambda: "Linux")
    monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
    # Point the pinned digest table at our synthetic zip's real digest.
    monkeypatch.setattr(lp, "_LLAMA_SERVER_SHA256",
                        {"llama-b7075-bin-ubuntu-x64.zip": digest})
    monkeypatch.setenv("NEXUS_AGENT_LLAMA_SERVER_SHA256", "")

    exe = lp.LocalProvider._download_llama_server(lp.LocalProvider())
    assert exe.endswith("llama-server")
    assert "build" in exe and "bin" in exe  # stayed nested, next to its libs
    assert not zip_path.exists()  # zip cleaned up
