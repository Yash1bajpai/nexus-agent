import sys

import pytest

from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.utils import doctor


@pytest.fixture
def tty(monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "INIT_FILE", tmp_path / ".initialized")
    monkeypatch.setattr(onboarding, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(sys, "stdin", type("S", (), {"isatty": lambda self: True})())
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEFAULT_PROVIDER",
              "NEXUS_AGENT_MODEL_REPO", "NEXUS_AGENT_MODEL_FILENAME"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path




CFG = {"repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "filename": "LFM2.5-1.2B-Instruct-Q4_0.gguf", "size": "~0.7 GB"}


def _answers(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr(onboarding, "_input", lambda prompt="": next(it))
    monkeypatch.setattr(onboarding, "detect_system_specs", lambda: {"available_ram_gb": 16.0})


def test_model_choice_enter_keeps_suggestion_and_saves_nothing(tty, monkeypatch):
    tmp = tty
    _answers(monkeypatch, "")
    picked = onboarding._step_model_choice(dict(CFG))
    assert (picked["repo"], picked["filename"]) == (CFG["repo"], CFG["filename"])
    assert not (tmp / ".env").exists()


def test_model_selection_does_not_authorize_download_before_consent(tty, monkeypatch):
    _answers(monkeypatch, "2")
    picked = onboarding._step_model_choice(dict(CFG))
    assert picked["filename"] != CFG["filename"]
    assert not (tty / ".env").exists()
    assert "NEXUS_AGENT_MODEL_FILENAME" not in onboarding.os.environ


def test_model_choice_low_ram_needs_confirmation_and_decline_keeps_suggestion(tty, monkeypatch):
    tmp = tty
    it = iter(["2", "n"])
    monkeypatch.setattr(onboarding, "_input", lambda prompt="": next(it))
    monkeypatch.setattr(onboarding, "detect_system_specs", lambda: {"available_ram_gb": 1.0})
    picked = onboarding._step_model_choice(dict(CFG))
    assert picked["filename"] == CFG["filename"]
    assert not (tmp / ".env").exists()


def test_existing_model_is_preserved_when_enter_keeps_current(tty, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "custom/existing")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "selected.gguf")
    _answers(monkeypatch, "")
    picked = onboarding._step_model_choice(dict(CFG))
    assert (picked["repo"], picked["filename"]) == ("custom/existing", "selected.gguf")
    assert not (tty / ".env").exists()


def test_declining_alternative_preserves_existing_model_configuration(tty, monkeypatch):
    from nexus_agent_ai.providers import local_provider
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "custom/existing")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "selected.gguf")
    original = "NEXUS_AGENT_MODEL_REPO=custom/existing\nNEXUS_AGENT_MODEL_FILENAME=selected.gguf\n"
    (tty / ".env").write_text(original, encoding="utf-8")
    _answers(monkeypatch, "2", "n")
    monkeypatch.setattr(onboarding, "recommended_model_config", lambda specs: dict(CFG))
    monkeypatch.setattr(local_provider, "ensure_llama_server_binary",
                        lambda **kw: pytest.fail("download started after decline"))
    monkeypatch.setattr(local_provider.LocalProvider, "setup_model",
                        lambda *a, **kw: pytest.fail("weights started after decline"))
    assert onboarding._step_local_model_setup() is False
    assert onboarding.os.environ["NEXUS_AGENT_MODEL_REPO"] == "custom/existing"
    assert onboarding.os.environ["NEXUS_AGENT_MODEL_FILENAME"] == "selected.gguf"
    assert (tty / ".env").read_text(encoding="utf-8") == original


def test_approved_alternative_is_saved_and_downloaded(tty, monkeypatch):
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from dotenv import dotenv_values
    from nexus_agent_ai.providers import local_provider
    from nexus_agent_ai.utils import model_download

    body = b"local fixture weights"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("X-Linked-Size", str(len(body)))
            self.send_header("X-Linked-ETag", hashlib.sha256(body).hexdigest())
            self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NEXUS_AGENT_MODEL_REPO", "custom/existing")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "selected.gguf")
    _answers(monkeypatch, "2", "y")
    monkeypatch.setattr(onboarding, "recommended_model_config", lambda specs: dict(CFG))
    monkeypatch.setattr(model_download, "HF_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(local_provider, "ensure_llama_server_binary", lambda **kw: None)
    try:
        assert onboarding._step_local_model_setup() is True
        saved = dotenv_values(tty / ".env")
        assert saved["NEXUS_AGENT_MODEL_REPO"] == CFG["repo"]
        assert saved["NEXUS_AGENT_MODEL_FILENAME"] == CFG["filename"]
        assert model_download.local_path(CFG["repo"], CFG["filename"]).read_bytes() == body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_declining_wizard_download_never_starts_engine_or_weights(monkeypatch):
    from nexus_agent_ai.providers import local_provider
    _answers(monkeypatch, "", "n")
    monkeypatch.setattr(onboarding, "recommended_model_config", lambda specs: dict(CFG))
    monkeypatch.setattr(local_provider, "ensure_llama_server_binary",
                        lambda **kw: pytest.fail("engine download started without consent"))
    monkeypatch.setattr(local_provider.LocalProvider, "setup_model",
                        lambda self, **kw: pytest.fail("weights download started without consent"))
    onboarding._step_local_model_setup()


def test_path_advice_names_the_real_folder(tmp_path, monkeypatch):
    import os
    exe = "nexus-agent-ai.exe" if os.name == "nt" else "nexus-agent-ai"
    (tmp_path / exe).write_text("")
    monkeypatch.setattr(doctor, "_script_dirs", lambda: [str(tmp_path)])
    msg = doctor._path_advice()
    assert str(tmp_path) in msg and "-m nexus_agent_ai" in msg and "does not change PATH" in msg


def test_path_advice_when_script_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "_script_dirs", lambda: [str(tmp_path)])
    assert "-m nexus_agent_ai" in doctor._path_advice()
