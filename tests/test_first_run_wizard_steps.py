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
    calls = []
    for name in ("_show_welcome", "_step_api_keys", "_step_system_specs", "_step_default_provider", "_step_local_model_setup"):
        monkeypatch.setattr(onboarding, name, lambda n=name: calls.append(n))
    return calls, tmp_path


def test_first_run_asks_keys_provider_and_model_for_everyone(tty):
    calls, tmp = tty
    onboarding.run_if_first_time()
    assert calls == ["_show_welcome", "_step_api_keys", "_step_system_specs", "_step_default_provider", "_step_local_model_setup"]
    assert (tmp / ".initialized").exists()


def test_setup_command_runs_the_same_wizard(tty):
    calls, _ = tty
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app
    assert CliRunner().invoke(app, ["setup"]).exit_code == 0
    assert "_step_api_keys" in calls and "_step_default_provider" in calls and "_step_local_model_setup" in calls


CFG = {"repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "filename": "LFM2.5-1.2B-Instruct-Q4_0.gguf", "size": "~0.7 GB"}


def _answers(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr(onboarding, "_input", lambda prompt="": next(it))
    monkeypatch.setattr(onboarding, "detect_system_specs", lambda: {"available_ram_gb": 16.0})


def test_model_choice_enter_keeps_suggestion_and_saves_nothing(tty, monkeypatch):
    _, tmp = tty
    _answers(monkeypatch, "")
    assert onboarding._step_model_choice(dict(CFG)) == CFG
    assert not (tmp / ".env").exists()


def test_model_choice_pick_is_saved_as_explicit_choice(tty, monkeypatch):
    _, tmp = tty
    _answers(monkeypatch, "2")
    picked = onboarding._step_model_choice(dict(CFG))
    env = (tmp / ".env").read_text()
    assert picked["filename"] in env and "NEXUS_AGENT_MODEL_REPO=" + picked["repo"] in env
    assert picked["filename"] != CFG["filename"]


def test_model_choice_low_ram_needs_confirmation_and_decline_keeps_suggestion(tty, monkeypatch):
    _, tmp = tty
    it = iter(["2", "n"])
    monkeypatch.setattr(onboarding, "_input", lambda prompt="": next(it))
    monkeypatch.setattr(onboarding, "detect_system_specs", lambda: {"available_ram_gb": 1.0})
    assert onboarding._step_model_choice(dict(CFG)) == CFG
    assert not (tmp / ".env").exists()


def test_model_choice_skipped_when_user_already_configured_a_model(tty, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "x.gguf")
    monkeypatch.setattr(onboarding, "_input", lambda prompt="": pytest.fail("must not ask"))
    assert onboarding._step_model_choice(dict(CFG)) == CFG


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
