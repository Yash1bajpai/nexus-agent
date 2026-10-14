import os
import sys

import pytest

from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.utils import doctor


@pytest.fixture
def tty(monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding, "INIT_FILE", tmp_path / ".initialized")
    monkeypatch.setattr(sys, "stdin", type("S", (), {"isatty": lambda self: True})())
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEFAULT_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    calls = []
    for name in ("_show_welcome", "_step_api_keys", "_step_system_specs", "_step_default_provider", "_step_local_model_setup"):
        monkeypatch.setattr(onboarding, name, lambda n=name: calls.append(n))
    return calls, tmp_path / ".initialized"


def test_new_user_without_cloud_config_is_never_asked_for_keys_or_provider(tty):
    calls, init = tty
    onboarding.run_if_first_time()
    assert "_step_api_keys" not in calls and "_step_default_provider" not in calls
    assert "_step_local_model_setup" in calls
    assert init.exists()


@pytest.mark.parametrize("var", ["GEMINI_API_KEY", "OPENAI_API_KEY", "DEFAULT_PROVIDER"])
def test_existing_cloud_config_keeps_full_setup(tty, monkeypatch, var):
    calls, init = tty
    monkeypatch.setenv(var, "x" * 12)
    onboarding.run_if_first_time()
    assert calls.count("_step_api_keys") == 1 and calls.count("_step_default_provider") == 1
    assert init.exists()


def test_setup_command_runs_full_wizard(tty):
    calls, init = tty
    from typer.testing import CliRunner
    from nexus_agent_ai.cli.app import app
    r = CliRunner().invoke(app, ["setup"])
    assert r.exit_code == 0
    assert "_step_api_keys" in calls and "_step_default_provider" in calls and "_step_local_model_setup" in calls


def test_path_advice_names_the_real_folder(tmp_path, monkeypatch):
    exe = "nexus-agent-ai.exe" if os.name == "nt" else "nexus-agent-ai"
    (tmp_path / exe).write_text("")
    monkeypatch.setattr(doctor, "_script_dirs", lambda: [str(tmp_path)])
    msg = doctor._path_advice()
    assert str(tmp_path) in msg and "-m nexus_agent_ai" in msg and "does not change PATH" in msg


def test_path_advice_when_script_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "_script_dirs", lambda: [str(tmp_path)])
    assert "-m nexus_agent_ai" in doctor._path_advice()
