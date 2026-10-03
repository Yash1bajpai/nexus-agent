import os

from nexus_agent_ai.agent import tools as T
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.cli import onboarding
from tests.test_reliable_edits_and_keys import Scripted, fresh_tracking  # noqa: F401 (autouse fixture)


def test_wizard_does_not_pin_model_so_first_run_confirm_happens(tmp_path, monkeypatch):
    monkeypatch.delenv("NEXUS_AGENT_MODEL_REPO", raising=False)
    monkeypatch.delenv("NEXUS_AGENT_MODEL_FILENAME", raising=False)
    written = []
    monkeypatch.setattr(onboarding, "_write_env_key", lambda k, v: written.append(k))
    onboarding._step_system_specs()
    assert "NEXUS_AGENT_MODEL_REPO" not in written
    assert "NEXUS_AGENT_MODEL_FILENAME" not in written
    assert "NEXUS_AGENT_MODEL_REPO" not in os.environ


def test_size_label_has_single_tilde():
    assert onboarding._size_txt("~0.7 GB") == "~0.7 GB"
    assert onboarding._size_txt("0.7 GB") == "~0.7 GB"


def test_declining_wizard_download_says_you_will_be_asked_again(monkeypatch, capsys):
    monkeypatch.setattr(onboarding, "_input", lambda p: "n")
    monkeypatch.setattr("nexus_agent_ai.providers.local_provider.ensure_llama_server_binary", lambda verbose=False: None)
    onboarding._step_local_model_setup()
    out = capsys.readouterr().out
    assert "asked again" in out and "~~" not in out


def test_warns_when_edit_request_ends_without_write(tmp_path):
    (tmp_path / "f.txt").write_text("alpha")
    agent = Agent(Scripted([("read_file", {"path": "f.txt"}), None]), verbose=False)
    out = agent.run("fix f.txt")
    assert "no file was changed" in out


def test_no_warning_after_successful_patch(tmp_path):
    (tmp_path / "f.txt").write_text("alpha")
    agent = Agent(Scripted([("read_file", {"path": "f.txt"}),
                            ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None]), verbose=False)
    out = agent.run("fix f.txt")
    assert "no file was changed" not in out


def test_no_warning_for_plain_question(tmp_path):
    (tmp_path / "f.txt").write_text("alpha")
    agent = Agent(Scripted([("read_file", {"path": "f.txt"}), None]), verbose=False)
    out = agent.run("what is in f.txt")
    assert "no file was changed" not in out


def test_closest_hint_for_reworded_single_line(tmp_path):
    (tmp_path / "c.py").write_text("def add(a, b):\n    return a - b\n")
    T.execute_read_file("c.py")
    res = T.execute_patch_file("c.py", "    return a-b  # sub", "x")
    assert "Closest text" in res and "return a - b" in res
