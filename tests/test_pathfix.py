import os
from pathlib import Path

from nexus_agent_ai.utils import pathfix


def _fake_install(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.base_prefix", __import__("sys").prefix)  # behave like a non-venv Python
    d = tmp_path / "scripts"
    d.mkdir()
    (d / ("nexus-agent-ai.exe" if os.name == "nt" else "nexus-agent-ai")).write_text("x")
    monkeypatch.setattr("nexus_agent_ai.utils.doctor._script_dirs", lambda: [str(d)])
    return d


def test_detects_folder_missing_from_path_and_not_when_present(tmp_path, monkeypatch):
    d = _fake_install(tmp_path, monkeypatch)
    assert pathfix.folder_missing_from_path("/usr/bin" + os.pathsep + "/bin") == str(d)
    assert pathfix.folder_missing_from_path(os.pathsep.join(["/usr/bin", str(d)])) is None


def test_no_install_found_means_nothing_to_offer(tmp_path, monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.utils.doctor._script_dirs", lambda: [str(tmp_path)])
    assert pathfix.folder_missing_from_path("/usr/bin") is None


def test_default_answer_changes_nothing(tmp_path, monkeypatch):
    _fake_install(tmp_path, monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home").mkdir()
    out = []
    changed = pathfix.offer(ask=lambda p: "", say=out.append)
    assert changed is False
    assert not list((tmp_path / "home").iterdir())
    assert any("python -m nexus_agent_ai" in o for o in out)


def test_yes_appends_one_marked_line_idempotently(tmp_path, monkeypatch):
    if os.name == "nt":
        return
    d = _fake_install(tmp_path, monkeypatch)
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("PATH", "/usr/bin")
    assert pathfix.offer(ask=lambda p: "y", say=lambda *_: None) is True
    text = (home / ".zshrc").read_text()
    assert text.count(pathfix.MARK) == 1 and f'export PATH="{d}:$PATH"' in text
    pathfix.add_to_user_path(str(d))  # second time: no duplicate
    assert (home / ".zshrc").read_text().count(pathfix.MARK) == 1


def test_setup_path_command_exists_and_skips_wizard():
    from typer.testing import CliRunner
    import nexus_agent_ai.cli.app as appmod
    r = CliRunner().invoke(appmod.app, ["setup-path", "--help"])
    assert r.exit_code == 0 and "user PATH" in r.output


def test_inside_a_virtualenv_nothing_is_offered_and_nothing_is_changed(tmp_path, monkeypatch):
    _fake_install(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.prefix", str(tmp_path / "venv"))
    monkeypatch.setattr("sys.base_prefix", str(tmp_path / "base"))
    assert pathfix.folder_missing_from_path("/usr/bin") is None
    asked = []
    assert pathfix.offer(ask=lambda p: asked.append(p) or "y", say=lambda *_: None) is False
    assert not asked
