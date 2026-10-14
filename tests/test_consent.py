"""Per-folder permission to run the project's tests. Nothing is allowed without a person answering or the env variable."""
import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import consent, tools as T
from nexus_agent_ai.cli import app as appmod


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.delenv("NEXUS_ALLOW_PROJECT_EXECUTION", raising=False)
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "home"))
    consent._session_allowed.clear()
    consent._session_denied.clear()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    (work / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    yield


def _tty(monkeypatch, answers):
    monkeypatch.setattr(consent, "_interactive", lambda: True)
    it = iter(answers)
    monkeypatch.setattr(consent, "_ask", lambda prompt: next(it))


def test_non_interactive_is_refused_and_tool_not_offered(monkeypatch):
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    assert "not approved" in T.execute_run_tests()
    assert "run_tests" not in {t.name for t in T.get_all_tools()}


def test_declined_runs_nothing_and_is_not_asked_again(monkeypatch):
    _tty(monkeypatch, ["n"])           # a second question would raise StopIteration
    assert "declined" in T.execute_run_tests()
    assert "declined" in T.execute_run_tests()
    assert consent.list_trusted() == []


def test_empty_answer_means_no(monkeypatch):
    _tty(monkeypatch, [""])
    assert "not approved" in T.execute_run_tests()


def test_once_allows_this_session_only_and_saves_nothing(monkeypatch):
    _tty(monkeypatch, ["o"])
    assert "PASSED" in T.execute_run_tests()
    assert "PASSED" in T.execute_run_tests()   # no second prompt
    assert consent.list_trusted() == []
    consent._session_allowed.clear()           # "new session"
    _tty(monkeypatch, ["n"])
    assert "not approved" in T.execute_run_tests()


def test_always_is_saved_for_this_exact_folder_only(monkeypatch, tmp_path):
    _tty(monkeypatch, ["a"])
    assert "PASSED" in T.execute_run_tests()
    assert consent.list_trusted() == [consent.folder_key()]
    consent._session_allowed.clear()
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    assert "PASSED" in T.execute_run_tests()   # remembered, no prompt needed
    sub = tmp_path / "work" / "sub"
    sub.mkdir()
    (sub / "test_x.py").write_text("def test_x():\n    assert True\n")
    monkeypatch.chdir(sub)
    assert "not approved" in T.execute_run_tests()      # subfolder is not covered
    monkeypatch.chdir(tmp_path)
    assert "not approved" in T.execute_run_tests()      # nor the parent


def test_revoke_removes_the_grant(monkeypatch):
    consent.trust()
    assert consent.revoke() == 1
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    assert "not approved" in T.execute_run_tests()


def test_trust_cli_add_list_revoke():
    r = CliRunner()
    assert "No folders" in r.invoke(appmod.app, ["trust", "list"]).output
    assert r.invoke(appmod.app, ["trust", "add"]).exit_code == 0
    assert consent.folder_key() in r.invoke(appmod.app, ["trust", "list"]).output
    assert "Removed 1" in r.invoke(appmod.app, ["trust", "revoke"]).output
    assert "No folders" in r.invoke(appmod.app, ["trust", "list"]).output


def test_env_variable_still_allows(monkeypatch):
    monkeypatch.setattr(consent, "_interactive", lambda: False)
    monkeypatch.setenv("NEXUS_ALLOW_PROJECT_EXECUTION", "1")
    assert "PASSED" in T.execute_run_tests()
    assert "run_tests" in {t.name for t in T.get_all_tools()}


def test_run_tests_offered_when_a_person_is_at_the_terminal(monkeypatch):
    monkeypatch.setattr(consent, "_interactive", lambda: True)
    assert "run_tests" in {t.name for t in T.get_all_tools()}
