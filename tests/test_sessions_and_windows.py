import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent.persistence import SQLiteMemory, get_workspace_session_id
from nexus_agent_ai.cli import app as appmod
from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.providers import local_provider as lp
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse


class Echo(BaseProvider):
    model = "echo"

    def complete(self, messages, tools, system):
        return ProviderResponse(text="ok")

    def stream(self, **kw):
        yield self.complete(**kw)

    def format_tool_result_message(self, i, r):
        return {"role": "tool", "tool_call_id": i, "content": r}


@pytest.fixture(autouse=True)
def no_wizard(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance", lambda *a, **k: (Echo(), "echo"))


def run_repl(args, text):
    return CliRunner().invoke(appmod.app, ["repl", "-p", "local"] + args, input=text)


# ---- persistence on by default, --continue ---------------------------------

def test_repl_saves_by_default_and_resumes_in_same_folder():
    first = run_repl([], "hello there\n/exit\n")
    assert "New session" in first.output
    sessions = SQLiteMemory.list_sessions()
    assert len(sessions) == 1 and sessions[0]["message_count"] == 2
    second = run_repl([], "/exit\n")
    assert "Resumed session" in second.output and "2 messages" in second.output


def test_no_persist_saves_nothing():
    run_repl(["--no-persist"], "hello\n/exit\n")
    assert SQLiteMemory.list_sessions() == []


def test_continue_resumes_most_recent_session_from_any_folder():
    SQLiteMemory(session_id="older").add("user", "a")
    SQLiteMemory(session_id="newer").add("user", "b")
    import sqlite3
    db = Path.home() / ".nexus-agent" / "history.db"
    with sqlite3.connect(db) as c:
        c.execute("UPDATE conversation_history SET timestamp='2020-01-01 00:00:00' WHERE session_id='older'")
        c.execute("UPDATE conversation_history SET timestamp='2026-01-01 00:00:00' WHERE session_id='newer'")
    out = run_repl(["--continue"], "/exit\n")
    assert "Resumed session 'newer'" in out.output


def test_continue_with_no_history_starts_new():
    out = run_repl(["--continue"], "/exit\n")
    assert out.exit_code == 0
    assert "No earlier session" in out.output and "New session" in out.output


def test_continue_and_explicit_session_prefers_session():
    SQLiteMemory(session_id="mine").add("user", "x")
    SQLiteMemory(session_id="other").add("user", "y")
    out = run_repl(["--continue", "--session", "mine"], "/exit\n")
    assert "Resumed session 'mine'" in out.output


def test_one_shot_chat_stays_throwaway_unless_asked():
    CliRunner().invoke(appmod.app, ["chat", "hi", "-p", "local", "--no-stream"])
    assert SQLiteMemory.list_sessions() == []
    CliRunner().invoke(appmod.app, ["chat", "hi", "-p", "local", "--no-stream", "--continue"])
    assert len(SQLiteMemory.list_sessions()) == 1


# ---- Windows-oriented checks (simulated on Linux, see notes in the PR) ------

def test_python_dash_m_entry_point_runs():
    import subprocess
    out = subprocess.run([sys.executable, "-m", "nexus_agent_ai", "--version"], capture_output=True, text=True,
                         env={**os.environ, "HOME": str(Path.home()), "USERPROFILE": str(Path.home())})
    assert out.returncode == 0 and "Nexus-Agent CLI" in out.stdout


@pytest.mark.parametrize("machine,expected", [("AMD64", "win-cpu-x64.zip"), ("ARM64", "win-cpu-arm64.zip")])
def test_windows_engine_download_selection(monkeypatch, machine, expected):
    monkeypatch.setattr(lp.platform, "system", lambda: "Windows")
    monkeypatch.setattr(lp.platform, "machine", lambda: machine)
    url, exe = lp._get_llama_server_info()
    assert url.endswith(expected) and exe == "llama-server.exe"


def test_chmod_is_skipped_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(onboarding.os, "name", "nt")
    called = []
    monkeypatch.setattr(onboarding.os, "chmod", lambda *a: called.append(a))
    onboarding._restrict_permissions(tmp_path, 0o700)
    assert called == []


def test_session_id_ignores_drive_letter_case_and_slashes_on_windows(monkeypatch):
    # The hash input is normalised the same way for C:\Users\Me\proj and c:/users/me/proj.
    import hashlib
    monkeypatch.setattr("os.name", "nt")
    class FakePath:
        name = "proj"
        def __init__(self, s): self.s = s
        def resolve(self): return self
        def __str__(self): return self.s
    a = get_workspace_session_id(FakePath("C:\\Users\\Me\\proj"))
    b = get_workspace_session_id(FakePath("c:/users/me/proj"))
    assert a == b and a.startswith("proj_")


def test_paths_with_spaces_and_unicode(tmp_path, monkeypatch):
    from nexus_agent_ai.agent import tools as T
    folder = tmp_path / "my project \u00fc"
    folder.mkdir()
    monkeypatch.chdir(folder)
    T.reset_file_tracking()
    assert "Successfully wrote" in T.execute_write_file("na\u00efve notes.txt", "caf\u00e9\r\nline2\r\n")
    assert T.execute_read_file("na\u00efve notes.txt").startswith("caf\u00e9")
    assert "Successfully patched" in T.execute_patch_file("na\u00efve notes.txt", "line2", "second")
    assert get_workspace_session_id(folder).startswith("my_project_")


def test_doctor_suggests_python_dash_m_when_not_on_path(monkeypatch):
    from nexus_agent_ai.utils import doctor
    monkeypatch.setattr("shutil.which", lambda name: None)
    checks = doctor.run_checks(probe_network=False)
    cmd = [c for c in checks if c[1] == "Command"][0]
    assert cmd[0] == "warn" and "python -m nexus_agent_ai" in cmd[2]
