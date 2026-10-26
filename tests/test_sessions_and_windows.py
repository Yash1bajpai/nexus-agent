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
from nexus_agent_ai.utils import private_files


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
    called = []
    monkeypatch.setattr(private_files, "_restrict_windows", lambda path: called.append(path))
    monkeypatch.setattr(private_files.os, "chmod", lambda *a: pytest.fail("chmod cannot secure a Windows ACL"))
    monkeypatch.setattr(private_files.os, "name", "nt")
    private_files.restrict_permissions(tmp_path, 0o700)
    assert called == [tmp_path]


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


def test_sqlite_all_operations_close_connections_before_return(tmp_path, monkeypatch):
    import sqlite3
    from nexus_agent_ai.agent import persistence

    opened = []
    original_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def close(self):
            self.closed_explicitly = True
            super().close()

    def connect(*args, **kwargs):
        connection = original_connect(*args, factory=TrackedConnection, **kwargs)
        connection.closed_explicitly = False
        opened.append(connection)  # retain references so GC cannot hide leaked handles
        return connection

    monkeypatch.setattr(persistence.sqlite3, "connect", connect)
    path = tmp_path / "private" / "history.db"
    memory = SQLiteMemory(path, session_id="close")
    memory.add("user", "request")
    assert memory.get() == [{"role": "user", "content": "request"}]
    memory.replace([{"role": "user", "content": "replacement"}])
    assert SQLiteMemory.list_sessions(path)
    SQLiteMemory.delete_session("close", path)
    memory.clear()
    assert opened and all(connection.closed_explicitly for connection in opened)
    path.unlink()  # on Windows this fails while any SQLite handle is still open
    assert not path.exists()


@pytest.mark.parametrize("kind", ["anthropic", "gemini"])
def test_legacy_native_session_migrates_to_canonical_json_on_load(tmp_path, kind):
    import json
    import sqlite3
    from contextlib import closing

    path = tmp_path / "private" / "history.db"
    memory = SQLiteMemory(path, session_id="legacy")
    if kind == "anthropic":
        old = [
            {"role": "user", "content": "read a.py"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "old",
                                              "name": "read_file", "input": {"path": "a.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old", "content": "source"}]},
        ]
    else:
        old = [
            {"role": "user", "parts": [{"text": "read a.py"}]},
            {"role": "model", "parts": [{"function_call": {"name": "read_file", "args": {"path": "a.py"}}}]},
            {"role": "user", "parts": [{"function_response": {"name": "read_file", "response": {"result": "source"}}}]},
        ]
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executemany(
            "INSERT INTO conversation_history (session_id, role, message_json) VALUES (?, ?, ?)",
            [("legacy", message["role"], json.dumps(message)) for message in old],
        )
    loaded = memory.get()
    assert [message["role"] for message in loaded] == ["user", "assistant", "tool"]
    assert loaded[2]["tool_call_id"] == loaded[1]["tool_calls"][0]["id"]
    assert loaded[2]["content"] == "source"
    with closing(sqlite3.connect(path)) as conn:
        stored = [json.loads(row[0]) for row in conn.execute(
            "SELECT message_json FROM conversation_history ORDER BY id",
        )]
    assert stored == loaded  # migration is durable, not a read-only compatibility view
    assert memory.get() == loaded


def test_session_storage_is_private_before_sqlite_opens(tmp_path, monkeypatch):
    from nexus_agent_ai.agent import persistence

    restricted = []
    original = persistence.restrict_permissions

    def restrict(path, mode):
        original(path, mode)
        restricted.append((path, mode))

    monkeypatch.setattr(persistence, "restrict_permissions", restrict)
    path = tmp_path / "private" / "history.db"
    SQLiteMemory(path, session_id="private")
    assert restricted == [(path.parent, 0o700), (path, 0o600)]
    if os.name != "nt":
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600


def test_private_permissions_fail_closed(tmp_path, monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(private_files, "_restrict_windows",
                            lambda path: (_ for _ in ()).throw(OSError("ACL rejected")))
    else:
        monkeypatch.setattr(private_files.os, "chmod",
                            lambda *args: (_ for _ in ()).throw(OSError("chmod rejected")))
    with pytest.raises(OSError, match="rejected"):
        private_files.restrict_permissions(tmp_path, 0o700)
