import os
import stat
import sys

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import tools as T
from nexus_agent_ai.agent.core import Agent, parse_at_mentions
from nexus_agent_ai.cli import app as appmod
from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


@pytest.fixture(autouse=True)
def fresh_tracking(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    T.reset_file_tracking()
    yield
    T.reset_file_tracking()


# ---- read before edit -------------------------------------------------------

def test_patch_requires_prior_read(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    res = T.execute_patch_file("a.txt", "hello", "bye")
    assert res.startswith("ERROR: Read 'a.txt' with read_file")
    assert (tmp_path / "a.txt").read_text() == "hello world"
    T.execute_read_file("a.txt")
    assert "Successfully patched" in T.execute_patch_file("a.txt", "hello", "bye")


def test_write_over_existing_requires_read_but_new_file_does_not(tmp_path):
    (tmp_path / "a.txt").write_text("old")
    assert T.execute_write_file("a.txt", "new").startswith("ERROR: Read")
    assert "Successfully wrote" in T.execute_write_file("brand_new.txt", "x")
    T.execute_read_file("a.txt")
    assert "Successfully wrote" in T.execute_write_file("a.txt", "new")


def test_edit_refused_when_file_changed_after_read(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one two")
    T.execute_read_file("a.txt")
    p.write_text("one two three")  # changed outside the agent
    res = T.execute_patch_file("a.txt", "one", "1")
    assert "changed on disk" in res
    T.execute_read_file("a.txt")
    assert "Successfully patched" in T.execute_patch_file("a.txt", "one", "1")


def test_consecutive_patches_do_not_need_rereading(tmp_path):
    (tmp_path / "a.txt").write_text("aaa bbb")
    T.execute_read_file("a.txt")
    assert "Successfully" in T.execute_patch_file("a.txt", "aaa", "AAA")
    assert "Successfully" in T.execute_patch_file("a.txt", "bbb", "BBB")
    assert (tmp_path / "a.txt").read_text() == "AAA BBB"


def test_at_mention_counts_as_a_read(tmp_path):
    (tmp_path / "m.txt").write_text("keep me")
    parse_at_mentions("explain @m.txt")
    assert "Successfully patched" in T.execute_patch_file("m.txt", "keep", "kept")


# ---- wrong target recovery --------------------------------------------------

def test_crlf_file_patched_with_lf_target_keeps_crlf(tmp_path):
    p = tmp_path / "w.txt"
    p.write_bytes(b"line one\r\nline two\r\nline three\r\n")
    T.execute_read_file("w.txt")
    res = T.execute_patch_file("w.txt", "line one\nline two", "first\nsecond")
    assert "Successfully patched" in res
    assert p.read_bytes() == b"first\r\nsecond\r\nline three\r\n"


def test_trailing_whitespace_difference_is_tolerated(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("def f():   \n    return 1  \n")
    T.execute_read_file("t.py")
    res = T.execute_patch_file("t.py", "def f():\n    return 1", "def f():\n    return 2")
    assert "Successfully patched" in res and "trailing whitespace" in res
    assert "return 2" in p.read_text()


def test_not_found_shows_closest_real_text(tmp_path):
    p = tmp_path / "c.py"
    p.write_text("def compute_total(items):\n    return sum(items)\n")
    T.execute_read_file("c.py")
    res = T.execute_patch_file("c.py", "def compute_totals(items):\n    return sum(items)", "x")
    assert res.startswith("ERROR: Target content not found")
    assert "Closest text in the file" in res and "1: def compute_total(items):" in res


def test_not_found_without_similar_text_has_no_hint(tmp_path):
    (tmp_path / "c.txt").write_text("alpha beta gamma")
    T.execute_read_file("c.txt")
    res = T.execute_patch_file("c.txt", "zzzzzzzzzzzz qqqqqqqq", "x")
    assert res.startswith("ERROR: Target content not found")
    assert "Closest" not in res


# ---- exit code / unresolved errors -----------------------------------------

class Scripted(BaseProvider):
    def __init__(self, steps):
        self.steps, self.i = steps, 0
        self.model = "stub"

    def complete(self, messages, tools, system):
        step = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if step is None:
            return ProviderResponse(text="All done")
        name, args = step
        return ProviderResponse(tool_calls=[ToolCall(f"c{self.i}", name, args)],
                                raw_assistant_message={"role": "assistant", "content": "x"})

    def stream(self, **kw):
        yield self.complete(**kw)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def test_recovered_error_is_not_reported_as_failure(tmp_path):
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([
        ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}),  # refused: unread
        ("read_file", {"path": "f.txt"}),
        ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}),
        None,
    ])
    agent = Agent(prov, verbose=False)
    out = agent.run("fix f.txt")
    assert agent.unresolved_errors == []
    assert "Tool errors occurred" not in out
    assert (tmp_path / "f.txt").read_text() == "beta"


def test_unfixed_error_stays_and_chat_exits_nonzero(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None])
    agent = Agent(prov, verbose=False)
    out = agent.run("fix f.txt")
    assert agent.unresolved_errors and "Tool errors occurred" in out

    T.reset_file_tracking()
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance",
                        lambda *a, **k: (Scripted([("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None]), "stub"))
    result = CliRunner().invoke(appmod.app, ["chat", "fix f.txt", "-p", "local", "--no-stream"])
    assert result.exit_code == 1


def test_clean_run_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance", lambda *a, **k: (Scripted([None]), "stub"))
    result = CliRunner().invoke(appmod.app, ["chat", "hi", "-p", "local", "--no-stream"])
    assert result.exit_code == 0


# ---- key storage ------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_env_file_created_private(tmp_path, monkeypatch):
    env = tmp_path / "cfg" / ".env"
    monkeypatch.setattr(onboarding, "ENV_FILE", env)
    onboarding._write_env_key("GEMINI_API_KEY", "secret-value-123")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text() == "GEMINI_API_KEY=secret-value-123\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_existing_world_readable_env_is_tightened_and_updated(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=old\nOTHER=1")
    os.chmod(env, 0o644)
    monkeypatch.setattr(onboarding, "ENV_FILE", env)
    onboarding._write_env_key("OPENAI_API_KEY", "newvalue123")
    onboarding._write_env_key("ANTHROPIC_API_KEY", "abc12345")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text() == "OPENAI_API_KEY=newvalue123\nOTHER=1\nANTHROPIC_API_KEY=abc12345\n"
    assert not (tmp_path / ".env.tmp").exists()


def test_api_key_prompt_is_hidden(monkeypatch):
    seen = {"secret": [], "plain": []}
    monkeypatch.setattr(onboarding, "_secret_input", lambda p: seen["secret"].append(p) or "")
    monkeypatch.setattr(onboarding, "_input", lambda p: seen["plain"].append(p) or "")
    for k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    onboarding._step_api_keys()
    assert len(seen["secret"]) == 3 and not seen["plain"]


def test_secret_input_uses_getpass(monkeypatch):
    import getpass
    called = {}
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": called.setdefault("hit", True) and "k")
    assert onboarding._secret_input("Key: ") == "k"
    assert called["hit"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_doctor_warns_on_readable_key_file(tmp_path, monkeypatch):
    from nexus_agent_ai.utils import doctor
    home = tmp_path / "home"
    (home / ".nexus-agent").mkdir(parents=True)
    env = home / ".nexus-agent" / ".env"
    env.write_text("X=1\n")
    os.chmod(env, 0o644)
    monkeypatch.setattr(doctor.Path, "home", classmethod(lambda cls: home))
    checks = doctor.run_checks(probe_network=False)
    assert any(c[1] == "Key file" and c[0] == "warn" for c in checks)
