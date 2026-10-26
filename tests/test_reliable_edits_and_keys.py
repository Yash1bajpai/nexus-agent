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
from nexus_agent_ai.providers.messages import assistant_message


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
        calls = [ToolCall(f"c{self.i}", name, args)]
        return ProviderResponse(tool_calls=calls, raw_assistant_message=assistant_message("", calls))

    def stream(self, **kw):
        yield self.complete(**kw)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def test_recovered_error_is_not_reported_as_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")  # exercise the refused-then-read path itself
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
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
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


def test_reading_a_file_does_not_clear_a_failed_edit(tmp_path, monkeypatch):
    """Regression: a failed patch followed by a successful read_file of the same file used to count as fixed,
    so the CLI exited 0 while the file was unchanged."""
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([
        ("patch_file", {"path": "f.txt", "target": "nothere", "replacement": "beta"}),
        ("read_file", {"path": "f.txt"}),
        None,
    ])
    agent = Agent(prov, verbose=False)
    out = agent.run("fix f.txt")
    assert agent.unresolved_errors and "Tool errors occurred" in out
    assert (tmp_path / "f.txt").read_text() == "alpha"


class _Say(BaseProvider):
    def __init__(self, text):
        self.text, self.model = text, "stub"

    def complete(self, messages, tools, system):
        return ProviderResponse(text=self.text)

    def stream(self, **kw):
        yield self.complete(**kw)

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}


def test_false_claim_of_edit_without_write_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    agent = Agent(_Say("I've fixed the bug in f.txt."), verbose=False)
    out = agent.run("fix f.txt")
    assert agent.claimed_unapplied_edit
    assert "false" in out and (tmp_path / "f.txt").read_text() == "alpha"


@pytest.mark.parametrize("text", [
    "f.txt is already correct, nothing to change.",
    "No changes are needed here.",
    "I could not apply a fix because the target was not found.",
    "That is not a bug, it behaves as designed.",
])
def test_legitimate_no_change_answers_are_not_flagged(tmp_path, monkeypatch, text):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    agent = Agent(_Say(text), verbose=False)
    agent.run("fix f.txt")
    assert not agent.claimed_unapplied_edit


def test_real_edit_is_not_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None])
    agent = Agent(prov, verbose=False)
    agent.run("fix f.txt")
    assert not agent.claimed_unapplied_edit and (tmp_path / "f.txt").read_text() == "beta"


def test_cli_exits_nonzero_on_false_edit_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance", lambda *a, **k: (_Say("I've fixed the bug in f.txt."), "stub"))
    result = CliRunner().invoke(appmod.app, ["chat", "fix f.txt", "-p", "local", "--no-stream"])
    assert result.exit_code == 1


def test_cli_exits_zero_when_nothing_to_change(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance", lambda *a, **k: (_Say("f.txt is already correct."), "stub"))
    result = CliRunner().invoke(appmod.app, ["chat", "fix f.txt", "-p", "local", "--no-stream"])
    assert result.exit_code == 0


class _Local(Scripted):
    """Looks like a local model (name contains 'local') and first just talks, then acts when nudged."""
    def __init__(self, steps):
        super().__init__(steps)
        self.model = "lfm-stub"

    def complete(self, messages, tools, system):
        step = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if isinstance(step, str):
            return ProviderResponse(text=step)
        if step is None:
            return ProviderResponse(text="Done, patched.")
        name, args = step
        calls = [ToolCall(f"c{self.i}", name, args)]
        return ProviderResponse(tool_calls=calls, raw_assistant_message=assistant_message("", calls))


def test_local_model_that_only_talks_gets_nudged_to_edit(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("alpha")
    prov = _Local(["The bug is that alpha is wrong; fixing it will help.",
                   ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None])
    agent = Agent(prov, verbose=False)
    agent.run("fix f.txt")   # f.txt auto-attached, so patch_file is allowed
    assert (tmp_path / "f.txt").read_text() == "beta"


def test_local_model_saying_nothing_to_change_is_not_nudged(tmp_path):
    (tmp_path / "f.txt").write_text("alpha")
    prov = _Local(["f.txt is already correct, nothing to change."])
    agent = Agent(prov, verbose=False)
    out = agent.run("fix f.txt")
    assert prov.i == 1 and (tmp_path / "f.txt").read_text() == "alpha"


def test_claim_tests_pass_without_running_is_not_success(tmp_path, monkeypatch):
    """A written file is not enough: 'tests pass' without a passing run exits 1."""
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None])
    class P(Scripted):
        def complete(self, messages, tools, system):
            r = super().complete(messages, tools, system)
            if r.tool_calls == [] or not r.tool_calls:
                return ProviderResponse(text="Fixed the bug. All tests pass now.")
            return r
    agent = Agent(P(prov.steps), verbose=False)
    out = agent.run("fix f.txt")
    assert (tmp_path / "f.txt").read_text() == "beta"
    assert agent.claim_unverified and "unverified" in out
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: None)
    monkeypatch.setattr(appmod, "get_provider_instance", lambda *a, **k: (P(prov.steps), "stub"))
    T.reset_file_tracking()
    (tmp_path / "f.txt").write_text("alpha")
    result = CliRunner().invoke(appmod.app, ["chat", "fix f.txt", "-p", "local", "--no-stream"])
    assert result.exit_code == 1



def test_rereading_the_same_file_does_not_end_an_edit_request_early(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("read_file", {"path": "f.txt"}), ("read_file", {"path": "f.txt"}),
                     ("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}), None])
    Agent(prov, verbose=False).run("fix f.txt")
    assert (tmp_path / "f.txt").read_text() == "beta"


def test_forced_loop_stop_still_says_no_file_was_changed(tmp_path, monkeypatch):
    """The anti-loop stop removes the tools; the 'no file was changed' warning must survive that."""
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("read_file", {"path": "f.txt"})] * 4 + [None])
    out = Agent(prov, verbose=False).run("fix f.txt")
    assert (tmp_path / "f.txt").read_text() == "alpha"
    assert "no file was changed" in out


def test_no_change_after_edit_request_is_incomplete_but_valid_no_change_is_not(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    a = Agent(Scripted([("read_file", {"path": "f.txt"}), None]), verbose=False)
    a.run("fix f.txt")
    assert a.edit_not_applied is True
    b = Agent(_Local(["f.txt is already correct, nothing to change."]), verbose=False)
    b.run("fix f.txt")
    assert b.edit_not_applied is False
    c = Agent(Scripted([None]), verbose=False)
    c.run("explain what f.txt does")
    assert c.edit_not_applied is False


def test_chat_exits_1_when_edit_requested_and_nothing_changed(tmp_path, monkeypatch):
    import nexus_agent_ai.cli.app as appmod
    from typer.testing import CliRunner
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    monkeypatch.setattr(appmod, "get_provider_instance",
                        lambda *a, **k: (Scripted([("read_file", {"path": "f.txt"}), None]), "stub"))
    r = CliRunner().invoke(appmod.app, ["chat", "-p", "local", "fix f.txt"])
    assert r.exit_code == 1


def test_reread_after_a_successful_edit_and_a_failed_patch_is_not_a_loop(tmp_path, monkeypatch):
    """Transcript from the 2.6B diagnostic (sub.py): read, read again, patch ok, read, failed patch, read again.
    The last read is needed to recover and must not be treated as a repeat of the reads made before the edit."""
    monkeypatch.setenv("NEXUS_AGENT_AUTO_ATTACH", "0")
    (tmp_path / "f.txt").write_text("alpha")
    prov = Scripted([("read_file", {"path": "f.txt"}), ("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "gamma"}),
                     ("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "alpha", "replacement": "beta"}),  # stale target: fails
                     ("read_file", {"path": "f.txt"}),
                     ("patch_file", {"path": "f.txt", "target": "gamma", "replacement": "beta"}), None])
    Agent(prov, verbose=False).run("fix f.txt")
    assert (tmp_path / "f.txt").read_text() == "beta"
