"""Real redirected stdin; no console API mocks, model setup, or network access.

Onboarding's import legitimately creates an empty ~/.nexus-agent directory and
restricts its permissions. No other home artifacts are allowed by these tests.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_CHILD = textwrap.dedent(r'''
    import contextlib
    import io
    import json
    import os
    from pathlib import Path
    import sys

    # Fail before any input read or network connection, without replacing stdin,
    # input(), or GetConsoleMode. A broken guard must fail, not start setup.
    def forbid_external_interaction(event, args):
        if event == "builtins.input" or event in ("socket.connect", "socket.getaddrinfo"):
            raise AssertionError("Unexpected prompt or network access: " + event)
    sys.addaudithook(forbid_external_interaction)

    home = Path(os.environ["HOME"]).resolve()
    assert Path.home().resolve() == home, "Child did not use the isolated home"
    assert not list(home.iterdir()), "Child home must start empty"
    assert "NEXUS_AGENT_ASSUME_YES" not in os.environ
    assert "NEXUS_ALLOW_PROJECT_EXECUTION" not in os.environ

    from nexus_agent_ai.utils.interactive import stdin_is_interactive
    from nexus_agent_ai.agent import consent
    from nexus_agent_ai.cli import onboarding

    assert onboarding.INIT_FILE == home / ".nexus_agent_initialized"
    assert not onboarding.INIT_FILE.exists()
    assert stdin_is_interactive() is False, "Redirected stdin was treated as interactive"
    assert consent._interactive() is False, "Redirected stdin enabled consent prompts"

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = onboarding.run_if_first_time()
        allowed, reason = consent.execution_allowed()
    assert result is None, "Redirected stdin started first-run setup"
    assert allowed is False and "not approved" in reason, "Piped input granted execution"
    assert consent.execution_allowed_without_asking() is False
    assert consent.offer_run_tests() is False
    assert consent.list_trusted() == []
    assert not onboarding.INIT_FILE.exists(), "Skipped setup marked onboarding complete"

    # Only the empty directory created at onboarding import is legitimate.
    entries = sorted(p.relative_to(home).as_posix() for p in home.rglob("*"))
    assert entries == [".nexus-agent"], ("Unexpected home artifacts", entries)
    assert (home / ".nexus-agent").is_dir()

    report = {"output": output.getvalue(), "isatty": sys.stdin.isatty(), "entries": entries}
    if len(sys.argv) > 1 and sys.argv[1] == "require-nul-tty":
        if not sys.stdin.isatty():
            report["skip"] = (
                "Real Windows NUL stdin reports isatty()=False on this host; "
                "the TTY-like NUL precondition cannot be demonstrated"
            )
        else:
            assert sys.stdin.isatty() is True, "Expected the real Windows NUL TTY quirk"
            assert stdin_is_interactive() is False, "GetConsoleMode accepted NUL as a console"
    print(json.dumps(report))
''')


def _run_child(tmp_path, *, piped=False, require_nul_tty=False):
    # Preserve the parent's resolved import environment: pytest changes cwd and
    # its pythonpath setting is process-local; HOME also hides user-site installs.
    source = Path(__file__).resolve().parents[1] / "src"
    paths = [str(source)] + [str(Path(p).resolve()) for p in sys.path if p]
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               NEXUS_AGENT_HOME=str(tmp_path), PYTHONPATH=os.pathsep.join(paths),
               PYTHONIOENCODING="utf-8", NO_COLOR="1")
    env.pop("NEXUS_AGENT_ASSUME_YES", None)
    env.pop("NEXUS_ALLOW_PROJECT_EXECUTION", None)
    command = [sys.executable, "-B", "-c", _CHILD]
    if require_nul_tty:
        command.append("require-nul-tty")
    kwargs = dict(cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, timeout=60)
    if piped:
        # subprocess.run(input=...) creates a real stdin=PIPE and communicates
        # these bytes; it rejects passing stdin=PIPE explicitly at the same time.
        child = subprocess.run(command, input=b"y\n", **kwargs)
    else:
        with open(os.devnull, "rb") as null:
            child = subprocess.run(command, stdin=null, **kwargs)
    stdout = child.stdout.decode("utf-8", errors="replace")
    stderr = child.stderr.decode("utf-8", errors="replace")
    assert child.returncode == 0, f"Child exit {child.returncode}\n{stdout}\n{stderr}"
    report = json.loads(stdout)
    output = " ".join(report["output"].split())
    assert "Non-interactive run: skipping first-time setup." in output
    assert "Allow?" not in output
    assert "Download [y]es" not in output
    # An input attempt fails via the audit hook, even if the application catches
    # its exception; any printed wizard/consent output must also be absent.
    assert output.startswith("Non-interactive run:")
    assert output.endswith("in a terminal to set up keys.")
    assert not (tmp_path / ".nexus_agent_initialized").exists()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [".nexus-agent"]
    return report


def test_devnull_skips_first_run_and_refuses_consent(tmp_path):
    _run_child(tmp_path)


def test_piped_yes_skips_first_run_and_cannot_grant_consent(tmp_path):
    _run_child(tmp_path, piped=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires real Windows NUL stdin")
def test_windows_nul_isatty_does_not_make_it_a_console(tmp_path):
    report = _run_child(tmp_path, require_nul_tty=True)
    if "skip" in report:
        pytest.skip(report["skip"])
    assert report["isatty"] is True, "Child did not observe the Windows NUL TTY quirk"
