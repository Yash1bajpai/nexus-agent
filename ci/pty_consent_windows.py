"""Drive the run_tests permission prompt through a real Windows pseudo-console (ConPTY via pywinpty).

Controlled scratch test only: a throwaway folder with one known passing test and a throwaway
NEXUS_AGENT_HOME. The answers are typed by this script into the prompt, so this checks that the prompt
works on a Windows console. It is not evidence about real users, real models or untrusted projects.
Usage: python pty_consent_windows.py <python-exe-of-installed-wheel>
"""
import json
import os
import subprocess
import sys
import tempfile
import time

from winpty import PtyProcess

PY = sys.argv[1]
home = tempfile.mkdtemp()
work = tempfile.mkdtemp()
sub = os.path.join(work, "sub")
os.mkdir(sub)
with open(os.path.join(work, "test_ok.py"), "w") as f:
    f.write("def test_ok():\n    assert True\n")
CODE = ("from nexus_agent_ai.agent.tools import execute_run_tests as r;"
        "print('RESULT:', r().splitlines()[0][:70]);print('RESULT2:', r().splitlines()[0][:70])")


def run(answer, cwd=work):
    env = dict(os.environ, NEXUS_AGENT_HOME=home)
    env.pop("NEXUS_ALLOW_PROJECT_EXECUTION", None)
    p = PtyProcess.spawn([PY, "-c", CODE], cwd=cwd, env=env)
    out, sent, t = "", False, time.time()
    while time.time() - t < 90 and (p.isalive() or True):
        try:
            d = p.read(1024)
        except EOFError:
            break
        out += d
        if "Allow?" in out and not sent and answer is not None:
            p.write(answer + "\r\n")
            sent = True
    lines = [l.strip() for l in out.replace("\r", "").splitlines() if l.strip().startswith("RESULT")]
    return ("asked" if "Allow?" in out else "no-prompt"), lines


def ok(cond, label, res):
    print(("PASS " if cond else "FAIL ") + label, res)
    return cond


results = []
r = run("n"); results.append(ok(r[0] == "asked" and all("not approved" in l for l in r[1]) and len(r[1]) == 2, "denied", r))
r = run(""); results.append(ok(r[0] == "asked" and all("not approved" in l for l in r[1]), "empty = no", r))
r = run("o"); results.append(ok(r[0] == "asked" and all("PASSED" in l for l in r[1]) and not os.path.exists(os.path.join(home, "trusted_folders.json")), "once (not saved)", r))
r = run("n"); results.append(ok(r[0] == "asked", "once does not persist to a new process", r))
r = run("a"); saved = json.load(open(os.path.join(home, "trusted_folders.json")))
results.append(ok(r[0] == "asked" and all("PASSED" in l for l in r[1]) and len(saved["folders"]) == 1, "always (saved exact folder)", (r, saved)))
r = run(None); results.append(ok(r[0] == "no-prompt" and all("PASSED" in l for l in r[1]), "after always, no prompt", r))
r = run("n", cwd=sub); results.append(ok(r[0] == "asked", "subfolder not covered", r))
env = dict(os.environ, NEXUS_AGENT_HOME=home)
cli = os.path.join(os.path.dirname(PY), "nexus-agent-ai.exe")
print(subprocess.run([cli, "trust", "revoke", work], env=env, capture_output=True, text=True).stdout.strip())
r = run("n"); results.append(ok(r[0] == "asked", "after revoke asked again", r))
sys.exit(0 if all(results) else 1)
