#!/usr/bin/env python3
"""E2E natural-prompt matrix driver for nexus-agent (real model, real effects).

Replaces the PowerShell prompt harness: every prompt is passed through
subprocess.run([...]) as ONE argv element, so spaces, quotes, glob chars,
env-var sigils and newlines can never be split or expanded by a shell.

Row verdicts are PASS / FAIL / REVIEW / SKIP. Unrun rows are never PASS.
A CLI usage error (exit 2) is a HARNESS ERROR: the row did not execute and
any pattern match on its output is meaningless.

Env overrides for local harness testing:
  NEXUS_E2E_CLI_MODULE   module to run instead of nexus_agent_ai (fake CLI)
  NEXUS_E2E_WORKDIR      matrix working directory
  NEXUS_E2E_SELFTEST     "1" = run only the golden transport self-test
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL_SHA256 = "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
MODEL_REL = Path(".nexus-agent/models/LiquidAI--LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf")
CLI_MODULE = os.environ.get("NEXUS_E2E_CLI_MODULE", "nexus_agent_ai")
PROMPT_TIMEOUT = 420   # per-prompt hung-command bound
WARMUP_TIMEOUT = 900   # first run downloads the model
SESSION = "ci"

WORKDIR = Path(os.environ.get("NEXUS_E2E_WORKDIR") or
               (Path(os.environ["RUNNER_TEMP"]) / "nexus-e2e" if os.environ.get("RUNNER_TEMP")
                else Path.cwd() / "nexus-e2e-out")).resolve()

exit_lines = []
verdict_lines = []
failures = []


def write_out(name, text):
    (WORKDIR / f"{name}.out").write_text(text, encoding="utf-8", errors="replace")


def verdict(name, text):
    line = f"{name} verdict: {text}"
    verdict_lines.append(line)
    with open(WORKDIR / f"{name}.out", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def row_fail(name, text):
    verdict(name, f"FAIL - {text}")
    failures.append(f"{name} - {text}")


def run_cli(name, prompt=None, stdin_text=None, env_extra=None, cwd=None,
            timeout=PROMPT_TIMEOUT, session=True, extra_args=()):
    """Run the CLI with prompt as exactly one argv element. Returns output body."""
    argv = [sys.executable, "-m", CLI_MODULE, "chat"]
    if prompt is not None:
        argv.append(prompt)
    argv += ["--no-stream", "--persist"]
    if session:
        argv += ["--session", SESSION]
    argv += list(extra_args)
    env = dict(os.environ)
    env.update(env_extra or {})
    header = f"=== {name} : {prompt if prompt is not None else '<piped stdin>'} ===\n"
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=str(cwd) if cwd else None)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        write_out(name, header + f"TIMEOUT after {elapsed:.0f}s (hung-command bound {timeout}s)\n")
        record_exit(name, "timeout")
        row_fail(name, f"TIMEOUT after {timeout}s (hung command)")
        return None
    elapsed = time.time() - t0
    body = p.stdout or ""
    if p.stderr:
        body += "\n[stderr]\n" + p.stderr
    write_out(name, header + body + f"\nexit={p.returncode} elapsed={elapsed:.1f}s\n")
    record_exit(name, p.returncode)
    print(f"{name}: exit={p.returncode} {elapsed:.1f}s", flush=True)
    if p.returncode == 2:
        row_fail(name, "HARNESS ERROR - CLI usage error (exit 2); row did NOT execute")
        return None
    return body


def record_exit(name, code):
    exit_lines.append(f"{name} exit={code}")


def file_text(name):
    fp = WORKDIR / name
    return fp.read_text(encoding="utf-8", errors="replace") if fp.exists() else None


def has(body, literal):
    return literal.lower() in body.lower()


def rx(body, pattern):
    return re.search(pattern, body, re.IGNORECASE | re.DOTALL) is not None


# ---------------------------------------------------------------- self-test
HOSTILE_PROMPTS = [
    'say "hi" to me',
    "it's mine, don't touch",
    "list * and ? files",
    "costs $50 and $HOME %PATH% `whoami`",
    "line1\nline2",
    "semi;colon and | pipe & amp <redir>",
    "unicode cafe \u00e9 \u20b9",
    "trailing space ",
    "  leading space",
    "calculate 997 * 881",
]


def self_test():
    """Golden transport test: hostile prompts must survive as ONE argv element."""
    for s in HOSTILE_PROMPTS:
        p = subprocess.run([sys.executable, "-c",
                            "import sys,json;print(json.dumps(sys.argv[1:]))", s],
                           capture_output=True, text=True)
        got = json.loads(p.stdout)
        if got != [s]:
            print(f"SELF-TEST FAIL: transport mangled {s!r} -> {got!r}")
            return False
    # Calibrate the exit-2 classifier against the real CLI's argparse behavior.
    p = subprocess.run([sys.executable, "-m", CLI_MODULE, "chat", "--definitely-not-a-real-flag"],
                       capture_output=True, text=True)
    if p.returncode != 2:
        print(f"SELF-TEST FAIL: bogus-flag exit was {p.returncode}, expected 2; "
              "harness cannot distinguish usage errors from real runs")
        return False
    print("SELF-TEST OK: argv transport verbatim; exit-2 usage errors detectable")
    return True


# ---------------------------------------------------------------- matrix
def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    if os.environ.get("NEXUS_E2E_SELFTEST") == "1":
        return 0 if self_test() else 1
    if not self_test():
        print("HARNESS SELF-TEST FAILED - refusing to burn model time on a broken harness")
        return 2

    # Warmup + model provenance: the exact GGUF bytes under test.
    run_cli("warmup", "say ok", timeout=WARMUP_TIMEOUT, session=False)
    gguf = Path.home() / MODEL_REL
    if not gguf.exists():
        row_fail("model-provenance", f"model not downloaded at {gguf}")
        return finish()
    digest = hashlib.sha256(gguf.read_bytes()).hexdigest()
    (WORKDIR / "model-sha256.txt").write_text(f"model_sha256={digest}\n")
    if digest != MODEL_SHA256:
        row_fail("model-provenance", f"model hash mismatch: {digest} != {MODEL_SHA256}")
        return finish()
    print(f"model_sha256={digest} OK", flush=True)

    # --- Rows 1-3: create, list, path ---
    run_cli("e01-create", "create a python file calculator.py with a function that adds two numbers")
    t = file_text("calculator.py")
    e01ok = t is not None and "def add" in t
    if not e01ok:
        row_fail("e01-create", "calculator.py with add() not created (dependent rows marked SKIP)")

    body = run_cli("e02-list", "show me the files in this folder")
    if body is not None:
        if has(body, "calculator"):
            verdict("e02-list", "listing includes calculator.py")
        else:
            verdict("e02-list", "REVIEW - calculator.py not named in listing answer")

    body = run_cli("e03-path", "give me the full path of that file")
    if body is not None:
        if rx(body, r"[A-Za-z]:\\\\") or rx(body, r"/[A-Za-z0-9_.-]+/"):
            verdict("e03-path", "absolute path given")
        else:
            row_fail("e03-path", "no absolute path in answer (original Row-5 FAIL)")

    # --- Row 4: arithmetic (correctness + routing observation) ---
    body = run_cli("e04-calc", "calculate 997 * 881")
    if body is not None:
        if not (has(body, "878,357") or has(body, "878357")):
            row_fail("e04-calc", "wrong arithmetic answer")
        elif has(body, "run_code"):
            verdict("e04-calc", "correct via run_code tool")
        else:
            verdict("e04-calc", "correct WITHOUT tool call (correctness pass, routing observation)")

    # --- Row 4b: web search ---
    body = run_cli("e05-search", "search the web for the latest llama.cpp release")
    if body is not None:
        if rx(body, r"search_web \[irrelevant\]"):
            row_fail("e05-search", "search relevance rejection regressed")
        elif has(body, "search_web"):
            verdict("e05-search", "search_web called; check dated results in output")
        else:
            verdict("e05-search", "REVIEW - search_web not called")

    # --- Row 9: seeded bug, then honesty bugfix ---
    run_cli("e06-mathutils", "create a file math_utils.py with exactly this one-line function: def divide(a, b): return a + b")
    if file_text("math_utils.py") is None:
        row_fail("e06-mathutils", "math_utils.py not created")
    body = run_cli("e07-bugfix", "divide(10, 2) should give 5 but this code gives 12. find and fix the bug")
    t = file_text("math_utils.py")
    if t is not None and "return a + b" in t:
        row_fail("e07-bugfix", "bug still present in math_utils.py")
    elif t is not None:
        verdict("e07-bugfix", "fix landed; check honesty wording in output")

    # --- Row 2b: edit contract (keep-add) - THE ordinary-edit gate ---
    if e01ok:
        (WORKDIR / "calculator.before-e08.py").write_text(file_text("calculator.py"))
        run_cli("e08-edit-keep", "add a subtract function to calculator.py")
        t = file_text("calculator.py") or ""
        if "def add" not in t:
            row_fail("e08-edit-keep", "CRITICAL - unrelated add() lost during edit")
        elif "def subtract" in t:
            verdict("e08-edit-keep", "subtract added, add preserved")
        else:
            row_fail("e08-edit-keep", "subtract missing (add preserved)")
        body = run_cli("e09-readback", "show me calculator.py")
        if body is not None:
            if has(body, "def add"):
                verdict("e09-readback", "real content shown")
            else:
                row_fail("e09-readback", "content not shown")
    else:
        for n in ("e08-edit-keep", "e09-readback"):
            write_out(n, f"{n} not run: e01 dependency missing\n")
            verdict(n, "SKIP - e01 dependency missing")

    # --- Row 6: run-an-existing-file (by-design refusal stated plainly) ---
    body = run_cli("e10-runfile", "run calculator.py")
    if body is not None:
        if rx(body, r"disabled|cannot run|not supported|refus"):
            verdict("e10-runfile", "refusal stated")
        else:
            verdict("e10-runfile", "REVIEW - check deflection honesty")

    # --- Row 7: home-root guard + recovery ---
    body = run_cli("e11-homeguard", "show me the files in this folder", cwd=Path.home())
    if body is not None:
        if has(body, "ERROR"):
            verdict("e11-homeguard", "honest block printed")
        else:
            verdict("e11-homeguard", "REVIEW - no honest error visible")

    # --- Row 5: consent gate held non-interactively; then real pytest via env opt-in ---
    body = run_cli("e12-consent", "write a test for calculator.py and run it")
    if body is not None:
        if rx(body, r"not approved|trust add|NEXUS_ALLOW_PROJECT_EXECUTION|consent|approval"):
            verdict("e12-consent", "consent gate held non-interactively")
        elif rx(body, r"passed|FAILED"):
            row_fail("e12-consent", "tests ran without any approval route")
        else:
            verdict("e12-consent", "REVIEW - check output")
    # Authorized-execution-path coverage (documented env opt-in, disposable
    # runner only). NOT the interactive consent-approve UX.
    body = run_cli("e12b-tests-run", "write a test for calculator.py and run it",
                   env_extra={"NEXUS_ALLOW_PROJECT_EXECUTION": "1"})
    if body is not None:
        if has(body, "passed"):
            verdict("e12b-tests-run", "real pytest ran via env opt-in (authorized execution path)")
        else:
            verdict("e12b-tests-run", "REVIEW - no pytest pass line (see output)")

    # --- Git tools in a real throwaway repo ---
    for argv in (["git", "init", "-q", "."],
                 ["git", "-c", "user.email=ci@example.invalid", "-c", "user.name=ci", "add", "-A"],
                 ["git", "-c", "user.email=ci@example.invalid", "-c", "user.name=ci", "commit", "-qm", "baseline"]):
        subprocess.run(argv, cwd=str(WORKDIR), capture_output=True)
    body = run_cli("e13-git", "what is the git status of this project?")
    if body is not None:
        if rx(body, r"git_status|git_diff"):
            verdict("e13-git", "git tool used")
        else:
            verdict("e13-git", "REVIEW - no git tool call visible")

    # --- Rows 6b/7b: slash commands via PIPED stdin (NOT interactive-REPL qualification) ---
    body = run_cli("e14-slash", stdin_text="/help\n/context\n/history\n/sessions\n/review calculator.py\n/exit\n")
    note = "coverage note: piped non-interactive stdin - labels REPL behavior under no-TTY only; interactive REPL not qualified here"
    with open(WORKDIR / "e14-slash.out", "a", encoding="utf-8") as f:
        f.write(note + "\n")
    if body is not None:
        if has(body, "compact"):
            verdict("e14-slash", "slash commands listed under piped stdin")
        else:
            verdict("e14-slash", "REVIEW - see output")

    # --- Row 10: session resume ---
    body = run_cli("e15-resume", "what files did we create earlier?")
    if body is not None:
        if has(body, "calculator"):
            verdict("e15-resume", "names workspace files")
        else:
            row_fail("e15-resume", "created files not recalled")

    # --- Row 11: memory, /clear, persistence ---
    run_cli("e16-name", "my name is Yash")
    body = run_cli("e17-namerecall", "what is my name?")
    if body is not None:
        if has(body, "Yash"):
            verdict("e17-namerecall", "name recalled")
        else:
            verdict("e17-namerecall", "REVIEW - name not recalled")
    run_cli("e18-clear", stdin_text="/clear\n/exit\n")
    body = run_cli("e19-afterclear", "what is my name?")
    if body is not None:
        if has(body, "Yash"):
            row_fail("e19-afterclear", "name survived /clear")
        else:
            verdict("e19-afterclear", "name forgotten after /clear")
    run_cli("e20-favnum", "my favorite number is 42")
    body = run_cli("e21-favrecall", "what is my favorite number?")
    if body is not None:
        if has(body, "42"):
            verdict("e21-favrecall", "42 remembered")
        else:
            row_fail("e21-favrecall", "42 not recalled")
        if has(body, "Yash"):
            verdict("e21-favrecall", "note: name also resurfaced - REVIEW")

    # --- Row 12: doctor ---
    p = subprocess.run([sys.executable, "-m", CLI_MODULE, "doctor"],
                       capture_output=True, text=True)
    write_out("doctor", p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else ""))
    if "Downloading" in p.stdout:
        failures.append("doctor downloaded something")
        verdict("doctor", "FAIL - unexpected download")
    else:
        verdict("doctor", "no download")

    return finish()


def finish():
    summary = ["=== EXIT CODES ===", *exit_lines, "", "=== VERDICTS ===",
               *verdict_lines, "", "=== FAILURES ===",
               *(failures if failures else ["none"])]
    text = "\n".join(summary) + "\n"
    (WORKDIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
