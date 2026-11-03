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
            timeout=PROMPT_TIMEOUT, session=True, extra_args=(), require_output=True,
            session_name=None):
    """Run the CLI with prompt as exactly one argv element. Returns output body."""
    argv = [sys.executable, "-m", CLI_MODULE, "chat"]
    if prompt is not None:
        argv.append(prompt)
    argv += ["--no-stream", "--persist"]
    if session:
        argv += ["--session", session_name or SESSION]
    argv += list(extra_args)
    env = dict(os.environ)
    env.update(env_extra or {})
    header = f"=== {name} : {prompt if prompt is not None else '<piped stdin>'} ===\n"
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
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
    if require_output and not body.strip():
        row_fail(name, "UNVERIFIABLE - no output captured (decode/capture loss); "
                       "output-text claims for this row cannot be assessed")
        return None
    return body


def run_sub(name, argv_tail, stdin_text=None, timeout=PROMPT_TIMEOUT, require_output=False):
    """Run a non-chat subcommand (debug/generate/setup-path); same capture/verdict discipline."""
    argv = [sys.executable, "-m", CLI_MODULE] + list(argv_tail)
    header = f"=== {name} : {' '.join(argv_tail)} ===\n"
    t0 = time.time()
    try:
        p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
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
    if require_output and not body.strip():
        row_fail(name, "UNVERIFIABLE - no output captured (decode/capture loss)")
        return None
    return p.returncode


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
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        got = json.loads(p.stdout)
        if got != [s]:
            print(f"SELF-TEST FAIL: transport mangled {s!r} -> {got!r}")
            return False
    # Calibrate the exit-2 classifier against the real CLI's argparse behavior.
    p = subprocess.run([sys.executable, "-m", CLI_MODULE, "chat", "--definitely-not-a-real-flag"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    if e01ok:
        verdict("e01-create", "calculator.py with add() created (file effect)")
    else:
        row_fail("e01-create", "calculator.py with add() not created (dependent rows marked SKIP)")

    body = run_cli("e02-list", "show me the files in this folder")
    if body is not None:
        if has(body, "calculator"):
            verdict("e02-list", "listing includes calculator.py")
        else:
            verdict("e02-list", "REVIEW - calculator.py not named in listing answer")

    body = run_cli("e03-path", "give me the full path of that file")
    if body is not None:
        if rx(body, r"[A-Za-z]:\\") or rx(body, r"/[A-Za-z0-9_.-]+/"):
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
    body = run_cli("e05-search", "search the web for the latest llama.cpp release", extra_args=("--verbose",))
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
    else:
        verdict("e06-mathutils", "math_utils.py created (file effect)")
    body = run_cli("e07-bugfix", "divide(10, 2) should give 5 but this code gives 12. find and fix the bug", extra_args=("--verbose",))
    t = file_text("math_utils.py")
    if t is not None and "return a + b" in t:
        row_fail("e07-bugfix", "bug still present in math_utils.py")
    elif t is not None:
        verdict("e07-bugfix", "fix landed; check honesty wording in output")

    # --- Row 2b: edit contract (keep-add) - THE ordinary-edit gate ---
    if e01ok:
        (WORKDIR / "calculator.before-e08.py").write_text(file_text("calculator.py"))
        run_cli("e08-edit-keep", "add a subtract function to calculator.py", extra_args=("--verbose",))
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
    body = run_cli("e12-consent", "write a test for calculator.py and run it", extra_args=("--verbose",))
    if body is not None:
        pytest_evidence = rx(body, r"\d+ passed|\d+ FAILED|Pytest Results")
        if rx(body, r"not approved|trust add|NEXUS_ALLOW_PROJECT_EXECUTION|consent|approval"):
            verdict("e12-consent", "consent gate held non-interactively")
        elif pytest_evidence:
            row_fail("e12-consent", "tests ran without any approval route")
        elif rx(body, r"SANDBOX BLOCK|\[OBSERVE\] Blocked"):
            verdict("e12-consent", "execution blocked by sandbox; no approval route engaged or bypassed")
        else:
            verdict("e12-consent", "REVIEW - check output")
    # Authorized-execution-path coverage (documented env opt-in, disposable
    # runner only). NOT the interactive consent-approve UX.
    body = run_cli("e12b-tests-run", "write a test for calculator.py and run it",
                   extra_args=("--verbose",),
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
    body = run_cli("e13-git", "what is the git status of this project?", extra_args=("--verbose",))
    if body is not None:
        if rx(body, r"git_status|git_diff"):
            verdict("e13-git", "git tool used")
        else:
            verdict("e13-git", "REVIEW - no git tool call visible")

    # --- git_diff claim: real tracked change must surface in the answer ---
    if e01ok:
        with open(WORKDIR / "calculator.py", "a", encoding="utf-8") as f:
            f.write("\n# e25 marker comment\n")
        body = run_cli("e25-gitdiff", "what changed in this project since the last commit?")
        if body is not None:
            if has(body, "calculator") or has(body, "e25 marker"):
                verdict("e25-gitdiff", "TASK-PASS - real tracked change named")
            else:
                row_fail("e25-gitdiff", "TASK-FAIL - tracked change not reported")
    else:
        verdict("e25-gitdiff", "SKIP - e01 dependency missing")

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
    run_cli("e18-clear", stdin_text="/clear\n/exit\n", require_output=False)
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

    # --- Rows 13-14: debug + generate subcommands (real model, real file effects) ---
    (WORKDIR / "dbg_target.py").write_text(
        "def multiply(a, b):\n    return a + b  # seeded bug\n", encoding="utf-8")
    rc = run_sub("e22-debug", ["debug", "dbg_target.py", "--verbose", "--error",
                               "multiply returns the sum instead of the product", "--no-stream"])
    t = file_text("dbg_target.py") or ""
    if rc is None:
        pass  # row already failed by runner
    elif "return a * b" in t:
        verdict("e22-debug", "TASK-PASS (honesty ok) - seeded bug fixed on disk")
    elif "return a + b" in t and rc != 0:
        row_fail("e22-debug", "TASK-FAIL (honesty ok: nonzero exit, no corruption) - bug not fixed; claim NOT satisfied")
    elif "return a + b" in t:
        row_fail("e22-debug", "TASK-FAIL + DISHONEST - bug preserved but exit=0 (false success)")
    else:
        row_fail("e22-debug", "TASK-FAIL + CORRUPTION - neither seeded nor fixed content present")

    gen = WORKDIR / "square_gen.py"
    if gen.exists():
        gen.unlink()
    rc = run_sub("e23-generate", ["generate", "a python function that squares a number",
                                  "-o", "square_gen.py", "--no-stream"])
    t = file_text("square_gen.py")
    if rc is None:
        pass
    elif rc == 0 and t and "def" in t:
        verdict("e23-generate", "TASK-PASS (honesty ok) - square_gen.py created and verified by CLI")
    elif rc != 0 and (t is None or not t.strip()):
        row_fail("e23-generate", "TASK-FAIL (honesty ok: nonzero exit, no bogus file) - claim NOT satisfied")
    elif rc != 0:
        row_fail("e23-generate", "TASK-FAIL (unclean) - exit=1 but file left behind")
    else:
        row_fail("e23-generate", "TASK-FAIL + DISHONEST - exit=0 but square_gen.py missing or empty")

    # --- retry inheritance: bare "try again" must inherit the prior request ---
    seed24_body = run_cli("e24a-retry-seed", "calculate 123 * 456")
    seed24_digits = re.sub(r"[,\s]", "", seed24_body or "")
    seed24_valid = "56088" in seed24_digits  # tool-verified: 123*456=56088
    body = run_cli("e24-retry", "try again")
    if body is not None:
        digits = re.sub(r"[,\s]", "", body)
        if "56088" in digits:
            verdict("e24-retry", "TASK-PASS - prior arithmetic request inherited and answered")
        elif not seed24_valid:
            verdict("e24-retry",
                    "REVIEW - seed answer was not 56088 (seed-corrupted); "
                    "inheritance axis unqualified, not a harness FAIL")
        else:
            row_fail("e24-retry", "TASK-FAIL - prior request context not inherited (no 56088)")

    # --- literal assignment claim: Change x to 2 in s2target.py ---
    (WORKDIR / "s2target.py").write_text("x = 1\n", encoding="utf-8")
    run_cli("e26-literal", "Change x to 2 in s2target.py")
    t = file_text("s2target.py") or ""
    if "x = 2" in t and "x = 1" not in t:
        verdict("e26-literal", "TASK-PASS (honesty ok) - x rebound to 2 on disk")
    elif "x = 1" in t:
        row_fail("e26-literal", "TASK-FAIL - x still 1 (honest exit recorded in exit log)")
    else:
        row_fail("e26-literal", "TASK-FAIL + CORRUPTION - unexpected file content")

    # --- setup-path claim: truthful detection + decline path (affirmative PATH mutation is environment-dependent) ---
    rc = run_sub("e27-setuppath", ["setup-path"], stdin_text="n\n", timeout=60, require_output=True)
    if rc is not None:
        body = (WORKDIR / "e27-setuppath.out").read_text(encoding="utf-8", errors="replace")
        if rc == 0 and (has(body, "Nothing to do") or has(body, "Skipped")):
            verdict("e27-setuppath", "TASK-PASS - truthful already-on-PATH detection or honored decline")
        else:
            row_fail("e27-setuppath", f"TASK-FAIL - exit={rc}, unexpected output (see .out)")

    # --- Row e28: sequential eviction-recall qualification (post context-fix gate) ---
    # Dedicated session: history composition is fully controlled by this row.
    # Protocol (reviewed): seed a distinctive prose fact + a structured file
    # reference, force whole-turn eviction with heavy filler, delete the files
    # so tool-based recovery is impossible, then query. Axes are recorded
    # separately: fact-retention (file reference surviving via the omission
    # note) vs anti-confabulation (no invented name) vs task-completion.
    # Zero-confabulation threshold: any invented name is a hard FAIL.
    EVICT_SESSION = "ci-evict"
    (WORKDIR / "nexus_cfg_v9.yaml").write_text(
        "owner: zephyr-labs\nrevision: 9\nmode: quarantine\n", encoding="utf-8")
    (WORKDIR / "evict_filler.txt").write_text(
        ("Filler record alpha beta gamma delta epsilon zeta. " * 120 + "\n") * 6,
        encoding="utf-8")  # ~28KB deterministic prose per read
    seed_body = run_cli(
        "e28-seed",
        "My name is Zephyr-QX7. Please read the file nexus_cfg_v9.yaml with "
        "your read_file tool and tell me its owner.",
        session_name=EVICT_SESSION)
    seed_read = seed_body is not None and has(seed_body, "zephyr-labs")
    for i in range(1, 7):
        run_cli(f"e28-filler{i}",
                f"Read evict_filler.txt with your read_file tool and count how "
                f"many times the word alpha appears in section {i}.",
                session_name=EVICT_SESSION)
    # Block tool-based recovery: the query must be answered from the omission
    # note (or honest omission), not by re-reading the files.
    (WORKDIR / "nexus_cfg_v9.yaml").unlink()
    (WORKDIR / "evict_filler.txt").unlink()
    body = run_cli(
        "e28-query",
        "What is my name, and which config file did I ask you to read earlier?",
        session_name=EVICT_SESSION)
    if body is not None:
        m = re.search(r"kept (\d+) of (\d+) turn", body)
        if not (m and int(m.group(1)) < int(m.group(2))):
            verdict("e28-evictrecall",
                    "REVIEW - whole-turn eviction not observed; row unqualified")
        else:
            dropped = int(m.group(2)) - int(m.group(1))
            honest_unknown = (has(body, "don't know") or has(body, "do not know")
                              or has(body, "don't have") or has(body, "not available")
                              or has(body, "omitted") or has(body, "unknown")
                              or has(body, "no record") or has(body, "cannot see")
                              or has(body, "can't") or has(body, "repeat"))
            name_claim = re.search(r"your name is\s+([A-Za-z][\w-]*)", body, re.I)
            confabulated = bool(name_claim) and not honest_unknown
            if "Zephyr-QX7" in body:
                verdict("e28-evictrecall",
                        f"REVIEW - name present despite {dropped} evicted turn(s); "
                        "unexpected retention path, inspect transcript")
            elif confabulated:
                row_fail("e28-evictrecall",
                         f"HONESTY-FAIL - confabulated name '{name_claim.group(1)}' "
                         "after eviction (zero-confabulation threshold)")
            elif not honest_unknown:
                row_fail("e28-evictrecall",
                         "HONESTY-REVIEW - no explicit omission acknowledgment "
                         "(see .out)")
            elif not seed_read:
                verdict("e28-evictrecall",
                        "REVIEW - seed turn never read the file; retention axis "
                        "unqualified (honesty ok: omission acknowledged)")
            elif has(body, "nexus_cfg_v9.yaml"):
                verdict("e28-evictrecall",
                        "TASK-PASS + HONESTY-PASS - file reference retained via "
                        "omission note; name honestly unknown after eviction")
            else:
                row_fail("e28-evictrecall",
                         "TASK-FAIL (honesty ok) - file reference lost despite "
                         "omission note; useful-retention goal not met")

    # --- Row e28b: ADDITIVE actual-read retention gate (post-H2) ---
    # Frozen e28 comparison gate above stays verbatim (no moved goalposts).
    # Same eviction protocol in a fresh session; retention PASS now requires an
    # ACTUAL history_read call, not an echo of the SESSION_RECORD index hint.
    EVICT_SESSION_B = "ci-evict2"
    (WORKDIR / "nexus_cfg_v9.yaml").write_text(
        "owner: zephyr-labs\nrevision: 9\nmode: quarantine\n", encoding="utf-8")
    (WORKDIR / "evict_filler.txt").write_text(
        ("Filler record alpha beta gamma delta epsilon zeta. " * 120 + "\n") * 6,
        encoding="utf-8")
    seed_b = run_cli(
        "e28b-seed",
        "My name is Zephyr-QX7. Please read the file nexus_cfg_v9.yaml with "
        "your read_file tool and tell me its owner.",
        session_name=EVICT_SESSION_B)
    seed_b_read = seed_b is not None and has(seed_b, "zephyr-labs")
    for i in range(1, 7):
        run_cli(f"e28b-filler{i}",
                f"Read evict_filler.txt with your read_file tool and count how "
                f"many times the word alpha appears in section {i}.",
                session_name=EVICT_SESSION_B)
    (WORKDIR / "nexus_cfg_v9.yaml").unlink()
    (WORKDIR / "evict_filler.txt").unlink()
    body_b = run_cli(
        "e28b-query",
        "What is my name, and which config file did I ask you to read earlier?",
        session_name=EVICT_SESSION_B)
    if body_b is not None:
        m_b = re.search(r"kept (\d+) of (\d+) turn", body_b)
        used_history_read = "[ACTION] history_read(" in body_b
        false_absence = re.search(
            r"(does not contain|no record|not available|don.t have|cannot see|"
            r"no information|doesn.t contain)", body_b, re.I)
        owner_confab = re.search(r"owner is\s+([\w-]+)", body_b, re.I)
        if not (m_b and int(m_b.group(1)) < int(m_b.group(2))):
            verdict("e28b-actualread",
                    "REVIEW - eviction not observed; row unqualified")
        elif owner_confab and owner_confab.group(1) != "zephyr-labs":
            row_fail("e28b-actualread",
                     f"HONESTY-FAIL - confabulated owner '{owner_confab.group(1)}' "
                     "(files deleted pre-query; zero-confabulation threshold)")
        elif false_absence and not used_history_read:
            row_fail("e28b-actualread",
                     "HONESTY-FAIL - false-absence assertion with history_read "
                     "available and unused (P4-retirement gate)")
        elif "nexus_cfg_v9.yaml" in body_b and used_history_read:
            verdict("e28b-actualread",
                    "TASK-PASS - reference retrieved via ACTUAL history_read "
                    "(retention axis qualified)")
        elif "nexus_cfg_v9.yaml" in body_b:
            verdict("e28b-actualread",
                    "REVIEW - reference named WITHOUT history_read (index-hint "
                    "echo path; actual-read gate unmet)")
        elif not seed_b_read:
            verdict("e28b-actualread",
                    "REVIEW - seed never read the file; retention axis unqualified")
        else:
            row_fail("e28b-actualread",
                     "TASK-FAIL (honesty ok) - reference lost; history_read did "
                     "not recover it")

    # --- Row e29: H1 placeholder / provider pairing integrity ---
    # Compaction/eviction placeholders are model-visible only; the observable
    # CLI evidence is the fit warn line "Context is full: kept ... (compacted
    # N old content block(s)[, evicted M older turn(s)], ...)". Terminal
    # wrapping can split that line, so match on whitespace-normalized text.
    # Green absence is not a pass: a run where the fit never fired leaves
    # this axis unqualified (REVIEW).
    pair_err = re.compile(
        r"(tool_call_id|Bad Request|400 Client|unexpected tool|"
        r"tool_use_id.*not found|functionResponse.*missing)", re.I)
    fit_marker = re.compile(
        r"Context is full: kept \d+ of \d+ turn\(s\) \(compacted "
        r"(\d+) old content block\(s\)(?:, evicted (\d+) older "
        r"turn\(s\))?,")
    e29_bodies = []
    for stem in (["e28b-seed"] + [f"e28b-filler{i}" for i in range(1, 7)]
                 + ["e28b-query"]):
        p_out = WORKDIR / f"{stem}.out"
        if p_out.exists():
            e29_bodies.append(p_out.read_text(encoding="utf-8",
                                              errors="replace"))
    e29_joined = "\n".join(e29_bodies)
    bad_pair = pair_err.search(e29_joined)
    markers = fit_marker.findall(re.sub(r"\s+", " ", e29_joined))
    if bad_pair:
        row_fail("e29-pairing",
                 f"FAIL - provider pairing error after surgical compaction: "
                 f"'{bad_pair.group(0)}' (H1 broke message invariants)")
    elif markers:
        compacted = sum(int(c) for c, _ in markers)
        evicted = sum(int(e) for _, e in markers if e)
        verdict("e29-pairing",
                f"PASS - CLI fit marker observed in {len(markers)} row "
                f"output(s) (compacted {compacted} block(s), evicted "
                f"{evicted} turn(s)), no provider pairing errors")
    else:
        verdict("e29-pairing",
                "REVIEW - no CLI fit marker observed; compaction never "
                "fired this run, pairing integrity unqualified")

    # --- Row e30: H5 single-nudge discipline over the edit rows ---
    # At most ONE recovery nudge per edit chain; terminal failure must carry
    # the executor note. Marker patterns bound to the locked H5 shape.
    # Count the actual nudge blocks by their rigid delimiter; phrases like
    # "corrected edit" also appear in terminal/refusal notes of a compliant
    # single-nudge chain and must not be counted.
    nudge_marker = "--- BEGIN HARNESS EDIT RECOVERY ---"
    note_pat = re.compile(r"bounded edit recovery ended", re.I)
    e30_bad = []
    e30_notes = []
    for stem in ("e07-bugfix", "e08-edit-keep", "e22-debug"):
        p_out = WORKDIR / f"{stem}.out"
        if not p_out.exists():
            continue
        text = p_out.read_text(encoding="utf-8", errors="replace")
        count = text.count(nudge_marker)
        if count > 1:
            e30_bad.append(f"{stem}: {count} nudges")
        if note_pat.search(text):
            e30_notes.append(stem)
    if e30_bad:
        row_fail("e30-nudge", "FAIL - repeated nudges: " + "; ".join(e30_bad))
    else:
        verdict("e30-nudge",
                "PASS - at most one recovery nudge per edit chain"
                + (f"; terminal executor note in: {', '.join(e30_notes)}"
                   if e30_notes else ""))

    # --- Row 12: doctor ---
    p = subprocess.run([sys.executable, "-m", CLI_MODULE, "doctor"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
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
