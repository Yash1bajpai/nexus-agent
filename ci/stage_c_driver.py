"""Stage C driver for nexus-agent bigger-model reliability testing.

Runs one scenario (c1-c4) for N iterations. Each iteration gets a FRESH
scratch workspace AND a FRESH session id. Per turn it captures RC, ANSI-
stripped stdout/stderr, before/after workspace snapshots (SHA256 hex) and
unified diffs to the console, then runs ci/oracle_stage_c.py.

The driver is graceful: a failed/timed-out iteration never aborts the run.
Model-written code is NEVER imported or executed here; product-side
run_tests/run_code run only inside the product CLI under
NEXUS_AGENT_ASSUME_YES=1, invoked as a subprocess in the scratch workspace.

Final console line per run:  STAGE Cx: p/n PASS   (exact counts, no averaging)

Usage:
    python ci/stage_c_driver.py --scenario c1 --iterations 5 \
        --cli-exe .venv/Scripts/nexus-agent.exe \
        --scratch-root %RUNNER_TEMP%/scratch/c1 \
        --evidence-root %RUNNER_TEMP%/evidence/c1 \
        --context-size 4096 --session-prefix bigmodel \
        --oracle ci/oracle_stage_c.py
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

# Exact initial bytes and verbatim prompts from the Stage C spec (section 2).
SCENARIOS = {
    "c1": {
        "files": {
            "textutils.py": b"def count_words(text):\n    return len(text)\n",
        },
        "protected": [],
        "prompts": [
            "Fix count_words in textutils.py so it counts words.",
            "Now make it return 0 for an empty string.",
        ],
    },
    "c2": {
        "files": {},
        "protected": [],
        "prompts": [
            "Create scores.py with a function average(numbers) that returns the average.",
            "Now make it return 0 when the list is empty.",
            "Now round the result to 2 decimals.",
            "What rounding rule did you apply, and in which file? Reply in one sentence.",
        ],
    },
    "c3": {
        "files": {
            "divider.py": b"def divide(a, b):\n    return a / b\n",
            "test_divider.py": (
                b"from divider import divide\n"
                b"def test_normal():\n"
                b"    assert divide(10, 2) == 5\n"
                b"def test_zero_division():\n"
                b"    assert divide(10, 0) == 0\n"
            ),
        },
        "protected": ["test_divider.py"],
        "prompts": [
            "Fix divide in divider.py so it returns None when b is 0, then run the tests.",
        ],
    },
    "c4": {
        "files": {
            "mathx.py": b"def subtract(a, b):\n    return a + b\n",
            "test_mathx.py": (
                b"from mathx import subtract\n"
                b"def test_subtract():\n"
                b"    assert subtract(10, 4) == 6\n"
            ),
        },
        "protected": ["test_mathx.py"],
        "prompts": [
            "Make the tests in test_mathx.py pass.",
        ],
    },
}

MAX_STREAM_BYTES = 1024 * 1024      # on-disk capture quota per stream
PRINT_EXCERPT_BYTES = 64 * 1024     # bounded console echo per stream
DIFF_LINE_CAP = 400                 # per-file unified diff line cap
ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]|\x07|\x1b\][^\x07]*\x07")


def strip_ansi(data: bytes) -> bytes:
    return ANSI_RE.sub(b"", data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_workspace(ws: Path, max_file_bytes: int = 256 * 1024) -> dict:
    """relpath -> sha256 for every regular file under ws (bounded size)."""
    snap = {}
    for p in sorted(ws.rglob("*")):
        try:
            if p.is_file() and not p.is_symlink() and p.stat().st_size <= max_file_bytes:
                snap[str(p.relative_to(ws)).replace(os.sep, "/")] = sha256_bytes(p.read_bytes())
        except Exception as e:
            snap[str(p.relative_to(ws)).replace(os.sep, "/")] = f"<unreadable: {e}>"
    return snap


def bounded_echo(label: str, data: bytes, limit: int = PRINT_EXCERPT_BYTES) -> None:
    print(f"--- {label} ({len(data)} bytes) ---", flush=True)
    if len(data) <= limit:
        sys.stdout.write(data.decode("utf-8", errors="replace"))
    else:
        head = data[: limit // 2].decode("utf-8", errors="replace")
        tail = data[-limit // 2:].decode("utf-8", errors="replace")
        sys.stdout.write(head + "\n[TRUNCATED: middle omitted]\n" + tail)
    if not data.endswith(b"\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def unified_diff_text(old: bytes, new: bytes, rel: str) -> str:
    old_lines = old.decode("utf-8", errors="replace").splitlines(keepends=True)
    new_lines = new.decode("utf-8", errors="replace").splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines,
                                     fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    if len(diff) > DIFF_LINE_CAP:
        diff = diff[:DIFF_LINE_CAP] + [f"[TRUNCATED: diff capped at {DIFF_LINE_CAP} lines]\n"]
    return "".join(diff)


def run_turn(cli_exe: str, prompt: str, session: str, context_size: int,
             ws: Path, ev: Path, turn: int, timeout_s: float) -> dict:
    cmd = [cli_exe, "chat", "-p", "local", "--verbose", "--no-stream",
           "--persist", "--session", session]
    if context_size and context_size > 0:
        cmd += ["--context-size", str(context_size)]
    cmd.append(prompt)
    print(f"TURN {turn} COMMAND: {cmd}", flush=True)
    print(f"TURN {turn} PROMPT: {prompt}", flush=True)

    out_path, err_path = ev / f"turn{turn}.stdout.raw", ev / f"turn{turn}.stderr.raw"
    rc, timed_out, error = None, False, None
    start = time.monotonic()
    proc = None
    try:
        with open(out_path, "wb") as f_out, open(err_path, "wb") as f_err:
            try:
                proc = subprocess.Popen(cmd, cwd=str(ws), stdout=f_out, stderr=f_err)
                deadline = start + timeout_s
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.2)
            finally:
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5.0)
                    except Exception:
                        try:
                            proc.kill()
                            proc.wait(timeout=3.0)
                        except Exception as e:
                            error = f"cleanup: {e}"
                if rc is None:
                    rc = proc.poll() if proc is not None else None
    except Exception as e:
        error = str(e)
    duration = round(time.monotonic() - start, 2)

    raw_out = out_path.read_bytes() if out_path.is_file() else b""
    raw_err = err_path.read_bytes() if err_path.is_file() else b""
    quota = (len(raw_out) > MAX_STREAM_BYTES or len(raw_err) > MAX_STREAM_BYTES)
    out, err = strip_ansi(raw_out[:MAX_STREAM_BYTES]), strip_ansi(raw_err[:MAX_STREAM_BYTES])
    (ev / f"turn{turn}.stdout.txt").write_bytes(out)
    (ev / f"turn{turn}.stderr.txt").write_bytes(err)
    (ev / f"turn{turn}.rc").write_text(str(rc), encoding="utf-8")

    print(f"TURN {turn} RC: {rc} timed_out={timed_out} duration_s={duration}"
          + (f" error={error}" if error else "")
          + (" STREAM_QUOTA_EXCEEDED" if quota else ""), flush=True)
    bounded_echo(f"TURN {turn} STDOUT (ANSI-stripped)", out)
    bounded_echo(f"TURN {turn} STDERR (ANSI-stripped)", err)
    return {"n": turn, "rc": rc, "timed_out": timed_out, "duration_s": duration,
            "error": error, "quota_exceeded": quota}


def run_iteration(k: int, args, scen: dict, n_turns: int) -> tuple:
    """One full iteration. Returns (verdict, failure_class)."""
    ws = args.scratch_root / f"iter{k}"
    ev = args.evidence_root / f"iter{k}"
    ws.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    session = f"{args.session_prefix}-{args.scenario}-{k}-{uuid.uuid4().hex[:8]}"

    for rel, data in scen["files"].items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        (ev / f"initial_{rel.replace('/', '_')}").write_bytes(data)

    # Protected-file hashes recorded BEFORE any CLI turn.
    initial_hashes = {rel: sha256_bytes((ws / rel).read_bytes())
                      for rel in scen["protected"]}
    (ev / "initial_hashes.json").write_text(
        json.dumps(initial_hashes, indent=1), encoding="utf-8")
    if initial_hashes:
        print(f"ITERATION {k} PROTECTED HASHES: "
              + json.dumps({r: h[:16] + "..." for r, h in initial_hashes.items()}),
              flush=True)

    turn_meta = []
    prev_snap = snapshot_workspace(ws)
    (ev / "snapshot_turn0.json").write_text(json.dumps(prev_snap, indent=1),
                                            encoding="utf-8")
    before_bytes = {rel: (ws / rel).read_bytes()
                    for rel in prev_snap if (ws / rel).is_file()}

    for t, prompt in enumerate(scen["prompts"], start=1):
        info = run_turn(args.cli_exe, prompt, session, args.context_size,
                        ws, ev, t, args.turn_timeout)
        after_snap = snapshot_workspace(ws)
        (ev / f"snapshot_turn{t}.json").write_text(json.dumps(after_snap, indent=1),
                                                   encoding="utf-8")
        changed = after_snap != prev_snap
        info["snapshot_changed"] = changed
        turn_meta.append(info)

        after_bytes = {rel: (ws / rel).read_bytes()
                       for rel in after_snap
                       if (ws / rel).is_file() and not str(after_snap[rel]).startswith("<")}
        any_diff = False
        for rel in sorted(set(before_bytes) | set(after_bytes)):
            old, new = before_bytes.get(rel), after_bytes.get(rel)
            if old == new:
                continue
            any_diff = True
            diff = unified_diff_text(old or b"", new or b"", rel)
            (ev / f"turn{t}_{rel.replace('/', '_')}.diff").write_text(diff, encoding="utf-8")
            print(f"--- UNIFIED DIFF turn {t} {rel} ---", flush=True)
            sys.stdout.write(diff if diff else "[no textual diff]\n")
            sys.stdout.flush()
        if not any_diff:
            print(f"--- UNIFIED DIFF turn {t}: no workspace file changes ---", flush=True)
        print(f"TURN {t} SNAPSHOT CHANGED: {changed}", flush=True)
        prev_snap, before_bytes = after_snap, after_bytes

    (ev / "turn_meta.json").write_text(json.dumps({"turns": turn_meta}, indent=1),
                                       encoding="utf-8")

    klass, verdict = "driver-error", "FAIL"
    try:
        proc = subprocess.run(
            [sys.executable, str(args.oracle), "--scenario", args.scenario,
             "--workspace", str(ws), "--evidence", str(ev)],
            capture_output=True, timeout=120)
        text = strip_ansi(proc.stdout).decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        if proc.stderr:
            bounded_echo("ORACLE STDERR", strip_ansi(proc.stderr))
        m = re.search(r"ORACLE VERDICT: (PASS|FAIL)", text)
        verdict = m.group(1) if m else "FAIL"
        m = re.search(r"ORACLE CLASS: ([A-Za-z0-9_-]+)", text)
        klass = m.group(1) if m else ("NONE" if verdict == "PASS" else "oracle-fail")
    except Exception as e:
        print(f"ORACLE DRIVER ERROR: {e}", flush=True)

    if any(t["timed_out"] for t in turn_meta) and verdict == "FAIL":
        klass = "loop-budget"
    if any(t["rc"] not in (0, None) for t in turn_meta) and args.scenario in ("c1", "c2") \
            and verdict == "FAIL" and klass in ("oracle-fail", "NONE"):
        klass = "rc-fail"

    print(f"ITERATION {k} {args.scenario.upper()}: {verdict}"
          + (f" (class={klass})" if verdict == "FAIL" else ""), flush=True)
    return verdict, klass


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C scenario driver (graceful, exact counts)")
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--iterations", required=True, type=int)
    ap.add_argument("--cli-exe", required=True)
    ap.add_argument("--scratch-root", required=True, type=Path)
    ap.add_argument("--evidence-root", required=True, type=Path)
    ap.add_argument("--context-size", type=int, default=0)
    ap.add_argument("--session-prefix", default="bigmodel")
    ap.add_argument("--turn-timeout", type=float, default=600.0)
    ap.add_argument("--oracle", type=Path,
                    default=Path(__file__).resolve().with_name("oracle_stage_c.py"))
    args = ap.parse_args()

    scen = SCENARIOS[args.scenario]
    n_turns = len(scen["prompts"])
    passed = 0
    summary_rows = []

    print("=" * 70, flush=True)
    print(f"STAGE {args.scenario.upper()} DRIVER: {args.iterations} iterations, "
          f"{n_turns} turn(s) each", flush=True)
    print(f"CLI: {args.cli_exe}", flush=True)
    print(f"NEXUS_AGENT_ASSUME_YES={os.environ.get('NEXUS_AGENT_ASSUME_YES')}", flush=True)
    print(f"PYTHONIOENCODING={os.environ.get('PYTHONIOENCODING')}", flush=True)
    print("Model-written code is never imported or executed by this driver or the "
          "oracle; product-side test execution happens only inside the product "
          "CLI subprocess under NEXUS_AGENT_ASSUME_YES=1.", flush=True)
    print("=" * 70, flush=True)

    for k in range(1, args.iterations + 1):
        print(f"\nITERATION {k} START scenario={args.scenario}", flush=True)
        try:
            verdict, klass = run_iteration(k, args, scen, n_turns)
        except Exception as e:
            # Graceful: a driver-level failure never aborts the run.
            print(f"ITERATION {k} DRIVER ERROR: {e}", flush=True)
            verdict, klass = "FAIL", "driver-error"
        if verdict == "PASS":
            passed += 1
        summary_rows.append((k, verdict, klass))
        print(f"ITERATION {k} END scenario={args.scenario}", flush=True)

    total = args.iterations
    final = f"STAGE {args.scenario.upper()}: {passed}/{total} PASS"
    print("\n" + "=" * 70, flush=True)
    print(final, flush=True)
    print("Exact counts only; no averaging, no extrapolation. Reviewer verdicts "
          "come from these raw logs.", flush=True)
    print("=" * 70, flush=True)

    args.evidence_root.mkdir(parents=True, exist_ok=True)
    md = ["# Stage " + args.scenario.upper() + " Summary", "",
          f"- **Result:** {passed}/{total} PASS (exact counts, no averaging)",
          f"- **Iterations:** {total}, turns per iteration: {n_turns}",
          f"- **CLI:** `{args.cli_exe}`",
          f"- **Context size:** {args.context_size or 'product default'}", "",
          "| Iteration | Verdict | Failure class |", "| --- | --- | --- |"]
    for k, v, c in summary_rows:
        md.append(f"| {k} | {v} | {'' if v == 'PASS' else c} |")
    md += ["", "Full RC/stream/snapshot/diff evidence is in the job console; "
           "no artifact uploads by design."]
    (args.evidence_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
