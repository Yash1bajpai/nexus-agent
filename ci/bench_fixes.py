"""Tiny grounding benchmark: fix a one-line bug and run the tests, via the installed `nexus-agent` command.

Controlled scratch fixtures only (known code, own test files). NEXUS_ALLOW_PROJECT_EXECUTION=1 is set by
the caller so run_tests is offered to the model inside these throwaway folders. Prints one line per run
and a summary. A handful of runs is a small sample: do not read it as a measure of a model's ability.
Usage: python bench_fixes.py <nexus-agent exe> <runs per task> <timeout seconds per run>
"""
import os, subprocess, sys, tempfile, json

EXE, RUNS, TMO = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TASKS = {
    "calc": ("def add(a, b):\n    return a - b\n", "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n"),
    "util": ("def is_even(n):\n    return n % 2 == 1\n", "from util import is_even\ndef test_e():\n    assert is_even(4)\n    assert not is_even(3)\n"),
    "greet": ('def greet(name):\n    return "Hello " + name\n', 'from greet import greet\ndef test_g():\n    assert greet("Bo") == "Hello, Bo!"\n'),
    "lst": ("def last(xs):\n    return xs[len(xs)]\n", "from lst import last\ndef test_l():\n    assert last([1, 2, 3]) == 3\n"),
    "sub": ("def sub(a, b):\n    return b - a\n", "from sub import sub\ndef test_s():\n    assert sub(5, 2) == 3\n"),
}
home = tempfile.mkdtemp()
LOGS = '/tmp/bench_logs'; os.makedirs(LOGS, exist_ok=True)
env = dict(os.environ, NEXUS_AGENT_HOME=home)
fixed = total = 0
rows = []
for run in range(1, RUNS + 1):
    for name, (src, tst) in TASKS.items():
        d = tempfile.mkdtemp(prefix=f"{name}_")
        open(f"{d}/{name}.py", "w").write(src)
        open(f"{d}/test_{name}.py", "w").write(tst)
        prompt = f"Fix the bug in {name}.py so the tests in test_{name}.py pass. Run the tests to confirm."
        try:
            p = subprocess.run([EXE, "chat", "-p", "local", "-v", prompt], cwd=d, env=env, capture_output=True, text=True, timeout=TMO)
            code, log = p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            code, log = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d, capture_output=True, text=True)
        ok = t.returncode == 0
        open(f'{LOGS}/run{run}_{name}.txt', 'w').write(f'EXIT {code}\n--- agent output ---\n{log}\n--- source after ---\n' + open(f'{d}/{name}.py').read() + '\n--- pytest ---\n' + t.stdout[-1500:])
        fixed += ok; total += 1
        tools = [l.strip()[:90] for l in log.splitlines() if "[ACTION]" in l]
        print(f"run{run} {name}: exit={code} tests_pass={'yes' if ok else 'no'} actions={len(tools)} first={tools[:1]}", flush=True)
        rows.append({"run": run, "task": name, "exit": code, "fixed": ok, "actions": tools})
print(f"SUMMARY fixed {fixed}/{total}")
json.dump(rows, open("bench_result.json", "w"))
