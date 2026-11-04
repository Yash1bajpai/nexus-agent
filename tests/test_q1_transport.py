"""Real harmless transport and actual CLI usage-error checks; no inference."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(tmp_path):
    spec = importlib.util.spec_from_file_location('q1_transport_driver', ROOT / '.github/e2e/driver.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKDIR = tmp_path
    return module


def test_real_golden_selftest_uses_bound_alias(driver):
    records = []

    def owned(argv, **kwargs):
        records.append((list(argv), dict(kwargs)))
        return subprocess.run(argv, **kwargs)

    driver._run_command = owned
    assert driver.self_test()
    assert len(records) == len(driver.HOSTILE_PROMPTS) + 1
    assert [argv[-1] for argv, _ in records[:-1]] == driver.HOSTILE_PROMPTS
    assert all('timeout' not in kwargs for _, kwargs in records)
    assert records[-1][0] == [sys.executable, '-m', driver.CLI_MODULE, 'chat', '--definitely-not-a-real-flag']


@pytest.mark.parametrize('payload', ['say "hi"', 'é ₹\n$HOME %PATH% ; | & * ?', '  leading trailing  '])
def test_real_argv_stdin_and_eof(payload):
    script = 'import sys,json; print(json.dumps([sys.argv[1:],sys.stdin.read()]))'
    result = subprocess.run([sys.executable, '-I', '-c', script, payload],
                            input=payload, text=True, encoding='utf-8',
                            capture_output=True, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout) == [[payload], payload]


def test_actual_cli_usage_error_is_not_executed(driver):
    result = subprocess.run([sys.executable, '-m', driver.CLI_MODULE, 'chat',
                             '--definitely-not-a-real-flag'],
                            capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 2
    assert 'usage' in (result.stdout + result.stderr).lower()
    assert driver.run_cli('usage', extra_args=('--definitely-not-a-real-flag',)) is None
    assert driver.exit_lines == ['usage exit=2']
    assert 'row did NOT execute' in driver.failures[0]


def test_alias_is_local_not_global(driver):
    original = subprocess.run
    driver._run_command = lambda *args, **kwargs: None
    assert subprocess.run is original


def test_all_driver_launches_use_injectable_alias(driver):
    # Explicit frozen-source contract: argv, inputs and gates cannot drift.
    tree = ast.parse(Path(driver.__file__).read_text())
    direct = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
              and n.func.value.id == 'subprocess' and n.func.attr == 'run']
    assert direct == []
    launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == '_run_command']
    assert len(launches) == 6
    aliases = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_run_command' for t in n.targets)]
    assert len(aliases) == 1
    assert ast.dump(aliases[0].value) == "Attribute(value=Name(id='subprocess', ctx=Load()), attr='run', ctx=Load())"


def test_frozen_complete_call_schedule_and_inputs(driver):
    contract = json.loads((ROOT / 'tests/fixtures/q1/frozen_contract.json').read_text())
    original = ast.parse(contract['base_driver_source'])
    candidate = ast.parse(Path(driver.__file__).read_text())

    def calls(tree, alias):
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if alias and isinstance(node.func, ast.Name) and node.func.id == '_run_command':
                node.func = ast.Attribute(value=ast.Name(id='subprocess', ctx=ast.Load()),
                                          attr='run', ctx=ast.Load())
            # Reporting-only helpers introduce function_shape/false_edit_claim
            # calls; compare every frozen workload/transport call in source order.
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'subprocess' and node.func.attr == 'run'):
                found.append((node.lineno, ast.dump(node, include_attributes=False)))
            elif isinstance(node.func, ast.Name) and node.func.id in {'run_cli', 'run_sub'}:
                found.append((node.lineno, ast.dump(node, include_attributes=False)))
        return [dump for _, dump in sorted(found)]

    assert calls(candidate, True) == calls(original, False)
    assert driver.PROMPT_TIMEOUT == contract['timeout_schedule']['run_cli_default'] == 420
    assert driver.WARMUP_TIMEOUT == contract['timeout_schedule']['warmup'] == 900
    assert {key: contract['timeout_schedule'][key] for key in ('selftest', 'doctor', 'git')} == {
        'selftest': None, 'doctor': None, 'git': None}


def test_runner_execute_adapter_integration_with_selftest(driver, tmp_path, monkeypatch):
    """The production adapter owns real harmless calls on the current host."""
    runner_spec = importlib.util.spec_from_file_location("q1_runner", ROOT / ".github/qualification/runner.py")
    runner_mod = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_mod)
    proc_mod = runner_mod.load_module(ROOT / ".github/qualification/windows_process.py", "q1_transport_process")
    report_mod = runner_mod.load_module(ROOT / ".github/qualification/report.py", "q1_transport_report")

    if os.name == "nt":
        def unexpected_killpg(*args, **kwargs):
            pytest.fail("Native Windows transport must not call os.killpg")
        monkeypatch.setattr(proc_mod.os, "killpg", unexpected_killpg, raising=False)

    home = tmp_path / "home"
    home.mkdir()
    effects = tmp_path / "effects"
    original = "def add(a, b):\n    return a + b\n"
    edited = original + "\ndef subtract(a, b):\n    return a - b\n"
    calculator = tmp_path / "calculator.py"
    calculator.write_text(original, encoding="utf-8")
    axes = []
    effect_budget = [0]
    backend = runner_mod.BackendObserver(tmp_path / "unused-model.gguf", home)
    environment = dict(os.environ)
    environment.update(HOME=str(home), USERPROFILE=str(home), NEXUS_AGENT_HOME=str(home / ".nexus-agent"))
    count = len(driver.HOSTILE_PROMPTS) + 1

    with proc_mod.ProcessDomain(tmp_path / "workload", deadline=time.monotonic() + 120) as domain:
        assert domain.platform == os.name
        execute = runner_mod.make_execute_adapter(
            domain=domain, workspace=tmp_path, home=home, effects_root=effects,
            environment=environment, effect_budget=effect_budget, axes=axes,
            report=report_mod, backend=backend)
        driver._run_command = execute
        assert driver.self_test() is True
        assert driver.run_cli("usage-row", extra_args=("--definitely-not-a-real-flag",)) is None
        assert driver.self_test() is True
        assert driver.run_sub("help-row", ["--help"]) == 0

        def run_named(name, argv):
            return execute(argv, capture_output=True, text=True, encoding="utf-8", timeout=10)

        result = run_named("e08-edit-keep", [
            sys.executable, "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')",
            str(calculator), edited])
        assert result.returncode == 0
        for script in ('print("doctor")', 'print("git")', 'print("unclassified")'):
            assert execute([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=10).returncode == 0

    expected_names = (["selftest"] * count + ["usage-row"] + ["selftest"] * count
                      + ["help-row", "e08-edit-keep", "doctor", "git-baseline"])
    expected_names.append(f"command-{len(expected_names) + 1:04d}")
    assert [row["name"] for row in axes] == expected_names
    assert [record["command_id"] for record in domain.records] == list(range(1, len(expected_names) + 1))
    assert all(record["capture_complete"] and record["cleanup"]["confirmed"] for record in domain.records)
    for start in (0, count + 1):
        records = domain.records[start:start + count]
        assert [record["argv"][-1] for record in records[:-1]] == driver.HOSTILE_PROMPTS
        assert all(record["requested_timeout"] is None for record in records)
        assert records[-1]["argv"] == [
            sys.executable, "-m", driver.CLI_MODULE, "chat", "--definitely-not-a-real-flag"]
        assert axes[start + count - 1]["executor"]["status"] == "NOT_EXECUTED"
    assert axes[count]["executor"]["status"] == "NOT_EXECUTED"
    assert axes[-4]["precondition"]["status"] == "PASS"
    assert axes[-4]["task_effect"]["status"] == "REVIEW"
    assert {path.name for path in effects.iterdir()} == {
        f"{index:04d}-{row_name}-{phase}"
        for index, row_name in enumerate(expected_names, 1) for phase in ("before", "after")}
    edit_index = expected_names.index("e08-edit-keep") + 1
    assert (effects / f"{edit_index:04d}-e08-edit-keep-before/calculator.py").read_text(encoding="utf-8") == original
    assert (effects / f"{edit_index:04d}-e08-edit-keep-after/calculator.py").read_text(encoding="utf-8") == edited
    assert effect_budget[0] == sum(path.stat().st_size for path in effects.rglob("*") if path.is_file())
    assert backend.servers == {}


@pytest.mark.parametrize("qualified", [False, True])
def test_runner_execute_adapter_fake_host_evidence_and_backend_gate(tmp_path, qualified):
    """Exercise evidence and warmup gating without launching any model/process."""
    runner_spec = importlib.util.spec_from_file_location("q1_runner", ROOT / ".github/qualification/runner.py")
    runner_mod = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_mod)
    report_mod = runner_mod.load_module(ROOT / ".github/qualification/report.py", "q1_fake_transport_report")
    home = tmp_path / "home"
    (home / ".nexus-agent").mkdir(parents=True)
    database = home / ".nexus-agent/history.db"
    user_json = json.dumps({"role": "user", "content": "synthetic question"})
    assistant_json = json.dumps({"role": "assistant", "content": "synthetic answer"})
    with runner_mod.sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE conversation_history "
                           "(id INTEGER, session_id TEXT, role TEXT, message_json TEXT, timestamp TEXT)")
        connection.execute("INSERT INTO conversation_history VALUES (1, 's', 'user', ?, 'before')", (user_json,))
    effect = tmp_path / "effect.txt"
    effect.write_text("before", encoding="utf-8")
    calls = []
    evidence = []
    backend_calls = []
    lifecycle = {"capture_complete": True, "cleanup": {"ownership": "owned_windows_job", "confirmed": True}}
    argv = ["synthetic", "say ok"]
    result = subprocess.CompletedProcess(argv, 0, b"synthetic stdout", b"synthetic stderr")

    class FakeDomain:
        records = []

        def run(self, actual_argv, **kwargs):
            calls.append((actual_argv, kwargs))
            effect.write_text("after", encoding="utf-8")
            with runner_mod.sqlite3.connect(database) as connection:
                connection.execute("DELETE FROM conversation_history")
                connection.execute("INSERT INTO conversation_history VALUES (10, 's', 'user', ?, 'after')", (user_json,))
                connection.execute("INSERT INTO conversation_history VALUES (11, 's', 'assistant', ?, 'after')",
                                   (assistant_json,))
            self.records.append(lifecycle)
            return result

    class RecordingReport:
        def assess_row(self, name, **kwargs):
            evidence.append((name, kwargs))
            return report_mod.assess_row(name, **kwargs)

    class FakeBackend:
        def qualified(self):
            backend_calls.append(True)
            return qualified

    environment = {"BASE": "preserved", "OVERRIDE": "base"}
    override = {"OVERRIDE": "child"}
    effect_budget = [0]
    axes = []
    execute = runner_mod.make_execute_adapter(
        domain=FakeDomain(), workspace=tmp_path, home=home, effects_root=tmp_path / "effects",
        environment=environment, effect_budget=effect_budget, axes=axes,
        report=RecordingReport(), backend=FakeBackend())
    kwargs = dict(input="finite input", capture_output=True, text=True, encoding="utf-8",
                  errors="replace", timeout=17, cwd=str(tmp_path), env=override)
    if qualified:
        assert execute(argv, **kwargs) is result
    else:
        with pytest.raises(runner_mod.GateError, match="Actual owned backend/version/model unqualified"):
            execute(argv, **kwargs)

    assert calls == [(argv, dict(kwargs, env={"BASE": "preserved", "OVERRIDE": "child"}))]
    assert environment == {"BASE": "preserved", "OVERRIDE": "base"}
    assert override == {"OVERRIDE": "child"}
    assert backend_calls == [True]
    assert len(axes) == len(evidence) == 1
    row_name, captured = evidence[0]
    assert row_name == "command-0001"
    assert captured["before"] == {"effect.txt": "before"}
    assert captured["after"] == {"effect.txt": "after"}
    assert captured["stdout"] == "synthetic stdout"
    assert captured["stderr"] == "synthetic stderr"
    assert captured["lifecycle"] is lifecycle
    assert captured["capture_complete"] is True
    assert [row["id"] for row in captured["canonical"]] == [11]
    assert axes[0]["observations"]["canonical_role_facts"] == [
        {"index": 0, "role": "assistant", "content": "synthetic answer"}]
    assert axes[0]["native"]["status"] == "OBSERVED"
    assert (tmp_path / "effects/0001-command-0001-before/effect.txt").read_text(encoding="utf-8") == "before"
    assert (tmp_path / "effects/0001-command-0001-after/effect.txt").read_text(encoding="utf-8") == "after"
    assert effect_budget == [len(b"before") + len(b"after")]


def test_canonical_diff_handles_append_replace_prune():
    runner_spec = importlib.util.spec_from_file_location("q1_runner", ROOT / ".github/qualification/runner.py")
    runner_mod = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_mod)
    canonical_diff = runner_mod.canonical_diff

    m1 = {"id": 1, "session_id": "s1", "role": "user", "message_json": json.dumps({"role": "user", "content": "hi"})}
    m2 = {"id": 2, "session_id": "s1", "role": "assistant", "message_json": json.dumps({"role": "assistant", "content": "hello"})}
    m3 = {"id": 3, "session_id": "s1", "role": "user", "message_json": json.dumps({"role": "user", "content": "test"})}
    m4 = {"id": 4, "session_id": "s1", "role": "assistant", "message_json": json.dumps({"role": "assistant", "content": "reply"})}

    # 1. Normal append
    delta = canonical_diff([m1, m2], [m1, m2, m3, m4])
    assert delta == [m3, m4]

    # 2. Persistence._replace: whole session deleted and reinserted with new IDs
    m1_new_id = dict(m1, id=10)
    m2_new_id = dict(m2, id=11)
    m3_new_id = dict(m3, id=12)
    # m1..m3 already existed in before, m3 was added
    delta_replace = canonical_diff([m1, m2], [m1_new_id, m2_new_id, m3_new_id])
    assert delta_replace == [m3_new_id]

    # 3. Compaction/prune: older messages dropped, new summary added
    summary = {"id": 20, "session_id": "s1", "role": "system", "message_json": json.dumps({"role": "system", "content": "summary"})}
    delta_prune = canonical_diff([m1, m2, m3], [summary, m2_new_id, m3_new_id, m4])
    assert delta_prune == [summary, m4]

    # 4. No change
    assert canonical_diff([m1, m2], [m1, m2]) == []
