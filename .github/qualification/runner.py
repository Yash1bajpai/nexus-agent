"""Q1 local candidate: installed-wheel/native gate, then ONE frozen workload.

This harness is not execution approval. Native/model stages require the separately
reviewed literal route. No application runtime is imported from the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import zipfile

MODEL_SHA256 = "2ea801949d760cdf1a2cc04a54262c22c3c0c54f0769d57760c9adeb0e59233f"
MODEL_REL = Path(".nexus-agent/models/LiquidAI--LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf")
EFFECT_FILE_CAP = 4 * 1024 * 1024
EFFECT_TOTAL_CAP = 64 * 1024 * 1024
CANONICAL_CAP = 16 * 1024 * 1024
JSON_CAP = 2 * 1024 * 1024
HELPERS = (
    ".github/diagnostics/budget_diagnostic_runner.py", ".github/diagnostics/windows_job.py",
    ".github/diagnostics/verify_reviewed_candidate.py", ".github/workflows/macos.yml",
    ".github/workflows/budget-diagnostic.yml", ".github/e2e/driver.py",
    ".github/qualification/runner.py", ".github/qualification/windows_process.py",
    ".github/qualification/report.py", ".github/qualification/verify_candidate.py",
    ".github/workflows/q1-windows-qualification.yml",
)


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value, cap=JSON_CAP):
    data = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2).encode()
    require(len(data) <= cap, "Bounded evidence JSON overflow; no qualified result")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
    os.replace(temporary, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Reviewed helper unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    result = subprocess.run(["git", "--no-replace-objects", "-C", str(root), *args],
                            capture_output=True, check=True, timeout=15, env=env)
    return result.stdout


def contract(root):
    return json.loads((Path(root) / "tests/fixtures/q1/frozen_contract.json").read_text())


def verify_installed(wheel, source_root, commit):
    wheel, source_root = Path(wheel).resolve(strict=True), Path(source_root).resolve(strict=True)
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "Exact commit required")
    require(git(source_root, "rev-parse", "HEAD").decode().strip() == commit, "Checkout commit mismatch")
    require(sys.prefix != sys.base_prefix, "Fresh isolated venv required")
    import nexus_agent_ai
    package = Path(nexus_agent_ai.__file__).resolve().parent
    prefix = Path(sys.prefix).resolve()
    require(package.is_relative_to(prefix) and "site-packages" in package.parts
            and not package.is_relative_to(source_root), "Installed wheel import required; checkout import refused")
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for entry in archive.infolist():
            name = entry.filename
            require(name not in seen and ".." not in Path(name).parts and not Path(name).is_absolute(),
                    "Unsafe or duplicate wheel member")
            seen.add(name)
            if entry.is_dir() or not name.startswith("nexus_agent_ai/") or not name.endswith(".py"):
                continue
            relative = name.removeprefix("nexus_agent_ai/")
            installed = (package / relative).resolve()
            require(installed.is_relative_to(package), "Installed package path escape")
            expected = hashlib.sha256(archive.read(entry)).hexdigest()
            require(installed.is_file() and digest(installed) == expected, "Installed Python bytes differ from wheel")
            original = git(source_root, "show", commit + ":src/" + name)
            require(hashlib.sha256(original).hexdigest() == expected, "Wheel runtime differs from sealed Git source")
            members[relative] = expected
    actual = {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
    require(actual == set(members) and "__init__.py" in members, "Extra/missing installed Python source")
    return {"commit": commit, "tree": git(source_root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "wheel_sha256": digest(wheel), "wheel": wheel.name,
            "interpreter": sys.executable, "prefix": str(prefix), "package_root": str(package),
            "import_path": str(package / "__init__.py"), "python": sys.version,
            "pytest": importlib.metadata.version("pytest"), "version": importlib.metadata.version("nexus-agent-ai"),
            "package_py": members, "generated_metadata_and_bytecode_excluded": True}


def stage_tests(source_root, destination, frozen):
    source_root, destination = Path(source_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tracked = git(source_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "tests").decode().split("\0")
    paths = sorted({p for p in tracked if p and "/__pycache__/" not in p} | set(HELPERS))
    staged = {}
    for name in paths:
        require(not name.startswith("src/") and ".." not in Path(name).parts, "Unsafe staged helper")
        original = source_root / name
        require(original.is_file() and not original.is_symlink(), "Missing/nonregular reviewed test/helper")
        blob = original.read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        staged[name] = hashlib.sha256(blob).hexdigest()
    config = destination / "isolated-pytest.ini"
    config.write_text("[pytest]\npythonpath =\naddopts =\ntestpaths = tests\n", encoding="utf-8")
    require(not (destination / "src").exists(), "Source may not be staged")
    return {"files": staged, "config": str(config), "root": str(destination),
            "critical_native": frozen["critical_native"]}


def critical_outcomes(outcomes, critical, platform):
    details = []
    qualified = True
    for entry in critical:
        predicate = entry.get("platform", "all")
        if predicate not in ("all", platform):
            details.append({"nodeid": entry["nodeid"], "status": "not_applicable", "platform": predicate})
            continue
        prefix = entry["nodeid"]
        matches = {node: value for node, value in outcomes.items()
                   if node == prefix or (entry.get("match") == "parameterized" and node.startswith(prefix + "["))}
        expected = entry.get("count", 1)
        okay = len(matches) == expected and all(value == "passed" for value in matches.values())
        qualified &= okay
        details.append({"nodeid": prefix, "expected": expected, "actual": len(matches),
                        "outcomes": matches, "qualified": okay})
    return {"qualified": bool(qualified), "critical": details}


def child_environment(home, temporary):
    directories = [str(Path(sys.executable).parent)]
    result = {}
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            result[key] = os.environ[key]
            directories.extend((str(Path(os.environ[key]) / "System32"), os.environ[key]))
    if os.name != "nt":
        directories.extend(os.defpath.split(os.pathsep))
    git_exe = shutil.which("git")
    if git_exe:
        directories.append(str(Path(git_exe).resolve().parent))
    home, temporary = Path(home), Path(temporary)
    home.mkdir(parents=True, exist_ok=True); temporary.mkdir(parents=True, exist_ok=True)
    result.update(HOME=str(home), USERPROFILE=str(home), NEXUS_AGENT_HOME=str(home / ".nexus-agent"),
                  PATH=os.pathsep.join(dict.fromkeys(directories)), TMP=str(temporary), TEMP=str(temporary),
                  TMPDIR=str(temporary), PYTHONUTF8="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1",
                  GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0",
                  GIT_NO_LAZY_FETCH="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return result


class PremodelAudit:
    """Actual attempted network/process events, not a no-network environment claim."""
    def __init__(self, package, package_py):
        self.package = Path(package).resolve()
        self.package_py = package_py
        self.violations = []
        self.children = []
        self.executed_package_files = set()

    def violate(self, reason):
        if len(self.violations) < 64:
            self.violations.append(reason)
        raise OSError("Q1 premodel boundary rejected " + reason)

    def __call__(self, event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            address = args[1] if event == "socket.connect" else args[0] if event == "socket.getaddrinfo" else args[-1]
            host = address[0] if isinstance(address, tuple) else address
            if isinstance(host, str):
                try:
                    local = ipaddress.ip_address(host).is_loopback
                except ValueError:
                    local = host.lower() == "localhost"
                if not local:
                    self.violate("nonloopback_network_attempt")
        elif event == "subprocess.Popen":
            argv = args[1]
            if isinstance(argv, (list, tuple)):
                words = [str(word) for word in argv]
                self.children.append({"executable": str(args[0]), "argv_sha256": hashlib.sha256(json.dumps(words).encode()).hexdigest()})
                if len(self.children) > 8192:
                    self.violate("child_audit_cap")
                exe = Path(words[0]).name.lower() if words else ""
                if exe in ("curl", "curl.exe", "wget", "wget.exe", "llama-server", "llama-server.exe", "ollama", "ollama.exe"):
                    self.violate("fetch_or_model_process")
                if exe in ("git", "git.exe") and set(words[1:]) & {"fetch", "push", "pull", "ls-remote"}:
                    self.violate("remote_git_attempt")
                for word in words:
                    path = Path(word)
                    if path.name == "text_count_worker.py":
                        require(path.resolve() == self.package / "agent/text_count_worker.py", "Worker outside installed wheel")
                        for relative in ("agent/text_count_worker.py", "agent/_workspace_paths.py", "agent/text_count_scan.py", "agent/text_count_limits.py"):
                            require(digest(self.package / relative) == self.package_py[relative], "Worker import sibling byte mismatch")
        elif event == "exec":
            filename = getattr(args[0], "co_filename", "")
            if filename and "nexus_agent_ai" in Path(filename).parts:
                path = Path(filename).resolve()
                require(path.is_relative_to(self.package), "Package exec outside installed wheel")
                relative = path.relative_to(self.package).as_posix()
                if relative.endswith(".py"):
                    require(relative in self.package_py and digest(path) == self.package_py[relative], "Executed installed bytes mismatch")
                    self.executed_package_files.add(relative)


class GatePlugin:
    def __init__(self, package):
        self.package = Path(package).resolve()
        self.collected = []
        self.outcomes = {}

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed or report.skipped:
            value = "xfail" if hasattr(report, "wasxfail") else report.outcome
            previous = self.outcomes.get(report.nodeid)
            if previous not in ("failed", "skipped", "xfail"):
                self.outcomes[report.nodeid] = value

    def pytest_sessionfinish(self, session, exitstatus):
        import nexus_agent_ai
        require(Path(nexus_agent_ai.__file__).resolve().is_relative_to(self.package), "Final package import changed")


def pytest_phase(args):
    import pytest
    root = Path(args.root).resolve()
    spec = json.loads((root / "output/native-spec.json").read_text())
    provenance = verify_installed(args.wheel, args.source_root, args.commit)
    require(provenance["wheel_sha256"] == spec["provenance"]["wheel_sha256"], "Native wheel identity changed")
    audit = PremodelAudit(provenance["package_root"], provenance["package_py"])
    sys.addaudithook(audit)
    plugin = GatePlugin(provenance["package_root"])
    stage = Path(spec["staged"]["root"])
    os.chdir(stage)
    exit_code = pytest.main(["-c", spec["staged"]["config"], "--rootdir", str(stage),
                             "-o", "pythonpath=", "-o", "addopts=", "tests", "-q"], plugins=[plugin])
    final = critical_outcomes(plugin.outcomes, spec["staged"]["critical_native"], sys.platform)
    final.update(exit_code=int(exit_code), collected=len(plugin.collected), outcomes=plugin.outcomes,
                 violations=audit.violations, children=audit.children,
                 executed_package_files=sorted(audit.executed_package_files), platform=sys.platform)
    # Detect installed-file mutation too; generated RECORD/bytecode are not compared.
    verify_installed(args.wheel, args.source_root, args.commit)
    final["qualified"] &= int(exit_code) == 0 and not audit.violations and sys.platform == "win32"
    write_json(root / "output/native-results.json", final)
    return 0 if final["qualified"] else 1


def deadline_from_file(path):
    value = json.loads(Path(path).read_text())
    require(value.get("job_limit_minutes") == 120 and value.get("reserve_minutes") == 10, "Wrong overall/teardown schedule")
    remaining = value["work_deadline_utc"] - time.time()
    require(0 < remaining <= 110 * 60, "Overall work deadline exhausted or invalid")
    return time.monotonic() + remaining


def owned_module(source_root):
    return load_module(Path(source_root) / ".github/qualification/windows_process.py", "q1_owned_process")


def native_gate(args):
    require(sys.platform == "win32", "Native Windows gate is unqualified on this platform")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(exist_ok=False)
    provenance = verify_installed(args.wheel, args.source_root, args.commit)
    staged = stage_tests(args.source_root, root / "reviewed-tests", contract(args.source_root))
    write_json(root / "output/native-spec.json", {"provenance": provenance, "staged": staged})
    environment = child_environment(root / "state/test-home", root / "state/test-tmp")
    environment["NEXUS_Q1_OUTPUT_DIR"] = str(root / "output")
    owner = owned_module(args.source_root)
    command = [sys.executable, "-I", "-X", "utf8", str(Path(args.source_root) / ".github/qualification/runner.py"),
               "_pytest", "--root", str(root), "--wheel", str(args.wheel),
               "--source-root", str(args.source_root), "--commit", args.commit]
    with owner.ProcessDomain(root / "output/native-process", deadline=deadline_from_file(args.deadline_file)) as domain:
        result = domain.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment)
        write_json(root / "output/native-lifecycle.json", domain.records)
        # Collect any test-generated diagnostic artifacts into dedicated output before failure upload
        for p in (root / "state").rglob("*diagnostic*.json"):
            shutil.copy2(p, root / "output" / p.name)
    results_path = root / "output/native-results.json"
    require(results_path.is_file(), "Native test outcomes unavailable; no model fetch")
    results = json.loads(results_path.read_text())
    qualified = result.returncode == 0 and results.get("qualified") is True
    # Golden transport/real CLI exit2 belongs here, not a model warmup.
    if qualified:
        driver = load_module(Path(args.source_root) / ".github/e2e/driver.py", "q1_native_golden_driver")
        with owner.ProcessDomain(root / "output/native-golden", deadline=deadline_from_file(args.deadline_file)) as domain:
            driver._run_command = lambda argv, **kw: domain.run(argv, env=kw.pop("env", environment), **kw)
            old_cwd = Path.cwd()
            try:
                os.chdir(root / "state")
                qualified = driver.self_test() is True
            finally:
                os.chdir(old_cwd)
            write_json(root / "output/native-golden.json", domain.records)
    gate = {"success": qualified, "qualified": qualified, "commit": args.commit, "wheel_sha256": provenance["wheel_sha256"],
            "root": str(root), "platform": sys.platform, "native_results_sha256": digest(results_path),
            "provenance": provenance, "model_fetch_permitted_by_gate": qualified,
            "external_execution_approval": False}
    write_json(root / "output/native-gate.json", gate)
    return 0 if qualified else 1


class BackendObserver:
    def __init__(self, model, home):
        self.model = Path(model).resolve()
        self.home = Path(home).resolve()
        self.servers = {}

    def __call__(self, owned):
        import psutil
        for fact in owned:
            pid = fact.get("pid")
            if type(pid) is not int:
                continue
            try:
                process = psutil.Process(pid)
                executable = Path(process.exe()).resolve()
                if executable.name.lower() not in ("llama-server", "llama-server.exe"):
                    continue
                argv = process.cmdline()
                key = str(pid) + ":" + str(process.create_time())
                if key not in self.servers:
                    self.servers[key] = {"pid": pid, "executable": str(executable), "argv": argv,
                                         "exe_sha256": digest(executable), "observed_owned": True}
                row = self.servers[key]
                model_arg = argv[argv.index("-m") + 1] if "-m" in argv else None
                context = argv[argv.index("-c") + 1] if "-c" in argv else None
                row["model_matches"] = model_arg is not None and Path(model_arg).resolve() == self.model
                row["context"] = context
                row["exe_inside_fresh_home"] = executable.is_relative_to(self.home)
                versions = []
                for log in sorted((self.home / ".nexus-agent/llama-server").glob("server-*.log")):
                    if log.is_symlink() or log.stat().st_size > EFFECT_FILE_CAP:
                        continue
                    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                        if re.search(r"\b(?:build|version)\s*[:=]", line, re.IGNORECASE):
                            versions.append(line[:256])
                row["version_evidence"] = versions[:8]
                row["qualified"] = bool(row["model_matches"] and context == "4096"
                                        and row["exe_inside_fresh_home"] and row["version_evidence"])
            except (psutil.Error, OSError, ValueError, IndexError):
                continue
        return {"servers": list(self.servers.values())[:32]}

    def qualified(self):
        return bool(self.servers) and all(row.get("qualified") is True for row in self.servers.values())


def snapshot_effects(workspace, destination, budget):
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for path in sorted(workspace.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        if not (path.suffix in (".py", ".yaml", ".txt") or path.name.endswith(".py.bak")):
            continue
        size = path.stat().st_size
        require(size <= EFFECT_FILE_CAP and budget[0] + size <= EFFECT_TOTAL_CAP, "Effect evidence cap; qualification incomplete")
        data = path.read_bytes()
        require(len(data) == size, "Effect changed during capture")
        budget[0] += size
        (destination / path.name).write_bytes(data)
        result[path.name] = data.decode("utf-8", errors="replace")
    return result


def synthetic_canonical(home):
    database = home / ".nexus-agent/history.db"
    if not database.exists():
        return None
    require(database.is_file() and not database.is_symlink() and database.stat().st_size <= CANONICAL_CAP,
            "Synthetic canonical evidence unavailable or exceeds cap")
    # A snapshot is attribution data, never private RunnerStatus reconstruction.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "conversation_history" not in tables:
            return None
        cursor = connection.execute("SELECT id, session_id, role, message_json, timestamp FROM conversation_history ORDER BY id ASC")
        rows = []
        for r in cursor:
            row_id, session_id, role, message_json, ts = r
            content = None
            tool_calls = None
            if message_json:
                try:
                    payload = json.loads(message_json)
                    if isinstance(payload, dict):
                        content = payload.get("content")
                        tool_calls = payload.get("tool_calls")
                except Exception:
                    pass
            rows.append({
                "id": row_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "message_json": message_json,
                "tool_calls": tool_calls,
                "timestamp": ts,
            })
    require(len(json.dumps(rows).encode()) <= CANONICAL_CAP, "Synthetic canonical record cap")
    return rows
def canonical_diff(canonical_before, canonical_after):
    """Compute newly added or rewritten messages attributing this turn.

    Handles session rewrites/replaces, prunes, and new sessions stably
    without assuming static SQL IDs or static list length.
    """
    if not canonical_after:
        return []
    if not canonical_before:
        return list(canonical_after)

    sessions_before = {}
    for r in canonical_before:
        sessions_before.setdefault(r["session_id"], []).append(r)

    sessions_after = {}
    for r in canonical_after:
        sessions_after.setdefault(r["session_id"], []).append(r)

    delta = []
    for sid, after_list in sessions_after.items():
        before_list = sessions_before.get(sid, [])
        if not before_list:
            delta.extend(after_list)
            continue

        prefix_len = 0
        while (prefix_len < len(before_list) and prefix_len < len(after_list) and
               before_list[prefix_len]["role"] == after_list[prefix_len]["role"] and
               before_list[prefix_len]["message_json"] == after_list[prefix_len]["message_json"]):
            prefix_len += 1

        if prefix_len == len(before_list):
            delta.extend(after_list[prefix_len:])
        else:
            before_multiset = [b["message_json"] for b in before_list]
            for r in after_list:
                if r["message_json"] in before_multiset:
                    before_multiset.remove(r["message_json"])
                else:
                    delta.append(r)
    return delta


def make_execute_adapter(*, domain, workspace, home, effects_root, environment,
                         effect_budget, axes, report, backend):
    """Bind the frozen driver's direct command alias to owned evidence capture.

    Keep execute as the alias itself: its immediate caller supplies the semantic
    row name (or the self-test frame), and one counter spans all driver calls.
    Snapshots, canonical attribution, reporting and backend gating share this
    same boundary in production and harmless transport integration tests.
    """
    index = 0

    def execute(argv, **kwargs):
        nonlocal index
        index += 1
        import inspect
        frame = inspect.currentframe().f_back
        try:
            row_name = frame.f_locals.get("name") if frame else None
            if not row_name:
                co_name = frame.f_code.co_name if frame else ""
                if co_name == "self_test":
                    row_name = "selftest"
                elif any("doctor" in str(a) for a in argv):
                    row_name = "doctor"
                elif any("git" in str(a) for a in argv):
                    row_name = "git-baseline"
                else:
                    row_name = f"command-{index:04d}"
        finally:
            del frame
        before = snapshot_effects(workspace, effects_root / f"{index:04d}-{row_name}-before", effect_budget)
        canonical_before = synthetic_canonical(home) or []
        local_environment = dict(environment)
        local_environment.update(kwargs.pop("env", {}))
        # driver env derives from this sanitized parent, never runner host credentials.
        result = domain.run(argv, env=local_environment, **kwargs)
        after = snapshot_effects(workspace, effects_root / f"{index:04d}-{row_name}-after", effect_budget)
        canonical_after = synthetic_canonical(home) or []
        row_canonical = canonical_diff(canonical_before, canonical_after)
        record = domain.records[-1]
        text = result.stdout if isinstance(result.stdout, str) else (result.stdout or b"").decode("utf-8", errors="replace")
        error = result.stderr if isinstance(result.stderr, str) else (result.stderr or b"").decode("utf-8", errors="replace")
        axes.append(report.assess_row(row_name, before=before, after=after,
                    stdout=text, stderr=error, exit_code=result.returncode,
                    lifecycle=record, capture_complete=record.get("capture_complete", False), canonical=row_canonical))
        if "say ok" in argv:
            require(backend.qualified(), "Actual owned backend/version/model unqualified; no remaining rows")
        return result

    return execute


def run_workload(args):
    require(sys.platform == "win32", "Real workload requires independently gated native Windows")
    root = Path(args.root).resolve(strict=True)
    provenance = verify_installed(args.wheel, args.source_root, args.commit)
    gate = json.loads((root / "output/native-gate.json").read_text())
    require(gate.get("qualified") is True and gate.get("platform") == "win32"
            and gate.get("commit") == args.commit and gate.get("root") == str(root)
            and gate.get("wheel_sha256") == provenance["wheel_sha256"]
            and gate.get("native_results_sha256") == digest(root / "output/native-results.json"),
            "Native gate identity/outcomes not qualified; no model run")
    source = Path(args.model_source).resolve(strict=True)
    require(source.is_file() and not source.is_symlink() and digest(source) == MODEL_SHA256, "Exact reviewed model bytes required")
    home, workspace = root / "state/model-home", root / "state/workspace"
    home.mkdir(parents=True, exist_ok=False); workspace.mkdir(parents=True, exist_ok=False)
    model = home / MODEL_REL
    model.parent.mkdir(parents=True)
    with source.open("rb") as incoming, model.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    require(digest(model) == MODEL_SHA256, "Staged model mismatch")
    (home / ".nexus_agent_initialized").write_text("Nexus-Agent initialized.\n")
    environment = child_environment(home, root / "state/model-tmp")
    environment.update(NEXUS_AGENT_MODEL_REPO="LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
                       NEXUS_AGENT_MODEL_FILENAME="LFM2.5-1.2B-Instruct-Q4_0.gguf",
                       NEXUS_E2E_WORKDIR=str(workspace))
    owner = owned_module(args.source_root)
    driver = load_module(Path(args.source_root) / ".github/e2e/driver.py", "q1_full_frozen_driver")
    report = load_module(Path(args.source_root) / ".github/qualification/report.py", "q1_axes_report")
    driver.WORKDIR = workspace
    backend = BackendObserver(model, home)
    effect_budget = [0]
    axes = []
    summary = {"commit": args.commit, "provenance": provenance, "model_sha256": MODEL_SHA256,
               "status": "incomplete", "model_readiness": "not_adjudicated", "rows": axes}
    old_cwd = Path.cwd()
    try:
        with owner.ProcessDomain(root / "output/workload", deadline=deadline_from_file(args.deadline_file), observer=backend) as domain:
            driver._run_command = make_execute_adapter(
                domain=domain, workspace=workspace, home=home,
                effects_root=root / "output/effects", environment=environment,
                effect_budget=effect_budget, axes=axes, report=report, backend=backend)
            # Freeze ambient driver environment to the exact synthetic policy, restoring on exit.
            ambient = dict(os.environ)
            try:
                os.environ.clear(); os.environ.update(environment)
                summary["driver_exit"] = driver.main()
            finally:
                os.environ.clear(); os.environ.update(ambient)
            summary.update(status="completed", lifecycle=domain.records, backend=list(backend.servers.values()))
        require(digest(model) == MODEL_SHA256, "Selected model bytes changed")
        canonical = synthetic_canonical(home)
        if canonical is not None:
            write_json(root / "output/synthetic-canonical.json", canonical, CANONICAL_CAP)
    except BaseException as error:
        summary.update(status="incomplete", error_class=type(error).__name__, backend=list(backend.servers.values()))
        raise
    finally:
        os.chdir(old_cwd)
        write_json(root / "output/qualification.json", summary)
        # Copy only bounded regular legacy artifacts, never HOME/model/engine weights.
        artifacts = root / "output/legacy"
        artifacts.mkdir(exist_ok=True)
        total = 0
        for path in sorted(workspace.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix not in (".out", ".txt"):
                continue
            size = path.stat().st_size
            if size > EFFECT_FILE_CAP or total + size > EFFECT_TOTAL_CAP:
                write_json(root / "output/artifact-incomplete.json", {"complete": False, "reason": "legacy_artifact_cap"})
                break
            shutil.copyfile(path, artifacts / path.name); total += size
    return int(summary.get("driver_exit", 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("native-gate", "workload", "_pytest"))
    for name in ("source-root", "root", "wheel", "commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-file")
    parser.add_argument("--model-source")
    args = parser.parse_args(argv)
    if args.phase == "_pytest":
        return pytest_phase(args)
    require(args.deadline_file is not None, "Reviewed absolute deadline required")
    if args.phase == "native-gate":
        return native_gate(args)
    require(args.model_source is not None, "Explicit reviewed model source required")
    return run_workload(args)


if __name__ == "__main__":
    raise SystemExit(main())
