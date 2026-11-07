"""Q1 untraced Windows native functional route: installed-wheel and native gate.

This harness executes untraced native functional tests only; automatic diagnostic
trace worker wrapping is disabled. No model download, warmup, or workload execution
is permitted. Real-model qualification requires separate independent approval.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

RUNNER_SHA256 = "6bf1c0f1eff306eec2b73f6b77a9661e4b077d9846f200e0f692472491db97db"
WINDOWS_PROCESS_SHA256 = "c64489a55a6937ea8714bb663493279d93f03ed60294d7eb38d1413e7cbc24d2"


def _load_old_runner():
    path = Path(__file__).resolve().with_name("runner.py")
    if not path.is_file():
        raise RuntimeError("Old runner not found at " + str(path))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("Old runner SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("q1_immutable_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load spec for old runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_old_runner = _load_old_runner()
GateError = _old_runner.GateError
require = _old_runner.require
digest = _old_runner.digest
write_json = _old_runner.write_json
git = _old_runner.git
contract = _old_runner.contract
verify_installed = _old_runner.verify_installed
critical_outcomes = _old_runner.critical_outcomes
child_environment = _old_runner.child_environment
PremodelAudit = _old_runner.PremodelAudit
GatePlugin = _old_runner.GatePlugin
deadline_from_file = _old_runner.deadline_from_file
owned_module = _old_runner.owned_module
load_module = _old_runner.load_module

NATIVE_HELPERS = _old_runner.HELPERS + (
    ".github/qualification/native_functional.py",
    ".github/qualification/verify_windows_native.py",
    ".github/workflows/q1-windows-native.yml",
    ".github/qualification/interactive_assess.py",
    ".github/qualification/interactive_core.py",
    ".github/qualification/verify_windows_interactive.py",
    ".github/workflows/q1-windows-interactive.yml",
)


def _check_windows_process(source_root):
    path = Path(source_root) / ".github/qualification/windows_process.py"
    require(path.is_file(), "windows_process.py unavailable")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == WINDOWS_PROCESS_SHA256,
            "windows_process.py SHA256 mismatch")


def resolve_approved_seal(approved_commit=None, approved_tree=None):
    commit = approved_commit or os.environ.get("NEXUS_APPROVED_COMMIT")
    tree = approved_tree or os.environ.get("NEXUS_APPROVED_TREE")
    if not commit or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == "0" * 40:
        raise GateError("Mandatory external approved commit seal is missing, unresolved, or zero; execution rejected")
    if not tree or not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree) or tree == "0" * 40:
        raise GateError("Mandatory external approved tree seal is missing, unresolved, or zero; execution rejected")
    return commit, tree


def stage_tests(source_root, destination, frozen):
    source_root, destination = Path(source_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tracked = git(source_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "tests").decode().split("\0")
    paths = sorted({p for p in tracked if p and "/__pycache__/" not in p} | set(NATIVE_HELPERS))
    staged = {}
    for name in paths:
        require(not name.startswith("src/") and ".." not in Path(name).parts, "Unsafe staged helper")
        original = source_root / name
        require(original.is_file() and not original.is_symlink(), "Missing/nonregular reviewed test/helper: " + name)
        blob = original.read_bytes()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        staged[name] = hashlib.sha256(blob).hexdigest()

    # Stage the candidate payloads as read-only neutral data for route tests
    # Never placed in importable src/
    candidate_gate = load_module(source_root / ".github/qualification/verify_windows_native.py", "native_gate_for_staging")
    interactive_gate = load_module(source_root / ".github/qualification/verify_windows_interactive.py", "interactive_gate_for_staging")
    candidate_payloads = {}
    for name in sorted(candidate_gate.ALLOWED_PATHS | interactive_gate.ALLOWED_PATHS):
        p_src = source_root / name
        require(p_src.is_file() and not p_src.is_symlink(), "Missing candidate payload file: " + name)
        p_blob = p_src.read_bytes()
        p_dst = destination / "q1-candidate-payload" / name
        p_dst.parent.mkdir(parents=True, exist_ok=True)
        p_dst.write_bytes(p_blob)
        candidate_payloads[name] = hashlib.sha256(p_blob).hexdigest()

    config = destination / "isolated-pytest.ini"
    config.write_text("[pytest]\npythonpath =\naddopts =\ntestpaths = tests\n", encoding="utf-8")
    require(not (destination / "src").exists(), "Source may not be staged")
    return {"files": staged, "candidate_payloads": candidate_payloads,
            "config": str(config), "root": str(destination),
            "critical_native": frozen["critical_native"]}


def pytest_phase(args):
    require("NEXUS_Q1_OUTPUT_DIR" not in os.environ, "NEXUS_Q1_OUTPUT_DIR must not be set in untraced native functional tests")
    import pytest
    root = Path(args.root).resolve()
    spec = json.loads((root / "output/native-spec.json").read_text())
    provenance = verify_installed(args.wheel, args.source_root, args.commit)
    require(provenance["wheel_sha256"] == spec["provenance"]["wheel_sha256"], "Native wheel identity changed")
    require(sys.gettrace() is None, "Trace function active before native functional tests")
    audit = PremodelAudit(provenance["package_root"], provenance["package_py"])
    sys.addaudithook(audit)
    plugin = GatePlugin(provenance["package_root"])
    stage = Path(spec["staged"]["root"])
    old_cwd = os.getcwd()
    os.environ["NEXUS_Q1_NATIVE_RECEIPT_DIR"] = str(root / "output")
    try:
        os.chdir(stage)
        exit_code = pytest.main(["-c", spec["staged"]["config"], "--rootdir", str(stage),
                                 "-o", "pythonpath=", "-o", "addopts=", "tests", "-q"], plugins=[plugin])
    finally:
        os.chdir(old_cwd)
        os.environ.pop("NEXUS_Q1_NATIVE_RECEIPT_DIR", None)
    require(sys.gettrace() is None, "Trace function leaked into harness after tests")
    final = critical_outcomes(plugin.outcomes, spec["staged"]["critical_native"], sys.platform)
    termination_nodeid = ("tests/test_q1_interactive_core.py::"
                          "test_terminate_verified_real_os_handle_identity")
    termination_outcome = plugin.outcomes.get(termination_nodeid)
    termination_receipt = root / "output" / "termination_native_test.json"
    if sys.platform == "win32":
        require(termination_outcome != "passed" or termination_receipt.is_file(),
                "Passed native termination test without a retained receipt")
        require(termination_outcome == "passed" or not termination_receipt.is_file(),
                "Retained native termination receipt without a passing outcome")
    else:
        require(not termination_receipt.is_file(),
                "Native termination receipt retained off Windows")
    final.update(
        exit_code=int(exit_code),
        collected=len(plugin.collected),
        outcomes=plugin.outcomes,
        violations=audit.violations,
        children=audit.children,
        executed_package_files=sorted(audit.executed_package_files),
        platform=sys.platform,
        native_functional_only=True,
        final_qualification=False,
        model_execution_permitted=False,
        untraced_native=True,
        trace_function_active=False,
        termination_receipt_sha256=(digest(termination_receipt)
                                    if termination_receipt.is_file() else None),
    )
    # Detect installed-file mutation too; generated RECORD/bytecode are not compared.
    verify_installed(args.wheel, args.source_root, args.commit)
    final["qualified"] = bool(final["qualified"] and int(exit_code) == 0 and not audit.violations and sys.platform == "win32")
    write_json(root / "output/native-results.json", final)
    return 0 if final["qualified"] else 1


def native_gate(args):
    require(sys.platform == "win32", "Native Windows gate is unqualified on this platform")
    _check_windows_process(args.source_root)
    approved_commit, approved_tree = resolve_approved_seal(args.approved_commit, args.approved_tree)
    require(args.commit == approved_commit, "Native gate run commit differs from approved commit seal")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=False)

    # Copy built wheel into output directory so it is collected in native upload artifact
    wheel_path = Path(args.wheel).resolve(strict=True)
    wheel_dest_dir = output_dir / "wheel"
    wheel_dest_dir.mkdir(parents=True, exist_ok=True)
    dest_wheel = wheel_dest_dir / wheel_path.name
    if not dest_wheel.exists():
        shutil.copy2(wheel_path, dest_wheel)

    provenance = verify_installed(args.wheel, args.source_root, args.commit)
    require(provenance.get("commit") == approved_commit, "Observed installed commit differs from approved commit seal")
    require(provenance.get("tree") == approved_tree, "Observed installed tree differs from approved tree seal")
    staged = stage_tests(args.source_root, root / "reviewed-tests", contract(args.source_root))
    write_json(output_dir / "native-spec.json", {"provenance": provenance, "staged": staged})
    environment = child_environment(root / "state/test-home", root / "state/test-tmp")
    # Crucial: NEXUS_Q1_OUTPUT_DIR is deliberately NOT set to preserve untraced execution.
    owner = owned_module(args.source_root)
    this_script = str(Path(__file__).resolve())
    command = [
        sys.executable, "-I", "-X", "utf8", this_script,
        "_pytest", "--root", str(root), "--wheel", str(args.wheel),
        "--source-root", str(args.source_root), "--commit", args.commit,
    ]
    with owner.ProcessDomain(output_dir / "native-process", deadline=deadline_from_file(args.deadline_file)) as domain:
        try:
            result = domain.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment)
        finally:
            write_json(output_dir / "native-lifecycle.json", domain.records)
    results_path = output_dir / "native-results.json"
    require(results_path.is_file(), "Native test outcomes unavailable")
    results = json.loads(results_path.read_text())
    qualified = result.returncode == 0 and results.get("qualified") is True
    # Golden transport/real CLI exit2 belongs here, not a model warmup.
    if qualified:
        driver = load_module(Path(args.source_root) / ".github/e2e/driver.py", "q1_native_golden_driver")
        with owner.ProcessDomain(output_dir / "native-golden", deadline=deadline_from_file(args.deadline_file)) as domain:
            try:
                driver._run_command = lambda argv, **kw: domain.run(argv, env=kw.pop("env", environment), **kw)
                old_cwd = Path.cwd()
                try:
                    os.chdir(root / "state")
                    qualified = driver.self_test() is True
                finally:
                    os.chdir(old_cwd)
            finally:
                write_json(output_dir / "native-golden.json", domain.records)
    gate = {
        "success": qualified,
        "qualified": qualified,
        "commit": approved_commit,
        "tree": approved_tree,
        "wheel_sha256": provenance["wheel_sha256"],
        "root": str(root),
        "platform": sys.platform,
        "native_results_sha256": digest(results_path),
        "provenance": provenance,
        "native_functional_only": True,
        "final_qualification": False,
        "model_execution_permitted": False,
        "untraced_native": True,
        "model_fetch_permitted_by_gate": False,
        "external_execution_approval": False,
    }
    write_json(output_dir / "native-gate.json", gate)
    return 0 if qualified else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pytest_cmd = commands.add_parser("_pytest")
    pytest_cmd.add_argument("--root", required=True)
    pytest_cmd.add_argument("--wheel", required=True)
    pytest_cmd.add_argument("--source-root", required=True)
    pytest_cmd.add_argument("--commit", required=True)
    native_cmd = commands.add_parser("native-run")
    native_cmd.add_argument("--source-root", required=True)
    native_cmd.add_argument("--root", required=True)
    native_cmd.add_argument("--wheel", required=True)
    native_cmd.add_argument("--commit", required=True)
    native_cmd.add_argument("--deadline-file", required=True)
    native_cmd.add_argument("--approved-commit")
    native_cmd.add_argument("--approved-tree")
    args = parser.parse_args(argv)
    if args.command == "_pytest":
        return pytest_phase(args)
    return native_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
