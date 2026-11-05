"""Interactive candidate/event verifier, native functional prerequisite check and bounded publication.

Independent review binds candidate commit, parent ancestry, native gate and publication
caps. Local preparation seals the full 18-path candidate invariant and 16 payload pins
without circular self-hashing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

PARENT = "ae1c0c2e9fddf9e9078a6d87e8c1b2202930ce0b"
PARENT_TREE = "95efded383cf137850d7f3144f0063455bead5e0"
REVIEWED_REF = "refs/heads/qualification/q1-windows-realmodel-v10-20261105-1850-62b4ad17"
NATIVE_BRANCH = "qualification/q1-windows-native-v10-20261105-1850-62b4ad17"
NATIVE_WORKFLOW = ".github/workflows/q1-windows-native.yml"

GATE_PATH = ".github/qualification/verify_windows_interactive.py"
NATIVE_GATE_PATH = ".github/qualification/verify_windows_native.py"
TOTAL_CAP = 48 * 1024 * 1024
FILE_CAP = 16 * 1024 * 1024
MANIFEST_RESERVE = 1024 * 1024
OLD_GATE_SHA256 = "d166b42450ef01ec9dd408b30ca691ec003ef2d4c349c5a7b8d8cbf6ef175f17"
ZERO_SHA = "0" * 40

VERIFIER_PATHS = frozenset({
    GATE_PATH,
    NATIVE_GATE_PATH,
})

ADDED_PATHS = frozenset({
    ".github/qualification/native_functional.py",
    ".github/qualification/verify_windows_native.py",
    ".github/workflows/q1-windows-native.yml",
    "tests/fixtures/q1_search_recorded_1712.json",
    "tests/test_q1_native_route.py",
})

MODIFIED_PATHS = frozenset({
    ".github/qualification/interactive_core.py",
    ".github/qualification/verify_windows_interactive.py",
    ".github/workflows/q1-windows-interactive.yml",
    "README.md",
    "src/nexus_agent_ai/agent/core.py",
    "src/nexus_agent_ai/agent/result_budget.py",
    "tests/test_approved_integration.py",
    "tests/test_capacity_recovery.py",
    "tests/test_count_text_agent.py",
    "tests/test_guardrails.py",
    "tests/test_q1_interactive_route.py",
    "tests/test_runner_evidence_d1.py",
    "tests/test_tool_result_guard.py",
})

ALLOWED_PATHS = ADDED_PATHS | MODIFIED_PATHS

# Exactly 16 non-verifier candidate files pinned; both verifiers bound out of band.
PAYLOAD_SHA256 = {
    ".github/qualification/interactive_core.py": "40aa087dcb64baf2ae6e19307bb13cc61d5f120ee5593303257c96f9d55ba214",
    ".github/qualification/native_functional.py": "2896285b7ee42b0d335dc0cd88c01a13ff554a1db299b7d09bc850a9fbe12efc",
    ".github/workflows/q1-windows-interactive.yml": "63b1b4a8dd17fe607c3aeb8e7a51835ebc80bc1fcf31cb20bd9b80ceaf8d8de5",
    ".github/workflows/q1-windows-native.yml": "8d2d38b45dd45c9f39b0c14154fefa0945a0fdc5b124cb02351c0a7032f02ec8",
    "README.md": "d9842d16b578ac25a9ca4f25d0373eac393b96bbeb13de199979c38bba460199",
    "src/nexus_agent_ai/agent/core.py": "efc18695767f0da44da4dc5eb0e29868f9cb148c505abf09c84d75afc5a04bb7",
    "src/nexus_agent_ai/agent/result_budget.py": "bd6be4637dc005b219812144e1eaf920a1f26a3bb0a0de24d69ba86f0e9e6f41",
    "tests/fixtures/q1_search_recorded_1712.json": "06388e71c69d267a95584ea57d930d51412c78ad917a455d0734dd2ed1b39949",
    "tests/test_approved_integration.py": "8613c000b51fdc25f26963b311d2661e9a7a927fa7ff463b6aa928b40f549d6d",
    "tests/test_capacity_recovery.py": "700879c8f451862308921657741e6f451be7347aeaeddaf807301c0f2506d1cc",
    "tests/test_count_text_agent.py": "93990093340806bbb47f231048f84d675ea1b42ca302bddbd3a68d7a0a3b02f8",
    "tests/test_guardrails.py": "b0988e904d1a224b7f1e0ae736a4e9b63261df0b34afcc64f2f9a97eba6f680e",
    "tests/test_q1_interactive_route.py": "b7654622c2f800cb4e7578eff19574f5df0cbc7dc9aab42e1eb60b64b21ec943",
    "tests/test_q1_native_route.py": "5e0546e06a20b78b25af6071fc41703adf975623523e7efa97f543255255fbe2",
    "tests/test_runner_evidence_d1.py": "71bec0225112391c9ca6abe582183f1bfffe6adf480def2f7fd3c3fd6e1a6adf",
    "tests/test_tool_result_guard.py": "196bc98c8a55fc7e383dd4302e72ef0003664972c694477f8a0777fa66570739",
}


def _load(alias, relpath, expected):
    path = Path(__file__).resolve().with_name(relpath)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError("Hash mismatch for " + relpath)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = _load("b_immutable_primitives", "verify_candidate.py", OLD_GATE_SHA256)
require = old._require
ReviewError = old.ReviewError


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_approved_seal(approved_commit=None, approved_tree=None):
    commit = approved_commit or os.environ.get("NEXUS_APPROVED_COMMIT")
    tree = approved_tree or os.environ.get("NEXUS_APPROVED_TREE")
    if not commit or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == ZERO_SHA:
        raise ReviewError("Mandatory external approved commit seal is missing, unresolved, or zero; execution rejected")
    if not tree or not isinstance(tree, str) or not re.fullmatch(r"[0-9a-f]{40}", tree) or tree == ZERO_SHA:
        raise ReviewError("Mandatory external approved tree seal is missing, unresolved, or zero; execution rejected")
    return commit, tree


def verify_tree(repo, tree, approved_tree=None):
    repo = Path(repo)
    if approved_tree is not None:
        require(tree == approved_tree, f"Candidate tree {tree} differs from approved tree seal {approved_tree}")
    require(old._commit_headers(repo, PARENT)[0] == PARENT_TREE, "Parent tree mismatch")
    before, after = old._entries(repo, PARENT_TREE), old._entries(repo, tree)
    ancestors = {str(p) for name in ALLOWED_PATHS for p in PurePosixPath(name).parents if str(p) != "."}
    changed = set()
    for name in before.keys() | after.keys():
        a, b = before.get(name), after.get(name)
        if a == b:
            continue
        if name in ancestors:
            require(b is not None and b[:2] == ("040000", "tree") and
                    (a is None or a[:2] == ("040000", "tree")), "Non-directory ancestor: " + name)
            continue
        require(name in ALLOWED_PATHS, "Change outside exact candidate paths: " + name)
        if name in ADDED_PATHS:
            require(a is None, "Added path already existed in baseline: " + name)
            require(b is not None and b[:2] == ("100644", "blob"), "Nonregular added file: " + name)
        else:
            require(a is not None and a[:2] == ("100644", "blob"), "Modified path missing from baseline: " + name)
            require(b is not None and b[:2] == ("100644", "blob"), "Nonregular modified file: " + name)
        changed.add(name)
    require(changed == ALLOWED_PATHS, "Exact eighteen candidate paths required")
    require(set(PAYLOAD_SHA256) == ALLOWED_PATHS - VERIFIER_PATHS, "Complete payload pins required for all 16 candidate files")
    for name, expected in PAYLOAD_SHA256.items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid payload pin: " + name)
        blob = old._git(repo, "cat-file", "blob", after[name][2])
        require(hashlib.sha256(blob).hexdigest() == expected, "Payload mismatch: " + name)
    return {
        "tree": tree, "parent": PARENT, "parent_tree": PARENT_TREE,
        "exact_delta_paths": sorted(changed), "payload_sha256": PAYLOAD_SHA256,
        "interactive_verifier_sha256": hashlib.sha256(old._git(repo, "cat-file", "blob", after[GATE_PATH][2])).hexdigest(),
        "native_verifier_sha256": hashlib.sha256(old._git(repo, "cat-file", "blob", after[NATIVE_GATE_PATH][2])).hexdigest(),
        "real_model_coverage_candidate": True, "qualified": False,
        "external_execution_approval": False,
    }


def verify_commit(repo, sha, approved_commit=None, approved_tree=None):
    commit, tree = resolve_approved_seal(approved_commit, approved_tree)
    require(sha == commit, f"Commit {sha} differs from approved commit seal {commit}")
    tree_obj, parents = old._commit_headers(Path(repo), old._object_id(sha))
    require(parents == [PARENT], "Exactly declared single parent required")
    require(tree_obj == tree, f"Commit tree {tree_obj} differs from approved tree seal {tree}")
    return dict(verify_tree(repo, tree_obj, approved_tree=tree), commit=sha)


def verify_event(event, *, event_name, ref, sha, run_attempt):
    old._object_id(sha)
    require(event_name == "push", "Only push is permitted")
    require(ref == REVIEWED_REF, "Branch ref mismatch")
    require(str(run_attempt) == "1", "Only first attempt is permitted")
    require(isinstance(event, dict), "Invalid push event")
    require(event.get("ref") == REVIEWED_REF, "Event ref mismatch")
    require(event.get("created") is True, "Only branch creation is permitted")
    require(event.get("before") == ZERO_SHA, "Non-zero before hash on creation")
    require(event.get("deleted") is False, "Deletion is not permitted")
    require(event.get("forced") is False, "Force push is not permitted")
    require(event.get("after") == sha, "Event SHA mismatch")
    return sha


def verify_checkout(repo, *, local=False, approved_commit=None, approved_tree=None):
    repo = Path(repo)
    commit, tree = resolve_approved_seal(approved_commit, approved_tree)
    sha = old._git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    require(sha == commit, f"HEAD {sha} differs from approved commit seal {commit}")
    result = verify_commit(repo, sha, approved_commit=commit, approved_tree=tree)
    if local:
        require("GITHUB_ACTIONS" not in os.environ, "Local verification forbidden inside GitHub Actions")
    else:
        env_sha = os.environ.get("GITHUB_SHA", "")
        require(env_sha == commit, f"GITHUB_SHA {env_sha} differs from approved commit seal {commit}")
        path = os.environ.get("GITHUB_EVENT_PATH")
        require(bool(path), "Explicit push event required")
        with open(path, "r", encoding="utf-8") as f:
            event = json.load(f)
        verify_event(event, event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
                     ref=os.environ.get("GITHUB_REF", ""), sha=sha,
                     run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    require(not old._git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip(), "Dirty checkout")
    executing_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(executing_hash == result["interactive_verifier_sha256"], "Executing verifier differs from candidate blob")
    return result


def select_native_run(metadata, repository, approved_commit=None, approved_tree=None):
    """Select only an actual successful attempt-one native workflow run matching approved commit and tree."""
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is not None, "Invalid repository")
    commit, tree = resolve_approved_seal(approved_commit, approved_tree)
    runs = metadata.get("workflow_runs", []) if isinstance(metadata, dict) else []
    matches = [r for r in runs if isinstance(r, dict) and
               r.get("head_sha") == commit and
               r.get("head_branch") == NATIVE_BRANCH and
               r.get("path") == NATIVE_WORKFLOW and
               r.get("event") == "push" and r.get("run_attempt") == 1 and
               r.get("status") == "completed" and r.get("conclusion") == "success" and
               r.get("repository", {}).get("full_name") == repository]
    require(len(matches) == 1, "Exactly one actual successful untraced native run required")
    run = matches[0]
    run_id = run.get("id")
    head_sha = run.get("head_sha")
    require(type(run_id) is int and run_id > 0, "Invalid native run identity")
    require(isinstance(head_sha, str) and re.fullmatch(r"[0-9a-f]{40}", head_sha) is not None, "Invalid head_sha in run")
    return {"run_id": run_id, "commit": head_sha, "tree": tree, "repository": repository}


def check_native_evidence(path, commit=None, tree=None, wheel=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= TOTAL_CAP, "Missing bounded native evidence")
    gate = json.loads(path.read_bytes())
    target_commit, target_tree = resolve_approved_seal(commit, tree)
    require(isinstance(gate, dict), "Invalid native gate payload")
    require(gate.get("commit") == target_commit, "Native gate commit mismatch")
    require(gate.get("platform") == "win32" and gate.get("success") is True and
            gate.get("qualified") is True and gate.get("untraced_native") is True and
            gate.get("native_functional_only") is True and gate.get("final_qualification") is False and
            gate.get("model_execution_permitted") is False and gate.get("model_fetch_permitted_by_gate") is False,
            "Actual untraced native PASS required; diagnostic/waiver/model permission forbidden")
    provenance = gate.get("provenance", {})
    require(provenance.get("commit") == target_commit, "Native provenance commit mismatch")
    require(provenance.get("tree") == target_tree, "Native provenance tree mismatch")
    package = provenance.get("package_py")
    require(isinstance(package, dict) and bool(package), "Missing native package identity")
    require(all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in package.items()), "Invalid native package pins")
    if wheel is not None:
        wheel_path = Path(wheel)
        require(wheel_path.is_file(), "Wheel file missing for pre-install digest check: " + str(wheel))
        actual_wheel_sha256 = digest(wheel_path)
        require(actual_wheel_sha256 == gate.get("wheel_sha256"), "Wheel digest mismatch against native gate")
    return gate


FILE_COUNT_CAP = 512
ENTRY_COUNT_CAP = 4096


def collect_output(source, destination):
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), "Interactive upload directory must be fresh")
    require(destination.resolve() != source.resolve() and
            not destination.resolve().is_relative_to(source.resolve()), "Upload destination overlaps source")
    destination.mkdir(parents=True)
    report = {
        "real_model_coverage_evidence": True,
        "final_qualification": False,
        "external_execution_approval": False,
        "complete": True,
        "files": [],
        "errors": [],
    }
    total, visited = 0, 0
    stack = [(source, 0)]
    try:
        while stack:
            directory, depth = stack.pop()
            if depth > 16:
                report["complete"] = False
                report["errors"].append(f"Directory depth exceeded at {directory.name}")
                continue
            if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
                report["complete"] = False
                report["errors"].append(f"Linked directory skipped: {directory.name}")
                continue
            try:
                entries_scan = os.scandir(directory)
            except OSError as error:
                report["complete"] = False
                report["errors"].append(f"Directory read error {directory.name}: {type(error).__name__}")
                continue
            with entries_scan as entries:
                for entry in entries:
                    visited += 1
                    if visited > ENTRY_COUNT_CAP:
                        report["complete"] = False
                        report["errors"].append("Entry count cap exceeded")
                        break
                    path = Path(entry.path)
                    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                        report["complete"] = False
                        report["errors"].append(f"Linked entry skipped: {path.name}")
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((path, depth + 1))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        report["complete"] = False
                        report["errors"].append(f"Nonregular entry skipped: {path.name}")
                        continue
                    name = path.relative_to(source).as_posix()
                    if len(name) > 256:
                        report["complete"] = False
                        report["errors"].append(f"Filename too long: {path.name[:32]}")
                        continue
                    if len(report["files"]) >= FILE_COUNT_CAP:
                        report["complete"] = False
                        report["errors"].append("File count cap exceeded")
                        break
                    try:
                        size = entry.stat(follow_symlinks=False).st_size
                    except OSError as error:
                        report["complete"] = False
                        report["errors"].append(f"Stat error {name}: {type(error).__name__}")
                        continue
                    if size > FILE_CAP:
                        report["complete"] = False
                        report["errors"].append(f"File {name} exceeds file cap ({size} > {FILE_CAP})")
                        continue
                    if total + size > TOTAL_CAP - MANIFEST_RESERVE:
                        report["complete"] = False
                        report["errors"].append(f"File {name} exceeds aggregate 48 MiB capacity cap")
                        continue
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copied, digest_val = 0, hashlib.sha256()
                    try:
                        with path.open("rb") as incoming, target.open("xb") as outgoing:
                            while True:
                                chunk = incoming.read(min(65536, size - copied + 1))
                                if not chunk:
                                    break
                                if copied + len(chunk) > size:
                                    raise OSError("File grew during copy")
                                if total + copied + len(chunk) > TOTAL_CAP - MANIFEST_RESERVE:
                                    raise OSError("Total capacity exceeded during copy")
                                outgoing.write(chunk)
                                digest_val.update(chunk)
                                copied += len(chunk)
                        if copied != size:
                            raise OSError("File shrank during copy")
                    except Exception as error:
                        target.unlink(missing_ok=True)
                        report["complete"] = False
                        report["errors"].append(f"Copy error {name}: {type(error).__name__}")
                        continue
                    total += copied
                    report["files"].append({"path": name, "bytes": copied, "sha256": digest_val.hexdigest()})
    except Exception as error:
        report["complete"] = False
        report["errors"].append(type(error).__name__ + ": " + str(error)[:256])
    raw = json.dumps(report, ensure_ascii=True, indent=2).encode("ascii")
    require(len(raw) <= MANIFEST_RESERVE and total + len(raw) <= TOTAL_CAP, "Final interactive upload manifest cap")
    (destination / "diagnostic-manifest.json").write_bytes(raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--source-root", default=os.getcwd())
    verify.add_argument("--local", action="store_true")
    verify.add_argument("--approved-commit")
    verify.add_argument("--approved-tree")
    select = commands.add_parser("select-native-run")
    select.add_argument("--metadata", required=True)
    select.add_argument("--repository", required=True)
    select.add_argument("--approved-commit", required=True)
    select.add_argument("--approved-tree", required=True)
    native = commands.add_parser("check-native")
    native.add_argument("--native-evidence", required=True)
    native.add_argument("--commit")
    native.add_argument("--tree")
    native.add_argument("--wheel")
    collect = commands.add_parser("collect")
    collect.add_argument("--output-root", required=True)
    collect.add_argument("--upload-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_checkout(args.source_root, local=args.local,
                                     approved_commit=args.approved_commit,
                                     approved_tree=args.approved_tree)
        elif args.command == "select-native-run":
            result = select_native_run(json.loads(Path(args.metadata).read_bytes()),
                                       args.repository, approved_commit=args.approved_commit,
                                       approved_tree=args.approved_tree)
        elif args.command == "check-native":
            gate = check_native_evidence(args.native_evidence, commit=args.commit,
                                         tree=args.tree, wheel=args.wheel)
            result = {"native_pass": True, "commit": gate["commit"], "wheel_sha256": gate.get("wheel_sha256"), "execution_approval": False}
        else:
            result = collect_output(args.output_root, args.upload_root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("complete", True) else 1
    except (ReviewError, OSError, ValueError) as error:
        print("INTERACTIVE VERIFICATION FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
