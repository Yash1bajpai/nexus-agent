"""Local Q1 Windows untraced native functional candidate verification.

Independent review must bind this verifier's SHA256 and the final commit/tree/
ref/run identities. Local preparation binds invariant structure and payload pins
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
import shutil
import subprocess
import sys
BASE_COMMIT = "ae1c0c2e9fddf9e9078a6d87e8c1b2202930ce0b"
REVIEWED_PARENT = "fe62609fe956baaead419c47dab7441e31d84f88"
PARENT_TREE = "293aa18714241a05871c8e91931bbee05fb43323"
NATIVE_BRANCH = os.environ.get("NEXUS_NATIVE_BRANCH", "")
REVIEWED_REF = f"refs/heads/{NATIVE_BRANCH}" if NATIVE_BRANCH else ""
INVARIANT = "Q1_WINDOWS_NATIVE_V2"
GATE_PATH = ".github/qualification/verify_windows_native.py"
INTERACTIVE_GATE_PATH = ".github/qualification/verify_windows_interactive.py"
OLD_GATE_SHA256 = "d166b42450ef01ec9dd408b30ca691ec003ef2d4c349c5a7b8d8cbf6ef175f17"
ZERO_SHA = "0" * 40

VERIFIER_PATHS = frozenset({
    GATE_PATH,
    INTERACTIVE_GATE_PATH,
})

ADDED_PATHS = frozenset({
    ".github/workflows/q1-windows-conpty.yml",
})

MODIFIED_PATHS = frozenset({
    ".github/qualification/interactive_assess.py",
    ".github/qualification/interactive_core.py",
    ".github/qualification/native_functional.py",
    ".github/qualification/verify_windows_interactive.py",
    ".github/qualification/verify_windows_native.py",
    ".github/workflows/q1-windows-interactive.yml",
    ".github/workflows/q1-windows-native.yml",
    "src/nexus_agent_ai/agent/tools.py",
    "src/nexus_agent_ai/providers/local_provider.py",
    "tests/test_budget_diagnostic_route.py",
    "tests/test_local_provider.py",
    "tests/test_q1_diagnostics_route.py",
    "tests/test_q1_interactive_core.py",
    "tests/test_q1_interactive_route.py",
    "tests/test_q1_native_route.py",
    "tests/test_q1_route.py",
    "tests/test_tools.py",
})

ALLOWED_PATHS = ADDED_PATHS | MODIFIED_PATHS

# Non-verifier candidate files pinned exactly; both verifiers bound out of band.
PAYLOAD_SHA256 = {
    ".github/qualification/interactive_assess.py": "8b18e9ea00b62c4d3181927456f28a73bc531c424df40b75dd542ca9c24088b5",
    ".github/qualification/interactive_core.py": "a96392bef56afc1ff8faf1fbc89648364c5e2fc2d7f12ba657d5d4ca0ba53da0",
    ".github/qualification/native_functional.py": "583f414e6ec84e47b7f94c4baf4e4f6ce9e549a957249cdcde7e3dc420a93af0",
    ".github/workflows/q1-windows-interactive.yml": "129d52c0409864fa983566d239607df6e8d2a2490168a2fbd3efa0ea31017c8f",
    ".github/workflows/q1-windows-native.yml": "481cdb7d7c1c6fbbf7a696ffd92bf1923ac41d66657be617308743ddf9910fdf",
    "src/nexus_agent_ai/agent/tools.py": "ef23ef0f39b6577f45de395ad9f1f10b5889c84b2f770b9a410f05ce7dad9c3e",
    "src/nexus_agent_ai/providers/local_provider.py": "11661be1d23f4ed53f4f97895c9b5472e9d9bd713d128ccd7c00d21ea6c0e48e",
    "tests/test_budget_diagnostic_route.py": "0b3fee9741081fc1dc09ddda974550a2afca454317138373df16ae04ae2b7370",
    "tests/test_local_provider.py": "8ec48936fe664ef0d991a24e8ba9e0de229f4984b8b7ecd3745c0d4cb5fcd416",
    "tests/test_q1_diagnostics_route.py": "4e203fb4f37dd202cfbfef9844a635e499852ee6f7141b167e5aebff212f050d",
    "tests/test_q1_interactive_core.py": "559038fc1b6fd6a94791c6cf9ab58fc1136358eb522a224dac02fb8374ba7ed3",
    "tests/test_q1_interactive_route.py": "49132c2ccb7a54228cce6f4f2b8c61f3910c5fd1e18175ef21e31ef8fc09d5f9",
    "tests/test_q1_native_route.py": "8247b31ce598818a00afb5e1d1123ca81e3a92044910f14b78b6b3ccc339e074",
    "tests/test_q1_route.py": "132a7366eb17ee1e461f76b8ae7219f502b95f915a8e09dd4a0badbeba9ce2cf",
    "tests/test_tools.py": "a396628e84a05266bd97e74628e7605f3955e1aa38d02d9500bfabaf1e0c9e98",
    ".github/workflows/q1-windows-conpty.yml": "cdfc69d60053b3f427f06e06b68504b4956a3926418b35970993b97ad00c4271"
}


def _old_primitives():
    path = Path(__file__).with_name("verify_candidate.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_GATE_SHA256:
        raise ValueError("Old immutable verifier bytes changed")
    spec = importlib.util.spec_from_file_location("q1_immutable_git_primitives", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_old = _old_primitives()
ReviewError = _old.ReviewError
_require = _old._require
_git = _old._git
_entries = _old._entries
_commit_headers = _old._commit_headers
_object_id = _old._object_id

def _check_ancestry(repo: Path, base: str, parent: str) -> None:
    directories = [str(Path(sys.executable).parent)]
    environment = {}
    if os.name == "nt":
        for name in ("SYSTEMROOT", "WINDIR"):
            root = os.environ.get(name)
            if root and Path(root).is_absolute():
                environment[name] = root
                directories.extend((str(Path(root) / "System32"), root))
    else:
        directories.extend(part for part in os.defpath.split(os.pathsep)
                           if part and Path(part).is_absolute())
    executable = shutil.which("git")
    _require(executable is not None and Path(executable).is_absolute(),
             "Local Git executable is unavailable")
    directories.append(str(Path(executable).resolve().parent))
    environment["PATH"] = os.pathsep.join(dict.fromkeys(directories))
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    try:
        result = subprocess.run(
            [executable, "--no-replace-objects", "-C", str(repo), "merge-base", base, parent],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise ReviewError("Local Git lookup timed out") from None
    except OSError:
        raise ReviewError("Cannot read local Git repository") from None
    if result.returncode == 0:
        ancestor = result.stdout.decode("ascii", errors="replace").strip()
        _require(ancestor == base, "Declared parent does not descend from grounded base")
    elif result.returncode == 1:
        raise ReviewError("Declared parent does not descend from grounded base")
    else:
        raise ReviewError("Local Git lookup failed")


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
        _require(tree == approved_tree, f"Candidate tree {tree} differs from approved tree seal {approved_tree}")
    parent_tree, _ = _commit_headers(repo, REVIEWED_PARENT)
    _require(parent_tree == PARENT_TREE, "Declared parent tree mismatch")
    _check_ancestry(repo, BASE_COMMIT, REVIEWED_PARENT)
    before, after = _entries(repo, PARENT_TREE), _entries(repo, tree)
    ancestors = {str(parent) for path in ALLOWED_PATHS for parent in PurePosixPath(path).parents
                 if str(parent) != "."}
    changed_files = set()
    for path in before.keys() | after.keys():
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        if path in ancestors:
            _require(new is not None and new[:2] == ("040000", "tree") and
                     (old is None or old[:2] == ("040000", "tree")), "Non-directory ancestor: " + path)
            continue
        _require(path in ALLOWED_PATHS, "Change outside exact candidate paths: " + path)
        if path in ADDED_PATHS:
            _require(old is None, "Added path already existed in baseline: " + path)
            _require(new is not None and new[:2] == ("100644", "blob"), "Nonregular added file: " + path)
        else:
            _require(old is not None and old[:2] == ("100644", "blob"), "Modified path missing from baseline: " + path)
            _require(new is not None and new[:2] == ("100644", "blob"), "Nonregular modified file: " + path)
        changed_files.add(path)
    _require(changed_files == ALLOWED_PATHS, "Delta must equal exactly the declared candidate paths")
    _require(set(PAYLOAD_SHA256) == ALLOWED_PATHS - VERIFIER_PATHS, "Payload table must pin exactly the non-verifier candidate files")
    for path, expected in PAYLOAD_SHA256.items():
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid SHA256 pin: " + path)
        blob = _git(repo, "cat-file", "blob", after[path][2])
        _require(hashlib.sha256(blob).hexdigest() == expected, "Payload SHA256 mismatch: " + path)
    return {
        "invariant": INVARIANT,
        "tree": tree,
        "base": BASE_COMMIT,
        "parent": REVIEWED_PARENT,
        "parent_tree": PARENT_TREE,
        "exact_delta_paths": sorted(changed_files),
        "payload_sha256": PAYLOAD_SHA256,
        "native_verifier_sha256": hashlib.sha256(_git(repo, "cat-file", "blob", after[GATE_PATH][2])).hexdigest(),
        "interactive_verifier_sha256": hashlib.sha256(_git(repo, "cat-file", "blob", after[INTERACTIVE_GATE_PATH][2])).hexdigest(),
        "native_functional_only": True,
        "untraced_native": True,
        "diagnostic_only": False,
        "final_qualification": False,
        "model_execution_permitted": False,
        "external_execution_approval": False,
    }


def verify_commit(repo, sha, approved_commit=None, approved_tree=None):
    commit, tree = resolve_approved_seal(approved_commit, approved_tree)
    _require(sha == commit, f"Commit {sha} differs from approved commit seal {commit}")
    tree_obj, parents = _commit_headers(Path(repo), _object_id(sha))
    _require(parents == [REVIEWED_PARENT], "Exactly declared single parent required")
    _require(tree_obj == tree, f"Commit tree {tree_obj} differs from approved tree seal {tree}")
    return dict(verify_tree(repo, tree_obj, approved_tree=tree), commit=sha)


def verify_event(event, *, event_name, ref, sha, run_attempt):
    _object_id(sha)
    _require(event_name == "push", "Only push is permitted")
    _require(REVIEWED_REF, "NEXUS_NATIVE_BRANCH not configured; refusing to bind a branch")
    _require(ref == REVIEWED_REF, "Branch ref mismatch")
    _require(str(run_attempt) == "1", "Only first attempt is permitted")
    _require(isinstance(event, dict), "Invalid push payload")
    _require(event.get("ref") == REVIEWED_REF, "Event ref mismatch")
    _require(event.get("created") is True, "Only branch creation is permitted")
    _require(event.get("before") == ZERO_SHA, "Non-zero before hash on creation")
    _require(event.get("deleted") is False, "Deletion is not permitted")
    _require(event.get("forced") is False, "Force push is not permitted")
    _require(event.get("after") == sha, "Event SHA mismatch")
    return sha


def verify_checkout(repo, *, local=False, approved_commit=None, approved_tree=None):
    repo = Path(repo)
    commit, tree = resolve_approved_seal(approved_commit, approved_tree)
    sha = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    _require(sha == commit, f"HEAD {sha} differs from approved commit seal {commit}")
    result = verify_commit(repo, sha, approved_commit=commit, approved_tree=tree)
    if local:
        _require("GITHUB_ACTIONS" not in os.environ, "Local verification forbidden inside GitHub Actions")
    else:
        env_sha = os.environ.get("GITHUB_SHA", "")
        _require(env_sha == commit, f"GITHUB_SHA {env_sha} differs from approved commit seal {commit}")
        path = os.environ.get("GITHUB_EVENT_PATH")
        _require(bool(path), "Explicit push event required")
        with open(path, "r", encoding="utf-8") as handle:
            event = json.load(handle)
        verify_event(
            event,
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            ref=os.environ.get("GITHUB_REF", ""),
            sha=sha,
            run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        )
    _require(not _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip(), "Dirty checkout")
    executing_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _require(executing_hash == result["native_verifier_sha256"], "Executing verifier differs from candidate blob")
    return result


FILE_CAP = 16 * 1024 * 1024
TOTAL_CAP = 16 * 1024 * 1024
MANIFEST_RESERVE = 1024 * 1024
FILE_COUNT_CAP = 512
ENTRY_COUNT_CAP = 4096


def collect_output(source, destination):
    source, destination = Path(source), Path(destination)
    _require(not destination.exists(), "Native upload directory must be fresh")
    _require(destination.resolve() != source.resolve() and
             not destination.resolve().is_relative_to(source.resolve()), "Upload destination overlaps source")
    destination.mkdir(parents=True)
    report = {
        "native_functional_only": True,
        "untraced_native": True,
        "diagnostic_only": False,
        "final_qualification": False,
        "model_execution_permitted": False,
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
                        report["errors"].append(f"File {name} exceeds total capacity cap")
                        continue
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copied, digest = 0, hashlib.sha256()
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
                                digest.update(chunk)
                                copied += len(chunk)
                        if copied != size:
                            raise OSError("File shrank during copy")
                    except Exception as error:
                        target.unlink(missing_ok=True)
                        report["complete"] = False
                        report["errors"].append(f"Copy error {name}: {type(error).__name__}")
                        continue
                    total += copied
                    report["files"].append({"path": name, "bytes": copied, "sha256": digest.hexdigest()})
    except Exception as error:
        report["complete"] = False
        report["errors"].append(type(error).__name__ + ": " + str(error)[:256])
    raw = json.dumps(report, ensure_ascii=True, indent=2).encode("ascii")
    _require(len(raw) <= MANIFEST_RESERVE and total + len(raw) <= TOTAL_CAP, "Final native upload manifest cap")
    (destination / "native-manifest.json").write_bytes(raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--source-root", default=os.getcwd())
    verify.add_argument("--local", action="store_true")
    verify.add_argument("--approved-commit")
    verify.add_argument("--approved-tree")
    collect = commands.add_parser("collect")
    collect.add_argument("--output-root", required=True)
    collect.add_argument("--upload-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_checkout(args.source_root, local=args.local,
                                     approved_commit=args.approved_commit,
                                     approved_tree=args.approved_tree)
        else:
            result = collect_output(args.output_root, args.upload_root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("complete", True) else 1
    except (ReviewError, OSError, ValueError) as error:
        print("NATIVE FUNCTIONAL VERIFICATION FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
