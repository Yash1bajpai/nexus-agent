"""Local Q1 Windows untraced native functional candidate verification.

Independent review must bind this verifier's SHA256 and the final commit/tree/
bundle OUT OF BAND. Payload pins do not authenticate their own verifier.
No remote is queried or changed here. Immediately before any separately approved
push, the executor must recover original owner authority and check live ref/run
absence. Never reuse/recreate the consumed route or rerun an attempt.
Native functional route is untraced; model execution is NOT permitted.
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

BASE_COMMIT = "79ac72dcb791f687f5512c71f64b96f761009093"
REVIEWED_PARENT = "ceee768ec7580489c35949498491e7b1cbbd5fa4"
PARENT_TREE = "3491413724c1f23e59c30168a088816692370194"
REVIEWED_REF = "refs/heads/qualification/q1-windows-native-ceee-20261104-2015-f7a93c61"
INVARIANT = "Q1_WINDOWS_NATIVE_V1"
GATE_PATH = ".github/qualification/verify_windows_native.py"
OLD_GATE_SHA256 = "d166b42450ef01ec9dd408b30ca691ec003ef2d4c349c5a7b8d8cbf6ef175f17"
ZERO_SHA = "0" * 40

ALLOWED_PATHS = frozenset({
    GATE_PATH,
    ".github/qualification/native_functional.py",
    ".github/workflows/q1-windows-native.yml",
    "tests/test_q1_native_route.py",
})

# Exactly three payload files; verifier hash is bound out of band after commit.
PAYLOAD_SHA256 = {
    ".github/qualification/native_functional.py": "948ccbd0412eff91a6dd6d489aff6dafeda05eb546070292b06e31efd9ca5003",
    ".github/workflows/q1-windows-native.yml": "9b5ad80c19c259a45483175e597dc1fe62e9750c61eee343cc34fd53e98f0716",
    "tests/test_q1_native_route.py": "d285114cdf90d972b062a1a3c530796173e41373e27f65cb2956d5bb8414dfa1",
}


def _old_primitives():
    # Reuse the old read-only Git/object primitives, not its old route invariant.
    # Check bytes BEFORE importing; never mutate the old module or old files.
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


def verify_tree(repo, tree):
    repo = Path(repo)
    parent_tree, _ = _commit_headers(repo, REVIEWED_PARENT)
    _require(parent_tree == PARENT_TREE, "Declared parent tree mismatch")
    ancestor = _git(repo, "merge-base", BASE_COMMIT, REVIEWED_PARENT).decode("ascii").strip()
    _require(ancestor == BASE_COMMIT, "Declared parent does not descend from grounded base")
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
        _require(path in ALLOWED_PATHS, "Change outside exact additive native paths: " + path)
        _require(old is None, "Existing file modified; only additive native paths permitted: " + path)
        _require(new is not None and new[:2] == ("100644", "blob"), "Nonregular reviewed file: " + path)
        changed_files.add(path)
    _require(changed_files == ALLOWED_PATHS, "Delta must equal all four additive native paths")
    _require(set(PAYLOAD_SHA256) == ALLOWED_PATHS - {GATE_PATH}, "Payload table must pin exactly three files")
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
        "verifier_sha256": hashlib.sha256(_git(repo, "cat-file", "blob", after[GATE_PATH][2])).hexdigest(),
        "native_functional_only": True,
        "untraced_native": True,
        "diagnostic_only": False,
        "final_qualification": False,
        "model_execution_permitted": False,
        "external_execution_approval": False,
    }


def verify_commit(repo, sha):
    tree, parents = _commit_headers(Path(repo), _object_id(sha))
    _require(parents == [REVIEWED_PARENT], "Candidate must have exactly the declared single parent")
    return dict(verify_tree(repo, tree), commit=sha)


def verify_event(event, *, event_name, ref, sha, run_attempt):
    _object_id(sha)
    _require(event_name == "push", "Only push is permitted")
    _require(ref == REVIEWED_REF, "Literal native ref mismatch")
    _require(run_attempt == "1", "Only first run attempt is permitted")
    _require(isinstance(event, dict), "Missing push event object")
    _require(event.get("ref") == REVIEWED_REF, "Event ref mismatch")
    _require(event.get("created") is True and event.get("before") == ZERO_SHA, "Only new-ref creation is permitted")
    _require(event.get("deleted") is False and event.get("forced") is False, "Deletion/force is forbidden")
    _require(event.get("after") == sha, "Event commit SHA mismatch")


def verify_checkout(repo, *, local=False):
    repo = Path(repo)
    head = _git(repo, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if local:
        _require(os.environ.get("GITHUB_ACTIONS", "").lower() != "true", "Local mode is forbidden in Actions")
        sha = head
    else:
        sha = os.environ.get("GITHUB_SHA", "")
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        _require(bool(event_path), "Explicit GitHub push event is required")
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        verify_event(event, event_name=os.environ.get("GITHUB_EVENT_NAME"),
                     ref=os.environ.get("GITHUB_REF"), sha=sha,
                     run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"))
        _require(sha == head, "Actual checkout HEAD does not match event SHA")
    _require(not _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip(), "Candidate checkout is dirty")
    result = verify_commit(repo, sha)
    executing_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _require(executing_hash == result["verifier_sha256"], "Executing verifier differs from candidate blob")
    return result


FILE_CAP = 16 * 1024 * 1024
TOTAL_CAP = 64 * 1024 * 1024
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
    _require(len(raw) <= MANIFEST_RESERVE and total + len(raw) <= TOTAL_CAP, "Final upload manifest cap")
    (destination / "native-manifest.json").write_bytes(raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--source-root", default=os.getcwd())
    verify.add_argument("--local", action="store_true")
    collect = commands.add_parser("collect")
    collect.add_argument("--output-root", required=True)
    collect.add_argument("--upload-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_checkout(args.source_root, local=args.local)
        else:
            result = collect_output(args.output_root, args.upload_root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("complete", True) else 1
    except (ReviewError, OSError, ValueError) as error:
        print("NATIVE FUNCTIONAL VERIFICATION FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
