"""B local candidate/event verifier, prior A gate check and bounded publication.

No remote calls or execution authorization. Main must approve this exact seal and
validate current free/account/model terms and live unused ref/run before push.
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

PARENT = "ceee768ec7580489c35949498491e7b1cbbd5fa4"
PARENT_TREE = "3491413724c1f23e59c30168a088816692370194"
REVIEWED_REF = "refs/heads/qualification/q1-windows-interactive-ceee-20261104-2040-6c83e21f"
A_COMMIT = "d8a7e95ed8141f25326092930a2ba2736023546a"
A_TREE = "beca126eb8c1507d0311515ad658f12dc9ec276f"
A_BRANCH = "qualification/q1-windows-native-ceee-20261104-2015-f7a93c61"
GATE_PATH = ".github/qualification/verify_windows_interactive.py"
EVIDENCE_CAP = 64 * 1024 * 1024
OLD_GATE_SHA256 = "d166b42450ef01ec9dd408b30ca691ec003ef2d4c349c5a7b8d8cbf6ef175f17"
OLD_COLLECTOR_SHA256 = "be04a2405f6fbb43880a69352166f3c0313486e0f1b34eae86988067b310df14"
ALLOWED_PATHS = frozenset({
    GATE_PATH, ".github/qualification/interactive_core.py",
    ".github/qualification/interactive_assess.py", ".github/workflows/q1-windows-interactive.yml",
    "tests/test_q1_interactive_core.py", "tests/test_q1_interactive_route.py",
})
# Final payload pins; this verifier's SHA256 is bound out of band.
PAYLOAD_SHA256 = {
    ".github/qualification/interactive_core.py": "1b19adbeca48d03baf728f19e35c908f20aa9b28632182ca614bf2f2966f6490",
    ".github/qualification/interactive_assess.py": "2efabff03bae221567f72e35c495d779bca5ad8eb6178cdca115763ac2dae285",
    ".github/workflows/q1-windows-interactive.yml": "43056edb3beac4eb4c463e47632f1f10b3b182e6ce4d437725e6dcbe86b3a9d1",
    "tests/test_q1_interactive_core.py": "97b63032652bc7c1aad98a68a47079f08a5ce523ac629ca0ffe7121875c8f578",
    "tests/test_q1_interactive_route.py": "181990fbdc25931c3c037d2497debee79eb66a48767b16292797bc3748be856c",
}


def _load(name, filename, expected):
    path = Path(__file__).with_name(filename)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Immutable helper changed: " + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = _load("b_immutable_git", "verify_candidate.py", OLD_GATE_SHA256)
ReviewError, require = old.ReviewError, old._require


def verify_tree(repo, tree):
    repo = Path(repo)
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
                    (a is None or a[:2] == ("040000", "tree")), "Non-directory ancestor")
            continue
        require(name in ALLOWED_PATHS and a is None and b is not None and
                b[:2] == ("100644", "blob"), "Only exact additive B files allowed: " + name)
        changed.add(name)
    require(changed == ALLOWED_PATHS, "Exact six-file additive delta required")
    require(set(PAYLOAD_SHA256) == ALLOWED_PATHS - {GATE_PATH}, "Complete payload pins required")
    for name, expected in PAYLOAD_SHA256.items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid payload pin")
        blob = old._git(repo, "cat-file", "blob", after[name][2])
        require(hashlib.sha256(blob).hexdigest() == expected, "Payload mismatch: " + name)
    return {"tree": tree, "parent": PARENT, "parent_tree": PARENT_TREE,
            "exact_delta_paths": sorted(changed), "payload_sha256": PAYLOAD_SHA256,
            "verifier_sha256": hashlib.sha256(old._git(repo, "cat-file", "blob", after[GATE_PATH][2])).hexdigest(),
            "real_model_coverage_candidate": True, "qualified": False,
            "external_execution_approval": False}


def verify_commit(repo, sha):
    tree, parents = old._commit_headers(Path(repo), old._object_id(sha))
    require(parents == [PARENT], "Exactly declared single parent required")
    return dict(verify_tree(repo, tree), commit=sha)


def verify_event(event, *, event_name, ref, sha, run_attempt):
    old._object_id(sha)
    require(event_name == "push" and ref == REVIEWED_REF and run_attempt == "1", "First literal push attempt only")
    require(isinstance(event, dict) and event.get("ref") == REVIEWED_REF, "Event ref mismatch")
    require(event.get("created") is True and event.get("before") == "0" * 40, "Fresh ref creation only")
    require(event.get("deleted") is False and event.get("forced") is False, "Deletion/force forbidden")
    require(event.get("after") == sha, "Event SHA mismatch")


def verify_checkout(repo, *, local=False):
    repo = Path(repo)
    head = old._git(repo, "rev-parse", "HEAD").decode().strip()
    if local:
        require(os.environ.get("GITHUB_ACTIONS", "").lower() != "true", "Local forbidden in Actions")
        sha = head
    else:
        path = os.environ.get("GITHUB_EVENT_PATH")
        require(bool(path), "Explicit push event required")
        sha = os.environ.get("GITHUB_SHA", "")
        verify_event(json.loads(Path(path).read_text()), event_name=os.environ.get("GITHUB_EVENT_NAME"),
                     ref=os.environ.get("GITHUB_REF"), sha=sha, run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"))
        require(sha == head, "Checkout HEAD mismatch")
    require(not old._git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip(), "Dirty checkout")
    result = verify_commit(repo, sha)
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == result["verifier_sha256"], "Executing verifier mismatch")
    return result


def select_native_run(metadata, repository):
    """Select only an actual successful attempt-one A workflow run, never a boolean waiver."""
    require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is not None, "Invalid repository")
    runs = metadata.get("workflow_runs", []) if isinstance(metadata, dict) else []
    matches = [r for r in runs if isinstance(r, dict) and
               r.get("head_sha") == A_COMMIT and r.get("head_branch") == A_BRANCH and
               r.get("path") == ".github/workflows/q1-windows-native.yml" and
               r.get("event") == "push" and r.get("run_attempt") == 1 and
               r.get("status") == "completed" and r.get("conclusion") == "success" and
               r.get("repository", {}).get("full_name") == repository]
    require(len(matches) == 1, "Exactly one actual successful untraced A run required")
    run_id = matches[0].get("id")
    require(type(run_id) is int and run_id > 0, "Invalid A run identity")
    return {"run_id": run_id, "commit": A_COMMIT, "repository": repository}


def check_native_evidence(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= EVIDENCE_CAP, "Missing bounded A evidence")
    gate = json.loads(path.read_bytes())
    require(isinstance(gate, dict) and gate.get("commit") == A_COMMIT and
            gate.get("platform") == "win32" and gate.get("success") is True and
            gate.get("qualified") is True and gate.get("untraced_native") is True and
            gate.get("native_functional_only") is True and gate.get("final_qualification") is False and
            gate.get("model_execution_permitted") is False and gate.get("model_fetch_permitted_by_gate") is False,
            "Actual untraced native A PASS required; diagnostic/waiver forbidden")
    provenance = gate.get("provenance", {})
    require(provenance.get("commit") == A_COMMIT and provenance.get("tree") == A_TREE, "A provenance mismatch")
    package = provenance.get("package_py")
    require(isinstance(package, dict) and bool(package), "Missing A package identity")
    require(all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in package.items()), "Invalid A package pins")
    return gate


def collect_output(source, destination):
    collector = _load("b_immutable_collector", "verify_windows_diagnostics.py", OLD_COLLECTOR_SHA256)
    collector.FILE_CAP = EVIDENCE_CAP
    report = collector.collect_output(source, destination)
    report.pop("diagnostic_only", None)
    report.pop("requires_untraced_native", None)
    report.update(real_model_coverage_evidence=True, external_execution_approval=False)
    data = json.dumps(report, ensure_ascii=True, indent=2).encode()
    require(sum(x["bytes"] for x in report["files"]) + len(data) <= EVIDENCE_CAP, "Evidence cap")
    (Path(destination) / "diagnostic-manifest.json").write_bytes(data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--source-root", default=os.getcwd())
    verify.add_argument("--local", action="store_true")
    select = commands.add_parser("select-native-run")
    select.add_argument("--metadata", required=True)
    select.add_argument("--repository", required=True)
    native = commands.add_parser("check-native")
    native.add_argument("--native-evidence", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output-root", required=True)
    collect.add_argument("--upload-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_checkout(args.source_root, local=args.local)
        elif args.command == "select-native-run":
            result = select_native_run(json.loads(Path(args.metadata).read_bytes()), args.repository)
        elif args.command == "check-native":
            gate = check_native_evidence(args.native_evidence)
            result = {"A_native_pass": True, "commit": gate["commit"], "execution_approval": False}
        else:
            result = collect_output(args.output_root, args.upload_root)
        print(json.dumps(result, indent=2))
        return 0 if result.get("complete", True) else 1
    except (ReviewError, OSError, ValueError) as error:
        print("B VERIFICATION FAILED: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
