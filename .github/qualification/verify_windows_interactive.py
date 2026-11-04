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

PARENT = "92312e6a862e1493ec24702d56f57c8c505d6691"
PARENT_TREE = "ac6a6c43d795c5ef8d17b94ed8ff3a0a290a2ad0"
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
    ".github/qualification/interactive_assess.py",
    "tests/test_q1_interactive_core.py", "tests/test_q1_interactive_route.py",
})
# Final payload pins; this verifier's SHA256 is bound out of band.
PAYLOAD_SHA256 = {
    ".github/qualification/interactive_core.py": "864f918b3c28d66ff38d118dd3f4a0e49d8bf78e76376f1527c309a7fa1c6c80",
    ".github/qualification/interactive_assess.py": "8fd2cc08d6273cfd52bc5efaef8c2be628042959b7d83cabb0f1442982c14d3b",
    "tests/test_q1_interactive_core.py": "29392f813338c50aa1039065645669c375c1a6fe4b93964430bae462627e779f",
    "tests/test_q1_interactive_route.py": "bb1e028fe10ad7a357297b0140983afe42b2606d538c434d57e9afe6130acad8",
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
        require(name in ALLOWED_PATHS and a is not None and a[:2] == ("100644", "blob")
                and b is not None and b[:2] == ("100644", "blob"),
                "Only exact reviewed B revision files allowed: " + name)
        changed.add(name)
    require(changed == ALLOWED_PATHS, "Exact five-file reviewed revision required")
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
