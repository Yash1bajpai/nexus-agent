"""Local-only Q1_FULL_WINDOWS_V1 candidate verifier.

Independent review and main approval must bind this script itself and the final
commit/tree/bundle out of band. Its payload hashes are not self-authenticating.
No remote is queried here: verify the literal remote ref is absent immediately
before ONE approved push. Never reuse/recreate the branch, rerun, or add commits.
GitHub first-push/first-attempt checks are not a server-side one-shot lock.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

INVARIANT = "Q1_FULL_WINDOWS_V1"
REVIEWED_PARENT = "f016eb836b88dfb52d4ff2c60c5fbc0bd404d93c"
REVIEWED_REF = "refs/heads/qualification/q1-f016eb8-8c73d412"
BASE_TREE = "67905b16532b51c2651af221844663d401e829b7"
ZERO_SHA = "0" * 40
GATE_PATH = ".github/qualification/verify_candidate.py"

ALLOWED_PATHS = frozenset({
    ".github/e2e/driver.py",
    ".github/qualification/runner.py",
    ".github/qualification/windows_process.py",
    ".github/qualification/report.py",
    GATE_PATH,
    ".github/workflows/q1-windows-qualification.yml",
    "tests/test_q1_reporting.py",
    "tests/test_q1_transport.py",
    "tests/test_q1_windows_process.py",
    "tests/test_q1_route.py",
    "tests/test_q1_installed_gate.py",
    "tests/fixtures/q1/frozen_contract.json",
})

# BEGIN REVIEWED PAYLOAD SHA256: integration owner seals all eleven final blobs.
# The gate itself is excluded; independent review binds its hash out of band.
PAYLOAD_SHA256 = {
    ".github/e2e/driver.py": "8be477fa74bf55ca2a1826b901a8f53b14a5221a2a4e411a7a67a6176f526f01",
    ".github/qualification/runner.py": "26bf384cff7e798eef2b61f84bb80cd1be0a54494eebc89e42a973f282114352",
    ".github/qualification/windows_process.py": "7622fb7180fffc8a12c4e9fa052b9847ca3ef3ac61181fcb28e27b3a94bcfe3f",
    ".github/qualification/report.py": "77809203ad33cd729f8237c1d68d5f6fb5fe4af75639ef1499c589db96758bcb",
    ".github/workflows/q1-windows-qualification.yml": "01859c9dd33bb89d7899fdd8fd56998d9e82ead6ef16c8530889bd9cb4209ff0",
    "tests/test_q1_reporting.py": "35784d45d5e591b671675abe6c5c1c87bd658de9f2fdfaee5a4db7039aa52697",
    "tests/test_q1_transport.py": "3a6337ec2f73870669058f19173dfec6e4d7836fbdaaca8971843dd2a0be0bfe",
    "tests/test_q1_windows_process.py": "ae1393ef03d22f2b382904971813a2e9506fed2ba29da7899ad04a8d45e54d68",
    "tests/test_q1_route.py": "1a2cb810d9ec95bcf678f3e616fa0d56252b50800e3039fddf188f6334147bdb",
    "tests/test_q1_installed_gate.py": "a37f993c4b094c88655bb8e669f2fcaebbb1053036b87d9182713c9a5512b5c8",
    "tests/fixtures/q1/frozen_contract.json": "e53160523ce458cf4a0fb83cfcc6397e6533dca325a97ab6deb87c89d2660274",
}
# END REVIEWED PAYLOAD SHA256

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReviewError(ValueError):
    """The candidate does not satisfy the bounded reviewed invariant."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReviewError(reason)


def _object_id(value: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             "Expected an exact lowercase 40-character Git object ID")
    return value


def _git(repo: Path, *args: str) -> bytes:
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
            [executable, "--no-replace-objects", "-C", str(repo), *args],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=15,
        )
    except subprocess.TimeoutExpired:
        raise ReviewError("Local Git lookup timed out") from None
    except OSError:
        raise ReviewError("Cannot read local Git repository") from None
    _require(result.returncode == 0, "Local Git lookup failed")
    return result.stdout


def _commit_headers(repo: Path, sha: str) -> tuple[str, list[str]]:
    _object_id(sha)
    _require(_git(repo, "cat-file", "-t", sha).strip() == b"commit",
             "Expected a commit object")
    headers = _git(repo, "cat-file", "commit", sha).split(b"\n\n", 1)[0].splitlines()
    trees = [line[5:].decode("ascii") for line in headers if line.startswith(b"tree ")]
    parents = [line[7:].decode("ascii") for line in headers if line.startswith(b"parent ")]
    _require(len(trees) == 1, "Commit must have exactly one tree header")
    return _object_id(trees[0]), [_object_id(parent) for parent in parents]


def _entries(repo: Path, tree: str) -> dict[str, tuple[str, str, str]]:
    _object_id(tree)
    _require(_git(repo, "cat-file", "-t", tree).strip() == b"tree",
             "Expected a tree object, not a commit/tag/blob")
    result = {}
    for record in _git(repo, "ls-tree", "-r", "-t", "-z", "--full-tree", tree).split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        name = path.decode("utf-8", errors="surrogateescape")
        _require(name not in result, "Duplicate tree entry")
        result[name] = (mode, kind, _object_id(oid))
    return result


def load_contract(source_root: Path | str) -> dict:
    contract_file = Path(source_root) / "tests/fixtures/q1/frozen_contract.json"
    _require(contract_file.is_file(), "Missing frozen contract fixture")
    return json.loads(contract_file.read_text(encoding="utf-8"))


def verify_tree(repo: str | Path, tree: str, parent: str | None = None,
                payload_hashes: dict | None = None) -> dict:
    repo = Path(repo)
    parent = REVIEWED_PARENT if parent is None else parent
    parent_tree, _ = _commit_headers(repo, parent)
    _require(parent_tree == BASE_TREE, "Parent tree does not match expected D1 tree")
    before, after = _entries(repo, parent_tree), _entries(repo, tree)
    changed = {path for path in before.keys() | after.keys()
               if before.get(path) != after.get(path)}
    ancestors = {str(Path(path).parent).replace("\\", "/") for path in ALLOWED_PATHS}
    ancestors |= {".github", ".github/workflows", ".github/qualification", "tests", "tests/fixtures", "tests/fixtures/q1"}
    changed_files = set()
    for path in changed:
        old, new = before.get(path), after.get(path)
        if path in ancestors:
            _require(new is not None and new[:2] == ("040000", "tree")
                     and (old is None or old[:2] == ("040000", "tree")),
                     f"Non-directory ancestor: {path}")
            continue
        _require(path in ALLOWED_PATHS, f"Change outside reviewed paths: {path}")
        _require(new is not None and new[:2] == ("100644", "blob"),
                 f"Reviewed paths must be regular 100644 A/M blobs: {path}")
        _require(old is None or old[:2] == ("100644", "blob"),
                 f"Reviewed path is not an A/M regular blob: {path}")
        changed_files.add(path)
    _require(changed_files == ALLOWED_PATHS,
             "Changed path set must equal the twelve reviewed paths exactly")
    hashes = PAYLOAD_SHA256 if payload_hashes is None else payload_hashes
    expected_payload_paths = ALLOWED_PATHS - {GATE_PATH}
    _require(set(hashes) == expected_payload_paths,
             "Reviewed payload hash table must seal exactly eleven files")
    for path, digest in hashes.items():
        _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
                 f"Invalid reviewed SHA256: {path}")
        blob = _git(repo, "cat-file", "blob", after[path][2])
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"Reviewed payload SHA256 mismatch: {path}")

    # Verify frozen contract blob
    contract_blob = _git(repo, "cat-file", "blob", after["tests/fixtures/q1/frozen_contract.json"][2])
    contract_data = json.loads(contract_blob.decode("utf-8"))
    _require(contract_data.get("invariant") == INVARIANT, "Frozen contract invariant mismatch")
    _require(contract_data.get("base_commit") == REVIEWED_PARENT, "Frozen contract base commit mismatch")
    _require(contract_data.get("base_tree") == BASE_TREE, "Frozen contract base tree mismatch")
    _require(contract_data.get("base_driver_blob") == before[".github/e2e/driver.py"][2],
             "Frozen contract base driver blob mismatch")

    # Verify driver calls in source order match frozen schedule
    driver_blob = _git(repo, "cat-file", "blob", after[".github/e2e/driver.py"][2])
    driver_tree = ast.parse(driver_blob.decode("utf-8"))
    launches = [n for n in ast.walk(driver_tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "_run_command"]
    _require(len(launches) == 6, "Driver must have exactly 6 calls to _run_command")
    direct_sub = [n for n in ast.walk(driver_tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "subprocess" and n.func.attr == "run"]
    _require(len(direct_sub) == 0, "Driver must not contain unaliased subprocess.run calls")

    # Verify base tree files outside .github/e2e/driver.py are unchanged
    for path, (mode, kind, oid) in before.items():
        if kind != "blob":
            continue
        if path == ".github/e2e/driver.py":
            continue
        _require(after.get(path) == (mode, kind, oid),
                 f"Base tree file {path} was modified or deleted")

    # Verify push route branches in workflow
    wf_blob = _git(repo, "cat-file", "blob", after[".github/workflows/q1-windows-qualification.yml"][2])
    wf_text = wf_blob.decode("utf-8")
    _require("qualification/q1-f016eb8-8c73d412" in wf_text, "Workflow must trigger only on reviewed ref")
    _require("workflow_dispatch:" not in wf_text, "Workflow must not permit workflow_dispatch")
    _require("pull_request:" not in wf_text, "Workflow must not permit pull_request")

    return {"invariant": INVARIANT, "tree": tree, "parent": parent}


def verify_commit(repo: str | Path, sha: str, parent: str | None = None,
                  payload_hashes: dict | None = None) -> dict:
    repo = Path(repo)
    parent = REVIEWED_PARENT if parent is None else parent
    tree, parents = _commit_headers(repo, sha)
    _require(parents == [parent], "Candidate must have exactly the reviewed single parent")
    result = verify_tree(repo, tree, parent=parent, payload_hashes=payload_hashes)
    return {**result, "commit": sha}


def verify_event(event: dict, *, event_name: str, ref: str, sha: str, run_attempt: str) -> None:
    _require(event_name == "push", f"Event name must be push, got {event_name}")
    _require(ref == REVIEWED_REF, f"Ref must be {REVIEWED_REF}, got {ref}")
    _require(str(run_attempt) == "1", f"Run attempt must be 1, got {run_attempt}")
    _require(isinstance(event, dict), "Push event payload must be an object")
    _require(event.get("ref") == REVIEWED_REF, "Push payload ref mismatch")
    _require(event.get("created") is True, "Only a newly created branch push is permitted")
    _require(event.get("before") == ZERO_SHA, "Push before must be all-zero object ID")
    _require(event.get("deleted") is False, "Branch deletion is forbidden")
    _require(event.get("forced") is False, "Forced pushes are forbidden")
    _require(event.get("after") == sha, "Push after must equal commit SHA")


def verify_candidate(repo: str | Path, *, event: dict | None = None,
                     event_name: str | None = None, ref: str | None = None,
                     sha: str | None = None, run_attempt: str | None = None,
                     payload_hashes: dict | None = None) -> dict:
    repo = Path(repo)
    if event is None and "GITHUB_EVENT_PATH" in os.environ:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    if event_name is None:
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if ref is None:
        ref = os.environ.get("GITHUB_REF") or _git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode("ascii").strip()
    if sha is None:
        sha = os.environ.get("GITHUB_SHA") or _git(repo, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    if run_attempt is None:
        run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    sha = _object_id(sha)
    if event is not None:
        verify_event(event, event_name=event_name, ref=ref, sha=sha, run_attempt=run_attempt)

    actual_head = _git(repo, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    _require(actual_head == sha, f"Actual checkout HEAD {actual_head} does not equal commit SHA {sha}")
    return verify_commit(repo, sha, payload_hashes=payload_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=os.getcwd(),
                        help="Root of repository under verification")
    args = parser.parse_args(argv)
    try:
        result = verify_candidate(args.source_root)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
