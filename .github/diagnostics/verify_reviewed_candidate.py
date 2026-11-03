"""Local-only R11_ROUTE_LIFECYCLE_V1 checks, not push authorization.

Independent review and main approval must bind this script itself and the final
commit/tree (or reviewed bundle). Its payload hashes are not self-authenticating.
No remote is queried here: verify the literal remote ref is absent immediately
before ONE approved push. Never reuse/recreate the branch, rerun, or add commits.
GitHub first-push/first-attempt checks are not a server-side one-shot lock.

Integrators may call verify_tree(repo, staged_tree) before committing. That API
checks Git invariants only and grants no approval or permission to push.
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

INVARIANT = "R11_ROUTE_LIFECYCLE_V1"
REVIEWED_PARENT = "590f748f20f9a2b3d17f3cce05a4d1bdea542d35"
REVIEWED_REF = "refs/heads/diagnostic/r11-budget-590f748-2d91c6a7"
ZERO_SHA = "0" * 40
GATE_PATH = ".github/diagnostics/verify_reviewed_candidate.py"
RUNNER_PATH = ".github/diagnostics/budget_diagnostic_runner.py"
ALLOWED_PATHS = frozenset({
    ".github/workflows/budget-diagnostic.yml",
    RUNNER_PATH,
    ".github/diagnostics/windows_job.py",
    GATE_PATH,
    "tests/test_budget_diagnostic_runner.py",
    "tests/test_windows_diagnostic_job.py",
    "tests/test_budget_diagnostic_route.py",
    "README.md",
})

# BEGIN REVIEWED PAYLOAD SHA256: integration owner seals all seven final blobs.
# The gate itself is deliberately excluded; independent approval binds its hash
# through the final commit/tree, never a self-referential embedded commit SHA.
PAYLOAD_SHA256 = {
    ".github/workflows/budget-diagnostic.yml": "c2284efc53e6c9414ea8b897ea763beb4643ca4737b966ed8111e588b6d2b61e",
    ".github/diagnostics/budget_diagnostic_runner.py": "c99aa2058bd54d889f8ddaa33ce250849e0827a2fa1e52edc6fd4ff55a14ad60",
    ".github/diagnostics/windows_job.py": "a662814d4a11c6f93487402278ef7b696ca95d862a4c5f15da05e744592f939d",
    "tests/test_budget_diagnostic_runner.py": "bb95f5f671a5599ef3f3d1595a8fe7df800757a4b4d06514c5991ddacbf1d940",
    "tests/test_windows_diagnostic_job.py": "623964bf6b780dcd02ae5ca39047c00f41add5ba0facd26cce0e660ac6be5be3",
    "tests/test_budget_diagnostic_route.py": "bb84988209eba2fc5ab0279ea616f248104b215097693a0609f192f8bef5f331",
    "README.md": "d65c849628b2c368a77cd317cb8796f44f757df0c457de4ac2bf608426131d0a",
}
# END REVIEWED PAYLOAD SHA256

_MUTABLE_RUNNER_FUNCTIONS = frozenset({
    "execute_row", "run_bundle", "_windows_job_factory", "_cleanup_posix_process",
})
_NEW_RUNNER_FUNCTIONS = frozenset({"_windows_job_factory", "_cleanup_posix_process"})
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
    # Retain only system/interpreter/discovered-Git search directories, never
    # ambient credentials, proxy settings, object overrides, or user Git config.
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
            check=False, timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise ReviewError("Local Git lookup timed out") from None
    except OSError:
        raise ReviewError("Cannot read local Git repository") from None
    _require(result.returncode == 0, "Local Git object/checkout lookup failed")
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
    # Include directory objects so crafted empty trees/type changes cannot hide
    # behind a recursive blob-only listing.
    for record in _git(repo, "ls-tree", "-r", "-t", "-z", "--full-tree", tree).split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        name = path.decode("utf-8", errors="surrogateescape")
        _require(name not in result, "Duplicate tree entry")
        result[name] = (mode, kind, _object_id(oid))
    return result


def _frozen_runner(source: bytes, *, candidate: bool) -> tuple[list[str], list[str]]:
    """Preserve every nonblank source line and AST node outside lifecycle edits."""
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeError, SyntaxError) as exc:
        raise ReviewError("Runner must remain valid UTF-8 Python") from exc
    ignored_lines = set()
    frozen_nodes = []
    names = []
    added_imports = 0
    for node in module.body:
        mutable = isinstance(node, ast.FunctionDef) and node.name in _MUTABLE_RUNNER_FUNCTIONS
        added_import = (candidate and isinstance(node, ast.Import)
                        and ast.dump(node, include_attributes=False)
                        == "Import(names=[alias(name='importlib.util')])")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        if mutable or added_import:
            first_line = min([node.lineno] + [decorator.lineno
                             for decorator in getattr(node, "decorator_list", [])])
            ignored_lines.update(range(first_line, node.end_lineno + 1))
            added_imports += int(added_import)
        else:
            frozen_nodes.append(ast.dump(node, include_attributes=False))
    _require(len(names) == len(set(names)), "Duplicate runner definitions are forbidden")
    if candidate:
        _require(added_imports == 1, "Only the required importlib.util addition is allowed")
        _require(_MUTABLE_RUNNER_FUNCTIONS.issubset(names), "Missing lifecycle functions")
    else:
        _require(not _NEW_RUNNER_FUNCTIONS.intersection(names), "Unexpected parent lifecycle helpers")
    frozen_lines = [line for number, line in enumerate(text.splitlines(), 1)
                    if number not in ignored_lines and line.strip()]
    return frozen_nodes, frozen_lines


def verify_tree(repo: str | Path, tree: str, parent: str | None = None) -> dict:
    """Check a staged/local tree against the reviewed parent; NOT authorization."""
    repo = Path(repo)
    parent = REVIEWED_PARENT if parent is None else parent
    parent_tree, _ = _commit_headers(repo, parent)
    before, after = _entries(repo, parent_tree), _entries(repo, tree)
    changed = {path for path in before.keys() | after.keys()
               if before.get(path) != after.get(path)}
    ancestors = {str(Path(path).parent).replace("\\", "/") for path in ALLOWED_PATHS}
    ancestors |= {".github"}
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
             "Changed path set must equal the eight reviewed paths exactly")
    _require(set(PAYLOAD_SHA256) == ALLOWED_PATHS - {GATE_PATH},
             "Reviewed payload hash table must seal exactly seven files")
    for path, digest in PAYLOAD_SHA256.items():
        _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
                 f"Invalid reviewed SHA256: {path}")
        blob = _git(repo, "cat-file", "blob", after[path][2])
        _require(hashlib.sha256(blob).hexdigest() == digest,
                 f"Reviewed payload SHA256 mismatch: {path}")
    _require(_frozen_runner(_git(repo, "cat-file", "blob", before[RUNNER_PATH][2]),
                            candidate=False)
             == _frozen_runner(_git(repo, "cat-file", "blob", after[RUNNER_PATH][2]),
                               candidate=True),
             "Runner changed outside lifecycle helpers/execute_row/run_bundle/importlib.util")
    return {"invariant": INVARIANT, "tree": tree, "parent": parent}


def verify_commit(repo: str | Path, sha: str, parent: str | None = None) -> dict:
    """Require the reviewed single nonmerge parent and verify its entire tree."""
    repo = Path(repo)
    parent = REVIEWED_PARENT if parent is None else parent
    tree, parents = _commit_headers(repo, sha)
    _require(parents == [parent], "Candidate must have exactly the reviewed single parent")
    result = verify_tree(repo, tree, parent=parent)
    return {**result, "commit": sha}


def verify_execution(repo: str | Path, environ=None) -> dict:
    """Production first-push checks, actual checkout identity, then Git invariants."""
    environment = os.environ if environ is None else environ
    _require(environment.get("GITHUB_EVENT_NAME") == "push", "Only push events are permitted")
    _require(environment.get("GITHUB_REF") == REVIEWED_REF, "Wrong literal diagnostic ref")
    _require(environment.get("GITHUB_RUN_ATTEMPT") == "1", "Reruns are forbidden")
    sha = _object_id(environment.get("GITHUB_SHA", ""))
    try:
        event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise ReviewError("A readable GitHub push event JSON is required") from exc
    _require(isinstance(event, dict), "Push event must be an object")
    _require(event.get("ref") == REVIEWED_REF, "Push payload ref mismatch")
    _require(event.get("created") is True, "Only a newly created branch is permitted")
    _require(event.get("before") == ZERO_SHA, "Push before must be the all-zero object ID")
    _require(event.get("deleted") is False, "Branch deletion is forbidden")
    _require(event.get("forced") is False, "Forced pushes are forbidden")
    _require(event.get("after") == sha, "Push after must equal GITHUB_SHA")
    actual = _git(Path(repo), "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    _require(actual == sha, "Actual checkout HEAD does not equal GITHUB_SHA")
    return verify_commit(repo, sha)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = verify_execution(args.repo)
    except (ReviewError, UnicodeError) as exc:
        print(f"{INVARIANT} rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**result, "push_authorization": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
