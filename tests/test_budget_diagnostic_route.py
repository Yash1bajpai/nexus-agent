"""Synthetic local Git objects only: no remote, workflow, model, or provider runs."""
from __future__ import annotations
import os as _route_os
_route_os.environ.setdefault("NEXUS_NATIVE_BRANCH", "qualification/q1-windows-native-v12-20261105-2335-8af3d120")
_route_os.environ.setdefault("NEXUS_REALMODEL_BRANCH", "qualification/q1-windows-realmodel-v12-20261105-2335-8af3d120")

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / ".github/diagnostics/verify_reviewed_candidate.py"
BASE_RUNNER = '''"""Frozen diagnostic fixture."""
from __future__ import annotations
import os
MODEL_REPO = "reviewed/model"
MODEL_SHA256 = "reviewed-model-hash"
PROMPT = "frozen synthetic prompt"

def frozen_argv(row):
    return ["chat", PROMPT, row]

def installed_provenance(wheel):
    return {"wheel": str(wheel), "source_tree_import_rejected": True}

def _drain(stream):
    return None

def execute_row(scope, row):
    return {"old": row}

def run_bundle(root):
    return {"old": str(root)}

def main():
    return 0
'''
CANDIDATE_RUNNER = (BASE_RUNNER.replace("import os\n", "import os\nimport importlib.util\n")
                    .replace('return {"old": row}', 'return {"owned": row}')
                    .replace('return {"old": str(root)}', 'return {"owned": str(root)}')
                    + '''
def _windows_job_factory():
    return "owned factory"

def _cleanup_posix_process(process):
    return {"confirmed": True}
''')


def load_gate():
    spec = importlib.util.spec_from_file_location("reviewed_candidate_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalRepository:
    def __init__(self, path):
        self.path = path
        self.path.mkdir()
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("GIT_")}
        self.environment.update({
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Synthetic review", "GIT_AUTHOR_EMAIL": "review@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic review", "GIT_COMMITTER_EMAIL": "review@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
            "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1",
        })
        self.git("init", "--quiet")

    def git(self, *args, data=None):
        return subprocess.run(
            ["git", "-C", str(self.path), "-c", "core.autocrlf=false",
             "-c", "core.hooksPath=" + os.devnull, *args],
            input=data, env=self.environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True,
        ).stdout

    def write(self, path, content):
        target = self.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)

    def stage_tree(self):
        self.git("add", "--all")
        return self.git("write-tree").decode().strip()

    def commit_tree(self, tree, *parents, message="synthetic candidate"):
        args = ["commit-tree", tree]
        for parent in parents:
            args.extend(["-p", parent])
        return self.git(*args, data=(message + "\n").encode()).decode().strip()

    def head(self, sha):
        self.git("update-ref", "HEAD", sha)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    gate = load_gate()
    repo = LocalRepository(tmp_path / "repository")
    base = {
        ".github/workflows/budget-diagnostic.yml": "original workflow\n",
        gate.RUNNER_PATH: BASE_RUNNER,
        "tests/test_budget_diagnostic_runner.py": "original runner tests\n",
        "README.md": "original documentation\n",
        "src/package.py": "runtime unchanged\n",
        ".github/e2e/driver.py": "authoritative driver unchanged\n",
        ".github/workflows/authoritative.yml": "existing workflow unchanged\n",
    }
    for path, content in base.items():
        repo.write(path, content)
    parent = repo.commit_tree(repo.stage_tree(), message="reviewed parent")
    repo.head(parent)
    payload = {path: f"reviewed payload for {path}\n" for path in gate.ALLOWED_PATHS}
    payload[gate.RUNNER_PATH] = CANDIDATE_RUNNER
    for path, content in payload.items():
        repo.write(path, content)
    tree = repo.stage_tree()
    sha = repo.commit_tree(tree, parent)
    repo.head(sha)
    monkeypatch.setattr(gate, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", {
        path: hashlib.sha256(content.encode()).hexdigest()
        for path, content in payload.items() if path != gate.GATE_PATH
    })
    event = {"ref": gate.REVIEWED_REF, "created": True, "before": gate.ZERO_SHA,
             "deleted": False, "forced": False, "after": sha}
    event_path = tmp_path / "push-event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    environment = {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": gate.REVIEWED_REF,
                   "GITHUB_SHA": sha, "GITHUB_RUN_ATTEMPT": "1",
                   "GITHUB_EVENT_PATH": str(event_path)}
    return SimpleNamespace(gate=gate, repo=repo, parent=parent, tree=tree, sha=sha,
                           base=base, payload=payload, event=event,
                           event_path=event_path, environment=environment)


def event_update(candidate, **changes):
    candidate.event.update(changes)
    candidate.event_path.write_text(json.dumps(candidate.event), encoding="utf-8")


def reseal_runner(candidate, source):
    candidate.repo.write(candidate.gate.RUNNER_PATH, source)
    candidate.gate.PAYLOAD_SHA256[candidate.gate.RUNNER_PATH] = hashlib.sha256(source.encode()).hexdigest()
    return candidate.repo.stage_tree()


def test_reviewed_tree_commit_and_first_push_pass_without_remote(candidate, monkeypatch):
    commands = []
    run = subprocess.run

    def local_only(command, **kwargs):
        commands.append(command)
        assert not {"fetch", "push", "pull", "clone", "ls-remote"}.intersection(command)
        return run(command, **kwargs)

    monkeypatch.setattr(candidate.gate.subprocess, "run", local_only)
    result = candidate.gate.verify_execution(candidate.repo.path, candidate.environment)
    assert result == {"invariant": "R11_ROUTE_LIFECYCLE_V1", "parent": candidate.parent,
                      "tree": candidate.tree, "commit": candidate.sha}
    assert commands
    assert candidate.gate.verify_commit(candidate.repo.path, candidate.sha) == result


def test_staged_tree_verifier_needs_neither_candidate_head_nor_event(candidate):
    candidate.repo.head(candidate.parent)
    assert candidate.gate.verify_tree(candidate.repo.path, candidate.tree) == {
        "invariant": "R11_ROUTE_LIFECYCLE_V1", "parent": candidate.parent,
        "tree": candidate.tree,
    }
    with pytest.raises(candidate.gate.ReviewError, match="checkout HEAD"):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("key,value", [
    ("GITHUB_EVENT_NAME", "workflow_dispatch"), ("GITHUB_EVENT_NAME", "pull_request"),
    ("GITHUB_EVENT_NAME", "schedule"), ("GITHUB_EVENT_NAME", ""),
    ("GITHUB_REF", "refs/heads/main"), ("GITHUB_REF", "refs/heads/diagnostic/other"),
    ("GITHUB_REF", "refs/heads/diagnostic/r11-budget-590f748-2d91c6a7/suffix"),
    ("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_RUN_ATTEMPT", "0"),
    ("GITHUB_RUN_ATTEMPT", "01"), ("GITHUB_RUN_ATTEMPT", ""),
    ("GITHUB_SHA", "A" * 40), ("GITHUB_SHA", "a" * 39),
    ("GITHUB_SHA", "a" * 41), ("GITHUB_SHA", "HEAD"),
])
def test_wrong_execution_context_rejected(candidate, key, value):
    candidate.environment[key] = value
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("key,value", [
    ("ref", "refs/heads/diagnostic/other"), ("created", False), ("created", 1),
    ("created", "true"), ("before", "1" * 40), ("before", "0" * 39),
    ("before", None), ("deleted", True), ("deleted", 0),
    ("forced", True), ("forced", 0), ("after", "1" * 40), ("after", None),
])
def test_wrong_push_payload_rejected(candidate, key, value):
    event_update(candidate, **{key: value})
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("key", ["ref", "created", "before", "deleted", "forced", "after"])
def test_missing_push_fields_rejected(candidate, key):
    del candidate.event[key]
    event_update(candidate)
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("content", ["not JSON", "[]", "null"])
def test_invalid_event_file_rejected(candidate, content):
    candidate.event_path.write_text(content, encoding="utf-8")
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


def test_missing_event_file_rejected(candidate):
    candidate.environment["GITHUB_EVENT_PATH"] += ".missing"
    with pytest.raises(candidate.gate.ReviewError, match="event JSON"):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("kind", ["wrong", "merge", "root", "followup"])
def test_wrong_or_additional_parent_rejected(candidate, kind):
    other = candidate.repo.commit_tree(candidate.tree, candidate.parent, message="not reviewed")
    parents = {"wrong": [other], "merge": [candidate.parent, other],
               "root": [], "followup": [candidate.sha]}[kind]
    sha = candidate.repo.commit_tree(candidate.tree, *parents, message=kind)
    with pytest.raises(candidate.gate.ReviewError, match="single parent"):
        candidate.gate.verify_commit(candidate.repo.path, sha)


def test_followup_cannot_be_disguised_as_new_branch_event(candidate):
    followup = candidate.repo.commit_tree(candidate.tree, candidate.sha, message="followup")
    candidate.repo.head(followup)
    candidate.environment["GITHUB_SHA"] = followup
    event_update(candidate, after=followup)
    with pytest.raises(candidate.gate.ReviewError, match="single parent"):
        candidate.gate.verify_execution(candidate.repo.path, candidate.environment)


@pytest.mark.parametrize("path", ["src/package.py", "outside.txt", ".github/e2e/driver.py",
                                  ".github/workflows/authoritative.yml"])
def test_any_change_outside_amendment_rejected(candidate, path):
    candidate.repo.write(path, "unreviewed change\n")
    with pytest.raises(candidate.gate.ReviewError, match="outside reviewed paths"):
        candidate.gate.verify_tree(candidate.repo.path, candidate.repo.stage_tree())


def test_all_eight_paths_must_actually_change(candidate):
    candidate.repo.write("README.md", candidate.base["README.md"])
    with pytest.raises(candidate.gate.ReviewError, match="eight reviewed paths"):
        candidate.gate.verify_tree(candidate.repo.path, candidate.repo.stage_tree())


@pytest.mark.parametrize("path", ["README.md", ".github/workflows/budget-diagnostic.yml",
                                  "tests/test_budget_diagnostic_route.py"])
def test_payload_hash_tampering_rejected(candidate, path):
    candidate.repo.write(path, "payload tampered after review\n")
    with pytest.raises(candidate.gate.ReviewError, match="SHA256 mismatch"):
        candidate.gate.verify_tree(candidate.repo.path, candidate.repo.stage_tree())


@pytest.mark.parametrize("mode", ["100755", "120000", "160000"])
def test_nonregular_blob_modes_rejected(candidate, mode):
    path = "README.md"
    oid = (candidate.parent if mode == "160000" else
           candidate.repo.git("rev-parse", candidate.tree + ":" + path).decode().strip())
    candidate.repo.git("update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}")
    tree = candidate.repo.git("write-tree").decode().strip()
    with pytest.raises(candidate.gate.ReviewError, match="regular 100644"):
        candidate.gate.verify_tree(candidate.repo.path, tree)


@pytest.mark.parametrize("rename", [False, True])
def test_deletes_and_renames_rejected(candidate, rename):
    oid = candidate.repo.git("rev-parse", candidate.tree + ":README.md").decode().strip()
    candidate.repo.git("update-index", "--force-remove", "README.md")
    if rename:
        candidate.repo.git("update-index", "--add", "--cacheinfo", f"100644,{oid},renamed.md")
    tree = candidate.repo.git("write-tree").decode().strip()
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_tree(candidate.repo.path, tree)


def test_empty_tree_injection_is_not_invisible(candidate):
    empty_tree = candidate.repo.git("mktree", data=b"").decode().strip()
    entries = candidate.repo.git("ls-tree", candidate.tree)
    entries += f"040000 tree {empty_tree}\tunreviewed-empty-directory\n".encode()
    tree = candidate.repo.git("mktree", data=entries).decode().strip()
    with pytest.raises(candidate.gate.ReviewError, match="outside reviewed paths"):
        candidate.gate.verify_tree(candidate.repo.path, tree)


@pytest.mark.parametrize("kind", ["commit", "blob", "missing", "symbolic"])
def test_tree_api_rejects_non_tree_objects(candidate, kind):
    value = {"commit": candidate.sha,
             "blob": candidate.repo.git("rev-parse", candidate.tree + ":README.md").decode().strip(),
             "missing": "1" * 40, "symbolic": "HEAD^{tree}"}[kind]
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_tree(candidate.repo.path, value)


@pytest.mark.parametrize("kind", ["missing", "extra", "malformed"])
def test_payload_seal_must_be_complete_and_literal(candidate, kind):
    if kind == "missing":
        candidate.gate.PAYLOAD_SHA256.pop("README.md")
    elif kind == "extra":
        candidate.gate.PAYLOAD_SHA256[candidate.gate.GATE_PATH] = "1" * 64
    else:
        candidate.gate.PAYLOAD_SHA256["README.md"] = "not a digest"
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_tree(candidate.repo.path, candidate.tree)


@pytest.mark.parametrize("old,new", [
    ('MODEL_REPO = "reviewed/model"', 'MODEL_REPO = "unreviewed/model"'),
    ('MODEL_SHA256 = "reviewed-model-hash"', 'MODEL_SHA256 = "changed"'),
    ('PROMPT = "frozen synthetic prompt"', 'PROMPT = "different prompt"'),
    ('return ["chat", PROMPT, row]', 'return ["chat", "extra request", row]'),
    ('"source_tree_import_rejected": True', '"source_tree_import_rejected": False'),
    ('def _drain(stream):\n    return None', 'def _drain(stream):\n    return "changed"'),
    ('import os\n', 'import os\nimport math\n'),
    ('import os\n', 'import os\nprint("outside lifecycle")\n'),
    ('import os\n', 'import os\n# unrelated source edit\n'),
    ('MODEL_REPO = "reviewed/model"', "MODEL_REPO = 'reviewed/model'"),
    ('def main():', 'class Unexpected:\n    pass\n\ndef main():'),
])
def test_frozen_runner_scope_rejected_even_if_payload_resealed(candidate, old, new):
    assert old in CANDIDATE_RUNNER
    tree = reseal_runner(candidate, CANDIDATE_RUNNER.replace(old, new))
    with pytest.raises(candidate.gate.ReviewError, match="outside lifecycle"):
        candidate.gate.verify_tree(candidate.repo.path, tree)


@pytest.mark.parametrize("suffix", [
    '\ndef execute_row(scope, row):\n    return "duplicate"\n',
    '\nimport importlib.util\n',
])
def test_duplicate_lifecycle_definition_or_import_rejected(candidate, suffix):
    tree = reseal_runner(candidate, CANDIDATE_RUNNER + suffix)
    with pytest.raises(candidate.gate.ReviewError):
        candidate.gate.verify_tree(candidate.repo.path, tree)


def test_lifecycle_bodies_are_the_only_mutable_runner_sections(candidate):
    source = CANDIDATE_RUNNER.replace('return {"owned": row}',
                                     'return {"owned": row, "cleanup_confirmed": True}')
    tree = reseal_runner(candidate, source)
    assert candidate.gate.verify_tree(candidate.repo.path, tree)["tree"] == tree


def test_gate_itself_requires_external_final_tree_approval_not_self_hash(candidate):
    # This is the deliberate trust boundary, not an authorization bypass: a
    # reviewer must bind the gate bytes with the final commit/tree externally.
    assert candidate.gate.GATE_PATH not in candidate.gate.PAYLOAD_SHA256
    candidate.repo.write(candidate.gate.GATE_PATH, "different gate requires new independent review\n")
    tree = candidate.repo.stage_tree()
    assert candidate.gate.verify_tree(candidate.repo.path, tree)["tree"] == tree


def test_cli_reports_invariant_success_but_never_push_authorization(candidate, monkeypatch, capsys):
    for key, value in candidate.environment.items():
        monkeypatch.setenv(key, value)
    assert candidate.gate.main(["--repo", str(candidate.repo.path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["push_authorization"] is False
    assert result["commit"] == candidate.sha
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    assert candidate.gate.main(["--repo", str(candidate.repo.path)]) == 1
    assert "Reruns are forbidden" in capsys.readouterr().err


def test_git_subprocess_receives_only_safe_path_and_fixed_local_controls(tmp_path, monkeypatch):
    gate = load_gate()
    secrets = {
        "AWS_SECRET_ACCESS_KEY": "aws-secret", "OPENAI_API_KEY": "provider-secret",
        "GH_TOKEN": "gh-secret", "GITHUB_TOKEN": "github-secret",
        "HTTPS_PROXY": "https://user:password@example.invalid",
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": "Authorization: secret", "GIT_ASKPASS": "credential-helper",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    untrusted = tmp_path / "ambient-untrusted"
    monkeypatch.setenv("PATH", str(untrusted))
    executable = str(tmp_path / "trusted-git" / "git")
    monkeypatch.setattr(gate.shutil, "which", lambda name: executable)
    captured = {}

    def capture(command, **kwargs):
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=b"local-object\n", stderr=b"")

    monkeypatch.setattr(gate.subprocess, "run", capture)
    assert gate._git(tmp_path, "cat-file", "-t", "1" * 40) == b"local-object\n"
    environment = captured["env"]
    assert not set(secrets).intersection(environment)
    assert set(environment) <= {
        "PATH", "SYSTEMROOT", "WINDIR", "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT", "GIT_NO_LAZY_FETCH",
    }
    assert str(untrusted) not in environment["PATH"].split(os.pathsep)
    assert str(Path(executable).parent) in environment["PATH"].split(os.pathsep)
    assert all(Path(part).is_absolute() for part in environment["PATH"].split(os.pathsep))
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert captured["timeout"] == 10
    assert captured["command"][:4] == [executable, "--no-replace-objects", "-C", str(tmp_path)]


@pytest.mark.parametrize("failure,message", [
    ("timeout", "Local Git lookup timed out"),
    ("oserror", "Cannot read local Git repository"),
    ("nonzero", "Local Git object/checkout lookup failed"),
])
def test_git_failures_are_bounded_and_do_not_echo_secrets(tmp_path, monkeypatch, failure, message):
    gate = load_gate()
    monkeypatch.setattr(gate.shutil, "which", lambda name: str(tmp_path / "git"))

    def fail(command, **kwargs):
        assert kwargs["timeout"] == 10
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10, output=b"secret stdout", stderr=b"secret stderr")
        if failure == "oserror":
            raise OSError("secret OS error")
        return subprocess.CompletedProcess(command, 1, stdout=b"secret stdout", stderr=b"secret stderr")

    monkeypatch.setattr(gate.subprocess, "run", fail)
    with pytest.raises(gate.ReviewError) as caught:
        gate._git(tmp_path, "cat-file", "-t", "1" * 40)
    assert str(caught.value) == message
    assert caught.value.__suppress_context__ or failure == "nonzero"
