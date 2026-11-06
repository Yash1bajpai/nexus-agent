"""Synthetic local Git objects only: no remote, workflow, model, or provider runs."""
from __future__ import annotations

import os as _route_os
_route_os.environ.setdefault("NEXUS_NATIVE_BRANCH", "qualification/q1-windows-native-v12-20261105-2335-8af3d120")
_route_os.environ.setdefault("NEXUS_REALMODEL_BRANCH", "qualification/q1-windows-realmodel-v12-20261105-2335-8af3d120")
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / ".github/qualification/verify_candidate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("q1_verify_candidate_under_test", GATE_PATH)
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
            "GIT_AUTHOR_NAME": "Synthetic Q1", "GIT_AUTHOR_EMAIL": "q1@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic Q1", "GIT_COMMITTER_EMAIL": "q1@example.invalid",
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
    contract_file = ROOT / "tests/fixtures/q1/frozen_contract.json"
    contract_data = json.loads(contract_file.read_text(encoding="utf-8"))

    repo = LocalRepository(tmp_path / "repository")
    base_driver = contract_data["base_driver_source"]
    base = {
        ".github/e2e/driver.py": base_driver,
        "src/runtime.py": "frozen runtime content\n",
        "pyproject.toml": "frozen pyproject\n",
    }
    for path, content in base.items():
        repo.write(path, content)
    parent_tree = repo.stage_tree()
    parent = repo.commit_tree(parent_tree, message="reviewed parent D1")
    repo.head(parent)

    candidate_driver = (ROOT / ".github/e2e/driver.py").read_text(encoding="utf-8")
    payload = {
        ".github/e2e/driver.py": candidate_driver,
        ".github/qualification/runner.py": "# runner stub\n",
        ".github/qualification/windows_process.py": "# process stub\n",
        ".github/qualification/report.py": "# report stub\n",
        ".github/qualification/verify_candidate.py": GATE_PATH.read_text(encoding="utf-8"),
        ".github/workflows/q1-windows-qualification.yml": (
            "name: Q1\non:\n  push:\n    branches:\n      - qualification/q1-f016eb8-8c73d412\n"
        ),
        "tests/test_q1_reporting.py": "# test stub\n",
        "tests/test_q1_transport.py": "# test stub\n",
        "tests/test_q1_windows_process.py": "# test stub\n",
        "tests/test_q1_route.py": "# test stub\n",
        "tests/test_q1_installed_gate.py": "# test stub\n",
        "tests/fixtures/q1/frozen_contract.json": json.dumps({
            **contract_data,
            "base_commit": parent,
            "base_tree": parent_tree,
            "base_driver_blob": repo.git("rev-parse", f"{parent}:.github/e2e/driver.py").decode().strip(),
        }),
    }
    for path, content in payload.items():
        repo.write(path, content)
    tree = repo.stage_tree()
    sha = repo.commit_tree(tree, parent)
    repo.head(sha)

    payload_hashes = {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in payload.items() if path != gate.GATE_PATH
    }
    monkeypatch.setattr(gate, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(gate, "BASE_TREE", parent_tree)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", payload_hashes)

    event = {
        "ref": gate.REVIEWED_REF,
        "created": True,
        "before": gate.ZERO_SHA,
        "deleted": False,
        "forced": False,
        "after": sha,
    }
    event_path = tmp_path / "push-event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    environment = {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": gate.REVIEWED_REF,
        "GITHUB_SHA": sha,
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_EVENT_PATH": str(event_path),
    }

    return SimpleNamespace(
        gate=gate, repo=repo, parent=parent, parent_tree=parent_tree,
        tree=tree, sha=sha, payload=payload, payload_hashes=payload_hashes,
        event=event, event_path=event_path, environment=environment,
    )


def test_valid_candidate_and_event_pass(candidate):
    result = candidate.gate.verify_candidate(
        candidate.repo.path,
        event=candidate.event,
        event_name="push",
        ref=candidate.gate.REVIEWED_REF,
        sha=candidate.sha,
        run_attempt="1",
        payload_hashes=candidate.payload_hashes,
    )
    assert result["invariant"] == candidate.gate.INVARIANT
    assert result["tree"] == candidate.tree
    assert result["commit"] == candidate.sha
    assert result["parent"] == candidate.parent


def test_candidate_from_environment(candidate, monkeypatch):
    for key, value in candidate.environment.items():
        monkeypatch.setenv(key, value)
    result = candidate.gate.verify_candidate(
        candidate.repo.path,
        payload_hashes=candidate.payload_hashes,
    )
    assert result["commit"] == candidate.sha


def test_wrong_parent_rejected(candidate):
    wrong_parent = candidate.repo.commit_tree(candidate.parent_tree, message="wrong parent")
    wrong_tree = candidate.repo.stage_tree()
    wrong_sha = candidate.repo.commit_tree(wrong_tree, wrong_parent)
    candidate.repo.head(wrong_sha)
    candidate.event["after"] = wrong_sha
    with pytest.raises(candidate.gate.ReviewError, match="single parent"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=wrong_sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


@pytest.mark.parametrize("path", [
    "src/nexus_agent_ai/agent/core.py",
    "pyproject.toml",
    "unauthorized_new_file.py",
    ".github/workflows/ci.yml",
])
def test_unauthorized_path_modification_rejected(candidate, path):
    candidate.repo.write(path, "unauthorized modification\n")
    new_tree = candidate.repo.stage_tree()
    new_sha = candidate.repo.commit_tree(new_tree, candidate.parent)
    candidate.repo.head(new_sha)
    candidate.event["after"] = new_sha
    with pytest.raises(candidate.gate.ReviewError, match="outside reviewed paths|Changed path set"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=new_sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


def test_tampered_payload_hash_rejected(candidate):
    candidate.repo.write("tests/test_q1_reporting.py", "# tampered test content\n")
    new_tree = candidate.repo.stage_tree()
    new_sha = candidate.repo.commit_tree(new_tree, candidate.parent)
    candidate.repo.head(new_sha)
    candidate.event["after"] = new_sha
    with pytest.raises(candidate.gate.ReviewError, match="payload SHA256 mismatch"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=new_sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


@pytest.mark.parametrize("key, bad_value, match", [
    ("ref", "refs/heads/main", "ref mismatch"),
    ("created", False, "newly created branch"),
    ("before", "1" * 40, "all-zero"),
    ("deleted", True, "deletion is forbidden"),
    ("forced", True, "Forced pushes are forbidden"),
    ("after", "0" * 40, "after must equal commit SHA"),
])
def test_invalid_event_attributes_rejected(candidate, key, bad_value, match):
    candidate.event[key] = bad_value
    with pytest.raises(candidate.gate.ReviewError, match=match):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=candidate.sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


@pytest.mark.parametrize("bad_name", ["pull_request", "workflow_dispatch", "schedule"])
def test_invalid_event_name_rejected(candidate, bad_name):
    with pytest.raises(candidate.gate.ReviewError, match="Event name must be push"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name=bad_name,
            ref=candidate.gate.REVIEWED_REF,
            sha=candidate.sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


def test_run_attempt_must_be_one(candidate):
    with pytest.raises(candidate.gate.ReviewError, match="Run attempt must be 1"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=candidate.sha,
            run_attempt="2",
            payload_hashes=candidate.payload_hashes,
        )


def test_head_mismatch_rejected(candidate):
    other_sha = candidate.repo.commit_tree(candidate.parent_tree, candidate.parent, message="other")
    candidate.repo.head(other_sha)
    with pytest.raises(candidate.gate.ReviewError, match="does not equal commit SHA"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=candidate.sha,
            run_attempt="1",
            payload_hashes=candidate.payload_hashes,
        )


def test_workflow_dispatch_rejected(candidate):
    bad_wf = (
        "name: Q1\non:\n  workflow_dispatch:\n  push:\n    branches:\n      - qualification/q1-f016eb8-8c73d412\n"
    )
    candidate.repo.write(".github/workflows/q1-windows-qualification.yml", bad_wf)
    hashes = dict(candidate.payload_hashes)
    hashes[".github/workflows/q1-windows-qualification.yml"] = hashlib.sha256(bad_wf.encode()).hexdigest()
    new_tree = candidate.repo.stage_tree()
    new_sha = candidate.repo.commit_tree(new_tree, candidate.parent)
    candidate.repo.head(new_sha)
    candidate.event["after"] = new_sha
    with pytest.raises(candidate.gate.ReviewError, match="must not permit workflow_dispatch"):
        candidate.gate.verify_candidate(
            candidate.repo.path,
            event=candidate.event,
            event_name="push",
            ref=candidate.gate.REVIEWED_REF,
            sha=new_sha,
            run_attempt="1",
            payload_hashes=hashes,
        )
