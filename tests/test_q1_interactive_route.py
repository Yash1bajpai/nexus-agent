"""Local B candidate/A-prerequisite/collector tests. Never launch a model or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_q1_route import LocalRepository

ROOT = Path(__file__).resolve().parents[1]


def gate_module():
    spec = importlib.util.spec_from_file_location("b_gate_under_test", ROOT / ".github/qualification/verify_windows_interactive.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    gate = gate_module()
    repo = LocalRepository(tmp_path / "repository")
    repo.write("src/preserved.py", "frozen production\n")
    for name in gate.ALLOWED_PATHS:
        repo.write(name, "historical reviewed B bytes\n")
    parent_tree = repo.stage_tree()
    parent = repo.commit_tree(parent_tree)
    repo.head(parent)
    payload = {}
    for name in gate.ALLOWED_PATHS:
        payload[name] = (ROOT / name).read_bytes()
        repo.write(name, payload[name])
    tree = repo.stage_tree()
    sha = repo.commit_tree(tree, parent)
    repo.head(sha)
    monkeypatch.setattr(gate, "PARENT", parent)
    monkeypatch.setattr(gate, "PARENT_TREE", parent_tree)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", {n: hashlib.sha256(v).hexdigest() for n, v in payload.items() if n != gate.GATE_PATH})
    return SimpleNamespace(gate=gate, repo=repo, sha=sha, parent=parent, tree=tree)


def event(gate, sha):
    return {"ref": gate.REVIEWED_REF, "created": True, "before": "0" * 40,
            "deleted": False, "forced": False, "after": sha}


def test_exact_candidate_and_clean_local_checkout(candidate, monkeypatch):
    c = candidate
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    result = c.gate.verify_checkout(c.repo.path, local=True)
    assert result["commit"] == c.sha and result["parent"] == c.parent
    assert result["qualified"] is False and result["external_execution_approval"] is False
    c.repo.write("src/preserved.py", "changed\n")
    with pytest.raises(c.gate.ReviewError, match="Dirty"):
        c.gate.verify_checkout(c.repo.path, local=True)


@pytest.mark.parametrize("change", ["production", "extra", "missing", "payload", "merge"])
def test_candidate_rejects_any_unsealed_delta(candidate, change):
    c = candidate
    if change == "production":
        c.repo.write("src/preserved.py", "modified\n")
    elif change == "extra":
        c.repo.write(".github/workflows/q1-windows-native.yml", "must not contaminate A\n")
    elif change == "missing":
        (c.repo.path / "tests/test_q1_interactive_route.py").unlink()
    elif change == "payload":
        c.repo.write(".github/qualification/interactive_core.py", "unreviewed\n")
    tree = c.repo.stage_tree()
    sha = c.repo.commit_tree(tree, c.parent, *([c.sha] if change == "merge" else []))
    with pytest.raises(c.gate.ReviewError):
        c.gate.verify_commit(c.repo.path, sha)


@pytest.mark.parametrize("field,value", [("created", False), ("created", 1), ("before", "1" * 40),
                                        ("forced", True), ("deleted", True), ("after", "2" * 40)])
def test_ref_creation_event_cannot_be_reused(field, value):
    g = gate_module()
    e = event(g, "1" * 40)
    e[field] = value
    with pytest.raises(g.ReviewError):
        g.verify_event(e, event_name="push", ref=g.REVIEWED_REF, sha="1" * 40, run_attempt="1")


def test_attempt_dispatch_and_local_actions_refused(candidate, monkeypatch):
    g = candidate.gate
    for name, attempt in [("workflow_dispatch", "1"), ("push", "2")]:
        with pytest.raises(g.ReviewError):
            g.verify_event(event(g, candidate.sha), event_name=name, ref=g.REVIEWED_REF,
                           sha=candidate.sha, run_attempt=attempt)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(g.ReviewError, match="forbidden"):
        g.verify_checkout(candidate.repo.path, local=True)


def actual_run(g):
    return {"id": 123, "head_sha": g.A_COMMIT, "head_branch": g.A_BRANCH,
            "path": ".github/workflows/q1-windows-native.yml", "event": "push",
            "run_attempt": 1, "status": "completed", "conclusion": "success",
            "repository": {"full_name": "owner/repo"}}


@pytest.mark.parametrize("field,value", [("head_sha", "1" * 40), ("run_attempt", 2),
                                        ("conclusion", "failure"), ("path", ".github/workflows/q1-windows-diagnostics.yml")])
def test_only_actual_successful_untraced_A_run_selected(field, value):
    g = gate_module()
    run = actual_run(g)
    assert g.select_native_run({"workflow_runs": [run]}, "owner/repo")["run_id"] == 123
    run[field] = value
    with pytest.raises(g.ReviewError):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo")


def test_missing_or_ambiguous_native_run_not_waived():
    g = gate_module()
    for runs in [[], [actual_run(g), actual_run(g)]]:
        with pytest.raises(g.ReviewError):
            g.select_native_run({"workflow_runs": runs}, "owner/repo")


def native_gate(g):
    return {"commit": g.A_COMMIT, "platform": "win32", "success": True, "qualified": True,
            "untraced_native": True, "native_functional_only": True, "final_qualification": False,
            "model_execution_permitted": False, "model_fetch_permitted_by_gate": False,
            "provenance": {"commit": g.A_COMMIT, "tree": g.A_TREE, "package_py": {"__init__.py": "a" * 64}}}


@pytest.mark.parametrize("field,value", [("untraced_native", False), ("qualified", False),
                                        ("commit", "ceee"), ("model_fetch_permitted_by_gate", True),
                                        ("platform", "linux")])
def test_diagnostic_or_failed_native_evidence_cannot_unlock_B(tmp_path, field, value):
    g = gate_module()
    path = tmp_path / "native-gate.json"
    gate = native_gate(g)
    path.write_text(json.dumps(gate))
    assert g.check_native_evidence(path)["commit"] == g.A_COMMIT
    gate[field] = value
    path.write_text(json.dumps(gate))
    with pytest.raises(g.ReviewError):
        g.check_native_evidence(path)


def test_collect_preserves_raw_output_and_retains_missing_output_failure(tmp_path):
    g = gate_module()
    source = tmp_path / "output"
    source.mkdir()
    raw = b"raw\r\n\xff\x00"
    (source / "terminal.txt").write_bytes(raw)
    report = g.collect_output(source, tmp_path / "upload")
    assert report["complete"] is True and report["final_qualification"] is False
    assert (tmp_path / "upload/terminal.txt").read_bytes() == raw
    missing = g.collect_output(tmp_path / "missing", tmp_path / "missing-upload")
    assert missing["complete"] is False and missing["errors"]
    assert (tmp_path / "missing-upload/diagnostic-manifest.json").is_file()
