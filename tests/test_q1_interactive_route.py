"""Local B candidate/A-prerequisite/collector tests. Never launch a model or network."""
from __future__ import annotations

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


def get_payload_blob(root: Path, name: str) -> bytes:
    """Read verified candidate payload from staged neutral data channel or local root fallback."""
    staged = root / "q1-candidate-payload" / name
    if staged.is_file():
        return staged.read_bytes()
    local = root / name
    if local.is_file():
        return local.read_bytes()
    raise FileNotFoundError(f"Required reviewed candidate payload missing: {name}")


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    gate = gate_module()
    repo = LocalRepository(tmp_path / "repository")

    # Base setup: baseline files present before candidate
    repo.write("src/preserved.py", "frozen production baseline\n")
    for name in gate.MODIFIED_PATHS:
        repo.write(name, "baseline unmodified content\n")
    parent_tree = repo.stage_tree()
    parent = repo.commit_tree(parent_tree, message="synthetic declared parent")
    repo.head(parent)

    # Candidate files: exact 18 paths (5 added, 13 modified)
    payload = {}
    for name in gate.ALLOWED_PATHS:
        blob = get_payload_blob(ROOT, name)
        payload[name] = blob
        repo.write(name, blob)
    tree = repo.stage_tree()
    sha = repo.commit_tree(tree, parent)
    repo.head(sha)

    monkeypatch.setattr(gate, "PARENT", parent)
    monkeypatch.setattr(gate, "PARENT_TREE", parent_tree)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", {
        n: hashlib.sha256(v).hexdigest() for n, v in payload.items()
        if n not in gate.VERIFIER_PATHS
    })
    return SimpleNamespace(gate=gate, repo=repo, sha=sha, parent=parent, tree=tree, payload=payload)


def event(gate, sha):
    return {"ref": gate.REVIEWED_REF, "created": True, "before": "0" * 40,
            "deleted": False, "forced": False, "after": sha}


def test_exact_candidate_and_clean_local_checkout(candidate, monkeypatch):
    c = candidate
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    result = c.gate.verify_checkout(c.repo.path, local=True, approved_commit=c.sha, approved_tree=c.tree)
    assert result["commit"] == c.sha and result["parent"] == c.parent
    assert len(result["exact_delta_paths"]) == 18
    assert result["qualified"] is False and result["external_execution_approval"] is False
    c.repo.write("src/preserved.py", "changed\n")
    with pytest.raises(c.gate.ReviewError, match="Dirty"):
        c.gate.verify_checkout(c.repo.path, local=True, approved_commit=c.sha, approved_tree=c.tree)


@pytest.mark.parametrize("bad_commit,bad_tree", [
    (None, None),
    ("0" * 40, "0" * 40),
    ("", ""),
    ("invalid_commit_hash", "invalid_tree_hash"),
])
def test_missing_or_zero_external_approved_seal_rejected_fail_closed(candidate, bad_commit, bad_tree):
    c = candidate
    with pytest.raises(c.gate.ReviewError, match="Mandatory external approved"):
        c.gate.verify_commit(c.repo.path, c.sha, approved_commit=bad_commit, approved_tree=bad_tree)


@pytest.mark.parametrize("change", ["production", "extra", "missing", "payload", "merge"])
def test_candidate_rejects_any_unsealed_delta(candidate, change):
    c = candidate
    if change == "production":
        c.repo.write("src/preserved.py", "modified\n")
    elif change == "extra":
        c.repo.write("unreviewed_extra.py", "unreviewed extra file\n")
    elif change == "missing":
        (c.repo.path / "tests/test_q1_interactive_route.py").unlink()
    elif change == "payload":
        c.repo.write(".github/qualification/interactive_core.py", "unreviewed\n")
    tree = c.repo.stage_tree()
    sha = c.repo.commit_tree(tree, c.parent, *([c.sha] if change == "merge" else []))
    with pytest.raises(c.gate.ReviewError):
        c.gate.verify_commit(c.repo.path, sha, approved_commit=sha, approved_tree=tree)


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
        g.verify_checkout(candidate.repo.path, local=True, approved_commit=candidate.sha, approved_tree=candidate.tree)


def actual_run(commit_sha):
    return {"id": 123, "head_sha": commit_sha, "head_branch": "qualification/q1-windows-native-e95f-20261105-0837-2f6a9c81",
            "path": ".github/workflows/q1-windows-native.yml", "event": "push",
            "run_attempt": 1, "status": "completed", "conclusion": "success",
            "repository": {"full_name": "owner/repo"}}


def test_select_native_run_requires_mandatory_approved_commit_and_tree():
    g = gate_module()
    commit = "a" * 40
    tree = "b" * 40
    run = actual_run(commit)
    # Zero or missing approved commit must fail closed
    with pytest.raises(g.ReviewError, match="Mandatory external approved"):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit=None, approved_tree=tree)
    with pytest.raises(g.ReviewError, match="Mandatory external approved"):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit="0" * 40, approved_tree=tree)
    # Zero or missing approved tree must fail closed
    with pytest.raises(g.ReviewError, match="Mandatory external approved"):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit=commit, approved_tree=None)
    with pytest.raises(g.ReviewError, match="Mandatory external approved"):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit=commit, approved_tree="0" * 40)

    # Valid matching run succeeds
    selected = g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit=commit, approved_tree=tree)
    assert selected["run_id"] == 123 and selected["commit"] == commit


@pytest.mark.parametrize("field,value", [("head_sha", "b" * 40), ("run_attempt", 2),
                                        ("conclusion", "failure"), ("path", ".github/workflows/q1-windows-diagnostics.yml")])
def test_only_actual_successful_untraced_A_run_selected(field, value):
    g = gate_module()
    commit = "a" * 40
    tree = "b" * 40
    run = actual_run(commit)
    run[field] = value
    with pytest.raises(g.ReviewError):
        g.select_native_run({"workflow_runs": [run]}, "owner/repo", approved_commit=commit, approved_tree=tree)


def test_missing_or_ambiguous_native_run_not_waived():
    g = gate_module()
    commit = "a" * 40
    tree = "b" * 40
    run = actual_run(commit)
    for runs in [[], [run, run]]:
        with pytest.raises(g.ReviewError):
            g.select_native_run({"workflow_runs": runs}, "owner/repo", approved_commit=commit, approved_tree=tree)


def native_gate(commit, tree, wheel_sha):
    return {
        "commit": commit,
        "platform": "win32",
        "success": True,
        "qualified": True,
        "untraced_native": True,
        "native_functional_only": True,
        "final_qualification": False,
        "model_execution_permitted": False,
        "model_fetch_permitted_by_gate": False,
        "wheel_sha256": wheel_sha,
        "provenance": {
            "commit": commit,
            "tree": tree,
            "package_py": {"nexus_agent_ai/__init__.py": "c" * 64},
        },
    }


def test_check_native_evidence_verifies_commit_tree_and_wheel(tmp_path):
    g = gate_module()
    gate_file = tmp_path / "native-gate.json"
    wheel_file = tmp_path / "nexus_agent_ai-2.10.4-py3-none-any.whl"
    wheel_file.write_bytes(b"synthetic wheel bytes")
    wheel_sha = hashlib.sha256(wheel_file.read_bytes()).hexdigest()

    commit = "a" * 40
    tree = "b" * 40
    gate_payload = native_gate(commit, tree, wheel_sha)
    gate_file.write_text(json.dumps(gate_payload))

    # Matching commit, tree, and wheel succeeds
    verified = g.check_native_evidence(gate_file, commit=commit, tree=tree, wheel=wheel_file)
    assert verified["commit"] == commit

    # Commit mismatch fails
    with pytest.raises(g.ReviewError, match="commit mismatch"):
        g.check_native_evidence(gate_file, commit="1" * 40, tree=tree, wheel=wheel_file)

    # Tree mismatch fails
    with pytest.raises(g.ReviewError, match="tree mismatch"):
        g.check_native_evidence(gate_file, commit=commit, tree="2" * 40, wheel=wheel_file)

    # Wheel mismatch fails
    bad_wheel = tmp_path / "bad.whl"
    bad_wheel.write_bytes(b"tampered")
    with pytest.raises(g.ReviewError, match="Wheel digest mismatch"):
        g.check_native_evidence(gate_file, commit=commit, tree=tree, wheel=bad_wheel)


@pytest.mark.parametrize("field,value", [("success", False), ("qualified", False),
                                        ("model_execution_permitted", True),
                                        ("final_qualification", True),
                                        ("platform", "linux")])
def test_diagnostic_or_failed_native_evidence_cannot_unlock_B(tmp_path, field, value):
    g = gate_module()
    gate_file = tmp_path / "native-gate.json"
    wheel_file = tmp_path / "test.whl"
    wheel_file.write_bytes(b"wheel")
    commit, tree = "a" * 40, "b" * 40
    gate = native_gate(commit, tree, hashlib.sha256(b"wheel").hexdigest())
    gate[field] = value
    gate_file.write_text(json.dumps(gate))
    with pytest.raises(g.ReviewError):
        g.check_native_evidence(gate_file, commit=commit, tree=tree, wheel=wheel_file)


def test_collect_preserves_raw_output_and_retains_missing_output_failure(tmp_path):
    g = gate_module()
    source = tmp_path / "output"
    source.mkdir()
    (source / "run.log").write_bytes(b"output log")
    upload = tmp_path / "upload"
    report = g.collect_output(source, upload)
    assert report["complete"] is True
    assert (upload / "run.log").read_bytes() == b"output log"
    assert (upload / "diagnostic-manifest.json").exists()


def test_collect_output_enforces_actual_aggregate_48_mib_cap(tmp_path, monkeypatch):
    g = gate_module()
    src = tmp_path / "src"
    dst = tmp_path / "upload"
    src.mkdir()

    # Create 5 files that individually fit under FILE_CAP, but whose sum exceeds aggregate cap
    # We monkeypatch TOTAL_CAP and FILE_CAP to test the exact aggregate threshold cleanly
    monkeypatch.setattr(g, "FILE_CAP", 1000)
    monkeypatch.setattr(g, "TOTAL_CAP", g.MANIFEST_RESERVE + 2500)  # total allowance = 2500 bytes for files

    # 4 files of 800 bytes = 3200 bytes total (exceeds 2500 bytes allowance, while each 800 <= 1000)
    for i in range(4):
        (src / f"file_{i}.bin").write_bytes(b"x" * 800)

    report = g.collect_output(src, dst)
    assert report["complete"] is False
    assert any("exceeds aggregate" in err for err in report["errors"])
    # Ensure total bytes in dst never exceeds TOTAL_CAP
    assert sum(p.stat().st_size for p in dst.iterdir()) <= g.TOTAL_CAP
