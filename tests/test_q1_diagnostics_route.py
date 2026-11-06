"""Local Git/event/artifact boundary tests; no native, network or model run."""
from __future__ import annotations

import os as _route_os
_route_os.environ.setdefault("NEXUS_NATIVE_BRANCH", "qualification/q1-windows-native-v12-20261105-2335-8af3d120")
_route_os.environ.setdefault("NEXUS_REALMODEL_BRANCH", "qualification/q1-windows-realmodel-v12-20261105-2335-8af3d120")
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_q1_route import LocalRepository

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / ".github/qualification/verify_windows_diagnostics.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("q1_diagnostic_gate_under_test", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    gate = load_gate()
    repo = LocalRepository(tmp_path / "repository")
    # Same nineteen-path policy as production; only synthetic ancestor identities
    # and content pins differ. Unlisted files must stay exactly frozen.
    repo.write("src/unchanged.py", "frozen source\n")
    repo.write("tests/fixtures/q1/frozen_contract.json", '{"frozen":true}\n')
    for name in gate.ALLOWED_PATHS:
        if name not in (gate.GATE_PATH, ".github/workflows/q1-windows-diagnostics.yml", "tests/test_q1_diagnostics_route.py"):
            repo.write(name, "old payload\n")
    base_tree = repo.stage_tree()
    base = repo.commit_tree(base_tree, message="synthetic grounded base")
    parent = repo.commit_tree(base_tree, base, message="synthetic declared parent")
    repo.head(parent)
    payload = {}
    for name in gate.ALLOWED_PATHS:
        src_file = ROOT / name
        if src_file.is_file():
            payload[name] = src_file.read_bytes()
        else:
            payload[name] = f"# synthetic reviewed payload for {name}\n".encode("utf-8")
        repo.write(name, payload[name])
    tree = repo.stage_tree()
    sha = repo.commit_tree(tree, parent)
    repo.head(sha)
    monkeypatch.setattr(gate, "BASE_COMMIT", base)
    monkeypatch.setattr(gate, "BASE_TREE", base_tree)
    monkeypatch.setattr(gate, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(gate, "PARENT_TREE", base_tree)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", {
        name: hashlib.sha256(blob).hexdigest() for name, blob in payload.items() if name != gate.GATE_PATH})
    return SimpleNamespace(gate=gate, repo=repo, base=base, tree=tree, parent=parent, sha=sha, payload=payload)


def push_event(gate, sha):
    return {"ref": gate.REVIEWED_REF, "created": True, "before": gate.ZERO_SHA,
            "deleted": False, "forced": False, "after": sha}


def test_exact_diagnostic_candidate_is_not_execution_or_qualification_approval(candidate):
    c = candidate
    result = c.gate.verify_commit(c.repo.path, c.sha)
    assert set(result["exact_delta_paths"]) == c.gate.ALLOWED_PATHS
    assert result["parent"] == c.parent and result["base"] == c.base
    assert result["diagnostic_only"] is True
    assert result["final_qualification"] is False
    assert result["requires_untraced_native"] is True
    assert result["external_execution_approval"] is False
    assert result["verifier_sha256"] == hashlib.sha256(c.payload[c.gate.GATE_PATH]).hexdigest()


@pytest.mark.parametrize("change", ["extra", "protected", "missing", "wrong_pin", "executable", "symlink"])
def test_candidate_refuses_scope_content_or_mode_change(candidate, change):
    c = candidate
    selected = "tests/test_action_outcomes.py"
    if change == "extra":
        c.repo.write("unreviewed.py", "unexpected\n")
    elif change == "protected":
        c.repo.write("tests/fixtures/q1/frozen_contract.json", "changed\n")
    elif change == "missing":
        (c.repo.path / selected).unlink()
    elif change == "wrong_pin":
        c.repo.write(selected, "unreviewed replacement\n")
    tree = c.repo.stage_tree()
    if change in ("executable", "symlink"):
        oid = c.repo.git("rev-parse", f"{tree}:{selected}").decode().strip()
        c.repo.git("update-index", "--cacheinfo", "100755" if change == "executable" else "120000", oid, selected)
        tree = c.repo.git("write-tree").decode().strip()
    with pytest.raises(c.gate.ReviewError):
        c.gate.verify_tree(c.repo.path, tree)


def test_payload_table_cannot_omit_a_reviewed_file(candidate, monkeypatch):
    c = candidate
    pins = dict(c.gate.PAYLOAD_SHA256)
    pins.pop("tests/test_action_outcomes.py")
    monkeypatch.setattr(c.gate, "PAYLOAD_SHA256", pins)
    with pytest.raises(c.gate.ReviewError, match="nineteen"):
        c.gate.verify_tree(c.repo.path, c.tree)


@pytest.mark.parametrize("parents", ["wrong", "merge"])
def test_candidate_rejects_wrong_or_multiple_parents(candidate, parents):
    c = candidate
    sha = c.repo.commit_tree(c.tree, *([c.base] if parents == "wrong" else [c.parent, c.base]))
    with pytest.raises(c.gate.ReviewError, match="single parent"):
        c.gate.verify_commit(c.repo.path, sha)


def test_declared_parent_must_really_descend_base(candidate, monkeypatch):
    c = candidate
    unrelated = c.repo.commit_tree(c.tree, message="unrelated base with no parent")
    monkeypatch.setattr(c.gate, "BASE_COMMIT", unrelated)
    monkeypatch.setattr(c.gate, "BASE_TREE", c.tree)
    with pytest.raises(c.gate.ReviewError):
        c.gate.verify_tree(c.repo.path, c.tree)


@pytest.mark.parametrize("field,value", [
    ("ref", "refs/heads/qualification/q1-f016eb8-8c73d412"),
    ("created", False), ("created", 1), ("before", "f" * 40),
    ("deleted", True), ("forced", True), ("after", "2" * 40),
])
def test_creation_event_rejects_reuse_force_delete_or_other_sha(field, value):
    gate = load_gate()
    event = push_event(gate, "1" * 40)
    event[field] = value
    with pytest.raises(gate.ReviewError):
        gate.verify_event(event, event_name="push", ref=gate.REVIEWED_REF, sha="1" * 40, run_attempt="1")


@pytest.mark.parametrize("name,ref,attempt", [
    ("workflow_dispatch", None, "1"), ("schedule", None, "1"),
    ("push", "refs/heads/main", "1"), ("push", None, "2"),
])
def test_event_context_rejects_manual_schedule_other_ref_or_rerun(name, ref, attempt):
    gate = load_gate()
    with pytest.raises(gate.ReviewError):
        gate.verify_event(push_event(gate, "1" * 40), event_name=name,
                          ref=gate.REVIEWED_REF if ref is None else ref, sha="1" * 40, run_attempt=attempt)


def test_checkout_requires_event_unless_explicit_local_and_local_is_forbidden_in_actions(candidate, monkeypatch):
    c = candidate
    for name in ("GITHUB_EVENT_PATH", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(c.gate.ReviewError, match="event"):
        c.gate.verify_checkout(c.repo.path)
    assert c.gate.verify_checkout(c.repo.path, local=True)["commit"] == c.sha
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with pytest.raises(c.gate.ReviewError, match="forbidden"):
        c.gate.verify_checkout(c.repo.path, local=True)


def test_checkout_checks_actual_head_and_clean_files(candidate, monkeypatch, tmp_path):
    c = candidate
    path = tmp_path / "event.json"
    path.write_text(json.dumps(push_event(c.gate, c.sha)), encoding="utf-8")
    for key, value in {"GITHUB_EVENT_PATH": str(path), "GITHUB_EVENT_NAME": "push",
                       "GITHUB_REF": c.gate.REVIEWED_REF, "GITHUB_SHA": c.sha,
                       "GITHUB_RUN_ATTEMPT": "1"}.items():
        monkeypatch.setenv(key, value)
    assert c.gate.verify_checkout(c.repo.path)["commit"] == c.sha
    c.repo.write("src/unchanged.py", "dirty worktree\n")
    with pytest.raises(c.gate.ReviewError, match="dirty"):
        c.gate.verify_checkout(c.repo.path)
    c.repo.head(c.parent)
    with pytest.raises(c.gate.ReviewError, match="HEAD"):
        c.gate.verify_checkout(c.repo.path)


def test_bounded_collection_preserves_raw_bytes_and_denies_qualification(tmp_path):
    gate = load_gate()
    source, target = tmp_path / "output", tmp_path / "upload"
    source.mkdir()
    raw = b"raw\r\n\xff\x00"
    (source / "native.stdout").write_bytes(raw)
    report = gate.collect_output(source, target)
    assert report["complete"] is True
    assert report["final_qualification"] is False and report["model_execution_permitted"] is False
    assert (target / "native.stdout").read_bytes() == raw
    assert report["files"] == [{"path": "native.stdout", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
    assert json.loads((target / "diagnostic-manifest.json").read_bytes()) == report


@pytest.mark.parametrize("kind", ["file", "total", "count", "link", "missing"])
def test_collection_caps_and_unavailable_output_are_explicit_not_truncated_success(tmp_path, monkeypatch, kind):
    gate = load_gate()
    source, target = tmp_path / "output", tmp_path / "upload"
    source.mkdir()
    if kind == "file":
        monkeypatch.setattr(gate, "FILE_CAP", 3)
        (source / "too-large").write_bytes(b"1234")
    elif kind == "total":
        monkeypatch.setattr(gate, "TOTAL_CAP", gate.MANIFEST_RESERVE + 3)
        (source / "too-large").write_bytes(b"1234")
    elif kind == "count":
        monkeypatch.setattr(gate, "FILE_COUNT_CAP", 0)
        (source / "first").write_bytes(b"1")
    elif kind == "link":
        outside = tmp_path / "not-output"
        outside.write_bytes(b"must not publish")
        try:
            (source / "alias").symlink_to(outside)
        except OSError:
            pytest.skip("synthetic symlink unavailable")
    else:
        source.rmdir()
    report = gate.collect_output(source, target)
    assert report["complete"] is False and report["errors"]
    assert sorted(path.name for path in target.iterdir()) == ["diagnostic-manifest.json"]
    assert sum(path.stat().st_size for path in target.iterdir()) <= gate.TOTAL_CAP


def test_collection_skips_offending_file_and_continues_valid_files(tmp_path, monkeypatch):
    gate = load_gate()
    source, target = tmp_path / "output", tmp_path / "upload"
    source.mkdir()
    (source / "good1.txt").write_bytes(b"valid content 1")
    monkeypatch.setattr(gate, "FILE_CAP", 20)
    (source / "bad_oversize.txt").write_bytes(b"very long content exceeding file cap")
    (source / "good2.txt").write_bytes(b"valid content 2")
    report = gate.collect_output(source, target)
    assert report["complete"] is False
    assert len(report["errors"]) == 1
    assert "bad_oversize.txt" in report["errors"][0]
    copied_names = {f["path"] for f in report["files"]}
    assert copied_names == {"good1.txt", "good2.txt"}
    assert (target / "good1.txt").read_bytes() == b"valid content 1"
    assert (target / "good2.txt").read_bytes() == b"valid content 2"
    assert not (target / "bad_oversize.txt").exists()


def test_collection_refuses_nested_or_existing_upload_directory(tmp_path):
    gate = load_gate()
    source = tmp_path / "output"
    source.mkdir()
    with pytest.raises(gate.ReviewError, match="overlaps"):
        gate.collect_output(source, source / "upload")
    with pytest.raises(gate.ReviewError, match="fresh"):
        gate.collect_output(source, source)
