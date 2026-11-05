"""Local Git/event/artifact boundary tests for untraced Windows native functional route."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.test_q1_route import LocalRepository

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / ".github/qualification/verify_windows_native.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("q1_native_gate_under_test", GATE)
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
    gate = load_gate()
    repo = LocalRepository(tmp_path / "repository")

    # Base setup: baseline files present before candidate
    repo.write("src/preserved.py", "frozen production baseline\n")
    for name in gate.MODIFIED_PATHS:
        repo.write(name, "baseline unmodified content\n")
    base_tree = repo.stage_tree()
    base = repo.commit_tree(base_tree, message="synthetic grounded base")
    parent = repo.commit_tree(base_tree, base, message="synthetic declared parent")
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

    monkeypatch.setattr(gate, "BASE_COMMIT", base)
    monkeypatch.setattr(gate, "REVIEWED_PARENT", parent)
    monkeypatch.setattr(gate, "PARENT_TREE", base_tree)
    monkeypatch.setattr(gate, "PAYLOAD_SHA256", {
        name: hashlib.sha256(blob).hexdigest() for name, blob in payload.items()
        if name not in gate.VERIFIER_PATHS
    })
    return SimpleNamespace(gate=gate, repo=repo, base=base, tree=tree, parent=parent, sha=sha, payload=payload)


def push_event(gate, sha):
    return {
        "ref": gate.REVIEWED_REF,
        "created": True,
        "before": gate.ZERO_SHA,
        "deleted": False,
        "forced": False,
        "after": sha,
    }


def test_exact_native_candidate_is_not_execution_or_qualification_approval(candidate):
    c = candidate
    result = c.gate.verify_commit(c.repo.path, c.sha, approved_commit=c.sha, approved_tree=c.tree)
    assert set(result["exact_delta_paths"]) == c.gate.ALLOWED_PATHS
    assert len(result["exact_delta_paths"]) == 18
    assert result["parent"] == c.parent and result["base"] == c.base
    assert result["native_functional_only"] is True
    assert result["untraced_native"] is True
    assert result["diagnostic_only"] is False
    assert result["final_qualification"] is False
    assert result["model_execution_permitted"] is False
    assert result["external_execution_approval"] is False
    assert result["native_verifier_sha256"] == hashlib.sha256(c.payload[c.gate.GATE_PATH]).hexdigest()


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


def test_github_sha_mismatch_against_approved_seal_rejected(candidate, monkeypatch, tmp_path):
    c = candidate
    monkeypatch.setenv("GITHUB_SHA", "1" * 40)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REF", c.gate.REVIEWED_REF)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(push_event(c.gate, c.sha)), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    with pytest.raises(c.gate.ReviewError, match="differs from approved commit seal"):
        c.gate.verify_checkout(c.repo.path, local=False, approved_commit=c.sha, approved_tree=c.tree)


@pytest.mark.parametrize("change", ["extra", "existing_modified", "missing", "wrong_pin", "executable", "symlink"])
def test_candidate_refuses_scope_content_or_mode_change(candidate, change):
    c = candidate
    selected = "tests/test_q1_native_route.py"
    if change == "extra":
        c.repo.write("unreviewed.py", "unexpected\n")
    elif change == "existing_modified":
        c.repo.write("src/preserved.py", "modified source\n")
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


def test_payload_table_must_pin_exact_sixteen_files(candidate, monkeypatch):
    c = candidate
    pins = dict(c.gate.PAYLOAD_SHA256)
    pins.pop("tests/test_q1_native_route.py")
    monkeypatch.setattr(c.gate, "PAYLOAD_SHA256", pins)
    with pytest.raises(c.gate.ReviewError, match="sixteen"):
        c.gate.verify_tree(c.repo.path, c.tree)


@pytest.mark.parametrize("parents", ["wrong", "merge"])
def test_candidate_rejects_wrong_or_multiple_parents(candidate, parents):
    c = candidate
    sha = c.repo.commit_tree(c.tree, *([c.base] if parents == "wrong" else [c.parent, c.base]))
    with pytest.raises(c.gate.ReviewError, match="single parent"):
        c.gate.verify_commit(c.repo.path, sha, approved_commit=sha, approved_tree=c.tree)


def test_declared_parent_must_really_descend_base(candidate, monkeypatch):
    c = candidate
    independent = c.repo.commit_tree(c.gate.PARENT_TREE, message="unrelated")
    monkeypatch.setattr(c.gate, "REVIEWED_PARENT", independent)
    with pytest.raises(c.gate.ReviewError, match="grounded base"):
        c.gate.verify_tree(c.repo.path, c.tree)


def test_event_requires_exact_creation_predicates(candidate):
    c = candidate
    ev = push_event(c.gate, c.sha)
    assert c.gate.verify_event(ev, event_name="push", ref=c.gate.REVIEWED_REF, sha=c.sha, run_attempt=1) == c.sha

    for attempt in (2, "2"):
        with pytest.raises(c.gate.ReviewError, match="attempt"):
            c.gate.verify_event(ev, event_name="push", ref=c.gate.REVIEWED_REF, sha=c.sha, run_attempt=attempt)

    bad_ev = dict(ev, created=False)
    with pytest.raises(c.gate.ReviewError, match="creation"):
        c.gate.verify_event(bad_ev, event_name="push", ref=c.gate.REVIEWED_REF, sha=c.sha, run_attempt=1)

    bad_ref = dict(ev, ref="refs/heads/wrong")
    with pytest.raises(c.gate.ReviewError, match="ref"):
        c.gate.verify_event(bad_ref, event_name="push", ref=c.gate.REVIEWED_REF, sha=c.sha, run_attempt=1)


def test_bounded_collection_preserves_raw_bytes_and_denies_qualification(tmp_path):
    gate = load_gate()
    source, target = tmp_path / "output", tmp_path / "upload"
    source.mkdir()
    raw = b"raw\r\n\xff\x00"
    (source / "native.stdout").write_bytes(raw)
    report = gate.collect_output(source, target)
    assert report["complete"] is True
    assert report["native_functional_only"] is True
    assert report["untraced_native"] is True
    assert report["diagnostic_only"] is False
    assert report["final_qualification"] is False
    assert report["model_execution_permitted"] is False
    assert (target / "native.stdout").read_bytes() == raw
    assert report["files"] == [{"path": "native.stdout", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
    assert json.loads((target / "native-manifest.json").read_bytes()) == report


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
        monkeypatch.setattr(gate, "FILE_COUNT_CAP", 1)
        (source / "first").write_bytes(b"1")
        (source / "second").write_bytes(b"2")
    elif kind == "link":
        target_file = source / "target"
        target_file.write_bytes(b"hello")
        try:
            (source / "symlink").symlink_to(target_file)
        except OSError:
            pytest.skip("Symlinks not supported in environment")
    elif kind == "missing":
        source.rmdir()
    report = gate.collect_output(source, target)
    assert report["complete"] is False
    assert len(report["errors"]) > 0
    assert (target / "native-manifest.json").is_file()
    assert sum(path.stat().st_size for path in target.iterdir()) <= gate.TOTAL_CAP


def test_native_staging_produces_isolated_suite_without_src(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "q1_native_functional_under_test",
        ROOT / ".github/qualification/native_functional.py",
    )
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)

    repo = LocalRepository(tmp_path / "source")
    for name in nf.NATIVE_HELPERS:
        repo.write(name, (ROOT / name).read_bytes())
    for name in ("tests/test_q1_native_route.py", "tests/fixtures/q1_search_recorded_1712.json"):
        repo.write(name, (ROOT / name).read_bytes())
    # Candidate payloads for staging
    candidate_gate = load_gate()
    for name in candidate_gate.ALLOWED_PATHS:
        repo.write(name, get_payload_blob(ROOT, name))
    repo.write("src/unstaged.py", "must not be imported or staged\n")
    repo.stage_tree()
    dest = tmp_path / "reviewed-tests"
    staged = nf.stage_tests(repo.path, dest, {"critical_native": []})

    assert not (dest / "src").exists()
    assert (dest / "isolated-pytest.ini").is_file()
    assert (dest / "q1-candidate-payload").is_dir()
    for expected in (
        ".github/qualification/native_functional.py",
        ".github/qualification/verify_windows_native.py",
        ".github/workflows/q1-windows-native.yml",
        "tests/test_q1_native_route.py",
    ):
        assert expected in staged["files"]
        assert (dest / expected).is_file()
        assert not (dest / expected).is_symlink()


def test_pytest_phase_rejects_active_output_env_before_invocation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "q1_native_functional_under_test",
        ROOT / ".github/qualification/native_functional.py",
    )
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)

    monkeypatch.setenv("NEXUS_Q1_OUTPUT_DIR", str(tmp_path / "leak"))
    args = SimpleNamespace(
        root=str(tmp_path),
        wheel="dummy.whl",
        source_root=str(ROOT),
        commit="0" * 40,
    )
    with pytest.raises(nf.GateError, match="NEXUS_Q1_OUTPUT_DIR must not be set"):
        nf.pytest_phase(args)


def test_native_functional_rejects_model_and_workload_args():
    spec = importlib.util.spec_from_file_location(
        "q1_native_functional_under_test",
        ROOT / ".github/qualification/native_functional.py",
    )
    nf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nf)

    with pytest.raises(SystemExit) as exc_info:
        nf.main([
            "native-run",
            "--source-root", ".",
            "--root", ".",
            "--wheel", "w.whl",
            "--commit", "0" * 40,
            "--deadline-file", "d.json",
            "--model-source", "m.gguf",
        ])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        nf.main([
            "workload",
            "--source-root", ".",
            "--root", ".",
            "--wheel", "w.whl",
            "--commit", "0" * 40,
            "--deadline-file", "d.json",
        ])
    assert exc_info.value.code == 2


def test_count_identity_diagnostics_inactive_without_output_dir(monkeypatch):
    monkeypatch.delenv("NEXUS_Q1_OUTPUT_DIR", raising=False)
    spec = importlib.util.spec_from_file_location(
        "count_identity_diag",
        ROOT / "tests/_count_identity_diagnostics.py",
    )
    cid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cid)

    import subprocess
    original_popen = subprocess.Popen
    trace_before = sys.gettrace()
    gen = cid.count_identity_diagnostics.__wrapped__(None, monkeypatch)
    next(gen)
    assert subprocess.Popen is original_popen
    assert sys.gettrace() == trace_before
    try:
        next(gen)
    except StopIteration:
        pass
