# tests/test_proof_gate_evaluate.py
"""Test suite for proof_gate_evaluate.py.

Demonstrates pass on valid fixture data (matching Real Run-3 structure) and
strict fail-closed behavior on granular negative cases for all schema rules and facts.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from proof_gate_evaluate import evaluate_prerequisites

# Real Run 3 Fixtures
REAL_RUN3_SURVIVOR_CHECK_RECORD = {
    "nonce": "probe_nonce_6aa653d560554bd2b467abe16abbb967",
    "tracked_pids": [6464, 7708, 8100],
    "historical_observation_seeds": [6464, 7708, 8100],
    "phase_a": {
        "conclusive": True,
        "error": None,
        "scan": {
            "conclusive": True,
            "clean": False,
            "confirmed": [
                {"pid": 6464, "state": "confirmed", "exe": "PING.EXE"},
                {"pid": 7708, "state": "confirmed", "exe": "conhost.exe"},
                {"pid": 8100, "state": "confirmed", "exe": "cmd.exe"},
            ],
            "reused": [],
            "unknown": [],
            "exited": [],
            "survivors": [
                {"pid": 6464, "state": "confirmed", "exe": "PING.EXE"},
                {"pid": 7708, "state": "confirmed", "exe": "conhost.exe"},
                {"pid": 8100, "state": "confirmed", "exe": "cmd.exe"},
            ],
        },
        "observed_snapshot_clean": False,
    },
    "phase_b": {
        "conclusive": True,
        "error": None,
        "scan": {
            "conclusive": True,
            "clean": False,
            "confirmed": [
                {"pid": 6464, "state": "confirmed", "exe": "PING.EXE"},
                {"pid": 7708, "state": "confirmed", "exe": "conhost.exe"},
                {"pid": 8100, "state": "confirmed", "exe": "cmd.exe"},
            ],
            "reused": [],
            "unknown": [],
            "exited": [],
            "survivors": [
                {"pid": 6464, "state": "confirmed", "exe": "PING.EXE"},
                {"pid": 7708, "state": "confirmed", "exe": "conhost.exe"},
                {"pid": 8100, "state": "confirmed", "exe": "cmd.exe"},
            ],
        },
        "observed_snapshot_clean": False,
        "observed_phase_b_no_survivors": False,
    },
    "cleanup_attempted": True,
    "cleanup_kills": [
        {"pid": 6464, "state": "confirmed", "opened": True, "terminated": True, "gle": 0, "status": "terminated"},
        {"pid": 7708, "state": "confirmed", "opened": True, "terminated": True, "gle": 0, "status": "terminated"},
        {"pid": 8100, "state": "confirmed", "opened": True, "terminated": True, "gle": 0, "status": "terminated"},
    ],
    "cleanup_unowned_refused": [],
    "phase_c": {
        "conclusive": True,
        "error": None,
        "scan": {
            "conclusive": True,
            "clean": True,
            "confirmed": [],
            "reused": [],
            "unknown": [],
            "exited": [
                {"pid": 6464, "state": "exited", "exe": "PING.EXE"},
                {"pid": 7708, "state": "exited", "exe": "conhost.exe"},
                {"pid": 8100, "state": "exited", "exe": "cmd.exe"},
            ],
            "survivors": [],
        },
        "observed_snapshot_clean": True,
    },
    "observed_phase_b_no_survivors": False,
    "observed_snapshot_clean": {"phase_a": False, "phase_b": False, "phase_c": True},
    "coverage": "known_lineage_observed",
    "coverage_phases": [],
    "full_tree_containment_proven": False,
    "clean": False,
}

REAL_RUN3_TIMEOUT_LOG = (
    "START pid=3336 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
    "SPAWNED_HAZARD_CHILD pid=8100 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
    "SPAWNED_HAZARD_TREE pids=8100,6464,7708 child_pid=8100 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 attempts=2 members=[] ts=2026-11-08T04:27:37Z\n"
    "BEAT nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 seq=0 ts=2026-11-08T04:27:37Z elapsed=0.00\n"
    "BEAT nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 seq=1 ts=2026-11-08T04:27:42Z elapsed=5.00\n"
    "BEAT nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 seq=2 ts=2026-11-08T04:27:47Z elapsed=10.00\n"
)

REAL_RUN3_VERIFICATION_JSON = {
    "nonce": "probe_nonce_6aa653d560554bd2b467abe16abbb967",
    "stuck_step_outcome": "failure",
    "timeout_seconds": 120,
    "interval_seconds": 5.0,
    "pass": True,
}

REAL_RUN3_ACCOUNT_RECEIPT = {
    "sid": "S-1-5-21-3289802278-1561746829-1498425070-1003",
    "username": "nexuse2e",
    "ts": "2026-11-08T04:27:37Z",
}

REAL_RUN3_ACCOUNT_CLEANUP = {
    "ts": "2026-11-08T04:27:38Z",
    "receipt_read": True,
    "target_sid": "S-1-5-21-3289802278-1561746829-1498425070-1003",
    "account_deleted": True,
    "post_sweep": {
        "clean": True,
        "child_pids": [],
        "indeterminate_pids": [],
    },
    "clean": True,
}


def _create_fixture_tree(
    root: Path,
    survivor_record: dict = REAL_RUN3_SURVIVOR_CHECK_RECORD,
    survivor_lines: list[str] | None = None,
    timeout_log: str = REAL_RUN3_TIMEOUT_LOG,
    verification_json: dict = REAL_RUN3_VERIFICATION_JSON,
    receipt_json: dict = REAL_RUN3_ACCOUNT_RECEIPT,
    cleanup_json: dict = REAL_RUN3_ACCOUNT_CLEANUP,
) -> None:
    t_dir = root / "timeout"
    a_dir = root / "account"
    t_dir.mkdir(parents=True, exist_ok=True)
    a_dir.mkdir(parents=True, exist_ok=True)

    if survivor_lines is not None:
        (t_dir / "survivor_check_step.log").write_text("\n".join(survivor_lines) + "\n", encoding="utf-8")
    else:
        (t_dir / "survivor_check_step.log").write_text(
            f"header line\nSURVIVOR_CHECK: {json.dumps(survivor_record)}\ntrailer line\n", encoding="utf-8"
        )

    (t_dir / "probe_timeout.log").write_text(timeout_log, encoding="utf-8")
    (t_dir / "probe_timeout_verification.json").write_text(json.dumps(verification_json), encoding="utf-8")
    (a_dir / "probe_account_receipt.json").write_text(json.dumps(receipt_json), encoding="utf-8")
    (a_dir / "probe_account_cleanup.json").write_text(json.dumps(cleanup_json), encoding="utf-8")


def test_real_run_3_pass_with_b_false(tmp_path: Path):
    """Real Run-3 PASS: B survivors were killed cleanly, observed_phase_b_no_survivors stays False."""
    _create_fixture_tree(tmp_path)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is True
    assert res["overall_pass"] is True
    assert res["observed_phase_b_no_survivors"] is False  # Preserved context
    assert len(res["errors"]) == 0
    for k, v in res["facts"].items():
        assert v["pass"] is True, f"Fact {k} should pass"


# -----------------------------------------------------------------------------
# Hold 1: observed_phase_b_no_survivors must be a boolean (accepts True or False)
# -----------------------------------------------------------------------------

def test_fact_2_missing_observed_phase_b_no_survivors_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    del bad_rec["observed_phase_b_no_survivors"]
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is False


def test_fact_2_non_boolean_observed_phase_b_no_survivors_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["observed_phase_b_no_survivors"] = "false"  # string, not bool
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is False


def test_fact_2_observed_phase_b_no_survivors_true_accepted(tmp_path: Path):
    good_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    good_rec["observed_phase_b_no_survivors"] = True
    _create_fixture_tree(tmp_path, survivor_record=good_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is True
    assert res["observed_phase_b_no_survivors"] is True


# -----------------------------------------------------------------------------
# Hold 2: coverage exact enum
# -----------------------------------------------------------------------------

def test_fact_2_coverage_invalid_enum_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["coverage"] = "full_coverage_claimed"
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is False


def test_fact_2_coverage_blind_window_accepted(tmp_path: Path):
    good_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    good_rec["coverage"] = "blind_window"
    good_rec["coverage_phases"] = ["Phase A"]
    _create_fixture_tree(tmp_path, survivor_record=good_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is True
    assert res["facts"]["2_survivor_record_schema_valid"]["coverage"] == "blind_window"


# -----------------------------------------------------------------------------
# Hold 3: coverage_phases members exact actual schema
# -----------------------------------------------------------------------------

def test_fact_2_coverage_phases_garbage_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["coverage_phases"] = ["phase_a", "garbage"]  # lowercase or unapproved names
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is False


def test_fact_2_coverage_phases_valid_members_accepted(tmp_path: Path):
    good_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    good_rec["coverage"] = "blind_window"
    good_rec["coverage_phases"] = ["Phase A", "Phase B"]
    _create_fixture_tree(tmp_path, survivor_record=good_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is True
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is True


# -----------------------------------------------------------------------------
# Hold 4: phase_c observed_snapshot_clean REQUIRED True (no legacy fallback)
# -----------------------------------------------------------------------------

def test_fact_3_phase_c_missing_observed_snapshot_clean_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    del bad_rec["phase_c"]["observed_snapshot_clean"]
    bad_rec["phase_c"]["clean"] = True  # legacy key present, but must not fall back
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["3_phases_conclusive_zero_unknown_reused"]["pass"] is False


def test_fact_3_phase_c_observed_snapshot_clean_false_fails(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["phase_c"]["observed_snapshot_clean"] = False
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["3_phases_conclusive_zero_unknown_reused"]["pass"] is False


# -----------------------------------------------------------------------------
# Hold 5: EVERY BEAT line must have non-empty nonce
# -----------------------------------------------------------------------------

def test_fact_5_beat_line_missing_nonce_fails(tmp_path: Path):
    bad_log = (
        "START pid=3336 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
        "BEAT seq=0 ts=2026-11-08T04:27:37Z elapsed=0.00\n"  # missing nonce=
    )
    _create_fixture_tree(tmp_path, timeout_log=bad_log)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["5_nonce_chain_verified"]["pass"] is False


def test_fact_5_beat_line_empty_nonce_fails(tmp_path: Path):
    bad_log = (
        "START pid=3336 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
        "BEAT nonce= seq=0 ts=2026-11-08T04:27:37Z elapsed=0.00\n"  # empty nonce=
    )
    _create_fixture_tree(tmp_path, timeout_log=bad_log)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["5_nonce_chain_verified"]["pass"] is False


# -----------------------------------------------------------------------------
# Hold 6: Require EXACTLY one START line
# -----------------------------------------------------------------------------

def test_fact_5_duplicate_start_line_fails(tmp_path: Path):
    bad_log = (
        "START pid=3336 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
        "START pid=3337 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:38Z\n"
        "BEAT nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 seq=0 ts=2026-11-08T04:27:37Z elapsed=0.00\n"
    )
    _create_fixture_tree(tmp_path, timeout_log=bad_log)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["5_nonce_chain_verified"]["pass"] is False
    assert any("Expected exactly 1 START line" in err for err in res["facts"]["5_nonce_chain_verified"]["errors"])


# -----------------------------------------------------------------------------
# Additional granular regressions
# -----------------------------------------------------------------------------

def test_fact_1_negative_missing_survivor_log(tmp_path: Path):
    _create_fixture_tree(tmp_path)
    (tmp_path / "timeout" / "survivor_check_step.log").unlink()
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["1_survivor_check_record_present_exact1"]["pass"] is False


def test_fact_1_negative_duplicate_survivor_lines(tmp_path: Path):
    lines = [
        f"SURVIVOR_CHECK: {json.dumps(REAL_RUN3_SURVIVOR_CHECK_RECORD)}",
        f"SURVIVOR_CHECK: {json.dumps(REAL_RUN3_SURVIVOR_CHECK_RECORD)}",
    ]
    _create_fixture_tree(tmp_path, survivor_lines=lines)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["1_survivor_check_record_present_exact1"]["pass"] is False
    assert res["facts"]["1_survivor_check_record_present_exact1"]["count"] == 2


def test_fact_2_negative_containment_proven_not_false(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["full_tree_containment_proven"] = True
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["2_survivor_record_schema_valid"]["pass"] is False


def test_fact_3_negative_phase_b_unknown_not_empty(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["phase_b"]["scan"]["unknown"] = [{"pid": 9999, "state": "unknown"}]
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["3_phases_conclusive_zero_unknown_reused"]["pass"] is False


def test_fact_3_negative_phase_c_survivors_not_empty(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["phase_c"]["scan"]["survivors"] = [{"pid": 8100, "state": "confirmed"}]
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["3_phases_conclusive_zero_unknown_reused"]["pass"] is False


def test_fact_4_negative_cleanup_unowned_refused(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["cleanup_unowned_refused"] = [{"pid": 1234, "state": "unknown"}]
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["4_cleanup_integrity_valid"]["pass"] is False


def test_fact_4_negative_kill_status_not_terminated(tmp_path: Path):
    bad_rec = copy.deepcopy(REAL_RUN3_SURVIVOR_CHECK_RECORD)
    bad_rec["cleanup_kills"] = [{"pid": 8100, "status": "terminate_failed"}]
    _create_fixture_tree(tmp_path, survivor_record=bad_rec)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["4_cleanup_integrity_valid"]["pass"] is False


def test_fact_5_negative_nonce_mismatch_in_beat(tmp_path: Path):
    bad_log = (
        "START pid=3336 nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 ts=2026-11-08T04:27:37Z\n"
        "BEAT nonce=probe_nonce_6aa653d560554bd2b467abe16abbb967 seq=0 ts=2026-11-08T04:27:37Z elapsed=0.00\n"
        "BEAT nonce=probe_nonce_TAMPERED seq=1 ts=2026-11-08T04:27:42Z elapsed=5.00\n"
    )
    _create_fixture_tree(tmp_path, timeout_log=bad_log)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["5_nonce_chain_verified"]["pass"] is False


def test_fact_6_negative_account_sid_mismatch(tmp_path: Path):
    bad_cleanup = copy.deepcopy(REAL_RUN3_ACCOUNT_CLEANUP)
    bad_cleanup["target_sid"] = "S-1-5-21-MISMATCHED-SID"
    _create_fixture_tree(tmp_path, cleanup_json=bad_cleanup)
    ok, res = evaluate_prerequisites(tmp_path)
    assert ok is False
    assert res["facts"]["6_account_lifecycle_clean"]["pass"] is False
