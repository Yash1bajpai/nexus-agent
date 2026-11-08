# proof_terminate_scope.py
"""Nexus e2e — Narrow proof entrypoint: terminate_and_confirm P1-P3 ONLY.

Scope:
- Runs Tier U (U1/U2 synthetic unit logic tests) and Tier W (W1/W2 direct contract
  tests) from contain_core_selftest.
- Does NOT execute Comp 1 items 1-6.
- Does NOT exercise spawn_in_job integration legs.
- Does NOT claim full component qualification.

Source Identity:
- Pins the accepted SHA-256 digests of contain_core.py and contain_core_selftest.py.
- Enforces an on-host hash check before creating ANY fixture or executing ANY effect.

Teardown & Cleanliness (R5, R6, C5, C7):
- Immediately upon creation, persists created SID atomically to proof_principal_receipt.json.
- Cleans up child processes by SID, deletes principal by SID, verifies post-sweep.
- Recursively removes temporary directory in finally, recording errors without masking.
- Clean exit (exit code 0) requires BOTH all items passed AND teardown verified clean.
- Primary harness errors ("error" outcome) are preserved across finally teardown.

Granularity:
- Aggregates tier results; writes incremental evidence after Tier U and Tier W.
- Mid-tier hang yields flushed heartbeat logs up to the hang point.

Status: SOURCE PREPARED, NOT RUN.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

EXPECTED_SHA256 = {
    "contain_core.py": "41ad522dbe3366d30e611a179ab176459af685fe2858ea9524e2f9d7a3cb53cd",
    "contain_core_selftest.py": "b0fb0fa27d119f90599cc1ba496d57553ed0caabcd23ab4a5686131caa3016c4",
}

OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else SELF_DIR / "proof_scope_evidence.json"
RECEIPT_PATH = SELF_DIR / "proof_principal_receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _beat(msg: str) -> None:
    print(f"[proof-scope {_utcnow()}] {msg}", flush=True)


def _emit(evidence: dict) -> None:
    OUT_PATH.write_text(json.dumps(evidence, indent=2))


def _persist_atomic_receipt(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    evidence = {
        "proof_scope": "terminate_and_confirm P1-P3 ONLY (Tier U + W1/W2 direct contract)",
        "not_qualification": True,
        "timestamp_utc": _utcnow(),
        "host_platform": sys.platform,
        "source_sha256": {},
        "items": [],
        "teardown": {},
        "all_passed": False,
        "teardown_clean": False,
        "observation": {"attempted": False, "outcome": "not_run", "ts": _utcnow()},
    }
    _emit(evidence)

    # R7 / F7: Verify source identity before ANY fixture or effect
    for name, want in EXPECTED_SHA256.items():
        target_file = SELF_DIR / name
        if not target_file.exists():
            got = "MISSING"
        else:
            got = _sha256(target_file)
        evidence["source_sha256"][name] = {"expected": want, "actual": got, "match": got == want}

    if not all(v["match"] for v in evidence["source_sha256"].values()):
        evidence["observation"] = {
            "attempted": False,
            "outcome": "aborted",
            "ts": _utcnow(),
            "detail": "source hash mismatch; aborted before fixtures or effects",
        }
        _emit(evidence)
        _beat("ABORT: source hash mismatch, no effects executed")
        return 2

    _beat("source hashes verified")
    if sys.platform != "win32":
        evidence["observation"] = {
            "attempted": False,
            "outcome": "not_run",
            "ts": _utcnow(),
            "detail": "non-Windows host",
        }
        _emit(evidence)
        return 2

    import contain_core
    import contain_core_selftest

    created_sid = None
    created_password = None
    temp_dir = tempfile.mkdtemp(prefix="nexus_proof_scope_")
    all_items_passed = False

    try:
        _beat("creating principal fixture")
        created_sid, created_password = contain_core.create_principal(contain_core.DEFAULT_PRINCIPAL_NAME)

        # C7: Atomically persist SID receipt immediately after creation, before any subsequent effect
        receipt_data = {
            "sid": created_sid,
            "username": contain_core.DEFAULT_PRINCIPAL_NAME,
            "ts": _utcnow(),
        }
        _persist_atomic_receipt(RECEIPT_PATH, receipt_data)
        evidence["teardown"]["created_sid"] = created_sid
        _beat(f"persisted atomic principal receipt to {RECEIPT_PATH.name}")

        expected_env = contain_core.build_child_env(Path(temp_dir) / "work")

        _beat("fixtures ready; running Tier U (SYNTHETIC UNIT)")
        u_items = contain_core_selftest.run_tier_u_synthetic_tests()
        evidence["items"] = list(u_items)
        evidence["observation"] = {"attempted": True, "outcome": "tier_u_complete", "ts": _utcnow()}
        _emit(evidence)
        _beat(f"Tier U complete: {sum(1 for i in u_items if i.get('pass') is True)}/{len(u_items)} passed; running Tier W")

        w_items = contain_core_selftest.run_tier_w_windows_observed_tests(
            created_sid, expected_env, created_password, temp_dir
        )
        items = list(u_items) + list(w_items)
        evidence["items"] = items
        all_items_passed = all(
            i.get("pass") is True for i in items if i.get("status") != "OMITTED"
        )
        _beat(f"Tier W complete: {sum(1 for i in w_items if i.get('pass') is True)}/{len(w_items)} passed")

    except Exception as e:
        evidence["items"].append({
            "item": "proof_scope_harness",
            "pass": False,
            "actual": {"error": f"{type(e).__name__}: {e}"},
            "observation": {
                "attempted": True,
                "outcome": "error",
                "ts": _utcnow(),
                "detail": f"{type(e).__name__}: {e}",
            },
        })
        evidence["observation"] = {
            "attempted": True,
            "outcome": "error",
            "ts": _utcnow(),
            "detail": f"{type(e).__name__}: {e}",
        }
    finally:
        # R5 / F5 & R6 / F6: Independent robust teardown recording
        teardown_errors = []

        if created_sid:
            try:
                res = contain_core.terminate_processes_by_sid(created_sid)
                evidence["teardown"]["residual_terminate"] = [
                    {"pid": p, "error": str(err)} for p, err in res
                ]
                if res:
                    teardown_errors.append(f"residual processes failed termination: {res}")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                evidence["teardown"]["residual_terminate_error"] = err_msg
                teardown_errors.append(f"terminate_processes_by_sid error: {err_msg}")

            try:
                deleted = contain_core.delete_principal(
                    captured_sid=created_sid, username=contain_core.DEFAULT_PRINCIPAL_NAME
                )
                evidence["teardown"]["principal_deleted"] = deleted
                if not deleted:
                    teardown_errors.append("principal was already absent during teardown")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                evidence["teardown"]["principal_delete_error"] = err_msg
                teardown_errors.append(f"delete_principal error: {err_msg}")

            try:
                post = contain_core.sweep_processes_by_sid(created_sid)
                evidence["teardown"]["post_sweep"] = {
                    "clean": post.clean,
                    "child_pids": post.child_pids,
                    "indeterminate_pids": post.indeterminate_pids,
                }
                if not post.clean:
                    teardown_errors.append(f"post-sweep not clean: child={post.child_pids}, indet={post.indeterminate_pids}")
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                evidence["teardown"]["post_sweep_error"] = err_msg
                teardown_errors.append(f"sweep_processes_by_sid error: {err_msg}")

        # R5: Cleanup temp_dir without raising
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                evidence["teardown"]["temp_dir_removed"] = True
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                evidence["teardown"]["temp_dir_error"] = err_msg
                teardown_errors.append(f"temp_dir rmtree error: {err_msg}")

        # R6: Teardown cleanliness required for overall pass
        teardown_clean = (
            (evidence["teardown"].get("principal_deleted") is True)
            and (evidence["teardown"].get("post_sweep", {}).get("clean") is True)
            and (len(teardown_errors) == 0)
        )
        evidence["teardown_clean"] = teardown_clean
        evidence["all_passed"] = bool(all_items_passed and teardown_clean)

        passed_count = sum(1 for i in evidence["items"] if i.get("pass") is True)
        total_count = len(evidence["items"])

        # C5: Preserve primary "error" outcome if set in except block
        if evidence["observation"].get("outcome") == "error":
            evidence["observation"]["detail"] = (
                f"{evidence['observation'].get('detail', '')}; teardown_clean={teardown_clean}"
            )
        else:
            evidence["observation"] = {
                "attempted": True,
                "outcome": "pass" if evidence["all_passed"] else "fail",
                "ts": _utcnow(),
                "detail": f"{passed_count}/{total_count} items passed; teardown_clean={teardown_clean}",
            }
        _emit(evidence)
        _beat(f"teardown recorded (clean={teardown_clean}); final evidence emitted")

    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
