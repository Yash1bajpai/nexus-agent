"""Test-only diagnostic capture for text_count_worker identity validation.

Zero production runtime impact: no extra stat calls, no receipt changes, no guard bypass.
Passive observation of already-consumed stat tuples and opened path identity checks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CountIdentityRecorder:
    """Bounded, test-only recorder for text_count_worker identity verification stages."""

    MAX_BYTES = 8192

    def __init__(self, workspace_root: Optional[Path | str] = None):
        if workspace_root is not None:
            self.workspace_root = os.path.normpath(os.path.abspath(os.fspath(workspace_root))).replace("\\", "/")
        else:
            self.workspace_root = None
        self.events: List[Dict[str, Any]] = []
        self.cap_exceeded: bool = False
        self.first_mismatch: Optional[Dict[str, Any]] = None

    def _append_event(self, event: Dict[str, Any]) -> None:
        if self.cap_exceeded:
            return
        candidate_events = self.events + [event]
        candidate_payload = {
            "events": candidate_events,
            "first_mismatch": self.first_mismatch,
            "cap_exceeded": False,
        }
        raw_size = len(json.dumps(candidate_payload, ensure_ascii=True).encode("utf-8"))
        if raw_size <= self.MAX_BYTES:
            self.events.append(event)
        else:
            self.cap_exceeded = True

    def _sanitize_path(self, path: Any) -> str:
        """Pure lexical path sanitization; zero filesystem stat/resolve/timing calls."""
        if path is None:
            return ""
        s = os.fspath(path)
        try:
            norm = os.path.normpath(os.path.abspath(s)).replace("\\", "/")
            if self.workspace_root:
                ws = self.workspace_root
                if norm.startswith(ws + "/"):
                    return norm[len(ws) + 1:]
                elif norm == ws:
                    return "."
            return os.path.basename(norm) or s
        except Exception:
            return str(path)

    @staticmethod
    def _tuple_dict(t: Optional[Tuple]) -> Optional[Dict[str, Any]]:
        if not t or len(t) != 6:
            return None
        return {
            "st_dev": t[0],
            "st_ino": t[1],
            "st_mode": t[2],
            "st_size": t[3],
            "st_mtime_ns": t[4],
            "st_ctime_ns": t[5],
        }

    def record_open(self, target_path: Any, initial_stat: Optional[Tuple], fd_fstat: Optional[Tuple], match: bool) -> None:
        event = {
            "phase": "open.initial_vs_fstat",
            "target": self._sanitize_path(target_path),
            "initial": self._tuple_dict(initial_stat),
            "fd_fstat": self._tuple_dict(fd_fstat),
            "match": match,
        }
        if not match and self.first_mismatch is None:
            differing = [
                k for k in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
                if (event["initial"] or {}).get(k) != (event["fd_fstat"] or {}).get(k)
            ]
            self.first_mismatch = {
                "phase": event["phase"],
                "differing_fields": differing,
            }
        self._append_event(event)

    def record_check_identity(self, phase: str, fd: int, original: Any, target: Any,
                              actual: Any, actual_allowed: Any, original_allowed: Any,
                              path_matches: Dict[str, bool],
                              before: Optional[Tuple],
                              fd_stat: Optional[Tuple] = None,
                              target_stat: Optional[Tuple] = None,
                              original_stat: Optional[Tuple] = None,
                              identity_match: bool = True) -> None:
        event = {
            "phase": phase,
            "paths": {
                "original": self._sanitize_path(original),
                "target": self._sanitize_path(target),
                "actual": self._sanitize_path(actual),
            },
            "path_matches": path_matches,
            "before_tuple": self._tuple_dict(before),
            "fd_stat": self._tuple_dict(fd_stat),
            "target_stat": self._tuple_dict(target_stat),
            "original_stat": self._tuple_dict(original_stat),
            "identity_match": identity_match,
        }
        if not identity_match and self.first_mismatch is None:
            self.first_mismatch = {
                "phase": phase,
                "path_matches": path_matches,
            }
        self._append_event(event)

    def record_postscan_bytes(self, scanned_bytes: int, expected_size: int, match: bool) -> None:
        event = {
            "phase": "postscan.bytes_vs_initial_size",
            "scanned_bytes": scanned_bytes,
            "expected_size": expected_size,
            "match": match,
        }
        if not match and self.first_mismatch is None:
            self.first_mismatch = {
                "phase": event["phase"],
                "scanned_bytes": scanned_bytes,
                "expected_size": expected_size,
            }
        self._append_event(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": self.events,
            "first_mismatch": self.first_mismatch,
            "cap_exceeded": self.cap_exceeded,
        }


class DiagnosticWorkerWrapper:
    """Test-only passive wrapper around text_count_worker identity checks.

    Zero production changes. Zero extra stat/fstat calls.
    Preserves exact order, returns, exceptions, receipts, and deadlines.
    """

    def __init__(self, recorder: CountIdentityRecorder, monkeypatch=None):
        self.recorder = recorder
        self.monkeypatch = monkeypatch
        self._patches: List[Tuple[Any, str, Any]] = []

    def install(self, worker_module: Any) -> None:
        recorder = self.recorder
        orig_check_identity = worker_module._check_identity
        orig_open_validated = worker_module._open_validated

        def wrapped_open_validated(path):
            orig_reliable = worker_module._reliable_identity
            tuples_seen = []

            def capturing_reliable(info):
                res = orig_reliable(info)
                tuples_seen.append(res)
                if len(tuples_seen) == 2:
                    # Both initial stat and fd fstat are captured before _check_identity runs
                    initial = tuples_seen[0]
                    before = tuples_seen[1]
                    recorder.record_open(path, initial, before, match=(initial == before))
                return res

            worker_module._reliable_identity = capturing_reliable
            try:
                return orig_open_validated(path)
            finally:
                worker_module._reliable_identity = orig_reliable

        def wrapped_check_identity(fd, original, target, before):
            orig_reliable = worker_module._reliable_identity
            check_tuples = []

            def capturing_check_reliable(info):
                res = orig_reliable(info)
                check_tuples.append(res)
                return res

            worker_module._reliable_identity = capturing_check_reliable
            try:
                orig_check_identity(fd, original, target, before)
                recorder.record_check_identity(
                    phase="check_identity.pass",
                    fd=fd,
                    original=original,
                    target=target,
                    actual=target,
                    actual_allowed=target,
                    original_allowed=target,
                    path_matches={"target": True},
                    before=before,
                    identity_match=True
                )
            except Exception:
                recorder.record_check_identity(
                    phase="check_identity.refusal",
                    fd=fd,
                    original=original,
                    target=target,
                    actual=None,
                    actual_allowed=None,
                    original_allowed=None,
                    path_matches={"target": False},
                    before=before,
                    fd_stat=check_tuples[0] if len(check_tuples) > 0 else None,
                    target_stat=check_tuples[1] if len(check_tuples) > 1 else None,
                    original_stat=check_tuples[2] if len(check_tuples) > 2 else None,
                    identity_match=False
                )
                raise
            finally:
                worker_module._reliable_identity = orig_reliable

        if self.monkeypatch:
            self.monkeypatch.setattr(worker_module, "_open_validated", wrapped_open_validated)
            self.monkeypatch.setattr(worker_module, "_check_identity", wrapped_check_identity)
        else:
            self._patches.append((worker_module, "_open_validated", orig_open_validated))
            self._patches.append((worker_module, "_check_identity", orig_check_identity))
            worker_module._open_validated = wrapped_open_validated
            worker_module._check_identity = wrapped_check_identity

    def uninstall(self) -> None:
        for mod, attr, orig in self._patches:
            setattr(mod, attr, orig)
        self._patches.clear()
