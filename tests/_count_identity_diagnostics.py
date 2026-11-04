"""Test-only diagnostic capture for text_count_worker identity validation.

Zero production runtime impact: no extra stat calls, no receipt changes, no guard bypass.
Passive observation of already-consumed stat tuples and opened path identity checks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Dict, List, Optional, Tuple


class CountIdentityRecorder:
    """Bounded, test-only recorder for text_count_worker identity verification stages."""

    MAX_BYTES = 8192

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root.resolve() if workspace_root is not None else None
        self.events: List[Dict[str, Any]] = []
        self.bytes_used: int = 2
        self.cap_exceeded: bool = False
        self.first_mismatch: Optional[Dict[str, Any]] = None

    def _append_event(self, event: Dict[str, Any]) -> None:
        raw = json.dumps(event, ensure_ascii=True).encode("utf-8")
        if self.bytes_used + len(raw) + 2 <= self.MAX_BYTES:
            self.events.append(event)
            self.bytes_used += len(raw) + 2
        else:
            self.cap_exceeded = True
    def _sanitize_path(self, path: Any) -> str:
        p = Path(path)
        try:
            resolved = p.resolve()
            if self.workspace_root is not None and resolved.is_relative_to(self.workspace_root):
                return str(resolved.relative_to(self.workspace_root)).replace("\\", "/")
            return p.name
        except Exception:
            return p.name

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

    def record_open(self, target_path: Any, initial_stat: Tuple, fd_fstat: Tuple, match: bool) -> None:
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
                              before: Tuple,
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
        result = {
            "events": self.events,
            "first_mismatch": self.first_mismatch,
        }
        serialized = json.dumps(result, ensure_ascii=True)
        if self.cap_exceeded:
            result["cap_exceeded"] = True
        return result
