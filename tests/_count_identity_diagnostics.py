"""Test-only observation of unchanged installed count-worker code.

The opt-in fixture wraps only the fixed worker launch in these test modules.
The isolated child executes the original worker bytes under a per-thread trace;
no production function, check, receipt, stat call or short-circuit is replaced.
Diagnostic timing/argv differ, so these artifacts are not qualification proof.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

MAX_BYTES = 8192
_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


class WorkerTrace:
    """Four bounded phases, with absent checks explicitly NOT_REACHED."""

    def __init__(self, filename, source_sha256=None):
        self.filename = os.path.abspath(filename)
        self.payload = {
            "worker_sha256": source_sha256,
            "phases": [{"phase": name, "status": "NOT_REACHED"} for name in
                       ("open", "after_ack", "after_scan", "receipt")],
            "cap_exceeded": False,
        }
        self.states = {}
        self.check_count = 0
        self.previous = None

    def _save(self, index, value):
        phases = list(self.payload["phases"])
        phases[index] = value
        candidate = dict(self.payload, phases=phases)
        if len(json.dumps(candidate, ensure_ascii=True).encode("ascii")) <= MAX_BYTES:
            self.payload = candidate
        else:
            self.payload["cap_exceeded"] = True
            self.payload["phases"][index] = {
                "phase": self.payload["phases"][index]["phase"], "status": "OMITTED_CAP"}

    def _trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.filename:
            return None
        name = frame.f_code.co_name
        if name not in ("_open_validated", "_check_identity", "_reliable_identity", "_allowed", "_child_main"):
            return None
        if event == "call":
            state = {"name": name}
            if name == "_open_validated":
                state.update(index=0, stats=[], checks=[])
            elif name == "_check_identity":
                parent = self.states.get(frame.f_back)
                index = 0 if parent and parent["name"] == "_open_validated" else min(2, self.check_count + 1)
                if index:
                    self.check_count += 1
                state.update(index=index, stats=[], allowed=[], before=list(frame.f_locals["before"]))
            elif name == "_reliable_identity":
                # Only read attributes of the stat_result ALREADY produced by the worker.
                info = frame.f_locals["info"]
                state["consumed"] = {key: getattr(info, key) for key in _FIELDS if hasattr(info, key)}
            self.states[frame] = state
        state = self.states.get(frame)
        if state is None:
            return self._trace
        if event == "exception":
            state["exception"] = arg[0].__name__  # No exception text or private file content.
            if arg[0].__name__ == "_PathRefusal":
                state["reason"] = arg[1].reason
            elif isinstance(arg[1], OSError):
                state["errno"] = arg[1].errno
        elif event == "return":
            self.states.pop(frame, None)
            parent = self.states.get(frame.f_back)
            if name == "_reliable_identity" and parent and "stats" in parent:
                order = ("initial_stat", "opened_fstat") if parent["name"] == "_open_validated" else (
                    "held_fstat", "target_stat", "original_stat")
                value = {"operation": order[len(parent["stats"])], "consumed": state["consumed"]}
                if "exception" in state:
                    value["error"] = state["exception"]
                    value.update({key: state[key] for key in ("reason", "errno") if key in state})
                else:
                    value["returned"] = list(arg)
                parent["stats"].append(value)
            elif name == "_allowed" and parent and parent["name"] == "_check_identity":
                parent["allowed"].append({"status": "ERROR" if "exception" in state else "RETURNED",
                                          "equals_target": arg == frame.f_back.f_locals["target"] if "exception" not in state else None})
            elif name == "_check_identity":
                local = frame.f_locals
                value = {"status": "ERROR" if "exception" in state else "PASS",
                         "stats": state["stats"], "before": state["before"],
                         "allowed": state["allowed"],
                         "actual_equals_target": local["actual"] == local["target"] if "actual" in local else "NOT_REACHED",
                         "actual_allowed_equals_target": local["actual_allowed"] == local["target"] if "actual_allowed" in local else "NOT_REACHED"}
                if "exception" in state:
                    value["error"] = state["exception"]
                    value.update({key: state[key] for key in ("reason", "errno") if key in state})
                if parent and parent["name"] == "_open_validated":
                    parent["checks"].append(value)
                else:
                    self._save(state["index"], dict(value, phase=self.payload["phases"][state["index"]]["phase"]))
            elif name == "_open_validated":
                value = {"phase": "open", "status": "ERROR" if "exception" in state else "PASS",
                         "stats": state["stats"], "checks": state["checks"]}
                if "exception" in state:
                    value["error"] = state["exception"]
                    value.update({key: state[key] for key in ("reason", "errno") if key in state})
                self._save(0, value)
            elif name == "_child_main":
                receipt = frame.f_locals.get("receipt")
                if isinstance(receipt, dict):
                    source = receipt.get("source", {})
                    self._save(3, {"phase": "receipt", "status": "RETURNED",
                                   "receipt_status": receipt.get("status"), "reason": receipt.get("reason"),
                                   "bytes": source.get("bytes"), "partial": source.get("partial"),
                                   "acknowledged": frame.f_locals.get("acknowledged")})
                    before = frame.f_locals.get("before")
                    if before is not None:
                        value = dict(self.payload["phases"][2], expected_bytes=before[3],
                                     observed_bytes=source.get("bytes"), partial=source.get("partial"))
                        self._save(2, value)
                elif "exception" in state:
                    self._save(3, {"phase": "receipt", "status": "ERROR", "error": state["exception"]})
        return self._trace

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)
        self.states.clear()

    def write(self, path):
        raw = json.dumps(self.payload, ensure_ascii=True).encode("ascii")
        if len(raw) > MAX_BYTES:
            raise AssertionError("diagnostic final envelope exceeds cap")
        with open(path, "xb") as stream:
            stream.write(raw)


def _diagnostic_child(worker_path, artifact):
    # Read once, compile and execute exact source with the original filename/main
    # bootstrap. All target filesystem checks are still performed by that code.
    worker_path = os.path.abspath(worker_path)
    source = Path(worker_path).read_bytes()
    capture = WorkerTrace(worker_path, hashlib.sha256(source).hexdigest())
    namespace = {"__name__": "__main__", "__file__": worker_path, "__package__": None}
    sys.argv = [worker_path]
    try:
        with capture:
            exec(compile(source, worker_path, "exec"), namespace)
    finally:
        capture.write(artifact)


def _prepare_worker_launch(argv, worker_path, helper, artifact, *, paths=None):
    """Match lexical identities only; do not probe files or resolve aliases."""
    paths = os.path if paths is None else paths
    if not isinstance(argv, (list, tuple)) or len(argv) < 5:
        return argv
    script = argv[4]
    if not isinstance(script, (str, bytes, os.PathLike)):
        return argv
    script = os.fsdecode(script)
    if paths.normcase(paths.basename(paths.normpath(script))) != paths.normcase("text_count_worker.py"):
        return argv

    def key(path):
        return paths.normcase(paths.abspath(paths.normpath(os.fsdecode(path))))

    shape_matches = len(argv) == 5 and list(argv[1:4]) == ["-I", "-S", "-B"]
    if shape_matches and key(script) == key(worker_path):
        return [*argv[:4], helper, os.fspath(argv[4]), str(artifact)]
    # No raw paths/arguments in evidence. Preserve the original launch unchanged,
    # but make a missing child trace explicit rather than silently skipping it.
    marker = {"status": "UNMATCHED_WORKER_LAUNCH",
              "reason": "path_mismatch" if shape_matches else "argv_shape"}
    with open(Path(artifact).with_suffix(".unmatched.json"), "xb") as stream:
        stream.write(json.dumps(marker, ensure_ascii=True).encode("ascii"))
    return argv


if __name__ == "__main__":
    _diagnostic_child(*sys.argv[1:])
else:
    import pytest

    @pytest.fixture(autouse=True)
    def count_identity_diagnostics(request, monkeypatch):
        """Opt-in at actual test invocation paths, including real child processes."""
        output = os.environ.get("NEXUS_Q1_OUTPUT_DIR")
        if not output:
            yield
            return
        import subprocess
        from nexus_agent_ai.agent import text_count_worker as worker
        output = Path(output) / "count-identity"
        output.mkdir(parents=True, exist_ok=True)
        original = subprocess.Popen
        helper = os.path.abspath(__file__)
        test_key = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]

        def launch(argv, *args, **kwargs):
            artifact = output / f"{test_key}-{uuid.uuid4().hex}.json"
            argv = _prepare_worker_launch(argv, worker.__file__, helper, artifact)
            return original(argv, *args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", launch)
        # The ninth original failure calls _open_validated in-process. Trace that
        # thread too; child workers have their own independent trace/state.
        capture = WorkerTrace(worker.__file__, hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest())
        try:
            with capture:
                yield
        finally:
            if any(phase["status"] != "NOT_REACHED" for phase in capture.payload["phases"]):
                capture.write(output / f"{test_key}-parent-{uuid.uuid4().hex}.json")
