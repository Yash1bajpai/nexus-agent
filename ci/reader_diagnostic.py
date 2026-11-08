"""Fixed local file-reader diagnostic. No Nexus/model/network execution."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from unittest.mock import patch

reader_path = Path(__file__).with_name("observe_normal_turns.py")
spec = importlib.util.spec_from_file_location("observed_reader", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def values(st):
    return {name: getattr(st, name, None) for name in
            ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
             "st_ctime_ns", "st_birthtime_ns", "st_file_attributes", "st_reparse_tag")}


def trace(path):
    print("PATH", str(path), flush=True)
    lst = path.lstat()
    print("LSTAT_BEFORE", json.dumps(values(lst)), flush=True)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        fst = os.fstat(fd)
        print("FSTAT_BEFORE", json.dumps(values(fst)), flush=True)
        print("DIFFERING_FIELDS", json.dumps({k: [values(lst)[k], values(fst)[k]]
              for k in values(lst) if values(lst)[k] != values(fst)[k]}), flush=True)
        print("FIXED_READ_BYTES", len(os.read(fd, 64)), flush=True)
        print("FSTAT_AFTER", json.dumps(values(os.fstat(fd))), flush=True)
    finally:
        os.close(fd)
    print("LSTAT_AFTER", json.dumps(values(path.lstat())), flush=True)


def rejected(name, path):
    try:
        reader.read_regular_bytes(path, 65536)
    except Exception as e:
        print("NEGATIVE", name, "REJECTED", type(e).__name__, str(e), flush=True)
        return True
    print("NEGATIVE", name, "UNEXPECTED_ACCEPT", flush=True)
    return False


print("PLATFORM", platform.platform(), "PYTHON", sys.version, flush=True)
failures = []
with tempfile.TemporaryDirectory(prefix="nexus-reader-fixed-") as directory:
    root = Path(directory)
    for name, data in [("empty.txt", b""), ("nonempty.txt", b"fixed fixture\n"),
                       ("stdout-like.txt", b"Starting engine\r\nfixed output\r\n" * 64)]:
        path = root / name
        with open(path, "wb") as stream:
            stream.write(data)
        trace(path)
        try:
            actual = reader.read_regular_bytes(path, 65536)
            print("READER", name, "ACCEPTED", len(actual), "EQUAL", actual == data, flush=True)
            if actual != data:
                failures.append(name + ": bytes mismatch")
        except Exception as e:
            print("READER", name, "ERROR", type(e).__name__, str(e), flush=True)
            failures.append(name + ": " + str(e))

    oversized = root / "oversized.txt"
    oversized.write_bytes(b"x" * 65537)
    if not rejected("oversize", oversized):
        failures.append("oversize accepted")
    folder = root / "directory"
    folder.mkdir()
    if not rejected("directory", folder):
        failures.append("directory accepted")

    link = root / "symlink.txt"
    try:
        link.symlink_to(root / "nonempty.txt")
    except OSError as e:
        print("NEGATIVE symlink UNVERIFIED creation unavailable", type(e).__name__, str(e), flush=True)
    else:
        if not rejected("symlink/reparse", link):
            failures.append("symlink accepted")

    # Simulate the existing Windows reparse flag without changing any OS state.
    original_lstat = Path.lstat
    target = root / "nonempty.txt"
    class ReparseStat:
        def __init__(self, st):
            self._st = st
        def __getattr__(self, name):
            return getattr(self._st, name)
        @property
        def st_file_attributes(self):
            return getattr(self._st, "st_file_attributes", 0) | 0x400
    def marked(path, *args, **kwargs):
        st = original_lstat(path, *args, **kwargs)
        return ReparseStat(st) if path == target else st
    with patch.object(Path, "lstat", marked):
        if not rejected("mocked reparse flag (not native junction proof)", target):
            failures.append("reparse flag accepted")

    race = root / "race.txt"
    race.write_bytes(b"before")
    original_fstat = os.fstat
    calls = [0]
    def mutate(fd):
        calls[0] += 1
        if calls[0] == 2:
            race.write_bytes(b"after modified")
        return original_fstat(fd)
    with patch.object(reader.os, "fstat", side_effect=mutate):
        if not rejected("metadata mutation during read", race):
            failures.append("race accepted")

# Root/lifecycle controls: no product or model. The open/flushed reader may
# reject stale metadata; record it, close the SAME file, and compare again.
roots = [("DEFAULT_TEMP", None), ("RUNNER_TEMP", os.environ.get("RUNNER_TEMP"))]
for label, base in roots:
    if label == "RUNNER_TEMP" and not base:
        print("ROOT RUNNER_TEMP UNVERIFIED environment absent", flush=True)
        continue
    with tempfile.TemporaryDirectory(prefix="reader-lifecycle-", dir=base) as directory:
        root = Path(directory)
        print("ROOT", label, str(root), flush=True)
        for state in ["closed_writer", "open_flushed_writer"]:
            path = root / (state + ".txt")
            writer = open(path, "wb")
            writer.write(b"fixed lifecycle bytes\n")
            writer.flush()
            if state == "closed_writer":
                writer.close()
            print("LIFECYCLE", label, state, "writer_closed", writer.closed, flush=True)
            trace(path)
            try:
                print("LIFECYCLE_READER", label, state, "ACCEPTED",
                      len(reader.read_regular_bytes(path, 65536)), flush=True)
            except Exception as e:
                print("LIFECYCLE_READER", label, state, "REJECTED", str(e), flush=True)
                if state == "closed_writer":
                    failures.append(label + ": closed writer rejected")
            finally:
                writer.close()
            print("SAME_FILE_AFTER_CLOSE", label, state, "writer_closed", writer.closed, flush=True)
            trace(path)
            try:
                print("AFTER_CLOSE_READER", label, state, "ACCEPTED",
                      len(reader.read_regular_bytes(path, 65536)), flush=True)
            except Exception as e:
                print("AFTER_CLOSE_READER", label, state, "REJECTED", str(e), flush=True)
                failures.append(label + ": after close rejected")

        # Fixed, harmless producer only. Existing observer callsite, no generated code.
        producer = root / "fixed_producer.py"
        producer.write_text("import sys\nfrom pathlib import Path\n"
                            "sys.stdout.write('fixed stdout\\n')\n"
                            "sys.stderr.write('fixed stderr\\n')\n"
                            "Path('calc.py').write_bytes(b'def multiply(a,b):\\n    return a+b\\n')\n",
                            encoding="utf-8")
        stdout_path = root / "producer_stdout.log"
        stderr_path = root / "producer_stderr.log"
        info = reader.run_observed_turn([sys.executable, str(producer)], root,
                                        stdout_path, stderr_path, timeout=5.0)
        print("PRODUCER_EXISTING_OBSERVER", label, json.dumps(info), flush=True)
        print("PRODUCER_PARENT_EXITED", info.get("parent_exited"),
              "observer returned after capture context closure; descendants UNKNOWN", flush=True)
        for path in [stdout_path, stderr_path, root / "calc.py"]:
            trace(path)
            try:
                print("PRODUCER_READER", label, path.name, "ACCEPTED",
                      len(reader.read_regular_bytes(path, 65536)), flush=True)
            except Exception as e:
                print("PRODUCER_READER", label, path.name, "REJECTED", str(e), flush=True)
        if info.get("exit_code") != 0:
            failures.append(label + ": harmless producer nonzero")

print("DIAGNOSTIC_FAILURES", json.dumps(failures), flush=True)
print("No model/product commands, credentials or artifact uploads. Fixed harmless fixture producer only; no model-generated module execution.", flush=True)
sys.exit(1 if failures else 0)
