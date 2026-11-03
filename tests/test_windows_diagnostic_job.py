"""Owned lifecycle tests: portable native-API doubles and harmless Windows children.

Run the dependency-free Windows preflight with this file's --native-preflight CLI.
No model, network, taskkill, PID enumeration, or unowned termination is used.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / ".github/diagnostics/windows_job.py"
_spec = importlib.util.spec_from_file_location("diagnostic_windows_job_under_test", MODULE_PATH)
windows_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(windows_job)


class FakeNativeAPI:
    """Real portable pipes, synthetic owned kernel handles, no child execution."""
    def __init__(self, fail=None, running=False):
        self.fail = fail
        self.running = running
        self.events = []
        self.open_handles = set()
        self.pipe_handles = set()
        self.descendant_writes = []
        self.code = None if running else 0
        self.active = 1
        self.inherited = None

    def event(self, name, *args):
        self.events.append((name, *args))
        if self.fail == name:
            raise OSError("synthetic native failure")

    def owned(self, handle):
        self.open_handles.add(handle)
        return handle

    def create_job(self):
        self.event("create_job")
        return self.owned("owned-job")

    def configure_job(self, job):
        self.event("configure_job", job)

    def pipe(self):
        self.event("pipe")
        read_handle, write_handle = os.pipe()
        self.pipe_handles.update((read_handle, write_handle))
        return self.owned(read_handle), self.owned(write_handle)

    def null_input(self):
        self.event("null_input")
        return self.owned("owned-null")

    def create_suspended(self, argv, cwd, env, handles):
        self.event("create_suspended", tuple(handles))
        self.inherited = list(handles)
        self.descendant_writes = [os.dup(handle) for handle in handles[1:]]
        return self.owned("owned-process"), self.owned("original-primary-thread"), 12345

    def assign(self, job, process):
        self.event("assign", job, process)

    def resume(self, thread):
        self.event("resume", thread)
        os.write(self.descendant_writes[0], b"harmless stdout\n")
        os.write(self.descendant_writes[1], b"harmless stderr\n")

    def read_stream(self, handle):
        try:
            self.event("read_stream", handle)
        except BaseException:
            self.close(handle)
            raise
        self.open_handles.remove(handle)
        self.pipe_handles.remove(handle)
        return os.fdopen(handle, "rb", buffering=0)

    def wait(self, process, timeout):
        self.event("wait", process, timeout)
        return self.code

    def _release(self):
        for handle in self.descendant_writes:
            os.close(handle)
        self.descendant_writes.clear()
        self.active = 0
        if self.code is None:
            self.code = 1

    def terminate_job(self, job):
        self.event("terminate_job", job)
        self._release()

    def terminate_process(self, process):
        self.event("terminate_process", process)
        self._release()

    def active_processes(self, job):
        self.event("active_processes", job)
        return self.active

    def close(self, handle):
        self.event("close", handle)
        self.open_handles.remove(handle)
        if handle in self.pipe_handles:
            self.pipe_handles.remove(handle)
            os.close(handle)
        if handle == "owned-job":
            self._release()  # Last-handle kill fallback.


def make_process(api):
    return windows_job.WindowsJobProcess(
        [sys.executable, "-I", "-c", "pass"], cwd=os.getcwd(), env={},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, api=api)


def collect_streams(process):
    results = {}
    threads = []
    def read(name, stream):
        try:
            chunks = []
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
            results[name] = b"".join(chunks)
        finally:
            stream.close()
    for name in ("stdout", "stderr"):
        thread = threading.Thread(target=read, args=(name, getattr(process, name)), daemon=True)
        thread.start()
        threads.append(thread)
    return results, threads


class PortableWindowsJobTests(unittest.TestCase):
    def test_native_call_contract_suspended_explicit_handles_and_no_breakaway(self):
        ctypes = windows_job.ctypes
        class Startup(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                        ("stdin", ctypes.c_void_p), ("stdout", ctypes.c_void_p),
                        ("stderr", ctypes.c_void_p)]
        class StartupEx(ctypes.Structure):
            _fields_ = [("startup", Startup), ("attributes", ctypes.c_void_p)]
        class ProcessInfo(ctypes.Structure):
            _fields_ = [("process", ctypes.c_void_p), ("thread", ctypes.c_void_p),
                        ("pid", ctypes.c_uint32)]
        class BasicLimits(ctypes.Structure):
            _fields_ = [("flags", ctypes.c_uint32)]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits)]
        class Kernel:
            def __init__(self):
                self.events = []
                self.fail_attribute = False
            def CreateJobObjectW(self, security, name):
                self.events.append(("job", security, name))
                return 99
            def SetInformationJobObject(self, job, kind, limits, size):
                self.events.append(("limits", job, kind, limits._obj.basic.flags))
                return 1
            def InitializeProcThreadAttributeList(self, attributes, count, flags, length):
                length._obj.value = 64
                return 1
            def UpdateProcThreadAttribute(self, attributes, flags, kind, handles, size, previous, returned):
                self.events.append(("inherit", kind, list(handles)))
                if self.fail_attribute:
                    raise OSError("explicit handle list unsupported")
                return 1
            def CreateProcessW(self, executable, command, process_security, thread_security,
                               inherit, flags, environment, cwd, startup, info):
                self.events.append(("create", executable, command.value, inherit, flags,
                                    startup._obj.startup.flags,
                                    (startup._obj.startup.stdin, startup._obj.startup.stdout,
                                     startup._obj.startup.stderr),
                                    [item for item in environment[:].split("\0") if item]))
                info._obj.process, info._obj.thread, info._obj.pid = 101, 102, 103
                return 1
            def DeleteProcThreadAttributeList(self, attributes):
                self.events.append(("delete_attributes",))
        api = windows_job._NativeAPI.__new__(windows_job._NativeAPI)
        api.StartupEx, api.ProcessInfo, api.ExtendedLimits = StartupEx, ProcessInfo, Limits
        api.kernel = Kernel()
        job = api.create_job()
        api.configure_job(job)
        result = api.create_suspended([sys.executable, "-I", "-c", "pass"], os.getcwd(),
                                      {"z": "two", "A": "one"}, [11, 12, 13])
        self.assertEqual(result, (101, 102, 103))
        self.assertIn(("job", None, None), api.kernel.events)
        self.assertIn(("limits", 99, 9, 0x2000), api.kernel.events)
        self.assertIn(("inherit", 0x20002, [11, 12, 13]), api.kernel.events)
        create = next(event for event in api.kernel.events if event[0] == "create")
        self.assertEqual(create[3:], (True, 0x00080404, 0x100, (11, 12, 13), ["A=one", "z=two"]))
        self.assertEqual(api.kernel.events[-1], ("delete_attributes",))
        api.kernel.fail_attribute = True
        before = len([event for event in api.kernel.events if event[0] == "create"])
        with self.assertRaises(OSError):
            api.create_suspended([sys.executable], os.getcwd(), {}, [11, 12, 13])
        self.assertEqual(len([event for event in api.kernel.events if event[0] == "create"]), before)
        self.assertEqual(api.kernel.events[-1], ("delete_attributes",))

    def test_create_suspended_assign_then_resume_original_owned_thread(self):
        api = FakeNativeAPI()
        process = make_process(api)
        try:
            names = [event[0] for event in api.events]
            self.assertLess(names.index("create_suspended"), names.index("assign"))
            self.assertLess(names.index("assign"), names.index("resume"))
            self.assertIn(("resume", "original-primary-thread"), api.events)
            self.assertEqual(api.inherited[0], "owned-null")
            self.assertEqual(len(api.inherited), 3)
            self.assertNotIn("owned-job", api.inherited)
            self.assertEqual(process.wait(timeout=1), 0)
            cleanup = process.cleanup(timeout=1)
            self.assertEqual({key: value for key, value in cleanup.items() if key != "duration_seconds"},
                             {"ownership": "owned_windows_job", "assigned": True,
                              "confirmed": True, "errors": []})
            self.assertGreaterEqual(cleanup["duration_seconds"], 0)
            self.assertLess(cleanup["duration_seconds"], 1)
            self.assertEqual(api.open_handles, set())
            self.assertNotIn("terminate_process", [event[0] for event in api.events])
            self.assertEqual(process.cleanup(timeout=1), cleanup)
        finally:
            process.cleanup(timeout=1)
            process.stdout.close()
            process.stderr.close()

    def test_inherited_pipe_eof_requires_descendant_cleanup_after_any_root_exit(self):
        for exit_code in (0, 7, 0xC0000005):
            with self.subTest(exit_code=exit_code):
                api = FakeNativeAPI()
                api.code = exit_code
                process = make_process(api)
                results, threads = collect_streams(process)
                self.assertEqual(process.wait(timeout=1), exit_code)
                self.assertTrue(any(thread.is_alive() for thread in threads))
                cleanup = process.cleanup(timeout=1)
                for thread in threads:
                    thread.join(timeout=1)
                self.assertTrue(cleanup["confirmed"])
                self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(results, {"stdout": b"harmless stdout\n", "stderr": b"harmless stderr\n"})

    def test_timeout_terminates_owned_job_and_reaps_process(self):
        api = FakeNativeAPI(running=True)
        process = make_process(api)
        results, threads = collect_streams(process)
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=0.001)
        self.assertTrue(process.cleanup(timeout=1)["confirmed"])
        self.assertEqual(process.returncode, 1)
        for thread in threads:
            thread.join(timeout=1)
        self.assertEqual(len(results), 2)
        self.assertFalse(any(thread.is_alive() for thread in threads))

    def test_setup_assignment_resume_failures_clean_every_owned_resource(self):
        for failure in ("create_job", "configure_job", "pipe", "null_input",
                        "create_suspended", "assign", "read_stream", "resume"):
            with self.subTest(failure=failure):
                api = FakeNativeAPI(fail=failure)
                with self.assertRaises(windows_job.WindowsJobError) as caught:
                    make_process(api)
                self.assertTrue(caught.exception.cleanup["confirmed"])
                self.assertEqual(api.open_handles, set())
                names = [event[0] for event in api.events]
                if failure == "assign":
                    self.assertIn(("terminate_process", "owned-process"), api.events)
                    self.assertNotIn("resume", names)
                if failure == "resume":
                    self.assertIn(("terminate_job", "owned-job"), api.events)
                self.assertFalse(any("sentinel" in str(event) for event in api.events))

    def test_cleanup_failure_closes_last_job_handle_without_false_confirmation(self):
        for failure in ("terminate_job", "active_processes", "wait"):
            with self.subTest(failure=failure):
                api = FakeNativeAPI()
                process = make_process(api)
                api.fail = failure
                cleanup = process.cleanup(timeout=0)
                self.assertFalse(cleanup["confirmed"])
                self.assertTrue(cleanup["errors"])
                self.assertEqual(api.open_handles, set())
                self.assertIn(("close", "owned-job"), api.events)
                process.stdout.close()
                process.stderr.close()

    def test_cleanup_timeout_is_unconfirmed_and_closes_owned_job(self):
        api = FakeNativeAPI()
        process = make_process(api)
        api.active_processes = lambda job: 1
        cleanup = process.cleanup(timeout=0)
        self.assertFalse(cleanup["confirmed"])
        self.assertIn("job_empty_timeout", cleanup["errors"])
        self.assertEqual(api.open_handles, set())
        process.stdout.close()
        process.stderr.close()

    def test_non_windows_default_is_fail_closed(self):
        if os.name == "nt":
            self.skipTest("portable platform rejection is not applicable on Windows")
        with self.assertRaises(OSError):
            windows_job.WindowsJobProcess([sys.executable], cwd=os.getcwd(), env={},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@unittest.skipUnless(os.name == "nt", "real Windows Job Object API requires Windows")
class NativeWindowsJobTests(unittest.TestCase):
    def _exercise_native(self, *, exit_code=0, timeout=False):
        environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
        # This independent sentinel is created before the owned job; its owned
        # Popen handle is the only handle used to terminate it in the test finally.
        sentinel = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(90)"],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, close_fds=True, env=environment)
        process = None
        try:
            with tempfile.TemporaryDirectory(prefix="diagnostic-owned-job-") as temporary:
                ready = Path(temporary) / "grandchild-ready"
                grandchild = ("import pathlib,time; print('grandchild', flush=True); pathlib.Path("
                              + repr(str(ready)) + ").write_text('ready'); time.sleep(90)")
                child = (
                    "import pathlib,subprocess,sys,time\n"
                    f"subprocess.Popen([sys.executable,'-I','-c',{grandchild!r}], close_fds=False)\n"
                    f"ready=pathlib.Path({str(ready)!r})\n"
                    "deadline=time.monotonic()+10\n"
                    "while not ready.exists():\n"
                    " if time.monotonic()>=deadline: raise RuntimeError('grandchild startup timed out')\n"
                    " time.sleep(.01)\n"
                    "print('child', flush=True)\n"
                    + ("time.sleep(90)\n" if timeout else f"sys.exit({exit_code})\n"))
                process = windows_job.WindowsJobProcess(
                    [sys.executable, "-I", "-c", child], cwd=temporary, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                results, threads = collect_streams(process)
                deadline = time.monotonic() + 15
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists(), "real grandchild must start inside the job")
                if timeout:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        process.wait(timeout=.05)
                else:
                    self.assertEqual(process.wait(timeout=10), exit_code)
                self.assertTrue(any(thread.is_alive() for thread in threads),
                                "grandchild must still hold an inherited pipe before cleanup")
                cleanup = process.cleanup(timeout=10)
                self.assertTrue(cleanup["confirmed"], cleanup)
                self.assertEqual(cleanup["errors"], [])
                for thread in threads:
                    thread.join(timeout=5)
                self.assertFalse(any(thread.is_alive() for thread in threads),
                                 "owned descendants must release inherited pipes")
                self.assertIn(b"grandchild", results["stdout"])
                self.assertIsNone(sentinel.poll(), "independently owned sentinel must survive cleanup")
        finally:
            if process is not None:
                process.cleanup(timeout=10)
            sentinel.terminate()
            sentinel.wait(timeout=10)

    def test_native_windows_normal_exit_descendant_pipe_and_sentinel(self):
        self._exercise_native(exit_code=0)

    def test_native_windows_abnormal_exit_descendant_pipe_and_sentinel(self):
        self._exercise_native(exit_code=7)

    def test_native_windows_timeout_descendant_pipe_and_sentinel(self):
        self._exercise_native(timeout=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-preflight"]:
        if os.name != "nt":
            raise SystemExit("Native Windows lifecycle preflight requires Windows; not verified here")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeWindowsJobTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
    unittest.main()
