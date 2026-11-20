import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("driver", Path(__file__).with_name("cloud_static_driver.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

class Tests(unittest.TestCase):
    def limiter(self):
        status = {"total_requests": 0, "retries": 0, "aborts": 0, "per_stage_requests": {}, "per_model_requests": {}}
        clock = [0.0]
        delays = []
        def sleep(sec):
            delays.append(sec)
            clock[0] += sec
        return d.Limiter(status, lambda: None, lambda: clock[0], sleep), status, delays

    def test_minute_cap(self):
        limiter, status, delays = self.limiter()
        for _ in range(19):
            limiter.take()
        self.assertEqual(status["total_requests"], 19)
        self.assertGreaterEqual(sum(delays), 60)

    def test_daily_cap(self):
        limiter, status, _ = self.limiter()
        for _ in range(45):
            limiter.take()
        with self.assertRaises(d.StopLeg):
            limiter.take()
        self.assertEqual(status["total_requests"], 45)

    def test_429_backoff(self):
        limiter, status, delays = self.limiter()
        class Response:
            status_code = 429
            headers = {"Retry-After": "9"}
            def close(self): pass
        with self.assertRaises(d.StopLeg):
            limiter.request(Response)
        self.assertEqual(status["total_requests"], 4)
        self.assertEqual(status["retries"], 3)
        self.assertEqual(delays, [9., 10, 20])

    def test_auth_no_retry(self):
        for code in (401, 402, 403):
            limiter, status, _ = self.limiter()
            class Response:
                status_code = code
                def close(self): pass
            with self.assertRaises(d.StopLeg):
                limiter.request(Response)
            self.assertEqual(status["total_requests"], 1)
            self.assertEqual(status["retries"], 0)

    def test_endpoint_free_fail_closed(self):
        for data in ({}, {"data": {"endpoints": [{"pricing": {"prompt": "0", "completion": "1"}}]}}):
            with self.assertRaises(d.StopLeg): d.check_endpoints(data)
        self.assertTrue(d.check_endpoints({"data": {"endpoints": [{"pricing": {"prompt": "0", "completion": "0"}}]}}))

    def test_secret_scan_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            root.mkdir()
            (root / "test.txt").write_bytes(b"sk-" + b"or-" + b"SYNTHETIC_CANARY")
            self.assertFalse(d.scan_and_delete(root))
            self.assertFalse(root.exists())

    def test_transport_exactly_one_retry(self):
        import httpx
        for exc in (httpx.ReadTimeout("synthetic"), httpx.ConnectError("synthetic")):
            limiter, status, _ = self.limiter()
            def fail(): raise exc
            with self.assertRaises(d.StopLeg): limiter.request(fail)
            self.assertEqual(status["total_requests"], 2)
            self.assertEqual(status["transport_retries"], 1)
            self.assertEqual(status["retries"], 1)
        limiter, status, _ = self.limiter()
        attempts = []
        class Response:
            status_code = 200
        def recover():
            attempts.append(1)
            if len(attempts) == 1: raise httpx.ReadTimeout("synthetic")
            return Response()
        self.assertEqual(limiter.request(recover).status_code, 200)
        self.assertEqual(status["total_requests"], 2)

    def test_symlink_scan_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            root.mkdir()
            (root / "link").symlink_to(Path(tmp) / "absent")
            self.assertFalse(d.scan_and_delete(root))
            self.assertFalse(root.exists())

    def test_ast_payload(self):
        import os
        source = os.environ["CLOUD_TEST_TOOLS_SOURCE"]
        names = [t["function"]["name"] for t in d.extract_payload(source)]
        self.assertEqual(names, list(d.ALLOW))
        self.assertNotIn("run_code", names)
        self.assertNotIn("run_tests", names)

if __name__ == "__main__":
    unittest.main()
