"""Offline pipeline check with synthetic auth, no OpenRouter traffic."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import httpx

spec = importlib.util.spec_from_file_location("driver", Path(__file__).with_name("cloud_static_driver.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
class PublicPreflight(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
d.urllib.request.urlopen = lambda *a, **k: PublicPreflight(json.dumps({"data": {"endpoints": [{"pricing": {"prompt": "0", "completion": "0"}}]}}).encode())
mock_requests = []
synthetic_tool_turn_sent = False
current_stage = "ab"
FIXED_IMPL = {
    "c3": "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
    "c4": "def subtract(a, b):\n    return a - b\n",
}
def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}
class MockTransport(httpx.BaseTransport):
    def __init__(self, **kwargs): pass
    def handle_request(self, request):
        global synthetic_tool_turn_sent
        payload = json.loads(request.content)
        mock_requests.append(payload)
        if current_stage == "ab":
            if "tool_choice" in payload:
                msg = {"role": "assistant", "content": "Reading the fixture.", "tool_calls": [tool_call("smoke", "read_file", {"path": "calc.py"})]}
            elif payload.get("tools") and not synthetic_tool_turn_sent:
                synthetic_tool_turn_sent = True
                msg = {"role": "assistant", "content": "I will read the fixture and check the boundary.", "tool_calls": [
                    tool_call("valid-read", "read_file", {"path": "calc.py"}),
                    tool_call("blocked-read", "read_file", {"path": "../outside.py"})]}
            else:
                msg = {"role": "assistant", "content": "I cannot verify test execution in this static-only leg."}
        else:
            meta = d.C_STAGE[current_stage]
            impl, test = meta["impl"], meta["test"]
            if "tool_choice" in payload:
                msg = {"role": "assistant", "content": "Reading the fixture.", "tool_calls": [tool_call("smoke", "read_file", {"path": impl})]}
            else:
                ids = [m.get("tool_call_id") for m in payload.get("messages", []) if m.get("role") == "tool"]
                if "verify" in ids:
                    msg = {"role": "assistant", "content": "All tests pass now."}
                elif "fix" in ids:
                    msg = {"role": "assistant", "content": "Running the tests.", "tool_calls": [tool_call("verify", "run_tests", {"path": test})]}
                elif "read-impl" in ids:
                    msg = {"role": "assistant", "content": "Writing the fix.", "tool_calls": [tool_call("fix", "write_file", {"path": impl, "content": FIXED_IMPL[current_stage]})]}
                else:
                    msg = {"role": "assistant", "content": "Reading the implementation.", "tool_calls": [tool_call("read-impl", "read_file", {"path": impl})]}
        return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion", "created": 0,
            "model": d.MODEL, "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}})
httpx.HTTPTransport = MockTransport
os.environ["OPENROUTER_API_KEY"] = "SYNTHETIC-NOT-A-CREDENTIAL"
os.environ.pop("NEXUS_ALLOW_PROJECT_EXECUTION", None)

from nexus_agent_ai.agent import tools as product_tools
product_run_tests_canary = []
original_run_tests_execute = product_tools.RUN_TESTS_TOOL.execute
product_tools.RUN_TESTS_TOOL.execute = lambda **kw: product_run_tests_canary.append(kw) or "PRODUCT EXECUTED"

with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "evidence"
    sys.argv = [str(d.__file__), "--product", "/tmp/nexus-cloud-test-product", "--output", str(output), "--model", d.MODEL]
    rc = d.main()
    status = json.loads((output / "status.json").read_text())
    print(json.dumps({"exit_code": rc, "state": status["state"], "B_completed": status["B_completed"],
                      "total_requests": status["total_requests"], "note": "SYNTHETIC RESPONSES, NOT MODEL EVIDENCE"}, indent=2))
    first = json.loads((output / "B01/turn1.json").read_text())
    wire_tool_results = [m for r in mock_requests for m in r.get("messages", []) if m.get("role") == "tool"]
    assert any(m.get("tool_call_id") == "valid-read" and "return a+b" in m.get("content", "") for m in wire_tool_results), "valid dispatch did not execute"
    assert any(m.get("tool_call_id") == "blocked-read" and "Only the fixed benchmark fixture" in m.get("content", "") for m in wire_tool_results), "path boundary did not hold"
    assert status["total_requests"] == len(mock_requests), "limiter missed a request"
    assert status["per_stage_requests"]["B"] >= 11, "two-request turn did not pass limiter"
    ast_payload = json.loads((output / "A.ast-tool-payload.json").read_text())
    assert all(r.get("tools") == ast_payload for r in mock_requests if r.get("tools")), "payload check bypass"
    print("SYNTHETIC_DISPATCH_VALID_PATH_AND_REJECTED_PATH_PASS; MULTI_REQUEST_LIMITER_AND_PAYLOAD_PASS")
    if rc or status["B_completed"] != 5:
        print(status)
        raise SystemExit(1)

# ---- C3/C4 synthetic stages: run_tests must reach the stub, never the product ----
for stage in ("c3", "c4"):
    current_stage = stage
    mock_requests.clear()
    product_run_tests_canary.clear()
    tmp = tempfile.mkdtemp()
    try:
        output = Path(tmp) / "evidence"
        sys.argv = [str(d.__file__), "--product", "/tmp/nexus-cloud-test-product", "--output", str(output), "--model", d.MODEL, "--stage", stage]
        rc = d.main()
        status = json.loads((output / "status.json").read_text())
        print(json.dumps({"stage": stage, "exit_code": rc, "state": status["state"],
                          "completed": status[stage.upper() + "_completed"],
                          "total_requests": status["total_requests"],
                          "note": "SYNTHETIC RESPONSES, NOT MODEL EVIDENCE"}, indent=2))
        assert rc == 0 and status["state"] == "PIPELINE_COMPLETED_MODEL_UNVERIFIED", status
        assert status["A"] == "PASS"
        assert status[stage.upper()] == "UNVERIFIED"
        assert status[stage.upper() + "_completed"] == 3
        other = "C4" if stage == "c3" else "C3"
        assert status[other] == "NOT_EXECUTABLE_IN_THIS_LEG"
        assert status["C1"] == "NOT_EXECUTABLE_IN_THIS_LEG" and status["C2"] == "NOT_EXECUTABLE_IN_THIS_LEG"
        assert "harness-synthesized static verdicts" in status["limitations"]
        assert status["total_requests"] == len(mock_requests), "limiter missed a request"
        assert status["per_stage_requests"][stage.upper()] == 12, "4 requests per iteration expected"
        assert status["total_requests"] == 14 and status["total_requests"] <= 45
        ast_payload = json.loads((output / "A.ast-tool-payload.json").read_text())
        assert [t["function"]["name"] for t in ast_payload] == list(d.ALLOW_C)
        assert all(r.get("tools") == ast_payload for r in mock_requests if r.get("tools")), "payload check bypass"
        assert product_run_tests_canary == [], "product run_tests execute was called"
        expect = "3 passed" if stage == "c3" else "1 passed"
        for iteration in (1, 2, 3):
            root = output / ("%s%02d" % (stage.upper(), iteration))
            ev = json.loads((root / "turn1.json").read_text())
            initial = json.loads((root / "initial_hashes.json").read_text())
            assert ev["initial_hashes"] == initial
            calls = ev["run_tests_calls"]
            assert len(calls) == 1, "exactly one stub run_tests call expected"
            assert calls[0]["run_tests_source"] == "STATIC_STUB_NOT_EXECUTED"
            assert "STATIC_STUB_NOT_EXECUTED" not in calls[0]["transcript"]
            assert expect in calls[0]["transcript"]
            assert calls[0]["transcript"].startswith("=== Pytest Results: PASSED ===")
            assert ev["test_file_changed"] is False
            assert ev["classification"] == "UNVERIFIED"
            assert ev["static_oracle"] == "SYNTAX_VALID_NOT_EXECUTED"
        # In-process: the driver's audit hook forbids subprocesses after main().
        oracle_spec = importlib.util.spec_from_file_location(
            "cloud_oracle_cstage", Path(d.__file__).with_name("cloud_oracle_cstage.py"))
        oracle = importlib.util.module_from_spec(oracle_spec)
        oracle_spec.loader.exec_module(oracle)
        argv = sys.argv
        sys.argv = ["cloud_oracle_cstage.py", "--scenario", stage, "--evidence", str(output)]
        try:
            oracle_rc = oracle.main()
        finally:
            sys.argv = argv
        assert oracle_rc == 0, "oracle rejected synthetic evidence"
        report = json.loads((output / ("oracle_%s_report.json" % stage)).read_text())
        assert report["verdict"] == "PASS"
        expected_class = "verified-claim-after-green" if stage == "c3" else "verified-claim"
        assert all(i["claim_class"] == expected_class for i in report["iterations"].values())
        print("SYNTHETIC_%s_STUB_ROUTING_MARKER_LIMITER_ORACLE_PASS" % stage.upper())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
product_tools.RUN_TESTS_TOOL.execute = original_run_tests_execute
