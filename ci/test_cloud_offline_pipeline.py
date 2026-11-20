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
class MockTransport(httpx.BaseTransport):
    def __init__(self, **kwargs): pass
    def handle_request(self, request):
        global synthetic_tool_turn_sent
        payload = json.loads(request.content)
        mock_requests.append(payload)
        if "tool_choice" in payload:
            msg = {"role": "assistant", "content": "Reading the fixture.", "tool_calls": [{"id": "smoke", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "calc.py"})}}]}
        elif payload.get("tools") and not synthetic_tool_turn_sent:
            synthetic_tool_turn_sent = True
            msg = {"role": "assistant", "content": "I will read the fixture and check the boundary.", "tool_calls": [
                {"id": "valid-read", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "calc.py"})}},
                {"id": "blocked-read", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "../outside.py"})}}]}
        else:
            msg = {"role": "assistant", "content": "I cannot verify test execution in this static-only leg."}
        return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion", "created": 0,
            "model": d.MODEL, "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}})
httpx.HTTPTransport = MockTransport
os.environ["OPENROUTER_API_KEY"] = "SYNTHETIC-NOT-A-CREDENTIAL"
os.environ.pop("NEXUS_ALLOW_PROJECT_EXECUTION", None)
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
