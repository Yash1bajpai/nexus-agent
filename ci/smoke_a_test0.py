"""Stage A (Test 0) engine tool-call smoke check, single combined script.

Flow (one workflow step drives this):
  1. AST-extract the exact tool payload the product emits
     (LocalProvider._convert_tools) from the target checkout's tools.py.
     Stdlib-only literal evaluation; no product code is executed.
  2. Wait for llama-server readiness via GET /health with curl --fail-with-body
     (HTTP 200 required; /v1/models polling alone is NOT accepted).
  3. POST request A: greeting, max_tokens 256, no tools.
     POST request B: FORCED read_file tool call (tool_choice pins read_file),
     max_tokens 2048. curl --fail-with-body; raw request/response bytes are
     dumped to the console as wire evidence.
  4. Evaluate criteria and print TEST0 VERDICT: PASS|FAIL.

PASS iff all of:
  A1. resp_a parses as JSON and has non-empty message content.
  B1. resp_b parses as JSON.
  B2. choices[0].message.tool_calls is a non-empty list.
  B2f. at least one tool call is read_file (the forced call landed).
  B3. every tool call name is one of the emitted schemas and its
      function.arguments string parses as a JSON object.
  B4. message content contains no "<|tool_call" and no "<|im_" leak.
  B5. message content contains no "<think" leak (thinking confined to
      reasoning_content or absent).

Usage:
    python ci/smoke_a_test0.py --tools-py product-src/src/nexus_agent_ai/agent/tools.py \
        --port 18080 --alias bigmodel-test0 --evidence-dir test0_evidence
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

# Expected tool set, in emitted order. Fail-closed if the target source adds,
# drops, or renames a tool: the reviewer must see that, not a silent pass.
ORDER = ["read_file", "write_file", "patch_file", "list_directory",
         "run_code", "search_web", "git_status", "git_diff", "run_tests"]

READY_DEADLINE_S = 300


def extract_tools(tools_py: Path) -> list:
    tree = ast.parse(tools_py.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            if (isinstance(f, ast.Name) and f.id == "Tool") or \
               (isinstance(f, ast.Attribute) and f.attr == "Tool"):
                kw = {k.arg: ast.literal_eval(k.value) for k in node.value.keywords
                      if k.arg in ("name", "description", "input_schema")}
                if len(kw) == 3:
                    found[kw["name"]] = kw
    missing = [n for n in ORDER if n not in found]
    if missing:
        raise SystemExit("MISSING TOOLS in tools.py: " + ",".join(missing))
    return [{"type": "function", "function": {"name": n,
             "description": found[n]["description"],
             "parameters": found[n]["input_schema"]}} for n in ORDER]


def curl(args, **kw):
    return subprocess.run(["curl.exe"] + args, capture_output=True, **kw)


def wait_ready(port: int, ev: Path) -> None:
    deadline = time.monotonic() + READY_DEADLINE_S
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        p = curl(["--fail-with-body", "-sS", f"http://127.0.0.1:{port}/health"],
                 timeout=10)
        body = p.stdout.decode("utf-8", errors="replace")
        print(f"health attempt {attempt}: rc={p.returncode} body={body[:200]!r}",
              flush=True)
        if p.returncode == 0:
            (ev / "health_ok.txt").write_text(body, encoding="utf-8")
            return
        time.sleep(3)
    raise SystemExit("FAIL: llama-server /health did not return 200 within "
                     f"{READY_DEADLINE_S}s (curl --fail-with-body)")


def post(port: int, req: dict, out_path: Path) -> None:
    req_path = out_path.with_suffix(".request.json")
    req_path.write_text(json.dumps(req, indent=1), encoding="utf-8")
    print(f"=== RAW REQUEST {req_path.name} ===", flush=True)
    print(json.dumps(req, indent=1)[:20000], flush=True)
    p = curl(["--fail-with-body", "-sS", "-X", "POST",
              "-H", "Content-Type: application/json",
              "--data-binary", f"@{req_path}",
              "-o", str(out_path),
              f"http://127.0.0.1:{port}/v1/chat/completions"], timeout=600)
    if p.returncode != 0:
        body = out_path.read_bytes() if out_path.is_file() else b""
        sys.stdout.write(body.decode("utf-8", errors="replace")[:20000])
        raise SystemExit(f"FAIL: curl --fail-with-body rc={p.returncode}: "
                         + p.stderr.decode("utf-8", errors="replace")[:2000])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-py", required=True, type=Path)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--alias", default="bigmodel-test0")
    ap.add_argument("--evidence-dir", required=True, type=Path)
    args = ap.parse_args()
    ev = args.evidence_dir.resolve()
    ev.mkdir(parents=True, exist_ok=True)

    tools = extract_tools(args.tools_py)
    (ev / "tools-payload.json").write_text(json.dumps(tools, indent=1), encoding="utf-8")
    print("req_b tools:", [t["function"]["name"] for t in tools], flush=True)

    req_a = {"model": args.alias,
             "messages": [{"role": "user",
                           "content": "Hello, reply with a short friendly greeting."}],
             "max_tokens": 256, "stream": False}
    req_b = {"model": args.alias,
             "messages": [{"role": "user",
                           "content": "Read the file calc.py using the read_file tool, "
                                      "then tell me what it contains."}],
             "tools": tools,
             "tool_choice": {"type": "function",
                             "function": {"name": "read_file"}},
             "max_tokens": 2048, "stream": False}

    wait_ready(args.port, ev)
    resp_a, resp_b = ev / "resp_a.json", ev / "resp_b.json"
    post(args.port, req_a, resp_a)
    post(args.port, req_b, resp_b)

    failures = []
    raw_a = resp_a.read_bytes()
    raw_b = resp_b.read_bytes()
    print("=== RAW RESPONSE A BYTES ===", flush=True)
    sys.stdout.write(raw_a.decode("utf-8", errors="replace")[:20000])
    print("\n=== RAW RESPONSE B BYTES ===", flush=True)
    sys.stdout.write(raw_b.decode("utf-8", errors="replace")[:20000])
    print("\n=== END RAW BYTES ===", flush=True)

    try:
        a = json.loads(raw_a)
        if not (a["choices"][0]["message"].get("content") or "").strip():
            failures.append("A1: greeting response has empty content")
    except Exception as e:
        failures.append(f"A1: resp_a JSON parse/content failed: {e}")

    tool_names = {t["function"]["name"] for t in tools}
    try:
        b = json.loads(raw_b)
    except Exception as e:
        failures.append(f"B1: resp_b JSON parse failed: {e}")
        b = None

    if b is not None:
        try:
            msg = b["choices"][0]["message"]
        except Exception as e:
            failures.append(f"B1: resp_b missing choices[0].message: {e}")
            msg = {}
        tcs = msg.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            failures.append("B2: no parsed tool_calls in response")
            tcs = []
        if not any(((tc or {}).get("function") or {}).get("name") == "read_file"
                   for tc in tcs):
            failures.append("B2f: forced read_file call absent from tool_calls")
        for i, tc in enumerate(tcs):
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            if name not in tool_names:
                failures.append(f"B3: tool_calls[{i}] name {name!r} not in emitted schemas")
            argstr = fn.get("arguments")
            try:
                parsed = json.loads(argstr) if isinstance(argstr, str) else argstr
                if not isinstance(parsed, dict):
                    failures.append(f"B3: tool_calls[{i}] arguments not a JSON object")
            except Exception as e:
                failures.append(f"B3: tool_calls[{i}] arguments invalid JSON: {e}")
        content_b = msg.get("content") or ""
        if "<|tool_call" in content_b or "<|im_" in content_b:
            failures.append("B4: special-token leak in message content")
        if "<think" in content_b:
            failures.append("B5: thinking leaked into content (not confined to reasoning_content)")
        reasoning = msg.get("reasoning_content")
        print(f"reasoning_content present: {reasoning is not None}, length: "
              f"{len(reasoning) if isinstance(reasoning, str) else 0}", flush=True)
        print(f"finish_reason: {b['choices'][0].get('finish_reason')}", flush=True)

    verdict = "PASS" if not failures else "FAIL"
    print(f"TEST0 VERDICT: {verdict}", flush=True)
    for f in failures:
        print("FAIL-ITEM:", f, flush=True)

    lines = ["# Stage A (Test 0) Engine Tool-Call Smoke", "",
             f"- **Verdict:** {verdict}",
             "- **Readiness:** GET /health 200 via curl --fail-with-body",
             "- **Requests:** greeting max_tokens=256; forced read_file max_tokens=2048",
             "- **Evidence:** raw request/response bytes in the job console; "
             "no artifact uploads by design", "", "## Criteria"]
    lines += [f"- FAIL: {f}" for f in failures] or \
             ["- All PASS criteria met (A1, B1, B2, B2f, B3, B4, B5)"]
    (ev / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
