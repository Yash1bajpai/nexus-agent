"""Static-only cloud candidate. No fixture is imported, evaluated or executed."""
from __future__ import annotations
import argparse
import ast
import collections
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import urllib.request

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
TARGET = "019e54eee0882ac7629869fa83959a1d8d55c582"
ALLOW = ("read_file", "write_file", "patch_file", "list_directory")
SECRET_PATTERN = re.compile(rb"sk-or-[A-Za-z0-9_-]+")
INITIAL = "def multiply(a,b):\n    return a+b\n"
PROMPTS = ("Fix multiply in calc.py so it multiplies.",
           "Now make it return 0 if either input is None.")

class StopLeg(BaseException):
    pass

class Limiter:
    def __init__(self, status, save, clock=time.monotonic, sleep=time.sleep):
        self.status, self.save, self.clock, self.sleep = status, save, clock, sleep
        self.recent = collections.deque()
        self.started = clock()
        self.stage = "A"

    def take(self):
        if self.clock() - self.started >= 2400:
            raise StopLeg("global wall-clock cap")
        if self.status["total_requests"] >= 45:
            raise StopLeg("run-local daily cap reached")
        now = self.clock()
        while self.recent and now - self.recent[0] >= 60:
            self.recent.popleft()
        if len(self.recent) >= 18:
            self.sleep(max(0, 60 - (now - self.recent[0])) + .01)
            return self.take()
        self.recent.append(now)
        self.status["total_requests"] += 1
        counts = self.status["per_stage_requests"]
        counts[self.stage] = counts.get(self.stage, 0) + 1
        self.status["per_model_requests"][MODEL] = self.status["total_requests"]
        self.save()

    def request(self, send):
        import httpx
        rate_retries = 0
        transport_retries = 0
        while True:
            self.take()
            try:
                response = send()
            except httpx.TransportError:
                if transport_retries == 1:
                    self.status["aborts"] += 1
                    self.save()
                    raise StopLeg("transport retry exhausted")
                transport_retries += 1
                self.status["retries"] += 1
                self.status["transport_retries"] = self.status.get("transport_retries", 0) + 1
                self.save()
                continue
            code = response.status_code
            if code in (401, 402, 403):
                self.status["aborts"] += 1
                self.save()
                response.close()
                raise StopLeg("hard HTTP abort " + str(code))
            if code != 429:
                return response
            try:
                hint = float(response.headers.get("Retry-After", "0"))
            except ValueError:
                hint = 0
            response.close()
            if rate_retries == 3:
                self.status["aborts"] += 1
                self.save()
                raise StopLeg("429 retries exhausted")
            self.status["retries"] += 1
            self.save()
            self.sleep(min(60, max(5 * 2 ** rate_retries, hint)))
            rate_retries += 1


def extract_payload(source):
    found = {}
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Tool":
            attrs = {k.arg: ast.literal_eval(k.value) for k in node.keywords
                     if k.arg in ("name", "description", "input_schema")}
            if attrs.get("name") in ALLOW:
                found[attrs["name"]] = attrs
    if set(found) != set(ALLOW):
        raise StopLeg("AST schema mismatch")
    return [{"type": "function", "function": {"name": name,
             "description": found[name]["description"],
             "parameters": found[name]["input_schema"]}} for name in ALLOW]


def check_endpoints(data):
    endpoints = data.get("data", {}).get("endpoints", [])
    if not endpoints:
        raise StopLeg("no live free endpoints")
    for endpoint in endpoints:
        prices = endpoint.get("pricing", {})
        for field in ("prompt", "completion", "request", "image", "web_search", "internal_reasoning"):
            if field in ("prompt", "completion") and field not in prices:
                raise StopLeg("missing endpoint price")
            if field in prices and float(prices[field]) != 0:
                raise StopLeg("nonzero endpoint price")
    return endpoints


def scan_and_delete(out):
    for path in out.rglob("*"):
        if path.is_symlink() or (path.is_file() and SECRET_PATTERN.search(path.read_bytes())):
            shutil.rmtree(out)
            return False
    return True


def deny_process_execution(event, args):
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "pty.spawn"):
        raise StopLeg("process execution prohibited in static-only leg: " + event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {"mode": "STATIC_ONLY_CORE_NOT_CLI", "model": args.model, "target_sha": TARGET,
              "state": "STARTED", "total_requests": 0, "retries": 0, "aborts": 0,
              "per_stage_requests": {}, "per_model_requests": {}, "B_completed": 0,
              "C1": "NOT_EXECUTABLE_IN_THIS_LEG", "C2": "NOT_EXECUTABLE_IN_THIS_LEG",
              "C3": "NOT_EXECUTABLE_IN_THIS_LEG", "C4": "NOT_EXECUTABLE_IN_THIS_LEG",
              "zone_B": "ABSENT", "fixture_execution": "PROHIBITED",
              "limitations": "Core API in process, static tool subset, not equivalent to unchanged CLI/local slate"}
    def save():
        (out / "status.json").write_text(json.dumps(status, indent=2))
    save()
    limiter = Limiter(status, save)
    original_cwd = Path.cwd()
    try:
        if args.model != MODEL or not args.model.endswith(":free"):
            raise StopLeg("unapproved model")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise StopLeg("missing repository-admin-provisioned key")
        if "NEXUS_ALLOW_PROJECT_EXECUTION" in os.environ:
            raise StopLeg("project execution flag must be unset")
        # Network preflight is public and cannot echo Authorization.
        url = "https://openrouter.ai/api/v1/models/" + MODEL + "/endpoints"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        check_endpoints(data)
        (out / "endpoint-preflight.json").write_text(json.dumps(data, indent=2))
        freeze = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "cloud-pip-freeze.txt"
        if freeze.is_file():
            shutil.copyfile(freeze, out / "pip-freeze.txt")
        product = args.product.resolve()
        tools_source = product / "src/nexus_agent_ai/agent/tools.py"
        payload = extract_payload(tools_source)
        (out / "A.ast-tool-payload.json").write_text(json.dumps(payload, indent=2))
        hashes = {}
        for name, path in (("driver", Path(__file__)), ("product_tools", tools_source),
                           ("product_core", product / "src/nexus_agent_ai/agent/core.py"),
                           ("local_observer", Path(__file__).with_name("observe_normal_turns.py")),
                           ("local_oracle", Path(__file__).with_name("oracle_stage_c.py"))):
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        status["sha256"] = hashes
        save()
        # Import installed pinned wheel only. Scratch working directories are never sys.path entries.
        sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != original_cwd]
        import httpx
        import openai
        from nexus_agent_ai.agent import core, tools
        from nexus_agent_ai.providers.openai_provider import OpenAIProvider
        class Transport(httpx.BaseTransport):
            def __init__(self):
                self.inner = httpx.HTTPTransport(retries=0)
            def handle_request(self, request):
                if (str(request.url) != "https://openrouter.ai/api/v1/chat/completions"
                        or request.method != "POST"):
                    raise StopLeg("unexpected API destination")
                body = json.loads(request.content)
                if body.get("model") != MODEL or body.get("stream"):
                    raise StopLeg("model or streaming bypass")
                names = [x["function"]["name"] for x in body.get("tools", [])]
                if any(n not in ALLOW for n in names):
                    raise StopLeg("tool payload bypass")
                if body.get("tools") and body["tools"] != payload:
                    raise StopLeg("offered tool payload differs from AST dump")
                return limiter.request(lambda: self.inner.handle_request(request))
            def close(self):
                self.inner.close()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(StopLeg("wall-clock deadline")))
        signal.alarm(2400)
        transport = Transport()
        client = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                               base_url="https://openrouter.ai/api/v1", max_retries=0,
                               timeout=180, http_client=httpx.Client(transport=transport, timeout=180))
        provider = OpenAIProvider(model=MODEL, base_url="https://openrouter.ai/api/v1",
                                  api_key="unused", context_size=4096, max_output_tokens=1024)
        provider.client.close()
        provider.client = client
        # Defense in depth: no processes, shell commands or git helpers after this point.
        # Preload SDK request models before entering the no-process measured zone.
        import platform
        platform.platform()
        platform.processor()
        sys.addaudithook(deny_process_execution)
        offered = [getattr(tools, name.upper() + "_TOOL") for name in ALLOW]
        functions = {t.name: t.execute for t in offered}
        active_root = None
        def dispatch(name, values):
            if name not in functions or not isinstance(values, dict):
                return "ERROR: Tool unavailable in static-only cloud leg."
            values = dict(values)
            target = values.get("path", ".")
            if not isinstance(target, str):
                return "ERROR: Invalid path."
            resolved = (active_root / target).resolve()
            expected = active_root if name == "list_directory" else active_root / "calc.py"
            if resolved != expected or resolved.is_symlink():
                return "ERROR: Only the fixed benchmark fixture is accessible."
            return functions[name](**values)
        core.execute_tool = dispatch
        tools.execute_tool = dispatch
        core.get_all_tools = lambda: offered
        tools.get_all_tools = lambda: offered
        # Health call is request 1, counted by the same limiter as all later calls.
        health = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Say hello."}], max_tokens=256)
        (out / "A.health.json").write_text(health.model_dump_json(indent=2))
        if not health.choices or not health.choices[0].message.content:
            raise StopLeg("health response empty")
        smoke = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Read calc.py."}],
                  tools=payload, tool_choice={"type": "function", "function": {"name": "read_file"}}, max_tokens=1024)
        (out / "A.forced-tool.json").write_text(smoke.model_dump_json(indent=2))
        msg = smoke.choices[0].message
        calls = msg.tool_calls or []
        try:
            invalid_calls = (not calls or not any(c.function.name == "read_file" for c in calls)
                or any(c.function.name not in ALLOW or not isinstance(json.loads(c.function.arguments), dict) for c in calls))
        except (ValueError, TypeError):
            invalid_calls = True
        if invalid_calls or any(x in (msg.content or "") for x in ("<|tool_call", "<|im_", "<think")):
            raise StopLeg("Stage A tool gate failed")
        status["A"] = "PASS"
        save()
        limiter.stage = "B"
        for iteration in range(1, 6):
            root = out / ("B%02d" % iteration)
            root.mkdir()
            active_root = root
            (root / "calc.py").write_text(INITIAL)
            os.chdir(root)
            agent = core.Agent(provider=provider, max_iterations=10, verbose=False, tools=offered)
            for turn, prompt in enumerate(PROMPTS, 1):
                started = time.monotonic()
                capture = io.StringIO()
                signal.alarm(min(480, max(1, int(2400 - (time.monotonic() - limiter.started)))))
                try:
                    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                        answer = agent.run(prompt, stream=False)
                finally:
                    signal.alarm(max(1, int(2400 - (time.monotonic() - limiter.started))))
                # History and fixture bytes are data only. Never import/exec/eval.
                evidence = {"prompt": prompt, "answer": answer, "console": capture.getvalue(),
                            "duration_seconds": time.monotonic() - started,
                            "history": agent.memory.get(), "calc.py": (root / "calc.py").read_text(),
                            "static_only": True, "model_verdict": "UNVERIFIED"}
                try:
                    ast.parse(evidence["calc.py"])
                    evidence["static_oracle"] = "SYNTAX_VALID_NOT_EXECUTED"
                except SyntaxError:
                    evidence["static_oracle"] = "SYNTAX_INVALID_NOT_EXECUTED"
                evidence["changed_from_initial"] = evidence["calc.py"] != INITIAL
                (root / ("turn%d.json" % turn)).write_text(json.dumps(evidence, indent=2, default=str))
                if evidence["duration_seconds"] > 480:
                    raise StopLeg("turn wall-clock cap")
            status["B_completed"] += 1
            save()
        status["state"] = "PIPELINE_COMPLETED_MODEL_UNVERIFIED"
    except StopLeg as exc:
        status["state"] = "STOPPED_OR_STAGE_VOID"
        status["reason"] = str(exc)
    except BaseException:
        # Exception strings and tracebacks may contain auth headers. Never emit them.
        status["state"] = "ERROR_REDACTED_STAGE_VOID"
    finally:
        signal.alarm(0)
        os.chdir(original_cwd)
        save()
        if not scan_and_delete(out):
            raise SystemExit(2)
    return 0 if status["state"] == "PIPELINE_COMPLETED_MODEL_UNVERIFIED" else 1

if __name__ == "__main__":
    raise SystemExit(main())
