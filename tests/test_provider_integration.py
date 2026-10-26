"""Offline regressions for canonical histories, backend switching and local SSE."""
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from nexus_agent_ai.providers.base import ProviderResponse, RateLimitError, ToolCall
from nexus_agent_ai.providers.messages import (
    anthropic_messages, assistant_message, extract_json_tool_calls,
    gemini_messages, normalize_messages, openai_messages, tool_result,
)
from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.providers.openai_provider import OpenAIProvider


def history():
    return [
        {"role": "user", "content": "Read then patch"},
        assistant_message("Inspecting", [ToolCall("opaque_call_42", "read_file", {"path": "a.py"})]),
        tool_result("opaque_call_42", "old"),
        assistant_message("Editing", [ToolCall("opaque_call_43", "patch_file", {"path": "a.py", "old": "old", "new": "new"})]),
        tool_result("opaque_call_43", "patched"),
    ]


def test_multiturn_translations_keep_names_ids_and_metadata_private():
    messages = history()
    messages[0]["metadata"] = {"internal": True}
    assert normalize_messages(messages)[0]["metadata"] == {"internal": True}
    assert "metadata" not in openai_messages(messages)[0]
    anthropic = anthropic_messages(messages)
    assert anthropic[1]["content"][1]["id"] == "opaque_call_42"
    assert anthropic[2]["content"][0]["tool_use_id"] == "opaque_call_42"
    gemini = gemini_messages(messages)
    assert gemini[1]["parts"][1]["function_call"] == {
        "id": "opaque_call_42", "name": "read_file", "args": {"path": "a.py"},
    }
    assert gemini[2]["parts"][0]["function_response"]["name"] == "read_file"
    assert gemini[4]["parts"][0]["function_response"]["id"] == "opaque_call_43"


def test_native_batched_results_and_missing_gemini_ids_migrate_without_loss():
    native = [
        {"role": "model", "parts": [
            {"function_call": {"name": "read_file", "args": {"path": "a"}}},
            {"function_call": {"name": "read_file", "args": {"path": "b"}}},
        ]},
        {"role": "user", "parts": [
            {"function_response": {"name": "read_file", "response": {"result": "a"}}},
            {"function_response": {"name": "read_file", "response": {"result": "b"}}},
        ]},
    ]
    normalized = normalize_messages(native)
    calls = normalized[0]["tool_calls"]
    assert len(normalized) == 3
    assert normalized[1]["tool_call_id"] == calls[0]["id"]
    assert normalized[2]["tool_call_id"] == calls[1]["id"]
    assert calls[0]["id"] != calls[1]["id"]
    assert normalize_messages(normalized) == normalized
    assert json.loads(json.dumps(normalized)) == normalized


def test_real_anthropic_sdk_payload_is_json_safe_for_sqlite():
    types = pytest.importorskip("anthropic.types")
    native = {"role": "assistant", "content": [
        types.TextBlock(type="text", text="Inspecting"),
        types.ToolUseBlock(type="tool_use", id="toolu_real", name="read_file", input={"path": "a.py"}),
    ]}
    messages = normalize_messages([native, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_real", "content": "file text"},
    ]}])
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE history (messages TEXT)")
        db.execute("INSERT INTO history VALUES (?)", (json.dumps(messages),))
        stored = json.loads(db.execute("SELECT messages FROM history").fetchone()[0])
    assert stored[0]["tool_calls"][0]["id"] == stored[1]["tool_call_id"] == "toolu_real"
    assert gemini_messages(stored)[1]["parts"][0]["function_response"]["name"] == "read_file"


def test_real_gemini_sdk_payload_migrates_name_binding():
    types = pytest.importorskip("google.genai.types")
    native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(
        name="patch_file", args={"path": "a.py", "old": "{}", "new": "{nested}"},
    ))])
    messages = normalize_messages([native, {"role": "user", "parts": [
        {"function_response": {"name": "patch_file", "response": {"result": "done"}}},
    ]}])
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"]
    json.dumps(messages)


@pytest.mark.parametrize("name,args", [
    ("write_file", {"path": "a.py", "content": 'x = {"nested": {"escaped": "\\\"}"}}\nprint("}")'}),
    ("patch_file", {"path": "a.py", "old": 'a = {"x": "}"}', "new": 'a = {"x": {"y": "{\\""}}'}),
])
def test_json_tool_parser_preserves_nested_and_escaped_content(name, args):
    text = "Before\n" + json.dumps({"name": name, "arguments": args}) + "\nAfter"
    cleaned, calls = extract_json_tool_calls(text)
    assert cleaned == "Before\n\nAfter"
    assert calls[0].name == name
    assert calls[0].args == args
    assert calls[0].id


def test_parameterless_json_tool_call_accepts_null_arguments():
    cleaned, calls = extract_json_tool_calls('{"name":"git_status","arguments":null}')
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].name == "git_status"
    assert calls[0].args == {}


def fake_openai(create):
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "offline-test"
    provider._openai = NS(RateLimitError=type("SDKRateLimit", (Exception,), {}))
    provider.client = NS(chat=NS(completions=NS(create=create)))
    return provider


def stream_chunk(content=None, calls=None, usage=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=calls))], usage=usage)


def test_openai_stream_generates_stable_id_and_preserves_argument_deltas():
    chunks = [
        stream_chunk(calls=[NS(index=0, id=None, function=NS(name="write_file", arguments='{"path":"a.py",'))]),
        stream_chunk(calls=[NS(index=0, id=None, function=NS(name=None, arguments='"content":"{}"}'))]),
        NS(choices=[], usage=NS(prompt_tokens=12, completion_tokens=8)),
    ]
    provider = fake_openai(lambda **kwargs: iter(chunks))
    result = list(provider.stream(history(), [], "system"))[-1]
    assert result.tool_calls[0].id
    assert result.raw_assistant_message["tool_calls"][0]["id"] == result.tool_calls[0].id
    assert result.tool_calls[0].args == {"path": "a.py", "content": "{}"}
    assert (result.input_tokens, result.output_tokens) == (12, 8)


@pytest.mark.parametrize("streaming", [False, True])
def test_openai_json_text_fallback_is_shared_by_complete_and_stream(streaming):
    args = {"path": "a.py", "content": 'x = {"nested": "}\\\""}'}
    payload = json.dumps({"name": "write_file", "arguments": args})
    if streaming:
        provider = fake_openai(lambda **kwargs: iter([stream_chunk(payload[:12]), stream_chunk(payload[12:])]))
        result = list(provider.stream([], [], ""))[-1]
    else:
        response = NS(choices=[NS(message=NS(content=payload, tool_calls=[]))], usage=None)
        provider = fake_openai(lambda **kwargs: response)
        result = provider.complete([], [], "")
    assert result.text == ""
    assert result.tool_calls[0].args == args
    assert result.raw_assistant_message["content"] == ""


@pytest.mark.parametrize("streaming", [False, True])
def test_fallback_reformats_full_history_on_switch_and_recovery(monkeypatch, streaming):
    from nexus_agent_ai.providers.fallback_provider import FallbackProvider
    seen = []
    primary = NS(model="primary", complete=lambda messages, tools, system: (_ for _ in ()).throw(RateLimitError("primary")))
    secondary = NS(model="secondary", context_size=2048, is_local=True,
                   complete=lambda messages, tools, system: seen.append(gemini_messages(messages)) or ProviderResponse(text="ok"))
    primary.stream = lambda messages, tools, system: iter([primary.complete(messages, tools, system)])
    secondary.stream = lambda messages, tools, system: iter([secondary.complete(messages, tools, system)])
    monkeypatch.setattr(FallbackProvider, "_init_first", lambda self: None)
    fb = FallbackProvider("primary")
    fb._chain = [("primary", lambda: primary), ("secondary", lambda: secondary)]
    fb._current_name, fb._current_provider = "primary", primary
    attempts = []
    fb.before_request = lambda provider, messages, tools, system: attempts.append(provider.model) or (messages, system)
    def invoke():
        return list(fb.stream(history(), [], "system"))[-1] if streaming else fb.complete(history(), [], "system")
    result = invoke()
    assert result.text == "ok"
    assert attempts == ["primary", "secondary"]
    assert fb.is_local and fb.context_size == 2048
    assert seen[0][2]["parts"][0]["function_response"]["name"] == "read_file"
    primary.complete = lambda messages, tools, system: seen.append(anthropic_messages(messages)) or ProviderResponse(text="restored")
    fb._fallback_time = 0
    fb._recovery_cooldown = 0
    assert invoke().text == "restored"
    assert seen[-1][2]["content"][0]["tool_use_id"] == "opaque_call_42"
    assert not fb.is_local and fb.context_size is None


def test_fallback_budget_error_does_not_trigger_switch(monkeypatch):
    from nexus_agent_ai.providers.fallback_provider import FallbackProvider
    monkeypatch.setattr(FallbackProvider, "_init_first", lambda self: None)
    fb = FallbackProvider("primary")
    fb._current_name, fb._current_provider = "primary", NS(model="primary")
    def budget(*args):
        raise ValueError("context budget")
    fb.before_request = budget
    with pytest.raises(ValueError, match="context budget"):
        fb.complete([], [], "")
    assert fb._current_name == "primary"


def test_local_http_sse_yields_before_response_finishes(monkeypatch):
    release = threading.Event()
    finished = threading.Event()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n')
            self.wfile.flush()
            release.wait(3)
            self.wfile.write(b'data: {"choices":[{"delta":{"content":" second","tool_calls":[{"index":0,"function":{"name":"read_file","arguments":"{\\"path\\":\\"a.py\\"}"}}]}}],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\ndata: [DONE]\n\n')
            self.wfile.flush()
            finished.set()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = LocalProvider()
    provider._model_instance = "llama_server"
    provider._server_port = server.server_port
    monkeypatch.setattr(provider, "_ensure_loaded", lambda: None)
    chunks = provider.stream(history(), [], "system")
    try:
        assert next(chunks) == "first"
        assert not finished.is_set(), "stream() buffered the response instead of yielding live SSE"
        release.set()
        remaining = list(chunks)
        assert remaining[0] == " second"
        final = remaining[-1]
        assert final.text == "first second"
        assert final.tool_calls[0].id
        assert final.tool_calls[0].args == {"path": "a.py"}
        assert final.raw_assistant_message["tool_calls"][0]["id"] == final.tool_calls[0].id
        assert (final.input_tokens, final.output_tokens) == (9, 4)
        assert requests[0]["stream"] is True
    finally:
        release.set()
        chunks.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_local_foreign_health_listener_not_reused_or_stopped(tmp_path, monkeypatch):
    import urllib.request
    import nexus_agent_ai.providers.local_provider as module
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        def log_message(self, *args):
            pass
    foreign = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=foreign.serve_forever, daemon=True)
    thread.start()
    provider = LocalProvider(context_size=8192)
    provider._server_port = foreign.server_port
    commands = []
    owned = MagicMock()
    owned.poll.return_value = None
    monkeypatch.setattr(module, "_SERVER_DIR", tmp_path)
    monkeypatch.setattr(provider, "_download_llama_server", lambda: "offline-server")
    monkeypatch.setattr(module.subprocess, "Popen", lambda command, **kwargs: commands.append(command) or owned)
    monkeypatch.setattr("time.sleep", lambda _: None)
    try:
        assert not provider._is_server_alive()
        monkeypatch.setattr(provider, "_is_server_alive", lambda: provider._server_proc is owned)
        provider._start_llama_server(str(tmp_path / "selected.gguf"))
        assert provider._server_port != foreign.server_port
        assert commands[0][commands[0].index("-c") + 1] == "8192"
        assert commands[0][commands[0].index("-m") + 1] == str((tmp_path / "selected.gguf").resolve())
        provider.stop()
        assert owned.terminate.called
        with urllib.request.urlopen(f"http://127.0.0.1:{foreign.server_port}/health") as response:
            assert response.status == 200
    finally:
        provider.stop()
        foreign.shutdown()
        foreign.server_close()
        thread.join(timeout=3)


def test_local_owned_server_wrong_identity_not_healthy(monkeypatch):
    provider = LocalProvider()
    provider._server_proc = MagicMock()
    provider._server_proc.poll.return_value = None
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.read.return_value = b'{"data":[{"id":"wrong-model"}]}'
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    assert not provider._is_server_alive()
    provider.stop()


def test_optional_engine_does_not_fake_token_chunks(monkeypatch):
    provider = LocalProvider()
    provider._model_instance = object()
    monkeypatch.setattr(provider, "_ensure_loaded", lambda: None)
    response = ProviderResponse(text="complete answer")
    monkeypatch.setattr(provider, "complete", lambda *args: response)
    assert list(provider.stream([], [], "")) == [response]


def test_gpu_inference_uses_json_decoder_for_real_file_arguments(monkeypatch):
    args = {"path": "a.py", "content": 'mapping = {"outer": {"inner": "}"}}'}
    provider = LocalProvider()
    provider._model_instance = NS(device="cpu", generate=lambda **kwargs: [[1, 2, 3]])
    class TokenRows(list):
        shape = (1, 2)
    class Inputs(dict):
        input_ids = TokenRows([[1, 2]])
        def to(self, device):
            return self
    inputs = Inputs(input_ids=[[1, 2]])
    tokenizer = MagicMock()
    tokenizer.return_value = inputs
    tokenizer.batch_decode.return_value = [json.dumps({"name": "write_file", "arguments": args})]
    provider._tokenizer = tokenizer
    monkeypatch.setattr(provider, "_ensure_loaded", lambda: None)
    response = provider.complete([], [], "")
    assert response.tool_calls[0].args == args
    assert response.text == ""
    assert json.loads(response.raw_assistant_message["tool_calls"][0]["function"]["arguments"]) == args


def test_startup_interrupt_stops_only_owned_process(tmp_path, monkeypatch):
    import nexus_agent_ai.providers.local_provider as module
    provider = LocalProvider()
    owned = MagicMock()
    owned.poll.return_value = None
    monkeypatch.setattr(module, "_SERVER_DIR", tmp_path)
    monkeypatch.setattr(provider, "_download_llama_server", lambda: "offline-server")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: owned)
    def interrupt(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr("time.sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        provider._start_llama_server(str(tmp_path / "model.gguf"))
    assert owned.terminate.called
    assert provider._server_proc is None
    assert provider._server_log is None


def test_parser_keeps_non_tool_text_unchanged():
    text = '  Plain response with {"nested": {"x": "}"}}.\\n'
    assert extract_json_tool_calls(text) == (text, [])


def test_explicit_model_verification_repairs_corrupt_cached_weights(tmp_path, monkeypatch):
    import hashlib
    from nexus_agent_ai.utils import model_download

    expected = b"verified fixture model"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("X-Linked-Size", str(len(expected)))
            self.send_header("X-Linked-ETag", hashlib.sha256(expected).hexdigest())
            self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(expected)))
            self.end_headers()
            self.wfile.write(expected)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(tmp_path / "models-home"))
    monkeypatch.setattr(model_download, "HF_BASE", f"http://127.0.0.1:{server.server_port}")
    target = model_download.local_path("fixture/model", "weights.gguf")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted")
    provider = LocalProvider(model_id="fixture/model", filename="weights.gguf")
    try:
        result = provider.setup_model(verify_download=True)
        assert Path(result).read_bytes() == expected
        assert target.read_bytes() == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
