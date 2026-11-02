"""Offline SDK boundaries exercise configured budgets through the real Agent."""
import json
import threading
import sys
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace as NS

import pytest

from nexus_agent_ai.agent import context, core
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.memory import ConversationMemory
from nexus_agent_ai.providers.anthropic_provider import AnthropicProvider
from nexus_agent_ai.providers.base import Tool
from nexus_agent_ai.providers.gemini_provider import GeminiProvider
from nexus_agent_ai.providers.openai_provider import OpenAIProvider


@pytest.fixture(autouse=True)
def clear_limit_environment(monkeypatch):
    for name in ("NEXUS_CONTEXT_SIZE", "NEXUS_MAX_OUTPUT_TOKENS", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "offline-key")


def cloud(monkeypatch, kind, *, cutoff=False, base_url=None, **limits):
    """Strict fake SDK endpoints: a tiny generation budget cuts off a tool batch."""
    requests = []
    rate_limit = type("SDKRateLimit", (Exception,), {})

    if kind == "openai":
        field = "max_completion_tokens" if not base_url or "api.openai.com" in base_url else "max_tokens"

        def create(**request):
            cap = request[field]
            assert ("max_tokens" if field == "max_completion_tokens" else "max_completion_tokens") not in request
            requests.append(request)
            used = cap if cutoff else 5
            calls = [NS(id="valid", function=NS(name="write_file", arguments='{"path":"safe.py","content":"x"}')),
                     NS(id="partial", function=NS(name="write_file", arguments='{"path":"unfinished'))] if cutoff else []
            reason = "length" if cutoff else "stop"
            usage = NS(prompt_tokens=7, completion_tokens=used)
            if not request.get("stream"):
                return NS(choices=[NS(message=NS(content="partial" if cutoff else "ok", tool_calls=calls),
                                      finish_reason=reason)], usage=usage)
            deltas = [NS(index=index, id=call.id, function=call.function) for index, call in enumerate(calls)]
            return iter([
                NS(choices=[NS(delta=NS(content="partial" if cutoff else "ok", tool_calls=deltas), finish_reason=None)], usage=None),
                NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason=reason)], usage=None),
                NS(choices=[], usage=usage),
            ])

        monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=lambda **kwargs: NS(chat=NS(completions=NS(create=create))), RateLimitError=rate_limit))
        provider = OpenAIProvider(base_url=base_url, **limits)
    elif kind == "anthropic":
        def message(request):
            cap = request["max_tokens"]
            requests.append(request)
            blocks = [NS(type="text", text="partial" if cutoff else "ok")]
            if cutoff:
                blocks += [NS(type="tool_use", id="valid", name="write_file", input={"path": "safe.py", "content": "x"}),
                           NS(type="tool_use", id="partial", name="write_file", input='{"path":"unfinished')]
            return NS(content=blocks, stop_reason="max_tokens" if cutoff else "end_turn",
                      usage=NS(input_tokens=7, output_tokens=cap if cutoff else 5))

        def stream(**request):
            final = message(request)
            return nullcontext(NS(text_stream=iter([final.content[0].text]), get_final_message=lambda: final))

        monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=lambda **kwargs: NS(messages=NS(stream=stream)), RateLimitError=rate_limit))
        provider = AnthropicProvider(**limits)
    else:
        def message(request):
            cap = request["config"].max_output_tokens
            requests.append(request)
            parts = [NS(text="partial" if cutoff else "ok", function_call=None)]
            if cutoff:
                parts += [NS(text=None, function_call=NS(id="valid", name="write_file", args={"path": "safe.py", "content": "x"})),
                          NS(text=None, function_call=NS(id="partial", name="write_file", args='{"path":"unfinished'))]
            return NS(text=parts[0].text,
                      candidates=[NS(content=NS(parts=parts), finish_reason="MAX_TOKENS" if cutoff else "STOP")],
                      usage_metadata=NS(prompt_token_count=7, candidates_token_count=cap if cutoff else 5))

        genai = ModuleType("google.genai")
        genai.types = NS(GenerateContentConfig=lambda **kwargs: NS(**kwargs),
                         FunctionDeclaration=lambda **kwargs: NS(**kwargs), Tool=lambda **kwargs: NS(**kwargs))
        genai.Client = lambda **kwargs: NS(models=NS(generate_content=lambda **request: message(request),
                                                    generate_content_stream=lambda **request: iter([message(request)])))
        google = ModuleType("google")
        google.genai = genai
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        provider = GeminiProvider(**limits)
    return provider, requests


def sent_messages(kind, request):
    if kind == "gemini":
        return [{"role": "assistant" if item["role"] == "model" else item["role"],
                 "content": "".join(part["text"] for part in item["parts"])} for item in request["contents"]]
    if kind == "anthropic":
        return [{"role": item["role"], "content": "".join(block["text"] for block in item["content"])}
                for item in request["messages"]]
    return [message for message in request["messages"] if message["role"] != "system"]


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
def test_nondefault_cloud_cap_stops_entire_native_batch_before_dispatch(monkeypatch, kind, stream):
    provider, requests = cloud(monkeypatch, kind, cutoff=True, context_size=8192, max_output_tokens=37)
    monkeypatch.setattr(core, "execute_tool", lambda *args: pytest.fail("a cutoff must not dispatch even the complete first call"))
    tool = Tool("write_file", "Write a file", {"type": "object"}, lambda **kwargs: pytest.fail("not dispatched"))
    agent = Agent(provider, tools=[tool], verbose=False)
    result = agent.run("write the requested artifact", stream=stream)
    assert len(requests) == 1
    assert agent.response_incomplete and "incomplete" in result.lower()
    assert agent.changed_files == set()
    assert not any(message["role"] == "tool" or message.get("tool_calls") for message in agent.memory.get())
    assert (agent.total_input_tokens, agent.total_output_tokens) == (7, 37)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
def test_cloud_context_fits_full_input_plus_actual_output_cap(monkeypatch, kind, stream):
    sent_sizes = []
    for cap in (512, 4096):
        provider, requests = cloud(monkeypatch, kind, context_size=8192, max_output_tokens=cap)
        memory = ConversationMemory(max_messages=500)
        memory.replace([{"role": role, "content": f"old {index} " + "x" * 600}
                        for index in range(50) for role in ("user", "assistant")])
        agent = Agent(provider, memory=memory, tools=[], verbose=False)
        agent.system = "Short test instructions."
        assert agent.run("explain this", stream=stream) == "ok"
        sent = sent_messages(kind, requests[0])
        assert sent[-1]["content"] == "explain this"
        assert context.history_tokens(sent) + context.estimate_tokens(agent.system) + cap <= 8192
        assert len(memory.get()) == 102
        assert (agent.total_input_tokens, agent.total_output_tokens) == (7, 5)
        sent_sizes.append(len(sent))
    assert sent_sizes[0] > sent_sizes[1]


@pytest.mark.parametrize("base_url,local,context_size,max_output", [
    (None, False, 128000, 16384),
    ("https://api.openai.com/v1", False, 128000, 16384),
    ("http://localhost:1234/v1", True, 4096, 2048),
    ("http://127.0.0.2:8000/v1", True, 4096, 2048),
    ("http://[::1]:8000/v1", True, 4096, 2048),
    ("https://compatible.example/v1", False, 128000, 16384),
])
@pytest.mark.parametrize("stream", [False, True])
def test_openai_official_and_compatible_endpoints_use_supported_caps(monkeypatch, base_url, local, context_size, max_output, stream):
    provider, requests = cloud(monkeypatch, "openai", base_url=base_url)
    assert provider.is_local is local
    assert (provider.context_size, provider.max_output_tokens) == (context_size, max_output)
    agent = Agent(provider, tools=[], verbose=False)
    agent.system = "Short instructions."
    assert agent.run("hello", stream=stream) == "ok"
    assert len(requests) == 1
    assert agent.total_tokens == 12


def test_openai_environment_base_url_gets_local_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    provider, _ = cloud(monkeypatch, "openai", base_url="http://localhost:8000/v1", context_size=16384, max_output_tokens=6000)
    assert (provider.context_size, provider.max_output_tokens) == (16384, 6000)
    agent = Agent(provider, tools=[], verbose=False)
    assert agent.run("hello") == "ok"
    assert agent.output_reserve == 6000
    # Constructor must also classify an environment-only URL before resolving limits.
    provider = OpenAIProvider()
    assert provider.is_local
    assert (provider.context_size, provider.max_output_tokens) == (4096, 2048)


def fallback(monkeypatch):
    """Use the built-in lazy factories, including a fresh primary on recovery."""
    from nexus_agent_ai.providers import fallback_provider

    cloud(monkeypatch, "openai")
    cloud(monkeypatch, "gemini")
    genai = sys.modules["google.genai"]
    create_client = genai.Client
    failing = [True]

    def client(**kwargs):
        sdk = create_client(**kwargs)
        for name in ("generate_content", "generate_content_stream"):
            original = getattr(sdk.models, name)

            def request(_original=original, **kwargs):
                if failing[0]:
                    failing[0] = False
                    raise RuntimeError("primary temporarily unavailable")
                return _original(**kwargs)

            setattr(sdk.models, name, request)
        return sdk

    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setattr(fallback_provider, "FALLBACK_CHAIN", [
        ("gemini", fallback_provider._make_gemini),
        ("openai", fallback_provider._make_openai),
    ])
    return fallback_provider.FallbackProvider


@pytest.mark.parametrize("stream", [False, True])
def test_fallback_updates_real_cloud_budgets_and_preserves_usage_on_recovery(monkeypatch, stream):
    provider = fallback(monkeypatch)("gemini")
    agent = Agent(provider, tools=[], verbose=False)
    agent.system = "Short instructions."
    assert (agent.context_window, agent.output_reserve) == (1048576, 65536)
    assert agent.run("hello", stream=stream) == "ok"
    assert (provider.context_size, provider.max_output_tokens) == (128000, 16384)
    assert (agent.context_window, agent.output_reserve) == (128000, 16384)
    assert (agent.total_input_tokens, agent.total_output_tokens) == (7, 5)
    provider._fallback_time = 0
    provider._recovery_cooldown = 0
    assert agent.run("continue", stream=stream) == "ok"
    assert (provider.context_size, provider.max_output_tokens) == (1048576, 65536)
    assert (agent.context_window, agent.output_reserve) == (1048576, 65536)
    assert (agent.total_input_tokens, agent.total_output_tokens) == (14, 10)
    assert provider.before_request is None


@pytest.mark.parametrize("overrides,expected", [
    ({}, (8192, 1024)),
    ({"max_output_tokens": 3072}, (8192, 3072)),
    ({"context_size": 16384, "max_output_tokens": 2048}, (16384, 2048)),
])
def test_fallback_constructor_overrides_and_environment_survive_all_activations(monkeypatch, overrides, expected):
    constructor = fallback(monkeypatch)
    monkeypatch.setenv("NEXUS_CONTEXT_SIZE", "8192")
    monkeypatch.setenv("NEXUS_MAX_OUTPUT_TOKENS", "1024")
    provider = constructor("gemini", **overrides)
    assert (provider.context_size, provider.max_output_tokens) == expected
    agent = Agent(provider, tools=[], verbose=False)
    agent.system = "Short instructions."
    assert agent.run("hello") == "ok"
    assert provider._current_name == "openai"
    assert (provider.context_size, provider.max_output_tokens) == expected
    provider._fallback_time = 0
    provider._recovery_cooldown = 0
    assert agent.run("continue") == "ok"
    assert provider._current_name == "gemini"
    assert (provider.context_size, provider.max_output_tokens) == expected
    assert agent.total_tokens == 24


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_default_large_cap_reaches_real_sdk_without_nonstreaming_guard(monkeypatch, stream):
    pytest.importorskip("anthropic")
    requests = []
    events = [
        {"type": "message_start", "message": {
            "id": "msg_offline", "type": "message", "role": "assistant", "content": [],
            "model": "claude-sonnet-4-6", "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    transcript = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(transcript)))
            self.end_headers()
            self.wfile.write(transcript)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    provider = None
    try:
        provider = AnthropicProvider()
        agent = Agent(provider, tools=[], verbose=False)
        assert agent.run("hello", stream=stream) == "ok"
        assert (agent.total_input_tokens, agent.total_output_tokens) == (7, 5)
        assert len(requests) == 1
        assert requests[0]["stream"] is True
        assert requests[0]["max_tokens"] == provider.max_output_tokens == 128000
    finally:
        if provider is not None:
            provider.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_fallback_reapplies_original_overrides_against_each_model_maximum(monkeypatch):
    constructor = fallback(monkeypatch)
    with pytest.warns(UserWarning):
        provider = constructor("gemini", context_size=1000000, max_output_tokens=100000)
    assert (provider.context_size, provider.max_output_tokens) == (1000000, 65536)
    agent = Agent(provider, tools=[], verbose=False)
    agent.system = "Short instructions."
    with pytest.warns(UserWarning):
        assert agent.run("hello") == "ok"
    assert (provider.context_size, provider.max_output_tokens) == (128000, 16384)
    provider._fallback_time = 0
    provider._recovery_cooldown = 0
    with pytest.warns(UserWarning):
        assert agent.run("continue") == "ok"
    assert (provider.context_size, provider.max_output_tokens) == (1000000, 65536)
    assert agent.total_tokens == 24
