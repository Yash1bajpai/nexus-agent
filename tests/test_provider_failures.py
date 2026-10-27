"""Offline native transcripts: reject malformed calls and preserve stop status."""
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from nexus_agent_ai.providers.anthropic_provider import AnthropicProvider
from nexus_agent_ai.providers.base import ProviderResponse
from nexus_agent_ai.providers.gemini_provider import GeminiProvider
from nexus_agent_ai.providers.local_provider import LocalProvider
from nexus_agent_ai.providers.openai_provider import OpenAIProvider


BACKENDS = ["llama_cpp", "llama_server", "openai_complete", "openai_stream"]


def wire_call(arguments="", name="write_file", call_id="native_1"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def native_response(monkeypatch, backend, calls, finish_reason="tool_calls", received=None, content=""):
    """Use fake libraries/SSE bytes, never a model or network connection."""
    if backend == "llama_cpp":
        response = {"choices": [{"message": {"content": content, "tool_calls": calls},
                                 "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 5}}
        provider = LocalProvider.__new__(LocalProvider)
        provider._llama = NS(create_chat_completion=lambda **kwargs: response)
        return provider._run_llama_cpp([], [], "")

    # Split both names and arguments across deltas to exercise real assembly.
    deltas = []
    for index, call in enumerate(calls):
        function = call["function"]
        name = function.get("name", "")
        arguments = function.get("arguments", "")
        name_parts = (name[:len(name) // 2], name[len(name) // 2:]) if isinstance(name, str) else (name, "")
        arg_parts = (arguments[:len(arguments) // 2], arguments[len(arguments) // 2:]) if isinstance(arguments, str) else (arguments, "")
        deltas.extend([
            {"index": index, "id": call["id"], "function": {
                "name": name_parts[0], "arguments": arg_parts[0]}},
            {"index": index, "function": {
                "name": name_parts[1], "arguments": arg_parts[1]}},
        ])

    if backend == "llama_server":
        events = [{"choices": [{"delta": {"tool_calls": [delta]}}]} for delta in deltas]
        events.insert(0, {"choices": [{"delta": {"content": content}}]})
        events.extend([
            {"choices": [{"delta": {}, "finish_reason": finish_reason}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5}},
        ])
        transcript = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        transcript += "data: [DONE]\n\n"
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(transcript.encode()))
        provider = LocalProvider.__new__(LocalProvider)
        provider._server_alias = "offline-test"
        provider._server_port = 1
        stream = provider._stream_via_server([], [], "")
    else:
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.model = "offline-test"
        provider._openai = NS(RateLimitError=type("SDKRateLimit", (Exception,), {}))
        if backend == "openai_complete":
            native_calls = [NS(id=c["id"], function=NS(**c["function"])) for c in calls]
            response = NS(choices=[NS(message=NS(content=content, tool_calls=native_calls),
                                      finish_reason=finish_reason)],
                          usage=NS(prompt_tokens=7, completion_tokens=5))
            provider.client = NS(chat=NS(completions=NS(create=lambda **kwargs: response)))
            return provider.complete([], [], "")
        chunks = [NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(
            index=d["index"], id=d.get("id"), function=NS(**d["function"]))]),
            finish_reason=None)], usage=None) for d in deltas]
        chunks.insert(0, NS(choices=[NS(delta=NS(content=content, tool_calls=None), finish_reason=None)], usage=None))
        chunks.extend([
            NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason=finish_reason)], usage=None),
            NS(choices=[], usage=NS(prompt_tokens=7, completion_tokens=5)),
        ])
        provider.client = NS(chat=NS(completions=NS(create=lambda **kwargs: iter(chunks))))
        stream = provider.stream([], [], "")

    final = None
    for chunk in stream:
        if isinstance(chunk, ProviderResponse):
            if received is not None:
                received.append(chunk)
            final = chunk
    assert final is not None
    return final


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("arguments", [
    '{"content":"private-payload",', "[1, 2]", '"private-payload"', "1", "true", "null",
], ids=["incomplete-json", "array", "string", "number", "boolean", "null"])
def test_native_bad_arguments_reject_whole_response(monkeypatch, backend, arguments):
    received = []
    # Even the valid first call must not escape in a response when a later call is bad.
    calls = [wire_call('{}', call_id="valid_first"), wire_call(arguments)]
    with pytest.raises(ValueError, match="JSON object") as error:
        native_response(monkeypatch, backend, calls, received=received)
    assert "private-payload" not in str(error.value)
    assert received == []


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["", "   "])
def test_native_incomplete_name_is_not_silently_dropped(monkeypatch, backend, name):
    with pytest.raises(ValueError, match="function name"):
        native_response(monkeypatch, backend, [wire_call('{}', name=name)])


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_valid_nested_arguments_survive_fragmentation(monkeypatch, backend):
    args = {"path": "a.py", "content": 'value = {"nested": "}\\\""}\n',
            "options": {"items": [1, {"escaped": "\\path"}], "enabled": True}}
    response = native_response(monkeypatch, backend, [wire_call(json.dumps(args))])
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.id, call.name, call.args) == ("native_1", "write_file", args)
    assert json.loads(response.raw_assistant_message["tool_calls"][0]["function"]["arguments"]) == args
    assert (response.input_tokens, response.output_tokens) == (7, 5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("arguments", ["", "  ", "{}"])
def test_native_parameterless_empty_arguments_remain_supported(monkeypatch, backend, arguments):
    response = native_response(monkeypatch, backend, [wire_call(arguments, name="git_status")])
    assert response.tool_calls[0].args == {}


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_missing_parameterless_arguments_remain_supported(monkeypatch, backend):
    response = native_response(monkeypatch, backend, [{"id": "native_1", "function": {"name": "git_status"}}])
    assert response.tool_calls[0].args == {}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reason", ["length", "stop", None, 1])
def test_native_finish_reason_reaches_response(monkeypatch, backend, reason):
    response = native_response(monkeypatch, backend, [wire_call('{}')], finish_reason=reason)
    assert response.finish_reason == (reason if isinstance(reason, str) else None)
    if reason == "length":
        assert response.tool_calls == []
    else:
        assert response.tool_calls[0].id == "native_1"




@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("arguments,name", [
    ('{"content":"unfinished', "write_file"), ("{}", ""), ("null", "write_file"),
    ([1], "write_file"), ("{}", 1), (None, None),
])
def test_length_preserves_status_without_decoding_native_calls(monkeypatch, backend, arguments, name):
    response = native_response(
        monkeypatch, backend, [wire_call(arguments, name=name)], finish_reason="length",
    )
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert "tool_calls" not in response.raw_assistant_message
    assert (response.input_tokens, response.output_tokens) == (7, 5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("content", [
    '{"name":"write_file","arguments":{"path":"a.py"}}',
    '{"name":"write_file","arguments":{"path":"unfinished',
])
def test_length_never_extracts_json_text_calls(monkeypatch, backend, content):
    response = native_response(monkeypatch, backend, [], finish_reason="length", content=content)
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert response.text == content
    assert "tool_calls" not in response.raw_assistant_message


def object_native_response(backend, arguments=None, function_call=None):
    """Exercise native object providers without importing their optional SDKs."""
    if backend.startswith("anthropic"):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.model = "offline-test"
        provider._anthropic = NS(RateLimitError=type("SDKRateLimit", (Exception,), {}))
        message = NS(content=[NS(type="tool_use", id="native_1", name="write_file", input=arguments)],
                     usage=NS(input_tokens=7, output_tokens=5))
        stream = NS(text_stream=iter(()), get_final_message=lambda: message)
        provider.client = NS(messages=NS(create=lambda **kwargs: message,
                                         stream=lambda **kwargs: nullcontext(stream)))
    else:
        provider = GeminiProvider.__new__(GeminiProvider)
        provider.model = "offline-test"
        provider._types = NS(GenerateContentConfig=lambda **kwargs: NS(**kwargs))
        if function_call is None:
            function_call = NS(id="native_1", name="write_file", args=arguments)
        response = NS(candidates=[NS(content=NS(parts=[NS(text=None, function_call=function_call)]))],
                      usage_metadata=None)
        provider.client = NS(models=NS(generate_content=lambda **kwargs: response,
                                        generate_content_stream=lambda **kwargs: iter([response])))
    if backend.endswith("stream"):
        return list(provider.stream([], [], ""))[-1]
    return provider.complete([], [], "")


@pytest.mark.parametrize("backend", ["anthropic_complete", "anthropic_stream", "gemini_complete", "gemini_stream"])
@pytest.mark.parametrize("arguments", [[], [1], False, 1, "null", "[]", "1"])
def test_object_native_providers_reject_nonobjects(backend, arguments):
    with pytest.raises(ValueError, match="JSON object"):
        object_native_response(backend, arguments)


@pytest.mark.parametrize("backend", ["anthropic_complete", "anthropic_stream", "gemini_complete", "gemini_stream"])
def test_object_native_providers_preserve_objects(backend):
    args = {"path": "a.py", "nested": {"items": ["}", "\\\""]}}
    assert object_native_response(backend, args).tool_calls[0].args == args


@pytest.mark.parametrize("backend", ["anthropic_complete", "anthropic_stream"])
def test_anthropic_explicit_null_arguments_are_invalid(backend):
    with pytest.raises(ValueError, match="JSON object"):
        object_native_response(backend, None)


@pytest.mark.parametrize("backend", ["gemini_complete", "gemini_stream"])
def test_gemini_omitted_parameterless_sdk_arguments_remain_supported(backend):
    try:
        from google.genai.types import FunctionCall
    except ImportError:
        # Faithful SDK omission: the optional args attribute defaults to None.
        function_call = NS(name="git_status", args=None, id=None)
    else:
        function_call = FunctionCall(name="git_status")
    response = object_native_response(backend, function_call=function_call)
    assert response.tool_calls[0].name == "git_status"
    assert response.tool_calls[0].args == {}
