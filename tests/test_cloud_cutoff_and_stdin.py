"""Offline: Anthropic/Gemini token-limit stops never yield tool calls; redirected stdin is not interactive."""
import sys
from types import SimpleNamespace as NS

import pytest

from nexus_agent_ai.providers.anthropic_provider import AnthropicProvider
from nexus_agent_ai.providers.base import ProviderResponse
from nexus_agent_ai.providers.gemini_provider import GeminiProvider
from nexus_agent_ai.utils import interactive


def _drain(stream):
    final = None
    for item in stream:
        if isinstance(item, ProviderResponse):
            final = item
    assert final is not None
    return final


def _anthropic(stop_reason, stream, arguments=None):
    message = NS(
        content=[NS(type="text", text="Writing the file."),
                 NS(type="tool_use", name="write_file", input={"path": "a.py", "content": "x"} if arguments is None else arguments, id="tu_1")],
        usage=NS(input_tokens=7, output_tokens=5), stop_reason=stop_reason)

    class Stream:
        text_stream = iter(["Writing the file."])
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self): return message

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "offline"
    provider.context_size = 8192
    provider.max_output_tokens = 3072
    provider._anthropic = NS(RateLimitError=type("RL", (Exception,), {}))
    provider.client = NS(messages=NS(stream=lambda **kw: Stream()))
    return _drain(provider.stream([], [], "")) if stream else provider.complete([], [], "")


def _gemini(finish_reason, stream, arguments=None, chunks=None):
    call = NS(name="write_file", args={"path": "a.py", "content": "x"} if arguments is None else arguments, id=None)
    parts = [NS(text="Writing the file.", function_call=None), NS(text=None, function_call=call)]
    candidate = NS(content=NS(parts=parts), finish_reason=finish_reason)
    usage = NS(prompt_token_count=7, candidates_token_count=5)
    response = NS(candidates=[candidate], usage_metadata=usage, text="Writing the file.", prompt_feedback=None)
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model = "offline"
    provider.context_size = 8192
    provider.max_output_tokens = 3072
    provider._types = NS(GenerateContentConfig=lambda **kw: NS(**kw))
    provider._format_messages = lambda messages: []
    provider._convert_tools = lambda tools: []
    provider.client = NS(models=NS(generate_content=lambda **kw: response,
                                   generate_content_stream=lambda **kw: iter([response] if chunks is None else chunks)))
    return _drain(provider.stream([], [], "")) if stream else provider.complete([], [], "")


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_max_tokens_blocks_tool_calls(stream):
    response = _anthropic("max_tokens", stream)
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert "tool_calls" not in response.raw_assistant_message
    assert response.text == "Writing the file."


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_normal_tool_use_still_works(stream):
    response = _anthropic("tool_use", stream)
    assert response.finish_reason == "tool_use"
    assert [c.name for c in response.tool_calls] == ["write_file"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", [NS(name="MAX_TOKENS"), "MAX_TOKENS", "FinishReason.MAX_TOKENS"])
def test_gemini_max_tokens_blocks_tool_calls(stream, reason):
    response = _gemini(reason, stream)
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert "tool_calls" not in response.raw_assistant_message


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", [NS(name="STOP"), None])
def test_gemini_normal_function_call_still_works(stream, reason):
    response = _gemini(reason, stream)
    assert response.finish_reason in ("stop", None)
    assert [c.name for c in response.tool_calls] == ["write_file"]


class _Stdin:
    def __init__(self, tty): self._tty = tty
    def isatty(self): return self._tty
    def fileno(self): return 0


def test_non_tty_stdin_is_not_interactive(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _Stdin(False))
    assert interactive.stdin_is_interactive() is False
    monkeypatch.setattr(sys, "stdin", None)
    assert interactive.stdin_is_interactive() is False


def test_windows_nul_device_is_not_interactive(monkeypatch):
    # Windows reports isatty() True for NUL; the console-mode check must override it.
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    monkeypatch.setattr(interactive, "os", NS(name="nt"))
    monkeypatch.setattr(interactive, "_is_windows_console", lambda stream: False)
    assert interactive.stdin_is_interactive() is False
    monkeypatch.setattr(interactive, "_is_windows_console", lambda stream: True)
    assert interactive.stdin_is_interactive() is True


def test_first_run_skips_wizard_when_stdin_not_a_console(monkeypatch, tmp_path):
    from nexus_agent_ai.cli import onboarding
    monkeypatch.setattr(onboarding, "INIT_FILE", tmp_path / "init")
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    monkeypatch.setattr(interactive, "os", NS(name="nt"))
    monkeypatch.setattr(interactive, "_is_windows_console", lambda stream: False)
    monkeypatch.setattr(onboarding, "run_full_setup", lambda: pytest.fail("wizard must not run"))
    assert onboarding.run_if_first_time() is None
    assert not (tmp_path / "init").exists()


@pytest.mark.parametrize("provider", [_anthropic, _gemini])
@pytest.mark.parametrize("stream", [False, True])
def test_cutoff_does_not_decode_truncated_arguments(provider, stream):
    reason = "max_tokens" if provider is _anthropic else "MAX_TOKENS"
    response = provider(reason, stream, arguments='{"path": "unfinished')
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert "tool_calls" not in response.raw_assistant_message


@pytest.mark.parametrize("arguments", [{"path": "a.py", "content": "x"}, '{"path": "unfinished'])
@pytest.mark.parametrize("trailing_reason", [None, "STOP"])
def test_gemini_stream_retains_late_cutoff(arguments, trailing_reason):
    call = NS(name="write_file", args=arguments, id=None)
    chunks = [
        NS(text=None, candidates=[NS(finish_reason=None, content=NS(parts=[
            NS(function_call=call),
        ]))]),
        NS(text=None, candidates=[NS(finish_reason="MAX_TOKENS", content=None)]),
        NS(text=None, candidates=[NS(finish_reason=trailing_reason, content=None)]),
    ]
    response = _gemini(None, True, chunks=chunks)
    assert response.finish_reason == "length"
    assert response.tool_calls == []
    assert "tool_calls" not in response.raw_assistant_message


def test_windows_nul_refuses_unapproved_execution_without_prompt(monkeypatch):
    from nexus_agent_ai.agent import consent
    monkeypatch.delenv("NEXUS_ALLOW_PROJECT_EXECUTION", raising=False)
    monkeypatch.setattr(consent, "_session_allowed", set())
    monkeypatch.setattr(consent, "_session_denied", set())
    monkeypatch.setattr(consent, "list_trusted", lambda: [])
    monkeypatch.setattr(consent, "_ask", lambda _: pytest.fail("must not prompt"))
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    monkeypatch.setattr(interactive, "os", NS(name="nt"))
    monkeypatch.setattr(interactive, "_is_windows_console", lambda stream: False)
    allowed, reason = consent.execution_allowed()
    assert not allowed and "not approved" in reason
    assert not consent.offer_run_tests()


def test_download_confirmation_keeps_original_isatty_policy(monkeypatch):
    from nexus_agent_ai.utils import model_select
    monkeypatch.delenv("NEXUS_AGENT_ASSUME_YES", raising=False)
    monkeypatch.setattr(sys, "stdin", _Stdin(True))
    monkeypatch.setattr(interactive, "os", NS(name="nt"))
    monkeypatch.setattr(interactive, "_is_windows_console", lambda stream: False)
    prompts = []
    def decline(prompt):
        prompts.append(prompt)
        return "n"
    choice = dict(model_select.SMALL_MODEL, reason="test")
    result = model_select.confirm_download(
        choice, ask=decline, speed_probe=lambda *args: None, out=lambda _: None,
    )
    assert result is None
    assert len(prompts) == 1
