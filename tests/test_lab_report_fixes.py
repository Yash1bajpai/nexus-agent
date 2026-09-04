"""Regression tests for the college-lab report fixes (v2.7.0)."""
import json

import pytest

from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, Tool, ToolCall
from nexus_agent_ai.agent.core import Agent


# ── Fix #1: tool-call infinite loop → forced final answer ──────────────────

class LoopingProvider(BaseProvider):
    """Mimics a stuck local model: repeats the same tool call forever."""

    def __init__(self):
        self.calls = 0
        self.saw_empty_tools = False

    def complete(self, messages, tools, system):
        self.calls += 1
        if not tools:
            self.saw_empty_tools = True
            return ProviderResponse(text="Here is your answer: reversed string")
        return ProviderResponse(
            text="",
            tool_calls=[ToolCall(id="call_1", name="read_file", args={"path": "string"})],
            raw_assistant_message={"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": "string"})}}
            ]},
        )

    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}

    def stream(self, messages, tools, system):
        res = self.complete(messages, tools, system)
        if res.text:
            yield res.text
        yield res


def test_tool_loop_forces_final_answer(monkeypatch):
    monkeypatch.setattr("nexus_agent_ai.agent.core.execute_tool", lambda name, args: "file content")
    prov = LoopingProvider()
    agent = Agent(provider=prov, verbose=False, max_iterations=10)
    out = agent.run("Write a python function that reverses a string", stream=False)
    assert "Max tool iterations reached" not in out
    assert "reversed string" in out
    # Tools must have been stripped to force the direct answer
    assert prov.saw_empty_tools is True
    # Must not need all 10 iterations to break the loop
    assert prov.calls <= 5


# ── Fix #4: pinned digest table + correct Linux asset name ─────────────────

def test_llama_server_digest_table_covers_all_platforms(monkeypatch):
    import platform as _plat
    from nexus_agent_ai.providers import local_provider as lp

    cases = [
        ("windows", "amd64", "llama-b7075-bin-win-cpu-x64.zip"),
        ("windows", "arm64", "llama-b7075-bin-win-cpu-arm64.zip"),
        ("darwin", "arm64", "llama-b7075-bin-macos-arm64.zip"),
        ("darwin", "x86_64", "llama-b7075-bin-macos-x64.zip"),
        ("linux", "x86_64", "llama-b7075-bin-ubuntu-x64.zip"),
    ]
    for system, machine, asset in cases:
        monkeypatch.setattr(_plat, "system", lambda s=system: s)
        monkeypatch.setattr(_plat, "machine", lambda m=machine: m)
        url, _ = lp._get_llama_server_info()
        assert url.endswith(asset), f"{system}/{machine}: {url}"
        digest = lp._LLAMA_SERVER_SHA256.get(asset)
        assert digest and len(digest) == 64, f"missing pinned digest for {asset}"
        assert all(c in "0123456789abcdef" for c in digest)


def test_no_stale_linux_asset_name():
    """b7075 has no linux-x64 asset — the old URL 404s."""
    from nexus_agent_ai.providers import local_provider as lp
    assert "linux-x64" not in json.dumps(lp._LLAMA_SERVER_SHA256)


# ── Fix #5: fatal error must be short and single-line friendly ─────────────

def test_no_engine_error_is_compact():
    from nexus_agent_ai.providers.local_provider import LocalProvider
    prov = LocalProvider.__new__(LocalProvider)
    prov._model_instance = "cpu_ollama_or_fallback"
    prov._llama = None
    with pytest.raises(RuntimeError) as excinfo:
        prov.complete([{"role": "user", "content": "hi"}], [], "sys")
    msg = str(excinfo.value)
    assert "━" not in msg
    assert len(msg) < 500


# ── Fix #6: reasoning must not leak into the final answer ───────────────────

class _FakeSSE:
    def __init__(self, lines):
        self._lines = [l.encode() for l in lines]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_reasoning_only_stream_returns_last_line(monkeypatch):
    from nexus_agent_ai.providers import local_provider as lp

    chunks = [
        {"choices": [{"delta": {"reasoning_content": "The user asks about France."}}]},
        {"choices": [{"delta": {"reasoning_content": " I don't need tools for this."}}]},
        {"choices": [{"delta": {"reasoning_content": "\nParis"}}]},
        {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]
    lines = ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=None: _FakeSSE(lines)
    )
    prov = lp.LocalProvider.__new__(lp.LocalProvider)
    prov._model_instance = "llama_server"
    prov._server_port = 8099
    res = prov.complete([{"role": "user", "content": "capital of France?"}], [], "sys")
    assert res.text == "Paris"
    assert "don't need tools" not in res.text
    assert res.input_tokens == 10 and res.output_tokens == 5


# ── Fix #2: startup timeout scales with model size ──────────────────────────

def test_startup_timeout_formula():
    """90s floor, 300s cap, scaled by model GB in between."""
    from nexus_agent_ai.providers.local_provider import _startup_timeout_s
    assert _startup_timeout_s(0.1) == 90
    assert 120 <= _startup_timeout_s(2.0) <= 180
    assert _startup_timeout_s(50.0) == 300


# ── Engine pre-install during onboarding ────────────────────────────────────

def test_ensure_llama_server_binary_skips_arm_linux(monkeypatch):
    """On Termux/ARM the prebuilt cannot run — must no-op, never download."""
    from nexus_agent_ai.providers import local_provider as lp
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: None)
    monkeypatch.setattr(lp, "system_machine_is_arm_linux", lambda: True)
    called = []
    monkeypatch.setattr(lp.LocalProvider, "_download_llama_server",
                        lambda self: called.append(1) or "/never")
    assert lp.ensure_llama_server_binary(verbose=False) is None
    assert called == []


def test_ensure_llama_server_binary_existing_short_circuits(monkeypatch):
    from nexus_agent_ai.providers import local_provider as lp
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: "/existing/llama-server")
    monkeypatch.setattr(lp, "system_machine_is_arm_linux", lambda: False)
    monkeypatch.setattr(lp.LocalProvider, "_download_llama_server",
                        lambda self: (_ for _ in ()).throw(AssertionError("must not download")))
    assert lp.ensure_llama_server_binary(verbose=False) == "/existing/llama-server"


def test_ensure_llama_server_binary_failure_is_soft(monkeypatch, capsys):
    """Download failure must not raise (onboarding continues)."""
    from nexus_agent_ai.providers import local_provider as lp
    monkeypatch.setattr(lp, "_find_llama_server_exe", lambda: None)
    monkeypatch.setattr(lp, "system_machine_is_arm_linux", lambda: False)
    monkeypatch.setattr(lp.LocalProvider, "_download_llama_server",
                        lambda self: (_ for _ in ()).throw(RuntimeError("network blocked")))
    assert lp.ensure_llama_server_binary(verbose=False) is None
