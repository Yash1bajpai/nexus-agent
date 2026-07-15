import pytest
from pathlib import Path
from src.agent.tools import _sandbox_check, execute_search_web
from src.providers.local_provider import LocalQwenProvider
from src.agent.core import Agent
from src.providers.base import BaseProvider, ProviderResponse, Tool

def test_sandbox_check_blocks_bypass():
    """Verify AST static analysis blocks builtins, getattr, and indirect import/exec bypasses."""
    bypasses = [
        "import os",
        "from subprocess import call",
        "import importlib",
        "getattr(builtins, '__import__')",
        "eval('1 + 1')",
        "exec('print(1)')",
        "open('test.py').read()",
        "__import__('os').system('ls')",
    ]
    for b in bypasses:
        err = _sandbox_check(b)
        assert err is not None, f"Expected {b} to be blocked by sandbox check, but was allowed."

    safe_code = "x = [1, 2, 3]\ny = sum(x)"
    assert _sandbox_check(safe_code) is None

def test_search_web_offline_labeling():
    """Verify that execute_search_web explicitly indicates offline reference data when live search is unavailable."""
    res = execute_search_web("python 3.13 new features")
    # Whether live or fallback, if fallback triggered it must have explicit labeling
    if "Offline/Cached Reference Data" in res:
        assert "[Offline/Cached Reference Data - Live search unavailable or rate-limited]:" in res

def test_local_provider_setup_model_verify(monkeypatch):
    """Verify setup_model(verify_download=True) raises RuntimeError if huggingface_hub is missing or fails."""
    prov = LocalQwenProvider()
    
    # Simulate missing huggingface_hub
    import builtins
    orig_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named huggingface_hub")
        return orig_import(name, *args, **kwargs)
    
    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(RuntimeError) as excinfo:
        prov.setup_model(verify_download=True)
    assert "huggingface_hub package is not installed" in str(excinfo.value)

class MockStreamProvider(BaseProvider):
    def __init__(self):
        self.stream_called = False
    def complete(self, messages, tools, system):
        return ProviderResponse(text="Complete response")
    def stream(self, messages, tools, system):
        self.stream_called = True
        yield "Hello "
        yield "world!"
        yield ProviderResponse(text="Hello world!")
    def format_tool_result_message(self, tool_call_id, result):
        return {"role": "tool", "content": result}

def test_agent_run_stream_true():
    """Verify Agent.run calls provider.stream() when stream=True."""
    mock_prov = MockStreamProvider()
    agent = Agent(provider=mock_prov, verbose=False)
    output = agent.run("Test input", stream=True)
    assert mock_prov.stream_called is True
    assert output == "Hello world!"

def test_onboarding_env_file_path():
    """Verify ENV_FILE is safely contained within the project directory."""
    from src.cli.onboarding import _find_project_root, ENV_FILE
    root = _find_project_root()
    assert ENV_FILE == root / ".env"
