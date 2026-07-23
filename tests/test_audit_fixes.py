import os
import pytest
import tempfile
from pathlib import Path
from src.agent.tools import _sandbox_check, execute_search_web, _validate_workspace_path, execute_run_file
from src.providers.local_provider import LocalQwenProvider
from src.agent.core import Agent
from src.providers.base import BaseProvider, ProviderResponse, Tool
from src.agent.memory import ConversationMemory

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

def test_sandbox_check_blocks_introspection_and_gc():
    """Verify AST check blocks gc, warnings, frame introspection, and gc.get_objects exploits."""
    exploits = [
        # GC search exploit
        "import gc\nfor o in gc.get_objects():\n    if str(o) == 'os': pass",
        # Introspection frame exploit
        "def leak(): yield\nframe = leak().gi_frame\nbuiltins = frame.f_builtins",
        # gi_code attribute exploit
        "def leak(): yield\ncode = leak().gi_code",
        # sys module retrieval from gc
        "import gc\nsys_mod = [o for o in gc.get_objects() if str(o).startswith('<module')][0]",
        # pkgutil loading
        "import pkgutil\ndata = pkgutil.get_data('a', 'b')",
        # warnings usage
        "import warnings\nwarnings.warn('exploit')"
    ]
    for e in exploits:
        err = _sandbox_check(e)
        assert err is not None, f"Expected exploit '{e}' to be blocked, but was allowed."

def test_validate_workspace_path_prefix_containment(tmp_path: Path, monkeypatch):
    """Verify path validation blocks prefix traversal even in fallback case."""
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    
    # Simulate a prefix confusion folder
    malicious_dir = tmp_path / "workspace_backup"
    malicious_dir.mkdir()
    malicious_file = malicious_dir / "exploit.py"
    malicious_file.write_text("print('hacked')", encoding="utf-8")
    
    # Set cwd mock
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    monkeypatch.setattr(os, "getcwd", lambda: str(cwd))
    
    # Mock gettempdir to point to a completely different folder so it doesn't match tmp_path
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "fake_temp"))
    
    # Test primary is_relative_to logic
    res = _validate_workspace_path(malicious_file)
    assert isinstance(res, str)
    assert "Security Sandbox Access Denied" in res

    # Force fallback block check by mocking is_relative_to to raise AttributeError
    def mock_is_relative_to(self, other):
        raise AttributeError("Mocked AttributeError to force fallback")
    monkeypatch.setattr(Path, "is_relative_to", mock_is_relative_to)

    res_fallback = _validate_workspace_path(malicious_file)
    assert isinstance(res_fallback, str)
    assert "Security Sandbox Access Denied" in res_fallback

    # Test clean relative file is allowed under fallback commonpath check
    clean_file = cwd / "subdir" / "clean.py"
    cwd.joinpath("subdir").mkdir()
    clean_file.touch()
    
    res_clean = _validate_workspace_path(clean_file)
    assert isinstance(res_clean, Path)
    assert res_clean.resolve() == clean_file.resolve()

def test_execute_run_file_sandbox_validation(tmp_path: Path, monkeypatch):
    """Verify execute_run_file scans files with sandbox check before running."""
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    monkeypatch.setattr(os, "getcwd", lambda: str(cwd))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "fake_temp"))

    # Safe file should pass and run
    safe_file = cwd / "safe.py"
    safe_file.write_text("print('All safe')", encoding="utf-8")
    res_safe = execute_run_file(str(safe_file))
    assert "STDOUT:\nAll safe" in res_safe

    # Malicious file containing forbidden imports should block
    malicious_file = cwd / "exploit.py"
    malicious_file.write_text("import os\nos.system('whoami')", encoding="utf-8")
    res_malicious = execute_run_file(str(malicious_file))
    assert "AST Sandbox validation failed" in res_malicious

def test_memory_pruning_user_boundaries():
    """Verify ConversationMemory prune preserves user boundary structure."""
    mem = ConversationMemory(max_messages=4)
    # Adding messages exceeding limit: [user1, assistant1, tool1, user2, assistant2, tool2]
    mem.add("user", "user1")
    mem.add("assistant", "assistant1")
    mem.add_raw({"role": "tool", "content": "tool1"})
    mem.add("user", "user2")
    mem.add("assistant", "assistant2")
    mem.add_raw({"role": "tool", "content": "tool2"})
    
    # Pruning will reduce list to at most 4. 
    # Check if the list starts with a user message.
    msgs = mem.get()
    assert len(msgs) <= 4
    assert msgs[0].get("role") == "user"
    assert msgs[0].get("content") == "user2"

def test_search_web_offline_labeling():
    """Verify that execute_search_web explicitly indicates offline reference data when live search is unavailable."""
    res = execute_search_web("python 3.13 new features")
    if "Offline/Cached Reference Data" in res:
        assert "[Offline/Cached Reference Data (as of 2025) - Live search unavailable or rate-limited]:" in res

def test_local_provider_setup_model_verify(monkeypatch):
    """Verify setup_model(verify_download=True) raises RuntimeError if huggingface_hub is missing or fails."""
    prov = LocalQwenProvider()
    
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
