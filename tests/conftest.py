import pytest
from pathlib import Path
import nexus_agent_ai.agent.tools as tools_module

@pytest.fixture(autouse=True)
def mock_workspace_path_validator(monkeypatch):
    """
    Since the agent correctly restricts file access to the actual workspace directory,
    we must bypass this check during tests to allow reading/writing to pytest's tmp_path.
    """
    def mock_validate(path):
        return Path(path).resolve()
        
    monkeypatch.setattr(tools_module, "_validate_workspace_path", mock_validate)
