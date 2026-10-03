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

@pytest.fixture(autouse=True)
def isolate_local_model_configuration(monkeypatch):
    """Do not let a developer's onboarding configuration change model defaults in tests."""
    monkeypatch.delenv("NEXUS_AGENT_MODEL_REPO", raising=False)
    monkeypatch.delenv("NEXUS_AGENT_MODEL_FILENAME", raising=False)


@pytest.fixture(autouse=True)
def isolate_home_directory(monkeypatch, tmp_path_factory):
    """Keep session history and config out of the developer's real ~/.nexus-agent."""
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
