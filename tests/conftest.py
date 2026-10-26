import pytest
from pathlib import Path
import nexus_agent_ai.agent.tools as tools_module

@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    """Exercise the real sandbox inside a fresh, per-test project directory."""
    monkeypatch.chdir(tmp_path)
    tools_module.reset_file_tracking()
    yield
    tools_module.reset_file_tracking()

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
    monkeypatch.setenv("NEXUS_AGENT_HOME", str(fake_home / ".nexus-agent"))
