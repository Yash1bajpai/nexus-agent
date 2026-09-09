import os
from pathlib import Path
from dotenv import load_dotenv

USER_CONFIG_DIR = Path.home() / ".nexus-agent"
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_ENV_FILE = USER_CONFIG_DIR / ".env"

def _find_project_root() -> Path:
    curr = Path.cwd().resolve()
    for p in [curr, *curr.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return curr

PROJECT_ROOT = _find_project_root()

# Load only the user-owned config by default. A repository-controlled .env must
# not be able to redirect API traffic or replace credentials implicitly.
if GLOBAL_ENV_FILE.exists():
    load_dotenv(GLOBAL_ENV_FILE, override=False)
if os.getenv("NEXUS_AGENT_ALLOW_PROJECT_ENV", "").lower() in {"1", "true", "yes"} and (PROJECT_ROOT / ".env").exists():
    load_dotenv(PROJECT_ROOT / ".env", override=True)

class ConfigError(Exception):
    """Raised when configuration or API keys are invalid or missing."""
    pass

def get_env_or_raise(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ConfigError(f"{key} not found in environment variables or .env file.")
    return val

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "local")
MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "60"))
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "10"))

PRICING = {
    "LiquidAI/LFM2.5-2.6B-GGUF": {"input": 0.0, "output": 0.0},
    "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
    "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
    "claude-sonnet-4-6": {"input": 0.000003, "output": 0.000015},
    "gpt-4o":             {"input": 0.0000025, "output": 0.00001},
    "gpt-4o-mini":        {"input": 0.00000015, "output": 0.0000006},
    "gemini-1.5-flash":   {"input": 0.000000075,"output": 0.0000003},
    "gemini-2.0-flash":   {"input": 0.000000075,"output": 0.0000003},
    "gemini-2.5-flash":   {"input": 0.000000075,"output": 0.0000003},
    "gemini-2.5-flash-lite": {"input": 0.0000000375,"output": 0.00000015},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"]) + (output_tokens * p["output"])

def get_package_version() -> str:
    """Get nexus-agent-ai package version dynamically."""
    try:
        from importlib.metadata import version
        return version("nexus-agent-ai")
    except Exception:
        pass
    return "2.7.3"

