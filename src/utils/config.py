import os
from pathlib import Path
from dotenv import load_dotenv

def _find_project_root() -> Path:
    curr = Path.cwd().resolve()
    for p in [curr, *curr.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / ".env").exists():
            return p
    f_curr = Path(__file__).resolve().parent
    for p in [f_curr, *f_curr.parents]:
        if (p / "pyproject.toml").exists() or (p / ".env").exists():
            return p
    return curr

PROJECT_ROOT = _find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

class ConfigError(Exception):
    """Raised when configuration or API keys are invalid or missing."""
    pass

def get_env_or_raise(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ConfigError(f"{key} not found in environment variables or .env file.")
    return val

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "local")
MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "20"))
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "10"))

PRICING = {
    "qwen2.5-7b-instruct-awq": {"input": 0.0, "output": 0.0},
    "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
    "claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
    "gpt-4o":             {"input": 0.0000025, "output": 0.00001},
    "gpt-4o-mini":        {"input": 0.00000015, "output": 0.0000006},
    "gemini-1.5-flash":   {"input": 0.000000075,"output": 0.0000003},
    "gemini-2.5-flash":   {"input": 0.000000075,"output": 0.0000003},
    "gemini-2.0-flash":   {"input": 0.000000075,"output": 0.0000003},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"]) + (output_tokens * p["output"])

def get_package_version() -> str:
    """Dynamically retrieve the installed version of nexus-agent-ai, falling back to pyproject.toml."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("nexus-agent-ai")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        pyproject = PROJECT_ROOT / "pyproject.toml"
        if pyproject.exists():
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version = "):
                    return line.split("=")[1].strip(" \"'")
    except Exception:
        pass
    return "2.2.8"
