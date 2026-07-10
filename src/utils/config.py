import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
