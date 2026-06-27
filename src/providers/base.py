from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class Tool:
    """Universal tool definition format."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    execute: Callable[..., str]

@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""
    id: str
    name: str
    args: Dict[str, Any]

@dataclass
class ProviderResponse:
    """Standardized response from any LLM provider."""
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_assistant_message: Any = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        """
        Send messages to LLM and return standardized response.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        """
        Stream tokens from LLM. Yields chunks or ProviderResponse updates.
        """
        pass

    @abstractmethod
    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        """
        Format a tool execution result into the provider's expected message structure.
        """
        pass
