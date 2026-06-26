from typing import Any, Dict, List
from src.utils.config import MAX_CONVERSATION_MESSAGES

class ConversationMemory:
    """Manages in-memory conversation history with a sliding window limit."""

    def __init__(self, max_messages: int = MAX_CONVERSATION_MESSAGES):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        """Add a standard user or assistant text message."""
        self.messages.append({"role": role, "content": content})
        self._prune()

    def add_raw(self, message: Dict[str, Any]):
        """Add a raw structured message (e.g. tool call or tool result)."""
        self.messages.append(message)
        self._prune()

    def _prune(self):
        """Enforce sliding window limit."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve a copy of the current message history."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
