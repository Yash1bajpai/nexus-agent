from typing import Any, Dict, List
from ..utils.config import MAX_CONVERSATION_MESSAGES

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
        """Enforce sliding window limit cleanly without breaking tool_call / tool_result pairs."""
        if len(self.messages) <= self.max_messages:
            return

        # Start looking from -self.max_messages for a safe split point
        target_idx = len(self.messages) - self.max_messages
        while target_idx < len(self.messages):
            if self.messages[target_idx].get("role") == "user":
                break
            target_idx += 1

        if target_idx >= len(self.messages):
            # If no user message found in the trailing window, fall back to max_messages
            target_idx = len(self.messages) - self.max_messages
            # Adjust forward if it lands on a tool result or inside an active sequence
            while target_idx < len(self.messages) and self.messages[target_idx].get("role") == "tool":
                target_idx += 1

        self.messages = self.messages[target_idx:]

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve a copy of the current message history."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
