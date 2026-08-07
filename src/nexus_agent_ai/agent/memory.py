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
        """Enforce sliding window limit cleanly by only slicing at user message boundaries."""
        if len(self.messages) <= self.max_messages:
            return

        target_idx = len(self.messages) - self.max_messages

        # Find the nearest 'user' message to target_idx
        # We scan both forward and backward to locate a user message index.
        # We want to pick the one that keeps the size reasonably close to max_messages.
        forward_idx = -1
        for i in range(target_idx, len(self.messages)):
            if self.messages[i].get("role") == "user":
                forward_idx = i
                break

        backward_idx = -1
        for i in range(target_idx - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                backward_idx = i
                break

        # Decide which index to use
        if forward_idx != -1 and backward_idx != -1:
            # Choose the one closer to target_idx
            if abs(forward_idx - target_idx) <= abs(backward_idx - target_idx):
                safe_idx = forward_idx
            else:
                safe_idx = backward_idx
        elif forward_idx != -1:
            safe_idx = forward_idx
        elif backward_idx != -1:
            safe_idx = backward_idx
        else:
            # If there are no user messages at all in the entire conversation history,
            # we cannot safely prune while ensuring it starts with user. We do not prune.
            return

        # Slicing at safe_idx means the new list starts at safe_idx.
        # Anthropic and OpenAI require the first message to be user (or system),
        # so starting with a 'user' message is structurally valid.
        if safe_idx > 0:
            self.messages = self.messages[safe_idx:]

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve a copy of the current message history."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
