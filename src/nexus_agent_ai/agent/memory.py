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
        """Enforce sliding window limit cleanly by only slicing at genuine user text message boundaries.
        Avoids cutting at tool_result messages (OpenAI role='tool', Gemini parts with function_response,
        Anthropic content with tool_result blocks) to prevent orphaned tool_use/tool_result pairs."""
        if len(self.messages) <= self.max_messages:
            return

        def _is_genuine_user_msg(msg: Dict[str, Any]) -> bool:
            if msg.get("role") != "user":
                return False
            # OpenAI tool results use role='tool', not 'user' — already excluded above.
            # Check for tool_call_id / tool_use_id / name at top level (OpenAI-style)
            if "tool_call_id" in msg or "tool_use_id" in msg or "name" in msg:
                return False
            # Gemini: parts containing function_response
            parts = msg.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and "function_response" in p:
                        return False
            # Anthropic: content is a list of blocks with type='tool_result'
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return False
            return True

        target_idx = len(self.messages) - self.max_messages

        forward_idx = -1
        for i in range(target_idx, len(self.messages)):
            if _is_genuine_user_msg(self.messages[i]):
                forward_idx = i
                break

        backward_idx = -1
        for i in range(target_idx - 1, -1, -1):
            if _is_genuine_user_msg(self.messages[i]):
                backward_idx = i
                break

        if forward_idx != -1 and backward_idx != -1:
            if abs(forward_idx - target_idx) <= abs(backward_idx - target_idx):
                safe_idx = forward_idx
            else:
                safe_idx = backward_idx
        elif forward_idx != -1:
            safe_idx = forward_idx
        elif backward_idx != -1:
            safe_idx = backward_idx
        else:
            return

        if safe_idx > 0:
            self.messages = self.messages[safe_idx:]

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve a copy of the current message history."""
        return self.messages.copy()

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
