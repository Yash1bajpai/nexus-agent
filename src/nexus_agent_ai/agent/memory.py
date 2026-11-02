import json

from typing import Any, Dict, List
from ..utils.config import MAX_CONVERSATION_MESSAGES
from ..providers.messages import normalize_messages

SESSION_METADATA_MAX_CHARS = 4096
SESSION_METADATA_MAX_KEYS = 16
SESSION_METADATA_VERSION = 1


def _session_metadata_key(key):
    if not isinstance(key, str) or not 1 <= len(key) <= 64:
        raise ValueError("Session metadata keys must contain 1 to 64 characters.")
    return key


def _session_metadata_json(value):
    """Validate and serialize a bounded JSON value without coercing arbitrary objects."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Session metadata must be JSON-safe.") from exc
    if len(encoded) > SESSION_METADATA_MAX_CHARS:
        raise ValueError("Session metadata value exceeds the bounded serialization limit.")
    return encoded


def _session_metadata_copy(value):
    return json.loads(_session_metadata_json(value))


def is_genuine_user_message(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    if msg.get("metadata", {}).get("internal"):
        return False
    # OpenAI tool results use role='tool', not 'user' — already excluded above.
    # Check for tool_call_id / tool_use_id / name at top level (OpenAI-style)
    if "tool_call_id" in msg or "tool_use_id" in msg or "name" in msg:
        return False
    # Gemini: parts containing function_response
    parts = msg.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, dict) and ("function_response" in p or "functionResponse" in p):
                return False
    # Anthropic: content is a list of blocks with type='tool_result'
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return False
    return True


class ConversationMemory:
    """Manages in-memory conversation history with a sliding window limit."""

    def __init__(self, max_messages: int = MAX_CONVERSATION_MESSAGES):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self._session_metadata = {}

    def add(self, role: str, content: str):
        """Add a standard user or assistant text message."""
        self.messages.append({"role": role, "content": content})
        self._prune()

    def add_raw(self, message: Dict[str, Any]):
        """Add a raw structured message (e.g. tool call or tool result)."""
        self.messages = normalize_messages([*self.messages, message])
        self._prune()

    def _prune(self):
        """Enforce sliding window limit cleanly by only slicing at genuine user text message boundaries.
        Avoids cutting at tool_result messages (OpenAI role='tool', Gemini parts with function_response,
        Anthropic content with tool_result blocks) to prevent orphaned tool_use/tool_result pairs."""
        if len(self.messages) <= self.max_messages:
            return

        _is_genuine_user_msg = is_genuine_user_message

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

    def replace(self, messages: List[Dict[str, Any]]):
        """Replace the whole history (used by /compact)."""
        self.messages = normalize_messages(messages)

    def get_session_metadata(self, key, default=None):
        """Return an isolated JSON value without modifying history or metadata."""
        _session_metadata_key(key)
        return _session_metadata_copy(self._session_metadata.get(key, default))

    def set_session_metadata(self, key, value):
        """Store bounded session-only metadata independently of conversation pruning."""
        _session_metadata_key(key)
        copied = _session_metadata_copy(value)
        if key not in self._session_metadata and len(self._session_metadata) >= SESSION_METADATA_MAX_KEYS:
            raise ValueError("Session metadata key limit reached.")
        self._session_metadata[key] = copied

    def clear(self):
        """Clear all messages from memory."""
        self.messages = []
        self._session_metadata.clear()
