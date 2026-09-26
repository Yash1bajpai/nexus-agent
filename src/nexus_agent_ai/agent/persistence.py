import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..utils.config import MAX_CONVERSATION_MESSAGES

def get_workspace_session_id(cwd: Optional[Path] = None) -> str:
    """Generate a stable, unique session ID based on the workspace directory path."""
    import hashlib
    p = (cwd or Path.cwd()).resolve()
    path_str = str(p).replace("\\", "/").rstrip("/").lower()
    path_hash = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:8]
    folder_name = p.name or "root"
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in folder_name)
    return f"{clean_name}_{path_hash}"

class SQLiteMemory:
    """Persistent SQLite-backed conversation buffer for Nexus-Agent sessions."""

    def __init__(self, db_path: Optional[Path] = None, session_id: Optional[str] = None, max_messages: int = MAX_CONVERSATION_MESSAGES):
        if db_path is None:
            home_dir = Path.home() / ".nexus-agent"
            home_dir.mkdir(parents=True, exist_ok=True)
            db_path = home_dir / "history.db"
        self.db_path = db_path
        self.session_id = session_id or get_workspace_session_id()
        self.max_messages = max_messages
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add(self, role: str, content: str):
        """Add a text message to persistent SQLite storage."""
        msg = {"role": role, "content": content}
        self.add_raw(msg)

    def add_raw(self, message: Dict[str, Any]):
        """Add a raw message object to SQLite storage and prune older entries."""
        role = message.get("role", "unknown")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversation_history (session_id, role, message_json) VALUES (?, ?, ?)",
                (self.session_id, role, json.dumps(message))
            )
            self._prune(conn)
            conn.commit()

    def _prune(self, conn):
        """Enforce sliding window limit cleanly without breaking tool_call / tool_result pairs.
        Detects Anthropic tool_result (content with type='tool_result'), Gemini function_response
        (parts with function_response), and OpenAI tool role to avoid orphaned messages."""
        cursor = conn.execute(
            "SELECT id, role, message_json FROM conversation_history WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,)
        )
        rows = cursor.fetchall()
        if len(rows) <= self.max_messages:
            return

        def _is_genuine_user_msg(role: str, msg_json: str) -> bool:
            if role != "user":
                return False
            try:
                msg = json.loads(msg_json)
            except Exception:
                return True
            if "tool_call_id" in msg or "tool_use_id" in msg or "name" in msg:
                return False
            parts = msg.get("parts")
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and "function_response" in p:
                        return False
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return False
            return True

        target_idx = len(rows) - self.max_messages
        while target_idx < len(rows):
            if _is_genuine_user_msg(rows[target_idx][1], rows[target_idx][2]):
                break
            target_idx += 1

        if target_idx >= len(rows):
            target_idx = len(rows) - self.max_messages

        if target_idx > 0 and target_idx < len(rows):
            cutoff_id = rows[target_idx][0]
            conn.execute("DELETE FROM conversation_history WHERE session_id = ? AND id < ?", (self.session_id, cutoff_id))

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve recent messages for the active session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT message_json FROM conversation_history WHERE session_id = ? ORDER BY id ASC",
                (self.session_id,)
            )
            rows = cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    def clear(self):
        """Clear session history from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (self.session_id,))
            conn.commit()

    @classmethod
    def list_sessions(cls, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all stored sessions with message counts and last activity."""
        if db_path is None:
            db_path = Path.home() / ".nexus-agent" / "history.db"
        if not db_path.exists():
            return []
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_active
                FROM conversation_history
                GROUP BY session_id
                ORDER BY last_active DESC
            """)
            return [
                {"session_id": row[0], "message_count": row[1], "last_active": row[2]}
                for row in cursor.fetchall()
            ]

    @classmethod
    def delete_session(cls, session_id: str, db_path: Optional[Path] = None):
        """Delete all messages for a specific session."""
        if db_path is None:
            db_path = Path.home() / ".nexus-agent" / "history.db"
        if not db_path.exists():
            return
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
            conn.commit()

