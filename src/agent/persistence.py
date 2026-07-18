import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from ..utils.config import MAX_CONVERSATION_MESSAGES

class SQLiteMemory:
    """Persistent SQLite-backed conversation buffer for Nexus-Agent sessions."""

    def __init__(self, db_path: Path = None, session_id: str = "default", max_messages: int = MAX_CONVERSATION_MESSAGES):
        if db_path is None:
            home_dir = Path.home() / ".nexus-agent"
            home_dir.mkdir(parents=True, exist_ok=True)
            db_path = home_dir / "history.db"
        self.db_path = db_path
        self.session_id = session_id
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
        """Enforce sliding window limit cleanly without breaking tool_call / tool_result pairs."""
        cursor = conn.execute(
            "SELECT id, role FROM conversation_history WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,)
        )
        rows = cursor.fetchall()
        if len(rows) <= self.max_messages:
            return

        target_idx = len(rows) - self.max_messages
        while target_idx < len(rows):
            if rows[target_idx][1] == "user":
                break
            target_idx += 1

        if target_idx >= len(rows):
            target_idx = len(rows) - self.max_messages
            while target_idx < len(rows) and rows[target_idx][1] == "tool":
                target_idx += 1

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
