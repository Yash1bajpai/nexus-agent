import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..providers.messages import normalize_messages
from ..utils.config import MAX_CONVERSATION_MESSAGES
from ..utils.private_files import restrict_permissions


def get_workspace_session_id(cwd: Optional[Path] = None) -> str:
    """Generate a stable, unique session ID based on the workspace directory path."""
    import hashlib
    import os
    p = (cwd or Path.cwd()).resolve()
    path_str = str(p).replace("\\", "/").rstrip("/")
    if os.name == "nt":
        path_str = path_str.lower()
    path_hash = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:8]
    folder_name = p.name or "root"
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in folder_name)
    return f"{clean_name}_{path_hash}"


class SQLiteMemory:
    """Persistent canonical history. Every operation closes its SQLite handle."""

    def __init__(self, db_path: Optional[Path] = None, session_id: Optional[str] = None, max_messages: int = MAX_CONVERSATION_MESSAGES):
        self.db_path = Path(db_path) if db_path is not None else Path.home() / ".nexus-agent" / "history.db"
        self.session_id = session_id or get_workspace_session_id()
        self.max_messages = max_messages
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The containing directory protects SQLite journal/sidecar files too.
        restrict_permissions(self.db_path.parent, 0o700)
        self.db_path.touch(exist_ok=True)
        restrict_permissions(self.db_path, 0o600)
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add(self, role: str, content: str):
        self.add_raw({"role": role, "content": content})

    def add_raw(self, message: Any):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            rows = self._rows(conn)
            old = [json.loads(row[1]) for row in rows]
            messages = normalize_messages([*old, message])
            from .memory import ConversationMemory
            memory = ConversationMemory(max_messages=self.max_messages)
            memory.messages = messages
            memory._prune()
            self._replace(conn, memory.messages)

    def _rows(self, conn):
        return conn.execute(
            "SELECT id, message_json FROM conversation_history WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()

    def get(self) -> List[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            old = [json.loads(row[1]) for row in self._rows(conn)]
            messages = normalize_messages(old)
            if messages != old:
                self._replace(conn, messages)
            return messages

    def _replace(self, conn, messages):
        conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (self.session_id,))
        conn.executemany(
            "INSERT INTO conversation_history (session_id, role, message_json) VALUES (?, ?, ?)",
            [(self.session_id, message["role"], json.dumps(message)) for message in messages],
        )

    def replace(self, messages: List[Dict[str, Any]]):
        messages = normalize_messages(messages)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            self._replace(conn, messages)

    def clear(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (self.session_id,))

    @classmethod
    def list_sessions(cls, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        db_path = Path(db_path) if db_path is not None else Path.home() / ".nexus-agent" / "history.db"
        if not db_path.exists():
            return []
        restrict_permissions(db_path.parent, 0o700)
        restrict_permissions(db_path, 0o600)
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute("""
                SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_active
                FROM conversation_history GROUP BY session_id ORDER BY last_active DESC
            """).fetchall()
            return [{"session_id": row[0], "message_count": row[1], "last_active": row[2]} for row in rows]

    @classmethod
    def delete_session(cls, session_id: str, db_path: Optional[Path] = None):
        db_path = Path(db_path) if db_path is not None else Path.home() / ".nexus-agent" / "history.db"
        if not db_path.exists():
            return
        restrict_permissions(db_path.parent, 0o700)
        restrict_permissions(db_path, 0o600)
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
