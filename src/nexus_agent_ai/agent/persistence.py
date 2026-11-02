import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..providers.messages import normalize_messages
from ..utils.config import MAX_CONVERSATION_MESSAGES
from ..utils.private_files import restrict_permissions
from .memory import (
    SESSION_METADATA_MAX_CHARS, SESSION_METADATA_MAX_KEYS, SESSION_METADATA_VERSION,
    _session_metadata_copy, _session_metadata_json, _session_metadata_key,
)


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

    _SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """

    _METADATA_SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS session_metadata (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            version INTEGER NOT NULL,
            value_json TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        )
    """

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            self._ensure_schema(conn)

    @classmethod
    def _ensure_schema(cls, conn):
        # Classmethods open their own handle; an interrupted run can leave an
        # existing db file without the table, so never assume it is there.
        conn.execute(cls._SCHEMA_SQL)
        conn.execute(cls._METADATA_SCHEMA_SQL)

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

    def get_session_metadata(self, key, default=None):
        """Read session metadata with SELECT only; never normalize or rewrite audit history."""
        _session_metadata_key(key)
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT version, value_json FROM session_metadata WHERE session_id = ? AND key = ?",
                (self.session_id, key),
            ).fetchone()
        if row is None or row[0] != SESSION_METADATA_VERSION or len(row[1]) > SESSION_METADATA_MAX_CHARS:
            return _session_metadata_copy(default)
        try:
            return _session_metadata_copy(json.loads(row[1]))
        except (ValueError, TypeError):
            return _session_metadata_copy(default)

    def set_session_metadata(self, key, value):
        """Write one bounded versioned JSON value without touching conversation rows."""
        _session_metadata_key(key)
        encoded = _session_metadata_json(value)
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            exists = conn.execute(
                "SELECT 1 FROM session_metadata WHERE session_id = ? AND key = ?",
                (self.session_id, key),
            ).fetchone()
            if exists is None:
                count = conn.execute(
                    "SELECT COUNT(*) FROM session_metadata WHERE session_id = ?", (self.session_id,),
                ).fetchone()[0]
                if count >= SESSION_METADATA_MAX_KEYS:
                    raise ValueError("Session metadata key limit reached.")
            conn.execute(
                "INSERT INTO session_metadata (session_id, key, version, value_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET version = excluded.version, value_json = excluded.value_json",
                (self.session_id, key, SESSION_METADATA_VERSION, encoded),
            )

    def clear(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (self.session_id,))

    @classmethod
    def list_sessions(cls, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        db_path = Path(db_path) if db_path is not None else Path.home() / ".nexus-agent" / "history.db"
        if not db_path.exists():
            return []
        restrict_permissions(db_path.parent, 0o700)
        restrict_permissions(db_path, 0o600)
        with closing(sqlite3.connect(db_path)) as conn, conn:
            cls._ensure_schema(conn)
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
            cls._ensure_schema(conn)
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
