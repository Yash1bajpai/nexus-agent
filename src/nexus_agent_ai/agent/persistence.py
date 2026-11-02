import hashlib
import re
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

    _ARTIFACT_SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS context_artifacts (
            artifact_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            byte_len INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, artifact_id)
        )
    """
    ARTIFACT_MAX_BYTES = 256 * 1024
    HISTORY_SCAN_ROWS = 500
    HISTORY_SCAN_BYTES = 2 * 1024 * 1024
    HISTORY_ROW_BYTES = 512 * 1024
    HISTORY_EXCERPT_CHARS = 2000

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            self._ensure_schema(conn)

    @classmethod
    def _ensure_schema(cls, conn):
        # Classmethods open their own handle; an interrupted run can leave an
        # existing db file without the table, so never assume it is there.
        conn.execute(cls._SCHEMA_SQL)
        conn.execute(cls._METADATA_SCHEMA_SQL)
        conn.execute(cls._ARTIFACT_SCHEMA_SQL)

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

    def store_artifacts(self, plans):
        """Atomically retain all pure fitter plans, or raise without changing state.

        Plans have exactly artifact_id/tool_name/byte_len/content; byte_len is
        the original string's UTF-8 length. IDs hash tool_name + NUL + content.
        The 256KiB per-session cap counts payload bytes. Existing artifacts are
        evicted by created_at, then rowid, oldest first; artifacts referenced by
        this batch are never evicted. An oversized batch fails closed.
        """
        if not isinstance(plans, list):
            raise ValueError("Artifact plans must be a list.")
        retained = {}
        for plan in plans:
            if not isinstance(plan, dict) or set(plan) != {"artifact_id", "tool_name", "byte_len", "content"}:
                raise ValueError("Invalid artifact plan fields.")
            content, name = plan["content"], plan["tool_name"]
            if not isinstance(content, str) or not isinstance(name, str):
                raise ValueError("Artifact content and tool name must be text.")
            size = len(content.encode("utf-8"))
            identity = hashlib.sha256((name + "\0" + content).encode("utf-8")).hexdigest()
            if (type(plan["byte_len"]) is not int or plan["byte_len"] != size
                    or plan["artifact_id"] != identity):
                raise ValueError("Artifact identity or UTF-8 byte length mismatch.")
            retained[identity] = plan
        if sum(plan["byte_len"] for plan in retained.values()) > self.ARTIFACT_MAX_BYTES:
            raise ValueError("Artifact batch exceeds the 256KiB session cap.")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            for identity, plan in retained.items():
                conn.execute(
                    "INSERT INTO context_artifacts (artifact_id, session_id, tool_name, byte_len, content) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id, artifact_id) DO NOTHING",
                    (identity, self.session_id, plan["tool_name"], plan["byte_len"], plan["content"]),
                )
                actual = conn.execute(
                    "SELECT tool_name, byte_len, content FROM context_artifacts WHERE session_id = ? AND artifact_id = ?",
                    (self.session_id, identity),
                ).fetchone()
                if actual != (plan["tool_name"], plan["byte_len"], plan["content"]):
                    raise ValueError("Artifact identity collision or corrupt retained content.")
            rows = conn.execute(
                "SELECT artifact_id, byte_len FROM context_artifacts WHERE session_id = ? "
                "ORDER BY created_at ASC, rowid ASC", (self.session_id,),
            ).fetchall()
            total = sum(row[1] for row in rows)
            for identity, size in rows:
                if total <= self.ARTIFACT_MAX_BYTES:
                    break
                if identity not in retained:
                    conn.execute("DELETE FROM context_artifacts WHERE session_id = ? AND artifact_id = ?",
                                 (self.session_id, identity))
                    total -= size
            if total > self.ARTIFACT_MAX_BYTES:
                raise ValueError("Artifact retention cap cannot retain this batch.")
        return list(retained)

    def history_entries(self, query, limit=5):
        """SELECT-only same-session canonical excerpts, never normalize audit rows.

        A literal case-insensitive substring ranks above all-keyword matches;
        each group is newest first. Empty query means recency. Scan at most 500
        recent rows / 2MiB, reject rows above 512KiB, return <=10 exact contiguous
        <=2000-character excerpts. Structured call paths, never result prose,
        supply provenance for the caller's current workspace/sensitive guards.
        """
        if not isinstance(query, str) or len(query) > 256:
            raise ValueError("History query must be text of at most 256 characters.")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("History limit must be between 1 and 10.")
        with closing(sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, role, message_json FROM "
                "(SELECT id, timestamp, role, message_json FROM conversation_history "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?) "
                "WHERE length(CAST(message_json AS BLOB)) <= ? ORDER BY id DESC",
                (self.session_id, self.HISTORY_SCAN_ROWS, self.HISTORY_ROW_BYTES),
            )
            rows, scanned = [], 0
            for row in cursor:
                scanned += len(row[3].encode("utf-8"))
                if scanned > self.HISTORY_SCAN_BYTES:
                    break
                rows.append(row)
        calls, candidates = {}, []
        query = query if query.strip() else ""
        words = re.findall(r"\w+", query)
        for entry_id, timestamp, role, raw in reversed(rows):
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if (not isinstance(message, dict) or role not in ("user", "assistant", "tool")
                    or message.get("role") != role):
                continue
            metadata = message.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            if role == "user" and not metadata.get("internal"):
                calls.clear()
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                        continue
                    function = call["function"]
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (ValueError, TypeError):
                            arguments = {}
                    paths = [arguments[key] for key in ("path", "file_path", "filepath", "filename")
                             if isinstance(arguments, dict) and isinstance(arguments.get(key), str) and arguments[key]]
                    calls[call.get("id")] = paths
            if metadata.get("internal") or metadata.get("outcome_correction"):
                continue
            text = message.get("content")
            if not isinstance(text, str) or not text:
                continue
            paths = calls.get(message.get("tool_call_id"), []) if role == "tool" else []
            attachment = (any(key in metadata for key in ("attachments", "files", "file_path", "path"))
                          or "[Context attached from @" in text)
            kind = "tool_result" if role == "tool" or attachment else "conversation"
            known = bool(paths) if kind == "tool_result" else True
            match = re.search(re.escape(query), text, re.IGNORECASE) if query else None
            # The historical index carries paths, while read_file returns raw
            # content without a path header. Search trusted call provenance too.
            path_text = "\n".join(paths)
            path_match = re.search(re.escape(query), path_text, re.IGNORECASE) if query and paths else None
            if query and match is None and path_match is None:
                search_text = text + "\n" + path_text
                matches = [re.search(re.escape(word), search_text, re.IGNORECASE) for word in words]
                if not matches or not all(matches):
                    continue
                position, rank = 0, 1
                for word in words:
                    content_match = re.search(re.escape(word), text, re.IGNORECASE)
                    if content_match:
                        position = content_match.start()
                        break
            else:
                position, rank = (match.start() if match else 0), 2
            start = max(0, position - 200)
            candidates.append((rank, entry_id, {
                "entry_id": entry_id, "timestamp": timestamp, "role": role,
                "content": text[start:start + self.HISTORY_EXCERPT_CHARS],
                "truncated": start > 0 or start + self.HISTORY_EXCERPT_CHARS < len(text),
                "paths": list(paths), "content_kind": kind, "provenance_known": known,
            }))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in candidates[:limit]]

    def clear(self):
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (self.session_id,))
            conn.execute("DELETE FROM context_artifacts WHERE session_id = ?", (self.session_id,))

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
            conn.execute("DELETE FROM context_artifacts WHERE session_id = ?", (session_id,))
