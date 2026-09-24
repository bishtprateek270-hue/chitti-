"""
Chitti Memory Database Module.
Provides SQLite persistent storage for long-term memories with full CRUD and schema management.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logging import log_debug, log_warning, log_error


def get_utc_now_iso() -> str:
    """Returns the current UTC timestamp formatted as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    """Represents a single persistent memory entity."""
    content: str
    memory_type: str = "fact"  # fact, preference, project, instruction, personal, context
    importance: int = 3       # 1 (very low) to 5 (very important)
    id: Optional[int] = None
    created_at: str = field(default_factory=get_utc_now_iso)
    updated_at: str = field(default_factory=get_utc_now_iso)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class MemoryDatabase:
    """Thread-safe SQLite database manager for Chitti long-term memory."""

    def __init__(self, db_path: str = "data/memory/chitti_memory.db"):
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            # Resolve relative to project root
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = root_dir / self.db_path

        self._lock = threading.Lock()
        self._ensure_database_dir()
        self._init_db()

    def _ensure_database_dir(self):
        """Creates the parent directory if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a new SQLite connection configured with standard row factories."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the database tables and indexes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL DEFAULT 'fact',
                        importance INTEGER NOT NULL DEFAULT 3,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        embedding TEXT,
                        metadata TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)
                """)
                conn.commit()
                log_debug(f"Memory database initialized at {self.db_path}")

    def insert_memory(self, record: MemoryRecord) -> int:
        """Inserts a new memory record and returns its generated ID."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = get_utc_now_iso()
                emb_json = json.dumps(record.embedding) if record.embedding is not None else None
                meta_json = json.dumps(record.metadata or {})

                cursor.execute(
                    """
                    INSERT INTO memories (content, memory_type, importance, created_at, updated_at, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.content.strip(),
                        record.memory_type.strip().lower(),
                        int(record.importance),
                        record.created_at or now,
                        now,
                        emb_json,
                        meta_json,
                    ),
                )
                conn.commit()
                record.id = cursor.lastrowid
                return record.id

    def get_memory(self, memory_id: int) -> Optional[MemoryRecord]:
        """Retrieves a memory by its ID."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_record(row)

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        memory_type: Optional[str] = None,
        importance: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Updates fields of an existing memory."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
                row = cursor.fetchone()
                if not row:
                    return False

                current = self._row_to_record(row)
                new_content = content.strip() if content is not None else current.content
                new_type = memory_type.strip().lower() if memory_type is not None else current.memory_type
                new_importance = int(importance) if importance is not None else current.importance
                new_emb = embedding if embedding is not None else current.embedding
                new_meta = metadata if metadata is not None else current.metadata
                now = get_utc_now_iso()

                cursor.execute(
                    """
                    UPDATE memories
                    SET content = ?, memory_type = ?, importance = ?, updated_at = ?, embedding = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        new_content,
                        new_type,
                        new_importance,
                        now,
                        json.dumps(new_emb) if new_emb is not None else None,
                        json.dumps(new_meta or {}),
                        memory_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0

    def delete_memory(self, memory_id: int) -> bool:
        """Deletes a single memory record by ID."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                return cursor.rowcount > 0

    def delete_by_category(self, memory_type: str) -> int:
        """Deletes all memories of a specific category and returns count deleted."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories WHERE memory_type = ?", (memory_type.strip().lower(),))
                conn.commit()
                return cursor.rowcount

    def list_all_memories(self, memory_type: Optional[str] = None) -> List[MemoryRecord]:
        """Lists all stored memories, optionally filtered by type."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if memory_type:
                    cursor.execute(
                        "SELECT * FROM memories WHERE memory_type = ? ORDER BY importance DESC, updated_at DESC",
                        (memory_type.strip().lower(),),
                    )
                else:
                    cursor.execute("SELECT * FROM memories ORDER BY importance DESC, updated_at DESC")
                rows = cursor.fetchall()
                return [self._row_to_record(row) for row in rows]

    def get_all_with_embeddings(self) -> List[MemoryRecord]:
        """Returns all memories that have computed embeddings."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories WHERE embedding IS NOT NULL")
                rows = cursor.fetchall()
                return [self._row_to_record(row) for row in rows]

    def clear_all_memories(self) -> int:
        """Permanently deletes all memories from the database."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memories")
                conn.commit()
                return cursor.rowcount

    def count(self) -> int:
        """Returns the total number of memories stored."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Converts an SQLite row to a MemoryRecord dataclass."""
        emb = json.loads(row["embedding"]) if row["embedding"] else None
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            embedding=emb,
            metadata=meta,
        )
