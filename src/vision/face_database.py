"""
Chitti Face Database Module.
Manages persistent SQLite storage of registered face identities and their 128-d feature embeddings.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logging import log_debug, log_warning


def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FaceDatabase:
    """Thread-safe SQLite database manager for registered face identities."""

    def __init__(self, db_path: str = "data/vision/faces.db"):
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = root_dir / self.db_path

        self._lock = threading.Lock()
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS registered_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        embedding TEXT NOT NULL,
                        sample_count INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_face_name ON registered_faces(name)
                """)
                conn.commit()
                log_debug(f"Face database initialized at {self.db_path}")

    def register_or_update_face(
        self,
        name: str,
        embedding: List[float],
        sample_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Registers a new person or updates their stored face embedding vector."""
        clean_name = name.strip()
        now = get_utc_now_iso()
        emb_json = json.dumps(embedding)
        meta_json = json.dumps(metadata or {})

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check if already exists
                cursor.execute("SELECT id, sample_count, embedding FROM registered_faces WHERE name = ?", (clean_name,))
                existing = cursor.fetchone()

                if existing:
                    person_id = existing["id"]
                    cursor.execute(
                        """
                        UPDATE registered_faces
                        SET embedding = ?, sample_count = ?, updated_at = ?, metadata = ?
                        WHERE id = ?
                        """,
                        (emb_json, sample_count, now, meta_json, person_id),
                    )
                    conn.commit()
                    return person_id
                else:
                    cursor.execute(
                        """
                        INSERT INTO registered_faces (name, embedding, sample_count, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (clean_name, emb_json, sample_count, now, now, meta_json),
                    )
                    conn.commit()
                    return cursor.lastrowid

    def get_face_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a registered face profile by name."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM registered_faces WHERE name = ?", (name.strip(),))
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "embedding": json.loads(row["embedding"]),
                    "sample_count": row["sample_count"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }

    def get_all_registered_faces(self) -> List[Tuple[int, str, List[float]]]:
        """Returns list of (person_id, name, embedding) for all registered individuals."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, embedding FROM registered_faces ORDER BY name ASC")
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    try:
                        emb = json.loads(row["embedding"])
                        results.append((row["id"], row["name"], emb))
                    except Exception:
                        pass
                return results

    def list_all_names(self) -> List[str]:
        """Returns list of all registered person names."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM registered_faces ORDER BY name ASC")
                rows = cursor.fetchall()
                return [r["name"] for r in rows]

    def delete_face(self, name: str) -> bool:
        """Deletes a registered person by name."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM registered_faces WHERE name = ?", (name.strip(),))
                conn.commit()
                return cursor.rowcount > 0

    def clear_all_faces(self) -> int:
        """Deletes all registered faces."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM registered_faces")
                conn.commit()
                return cursor.rowcount

    def count(self) -> int:
        """Returns the total number of registered people."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM registered_faces")
                return cursor.fetchone()[0]
