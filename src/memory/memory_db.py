"""
Visual Memory AI — SQLite Memory Database
Stores structured metadata about visual memories (object sightings).
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class MemoryEntry:
    """A single visual memory record."""
    memory_id: Optional[int]      # Auto-assigned by SQLite
    object_name: str              # e.g., 'laptop', 'cell phone'
    track_id: int                 # Persistent tracker ID
    timestamp: str                # ISO format timestamp
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    region: str                   # e.g., 'top-left', 'middle-center'
    confidence: float
    embedding_id: Optional[int]   # Reference to FAISS index position

    @property
    def bbox(self):
        return (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2)

    @property
    def time_ago(self) -> str:
        """Human-readable time since this memory was created."""
        try:
            mem_time = datetime.fromisoformat(self.timestamp)
            delta = datetime.now() - mem_time
            seconds = int(delta.total_seconds())
            if seconds < 60:
                return f"{seconds}s ago"
            elif seconds < 3600:
                return f"{seconds // 60}m ago"
            elif seconds < 86400:
                return f"{seconds // 3600}h ago"
            else:
                return f"{seconds // 86400}d ago"
        except Exception:
            return self.timestamp


class MemoryDatabase:
    """SQLite database for visual memory metadata."""

    def __init__(self, db_path: str = "data/memories.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        print(f"[MemoryDB] Connected to {db_path}")

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                region TEXT,
                confidence REAL,
                embedding_id INTEGER
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_object_name
            ON memories(object_name)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memories(timestamp DESC)
        """)
        self.conn.commit()

    def store_memory(self, entry: MemoryEntry) -> int:
        """Insert a new memory and return its ID."""
        cursor = self.conn.execute("""
            INSERT INTO memories
            (object_name, track_id, timestamp, bbox_x1, bbox_y1,
             bbox_x2, bbox_y2, region, confidence, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.object_name, entry.track_id, entry.timestamp,
            entry.bbox_x1, entry.bbox_y1, entry.bbox_x2, entry.bbox_y2,
            entry.region, entry.confidence, entry.embedding_id,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_memory(self, memory_id: int) -> Optional[MemoryEntry]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_memories_by_object(self, object_name: str, limit: int = 20) -> List[MemoryEntry]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE object_name = ? ORDER BY timestamp DESC LIMIT ?",
            (object_name, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_latest_memory(self, object_name: str) -> Optional[MemoryEntry]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE object_name = ? ORDER BY timestamp DESC LIMIT 1",
            (object_name,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_recent_memories(self, limit: int = 50) -> List[MemoryEntry]:
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_unique_objects(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT object_name FROM memories ORDER BY object_name"
        ).fetchall()
        return [r["object_name"] for r in rows]

    def get_total_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"]

    def get_last_store_time(self, object_name: str, track_id: int) -> Optional[str]:
        """Get the timestamp of the most recent memory for a specific object+track."""
        row = self.conn.execute(
            "SELECT timestamp FROM memories WHERE object_name = ? AND track_id = ? ORDER BY timestamp DESC LIMIT 1",
            (object_name, track_id),
        ).fetchone()
        return row["timestamp"] if row else None

    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            memory_id=row["memory_id"],
            object_name=row["object_name"],
            track_id=row["track_id"],
            timestamp=row["timestamp"],
            bbox_x1=row["bbox_x1"],
            bbox_y1=row["bbox_y1"],
            bbox_x2=row["bbox_x2"],
            bbox_y2=row["bbox_y2"],
            region=row["region"],
            confidence=row["confidence"],
            embedding_id=row["embedding_id"],
        )

    def close(self):
        self.conn.close()
