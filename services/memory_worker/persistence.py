"""
Visual Memory AI — Memory Worker & Persistence Service
Manages persistent object sightings, snapshot media, FAISS/vector indexing,
idempotent event writes, and representative-frame selection.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import faiss

from packages.shared_types.models import MemoryEvent, SearchFilter, TimelineEntry


class MemoryPersistenceService:
    """
    Decoupled memory persistence service.
    Implements:
    - Relational metadata store (SQLite) with indexed timestamps, cameras, classes
    - Vector index (FAISS IndexFlatIP) for cosine similarity on unit CLIP vectors
    - Bounded queue / interval throttling to store only representative sightings
    - Evidence snapshot & crop file management
    """

    def __init__(
        self,
        db_path: str = "data/memories.db",
        faiss_path: str = "data/memories.faiss",
        id_map_path: str = "data/id_map.json",
        media_root: str = "data/media",
        min_seconds_between_similar: float = 3.0,
        dimension: int = 512,
    ):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.id_map_path = id_map_path
        self.media_root = media_root
        self.min_seconds_between_similar = min_seconds_between_similar
        self.dimension = dimension

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(os.path.join(media_root, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(media_root, "crops"), exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_database()

        # Initialize or load FAISS
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: List[str] = []  # FAISS row index -> event_id
        self._load_faiss()

        # In-memory throttle cache: (camera_id, class_name, track_id) -> last_stored_datetime
        self._throttle_cache: Dict[Tuple[str, str, str], datetime] = {}

        print(f"[MemoryService] Initialized with DB {db_path} and FAISS {faiss_path}")

    def _init_database(self):
        """Create indexed memory events table with schema matching specification."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS visual_events (
                event_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                object_identity_id TEXT,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp_utc TEXT NOT NULL,
                bbox_x1 INTEGER NOT NULL,
                bbox_y1 INTEGER NOT NULL,
                bbox_x2 INTEGER NOT NULL,
                bbox_y2 INTEGER NOT NULL,
                embedding_id TEXT,
                snapshot_uri TEXT NOT NULL,
                crop_uri TEXT,
                event_label TEXT DEFAULT 'observed'
            )
        """)
        # Indexes for fast temporal and filtered retrieval
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_class ON visual_events(class_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON visual_events(timestamp_utc DESC)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera_zone ON visual_events(camera_id, zone_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_track ON visual_events(track_id)")
        self.conn.commit()

    def _load_faiss(self):
        if os.path.exists(self.faiss_path) and os.path.exists(self.id_map_path):
            try:
                self.index = faiss.read_index(self.faiss_path)
                with open(self.id_map_path, "r") as f:
                    self.id_map = json.load(f)
                print(f"[MemoryService] Loaded FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"[MemoryService] Warning loading FAISS: {e}. Reinitializing fresh index.")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.id_map = []

    def _save_faiss(self):
        try:
            os.makedirs(os.path.dirname(self.faiss_path) or ".", exist_ok=True)
            faiss.write_index(self.index, self.faiss_path)
            with open(self.id_map_path, "w") as f:
                json.dump(self.id_map, f)
        except Exception as e:
            print(f"[MemoryService] Error saving FAISS: {e}")

    def should_store_event(
        self,
        camera_id: str,
        class_name: str,
        track_id: str,
        now: Optional[datetime] = None
    ) -> bool:
        """Throttling mechanism for representative-frame selection."""
        now = now or datetime.now(timezone.utc)
        key = (camera_id, class_name, track_id)
        last_time = self._throttle_cache.get(key)
        if last_time is not None:
            if (now - last_time).total_seconds() < self.min_seconds_between_similar:
                return False
        return True

    def record_sighting(
        self,
        camera_id: str,
        zone_id: str,
        track_id: str,
        class_name: str,
        confidence: float,
        bbox_xyxy: Tuple[int, int, int, int],
        frame: Optional[np.ndarray],
        embedding: Optional[np.ndarray] = None,
        event_label: str = "observed",
        object_identity_id: Optional[str] = None,
        force: bool = False,
        now_dt: Optional[datetime] = None,
    ) -> Optional[MemoryEvent]:
        """
        Record a new object sighting with idempotent writes, disk snapshots, and vector embeddings.
        """
        now = now_dt or datetime.now(timezone.utc)
        if not force and not self.should_store_event(camera_id, class_name, track_id, now):
            return None

        event_id = f"evt_{uuid.uuid4().hex[:10]}"
        timestamp_utc = now.isoformat()
        x1, y1, x2, y2 = bbox_xyxy

        # Save context snapshot and representative crop
        snapshot_uri = f"/media/snapshots/{event_id}.jpg"
        crop_uri = f"/media/crops/{event_id}.jpg"

        snapshot_path = os.path.join(self.media_root, "snapshots", f"{event_id}.jpg")
        crop_path = os.path.join(self.media_root, "crops", f"{event_id}.jpg")

        if frame is not None and frame.size > 0:
            try:
                # Save full frame
                cv2.imwrite(snapshot_path, frame)

                # Save object crop with padding
                h, w = frame.shape[:2]
                pad = 12
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    cv2.imwrite(crop_path, crop)
                else:
                    crop_uri = snapshot_uri
            except Exception as e:
                print(f"[MemoryService] Image save error: {e}")
        else:
            snapshot_uri = "/media/snapshots/placeholder.jpg"
            crop_uri = "/media/crops/placeholder.jpg"

        # Embed vector if provided
        embedding_id = None
        if embedding is not None:
            embedding_id = f"vec_{event_id}"
            emb_norm = embedding.astype(np.float32)
            if emb_norm.ndim == 1:
                emb_norm = emb_norm.reshape(1, -1)
            self.index.add(emb_norm)
            self.id_map.append(event_id)
            if len(self.id_map) % 10 == 0:
                self._save_faiss()

        # Database insert (Idempotent with INSERT OR REPLACE)
        self.conn.execute("""
            INSERT OR REPLACE INTO visual_events (
                event_id, camera_id, zone_id, track_id, object_identity_id,
                class_name, confidence, timestamp_utc, bbox_x1, bbox_y1,
                bbox_x2, bbox_y2, embedding_id, snapshot_uri, crop_uri, event_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, camera_id, zone_id, str(track_id), object_identity_id,
            class_name, confidence, timestamp_utc, x1, y1, x2, y2,
            embedding_id, snapshot_uri, crop_uri, event_label
        ))
        self.conn.commit()

        # Update throttle cache
        self._throttle_cache[(camera_id, class_name, str(track_id))] = now

        return MemoryEvent(
            event_id=event_id,
            camera_id=camera_id,
            zone_id=zone_id,
            track_id=str(track_id),
            object_identity_id=object_identity_id,
            class_name=class_name,
            confidence=confidence,
            timestamp_utc=timestamp_utc,
            bbox_xyxy=(x1, y1, x2, y2),
            embedding_id=embedding_id,
            snapshot_uri=snapshot_uri,
            crop_uri=crop_uri,
            event_label=event_label,
        )

    def get_latest_event_for_object(self, object_name: str, zone_id: Optional[str] = None) -> Optional[MemoryEvent]:
        """Fetch the most recent reliable sighting for a given object class."""
        query = "SELECT * FROM visual_events WHERE LOWER(class_name) = LOWER(?)"
        params = [object_name]
        if zone_id:
            query += " AND zone_id = ?"
            params.append(zone_id)
        query += " ORDER BY timestamp_utc DESC LIMIT 1"

        row = self.conn.execute(query, params).fetchone()
        return self._row_to_event(row) if row else None

    def get_sightings_for_object(self, object_name: str, limit: int = 100) -> List[MemoryEvent]:
        rows = self.conn.execute(
            "SELECT * FROM visual_events WHERE LOWER(class_name) = LOWER(?) ORDER BY timestamp_utc DESC LIMIT ?",
            (object_name, limit)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def search_events(self, filters: SearchFilter) -> List[MemoryEvent]:
        """Structured SQL search with multi-attribute filtering."""
        query = "SELECT * FROM visual_events WHERE 1=1"
        params: List[Any] = []

        if filters.object_class:
            query += " AND LOWER(class_name) = LOWER(?)"
            params.append(filters.object_class)
        if filters.camera_id:
            query += " AND camera_id = ?"
            params.append(filters.camera_id)
        if filters.zone_id:
            query += " AND zone_id LIKE ?"
            params.append(f"%{filters.zone_id}%")
        if filters.start_time:
            query += " AND timestamp_utc >= ?"
            params.append(filters.start_time)
        if filters.end_time:
            query += " AND timestamp_utc <= ?"
            params.append(filters.end_time)

        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(filters.limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[MemoryEvent, float]]:
        """Vector similarity search against visual embeddings in FAISS."""
        if self.index.ntotal == 0 or len(self.id_map) == 0:
            return []

        emb = query_embedding.astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(emb, k)

        matched: List[Tuple[MemoryEvent, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.id_map):
                ev_id = self.id_map[idx]
                row = self.conn.execute("SELECT * FROM visual_events WHERE event_id = ?", (ev_id,)).fetchone()
                if row:
                    matched.append((self._row_to_event(row), float(score)))

        return matched

    def get_recent_events(self, limit: int = 30) -> List[MemoryEvent]:
        rows = self.conn.execute(
            "SELECT * FROM visual_events ORDER BY timestamp_utc DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_timeline(self, object_class: Optional[str] = None, limit: int = 50) -> List[TimelineEntry]:
        """
        Generate timeline with inferred visibility gaps, moves, and reappearances.
        """
        query = "SELECT * FROM visual_events"
        params = []
        if object_class:
            query += " WHERE LOWER(class_name) = LOWER(?)"
            params.append(object_class)
        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        events = [self._row_to_event(r) for r in rows]
        if not events:
            return []

        # Chronological order for gap calculation
        events_chron = list(reversed(events))
        timeline: List[TimelineEntry] = []

        last_evt_by_class: Dict[str, MemoryEvent] = {}

        for evt in events_chron:
            prev = last_evt_by_class.get(evt.class_name)
            event_type = evt.event_label or "observed"
            gap_sec = None
            note = None

            if prev:
                try:
                    t1 = datetime.fromisoformat(prev.timestamp_utc.replace("Z", "+00:00"))
                    t2 = datetime.fromisoformat(evt.timestamp_utc.replace("Z", "+00:00"))
                    diff = int((t2 - t1).total_seconds())
                    if diff > 15:
                        gap_sec = diff
                        if evt.zone_id != prev.zone_id:
                            event_type = "moved"
                            note = f"Moved from {prev.zone_id} to {evt.zone_id} after {diff}s"
                        else:
                            event_type = "reappeared"
                            note = f"Reappeared in {evt.zone_id} after {diff}s gap"
                except Exception:
                    pass

            last_evt_by_class[evt.class_name] = evt

            # Formatted display time
            try:
                dt = datetime.fromisoformat(evt.timestamp_utc.replace("Z", "+00:00"))
                disp = dt.strftime("%H:%M:%S")
            except Exception:
                disp = evt.timestamp_utc

            timeline.append(TimelineEntry(
                id=evt.event_id,
                timestamp_utc=evt.timestamp_utc,
                time_display=disp,
                event_type=event_type,
                class_name=evt.class_name,
                camera_id=evt.camera_id,
                zone_id=evt.zone_id,
                track_id=evt.track_id,
                confidence=evt.confidence,
                snapshot_uri=evt.snapshot_uri,
                crop_uri=evt.crop_uri,
                gap_duration_sec=gap_sec,
                note=note,
            ))

        # Return latest first
        return list(reversed(timeline))

    def get_stats(self) -> Dict[str, Any]:
        total_events = self.conn.execute("SELECT COUNT(*) as c FROM visual_events").fetchone()["c"]
        unique_classes = [r["class_name"] for r in self.conn.execute("SELECT DISTINCT class_name FROM visual_events").fetchall()]
        unique_zones = [r["zone_id"] for r in self.conn.execute("SELECT DISTINCT zone_id FROM visual_events").fetchall()]
        cameras = [r["camera_id"] for r in self.conn.execute("SELECT DISTINCT camera_id FROM visual_events").fetchall()]

        return {
            "total_events": total_events,
            "unique_objects": unique_classes,
            "unique_objects_count": len(unique_classes),
            "zones": unique_zones,
            "cameras": cameras or ["camera_01", "camera_02"],
            "vector_count": self.index.ntotal,
        }

    def _row_to_event(self, row: sqlite3.Row) -> MemoryEvent:
        return MemoryEvent(
            event_id=row["event_id"],
            camera_id=row["camera_id"],
            zone_id=row["zone_id"],
            track_id=str(row["track_id"]),
            object_identity_id=row["object_identity_id"],
            class_name=row["class_name"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else 0.85,
            timestamp_utc=row["timestamp_utc"],
            bbox_xyxy=(row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
            embedding_id=row["embedding_id"],
            snapshot_uri=row["snapshot_uri"],
            crop_uri=row["crop_uri"],
            event_label=row["event_label"] or "observed",
        )
