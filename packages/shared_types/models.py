"""
Visual Memory AI — Shared Types & Data Contracts
Standardized schemas for events, queries, tracking, cameras, and observability.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Bounding box coordinates [x1, y1, x2, y2]."""
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    """Single object detection."""
    bbox_xyxy: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    class_id: int
    zone_id: Optional[str] = "unknown"


class TrackItem(BaseModel):
    """Short-term tracked object from BoT-SORT / ByteTrack."""
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    zone_id: str
    timestamp_utc: str
    class_id: int = 0


class MemoryEvent(BaseModel):
    """
    Standard Visual Memory Event Schema.
    Associates object sightings with camera zone, timestamps, and media evidence.
    """
    event_id: str = Field(..., description="Unique event identifier e.g. evt_000184")
    camera_id: str = Field(..., description="Camera identifier e.g. camera_01")
    zone_id: str = Field(..., description="Human-readable zone e.g. study_desk, kitchen_counter")
    track_id: str = Field(..., description="Short-term tracking ID e.g. track_0042")
    object_identity_id: Optional[str] = Field(None, description="Persistent physical ID if resolved, else null")
    class_name: str = Field(..., description="Everyday object category e.g. laptop, phone, bottle")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp")
    bbox_xyxy: Tuple[int, int, int, int] = Field(..., description="[x1, y1, x2, y2]")
    embedding_id: Optional[str] = Field(None, description="Vector ID in Qdrant/FAISS")
    snapshot_uri: str = Field(..., description="URI to context scene frame snapshot")
    crop_uri: Optional[str] = Field(None, description="URI to representative object crop")
    event_label: str = Field(
        default="observed",
        description="Event type: first_seen, observed, moved, disappeared, reappeared, last_seen"
    )

    @property
    def formatted_time(self) -> str:
        try:
            dt = datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %H:%M:%S")
        except Exception:
            return self.timestamp_utc


class QueryRequest(BaseModel):
    """Natural language or structured query request."""
    question: str
    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    require_evidence: bool = True


class QueryResponse(BaseModel):
    """Evidence-backed visual memory response."""
    question: str
    answer: str
    object_class: Optional[str] = None
    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    timestamp_utc: Optional[str] = None
    time_ago: Optional[str] = None
    confidence: Optional[float] = None
    event_id: Optional[str] = None
    snapshot_uri: Optional[str] = None
    crop_uri: Optional[str] = None
    is_uncertain: bool = False
    disclaimer: Optional[str] = None
    sightings_count: Optional[int] = None
    all_known_objects: Optional[List[str]] = None


class SearchFilter(BaseModel):
    """Filter parameters for visual memory search."""
    object_class: Optional[str] = None
    query_text: Optional[str] = None
    camera_id: Optional[str] = None
    zone_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = 50


class TimelineEntry(BaseModel):
    """A timeline event representing appearance, gap, or reappearance."""
    id: str
    timestamp_utc: str
    time_display: str
    event_type: str  # 'first_seen', 'observed', 'moved', 'disappeared', 'reappeared', 'last_seen'
    class_name: str
    camera_id: str
    zone_id: str
    track_id: str
    confidence: float
    snapshot_uri: str
    crop_uri: Optional[str] = None
    gap_duration_sec: Optional[int] = None
    note: Optional[str] = None


class CameraInfo(BaseModel):
    """Registered camera metadata."""
    camera_id: str
    name: str
    zone_id: str
    status: str  # 'online', 'degraded', 'offline'
    fps: float
    resolution: str
    stream_url: str
    last_frame_time: Optional[str] = None


class WorkerStatus(BaseModel):
    """Status of an independent architecture service."""
    worker_id: str
    name: str
    status: str  # 'healthy', 'busy', 'idle', 'degraded'
    fps: float
    latency_ms: float
    queue_lag: int
    processed_count: int
    last_heartbeat: str


class SystemObservability(BaseModel):
    """System-wide telemetry and performance metrics."""
    cpu_usage_pct: float
    memory_usage_pct: float
    gpu_usage_pct: Optional[float] = None
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_events: int
    unique_objects_count: int
    active_tracks: int
    active_cameras: int
    dropped_frames: int
    queue_lag: int
    workers: List[WorkerStatus]
