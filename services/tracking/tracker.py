"""
Visual Memory AI — Tracking Service
BoT-SORT / ByteTrack multi-object tracking for short-term identity continuity.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO
from packages.shared_types.models import TrackItem
from services.detection.detector import DEFAULT_OBJECT_CLASSES


class TrackingService:
    """
    Decoupled tracking service maintaining short-term track IDs within a stream.
    Note: Per design requirements, track_id is scoped to current stream/session
    and is not treated as permanent physical identity without verification.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "botsort.yaml",
        confidence: float = 0.40,
        allowed_classes: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.confidence = confidence
        self.allowed_classes = allowed_classes or DEFAULT_OBJECT_CLASSES

        self.model = YOLO(model_name)
        self._name_to_id: Dict[str, int] = {v: k for k, v in self.model.names.items()}
        self._target_class_ids: List[int] = [
            self._name_to_id[c] for c in self.allowed_classes if c in self._name_to_id
        ]

        # In-memory track history: track_id -> list of (x, y, timestamp, zone)
        self.track_history: Dict[int, List[Tuple[int, int, str, str]]] = {}
        # Track initial observation time and last observation time
        self.track_lifecycle: Dict[int, Dict[str, Any]] = {}

        print(f"[TrackingService] Initialized with {tracker_type}")

    def track(
        self,
        frame: np.ndarray,
        camera_id: str = "camera_01",
        default_zone: str = "study_desk"
    ) -> List[TrackItem]:
        """
        Run tracking on a single frame and return persistent short-term tracks.
        """
        h, w = frame.shape[:2]
        now_utc = datetime.now(timezone.utc).isoformat()

        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self._target_class_ids,
            tracker=self.tracker_type,
            persist=True,
            verbose=False,
        )

        tracks: List[TrackItem] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes
            has_ids = boxes.id is not None
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                if has_ids:
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    # Fallback track ID when tracker hasn't assigned a persistent ID yet
                    track_id = 1000 + (i % 500)

                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                conf = float(boxes.conf[i].cpu().numpy())

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Compute spatial zone
                zone_h = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
                zone_id = f"{default_zone}_{zone_h}"

                track_item = TrackItem(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=round(conf, 3),
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    zone_id=zone_id,
                    timestamp_utc=now_utc,
                    class_id=class_id,
                )
                tracks.append(track_item)

                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                    self.track_lifecycle[track_id] = {
                        "first_seen": now_utc,
                        "class_name": class_name,
                        "camera_id": camera_id,
                    }
                self.track_history[track_id].append((cx, cy, now_utc, zone_id))
                self.track_lifecycle[track_id]["last_seen"] = now_utc
                self.track_lifecycle[track_id]["last_zone"] = zone_id

                # Cap history
                if len(self.track_history[track_id]) > 100:
                    self.track_history[track_id] = self.track_history[track_id][-100:]

        return tracks

    def load_model(self, model_path: str):
        """Dynamically load new YOLO model weights (e.g. custom trained weights)."""
        print(f"[TrackingService] Loading new model weights from: {model_path}")
        self.model_name = model_path
        self.model = YOLO(model_path)
        self._name_to_id = {v: k for k, v in self.model.names.items()}
        self._target_class_ids = [
            self._name_to_id[c] for c in self.allowed_classes if c in self._name_to_id
        ]
        self.track_history.clear()
        self.track_lifecycle.clear()
        print(f"[TrackingService] Successfully loaded: {model_path}")

    def get_track_history(self, track_id: int) -> List[Tuple[int, int, str, str]]:
        return self.track_history.get(track_id, [])

    def reset(self):
        self.track_history.clear()
        self.track_lifecycle.clear()
        self.model = YOLO(self.model_name)
